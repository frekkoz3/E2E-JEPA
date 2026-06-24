r"""
  _____ ____  _____          _ _____ ____   _    
 | ____|___ \| ____|        | | ____|  _ \ / \   
 |  _|   __) |  _| _____ _  | |  _| | |_) / _ \  
 | |___ / __/| |__|_____| |_| | |___|  __/ ___ \ 
 |_____|_____|_____|     \___/|_____|_| /_/   \_\
"""
import yaml
import time
import cv2 #will need for registration maybe
import argparse
import numpy as np
import torch
import shutil
from pathlib import Path

from src.jepa.transformers import VisualTransformer, Transformer
from src.game.snake import SnakeEnv, TOTAL_HEIGHT, GRID_HEIGHT, WIDTH, CELL_SIZE, BAR_HEIGHT
from src.policy.policy import Policy, PolicyDQN, PolicyPPO
from src.jepa.e2e_jepa import *
from src.utils.utils import *

action_dim = 4
EMBED_DIM = 64

GPU = "cuda"
CPU = "cpu"
XPU = "xpu"

if __name__ == '__main__':
    """
        Quick usage (from the root of the project)

        py -m src.train.train --config PATH-TO-CONFIG --run-name RUN-NAME
    """   
    parser = argparse.ArgumentParser(description="Active E2E-JEPA Training for Snake Game")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--run-name", type=str, required=True, help="name of the run.")

    args = parser.parse_args()

    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = flat_config(config)

    run_name = args.run_name
    default_save_location = f"models/e2e/{run_name}/"
    source = config_path
    destination = f"{default_save_location}/config.yaml"
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, destination) # we preserve the exact configuration

    # Training parameters
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    total_epochs = config.get("n_epochs", 200)
    steps_per_epoch = config.get("steps_per_epoch", 256)
    batch_size = config.get("batch_size", 32)
    refresh_buffer = config.get("refresh_buffer_freq", 8)
    where_save = config.get("save_path", default_save_location)
    epochs_per_checkpoint = config.get("epochs_per_checkpoint", total_epochs//2)
    clean_checkpoints = config.get("clean_checkpoints", True)

    # Environment parameters
    action_dim = config.get("action_dim", 4)
    cell_size = CELL_SIZE
    grid_width, grid_height = WIDTH, GRID_HEIGHT
    width = WIDTH
    total_height = TOTAL_HEIGHT
    img_size = (width, total_height, 3)
    n_obstacles = config.get("n_obstacles", 10)
    fps = config.get("fps", 10)
    difficulty = config.get("difficulty", 2)

    # Encoder parameters
    embed_dim = config.get("embedding_dim", 64)
    enc_mlp_dim = config.get("enc_mlp_dim", 256)
    enc_n_heads = config.get("enc_n_heads", 4)
    enc_depth = config.get("enc_depth", 3)
    enc_patch_size = config.get("enc_patch_size", 6)
    enc_patch_in_channels = config.get("enc_path_in_channels", 3)

    # Predictor parameters
    pred_hidden_dim = config.get("pred_hidden_dim", 64)
    pred_cond_dim = config.get("pred_cond_dim", 1)
    pred_mlp_dim = config.get("pred_mlp_dim", 256)
    pred_n_heads = config.get("pred_n_heads", 4)
    pred_depth = config.get("pred_depth", 3)
    use_adaLN = config.get("use_adaLN", True)
    dropout = config.get("dropout", 0.0)

    metrics_collector = MetricsCollector()

    trainer = E2EJEPA(
        env=SnakeEnv(**config),
        encoder=VisualTransformer(img_size=img_size,
                                  embed_dim=embed_dim,
                                  patch_size=cell_size,
                                  mlp_dim=enc_mlp_dim,
                                  num_heads=enc_n_heads,
                                  depth=enc_depth).to(device=device),
        predictor=Transformer(input_dim=embed_dim,
                              hidden_dim=pred_hidden_dim,
                              cond_dim=pred_cond_dim,
                              output_dim=embed_dim,
                              depth=pred_depth,
                              num_heads=pred_n_heads,
                              mlp_dim=pred_mlp_dim,
                              use_adaLN=use_adaLN).to(device=device),
        policy=eval(config["pol_type"])(**config),
        action_dim=action_dim,
        embed_dim=embed_dim
    )
    
    env = SnakeEnv(render_mode="rgb_array", observation_type="image", difficulty=difficulty)
    x_t, _ = env.reset()
    x_t = torch.tensor(np.expand_dims(x_t, 0)).float().to(device=device)

    for epoch in range(total_epochs):
        if epoch % refresh_buffer == 0:
            trainer.buffer.refresh()

        trainer.encoder.eval()
        trainer.predictor.eval()
        
        with torch.no_grad():

            for step in range(steps_per_epoch):
                z_t = trainer.encoder(x_t)[:, 0, :]
                
                # Choose action actively using current model state
                a_t, _ = trainer.get_action(z_t.detach().unsqueeze(0))
                
                # Step the real environment
                x_tp1, r_t, done, _, info = env.step(a_t)

                # Check if the action is actually legit
                if info:
                    a_t = info["act"]

                x_tp1 = torch.tensor(np.expand_dims(x_tp1, 0)).float().to(device=device)

                z_tp1 = trainer.encoder(x_tp1)[:, 0, :]

                # Stream data into the experience buffer seamlessly
                trainer.buffer.push(x_t.squeeze(0).to(device=CPU),
                                    torch.tensor(a_t).float().to(device=CPU),
                                    r_t,
                                    x_tp1.squeeze(0).to(device=CPU),
                                    float(done))
                
                if done:
                    # Reset        
                    x_t, _ = env.reset()
                    x_t = torch.tensor(np.expand_dims(x_t, 0)).float().to(device=device)
                else:
                    # Move forward
                    x_t = x_tp1
            
        # Optimize over collected transitions at the end of the epoch step block
        metrics = trainer.update_parameters(batch_size, device=device)
        if metrics:
            print(f"Epoch {epoch} Metrics -> "
                  f"Loss: {metrics['total_loss']:.8f} | "
                  f"Pred: {metrics['pred_loss']:.8f} | "
                  f"Policy : {metrics['policy_loss']:.8f} | "
                  f"SigReg: {metrics['sigreg_loss']:.8f}")
            metrics_collector.add_metric(metrics)
            metrics_collector.save_metrics(f"{where_save}metrics.csv", append = (epoch > 0))

        # Dynamically saving checkpoints and removing them
        if epoch%epochs_per_checkpoint == 0:
            save_results(f"{where_save}{epoch//epochs_per_checkpoint}.pkl",
                         trainer.predictor,
                         trainer.encoder,
                         trainer.policy.network)
            if clean_checkpoints:
                old = Path(f"{where_save}{epoch//epochs_per_checkpoint - 1}.pkl")
                if old.exists():
                    old.unlink()
    
    if clean_checkpoints:
        import math
        # Removing the last checkpoint searching for the correct index with floor and ceil functions
        old_ceil = Path(f"{where_save}{math.ceil(total_epochs//epochs_per_checkpoint)}.pkl")
        old_floor = Path(f"{where_save}{math.floor(total_epochs//epochs_per_checkpoint)}.pkl")
        if old_ceil.exists():
            old_ceil.unlink()
        if old_floor.exists():
            old_floor.unlink()

    save_results(f"{where_save}final.pkl",  trainer.predictor, trainer.encoder, trainer.policy.network)
    metrics_collector.save_metrics(f"{where_save}metrics.csv")