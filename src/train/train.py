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

from torch.optim import lr_scheduler, Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR

from src.jepa.transformers import VisualTransformer, Predictor
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
    try:
        shutil.copy(source, destination) # we preserve the exact configuration
    except Exception as e:
        print(e)

    # Training parameters
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    total_epochs = config.get("n_epochs", 200)
    steps_per_epoch = config.get("steps_per_epoch", 256)
    batch_size = config.get("batch_size", 32)
    refresh_buffer = config.get("refresh_buffer_freq", 8)
    where_save = config.get("save_path", default_save_location)
    epochs_per_checkpoint = config.get("epochs_per_checkpoint", total_epochs//2)
    clean_checkpoints = config.get("clean_checkpoints", True)
    load_checkpoints = config.get("load_checkpoints", False)
    load_checkpoints_path = config.get("load_checkpoints_path", f"{where_save}final.pkl")
    starting_epoch = config.get("starting_epoch", 0)

    # Optimizer parameters
    optimizer = config.get("optimizer", "Adam")
    lr_init = config.get("lr_init", 1e-4)
    lr_scheduler = config.get("lr_scheduler", "ExponentialLR")
    lr_step_size = config.get("lr_step_size", 10)
    lr_gamma = config.get("lr_gamma", 0.9)

    # Sequence parameters
    horizon = config.get("horizon", 3)
    history_size = config.get("history_size", 4)
    seq_len = history_size + horizon

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
    rescale_frames = config.get("rescale_frames", False)
    using_heuristic = config.get("using_heuristic", False)

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
        predictor=Predictor(embed_dim=embed_dim,
                            hidden_dim=pred_hidden_dim,
                            action_dim=action_dim,
                            depth=pred_depth,
                            num_heads=pred_n_heads,
                            mlp_dim=pred_mlp_dim,
                            use_adaLN=use_adaLN,
                            dropout=dropout).to(device=device),
        policy=eval(config["pol_type"])(**config),
        action_dim=action_dim,
        embed_dim=embed_dim,
        optimizer_name = optimizer,
        lr_init = lr_init,
        lr_scheduler = lr_scheduler,
        lr_gamma = lr_gamma,
        device=device,
        horizon=horizon,
    )
    if load_checkpoints:
        checkpoint_name = config.get("last_checkpoint")
        load_results(f"{default_save_location}/{checkpoint_name}",
                     trainer.predictor,
                     trainer.encoder,
                     trainer.policy.network,
                     trainer.optimizer,
                     trainer.scheduler,
                     trainer.policy.optimizer,
                     trainer.policy.scheduler,
                     trainer.policy.epsilon_strategy)
    
    env = SnakeEnv(**config)
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
                a_t, _ = trainer.get_action(z_t.detach().unsqueeze(1)) if not using_heuristic else env._heuristic_action()
                
                # Step the real environment
                x_tp1, r_t, done, _, info = env.step(a_t)

                a_t = F.one_hot(torch.tensor(a_t), action_dim).float() # one hot encoding of the action instead of one-dimensional one

                x_tp1 = torch.tensor(np.expand_dims(x_tp1, 0)).float().to(device=device)

                # Stream data into the experience buffer seamlessly
                trainer.buffer.push(x_t.squeeze(0).to(device=CPU),
                                    a_t.to(device=CPU),
                                    r_t,
                                    float(done))
                
                if done:
                    # Reset        
                    x_t, _ = env.reset()
                    x_t = torch.tensor(np.expand_dims(x_t, 0)).float().to(device=device)
                else:
                    # Move forward
                    x_t = x_tp1

        # Optimize over collected transitions at the end of the epoch step block
        seq_batch = trainer.buffer.sample_sequences(batch_size = batch_size, seq_len=seq_len, device=device)

        metrics = trainer.update_parameters(seq_batch) if seq_batch is not None else None

        if metrics:
            # Add also Learning rate and epsilon parameter to the metrics dictionary
            metrics.update({
                "learning_rate": trainer.optimizer.param_groups[0]['lr'],
                "epsilon": trainer.policy.epsilon_strategy.eps
            })
            print(metrics)
            metrics_collector.add_metric(metrics)
            if not load_checkpoints:
                metrics_collector.save_metrics(f"{where_save}metrics.csv", append = (epoch > 0))
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                config["save"]["load_checkpoints"] = True
                with open(config_path, 'w') as f:
                    yaml.safe_dump(config, f)
            else:
                metrics_collector.save_metrics(f"{where_save}metrics.csv", append = True)

        # Adjust learning rate scheduler
        if epoch%lr_step_size == 0:
            trainer.scheduler.step()

        # Dynamically saving checkpoints and removing them
        if epoch%epochs_per_checkpoint == 0:
            save_results(f"{where_save}latest.pkl",
                         trainer.predictor,
                         trainer.encoder,
                         trainer.policy.network,
                         trainer.optimizer,
                         trainer.scheduler,
                         trainer.policy.optimizer,
                         trainer.policy.scheduler)


            if clean_checkpoints:
                old = Path(f"{where_save}{(starting_epoch+epoch)//epochs_per_checkpoint - 1}.pkl")
                if old.exists():
                    old.unlink()
    
    if clean_checkpoints:
        # Removing the last checkpoint searching for the correct index with floor and ceil functions
        old_ceil = Path(f"{where_save}{(starting_epoch + total_epochs)//epochs_per_checkpoint-1}.pkl")
        old_floor = Path(f"{where_save}{(starting_epoch + total_epochs)//epochs_per_checkpoint+1}.pkl")
        if old_ceil.exists():
            old_ceil.unlink()
        if old_floor.exists():
            old_floor.unlink()

    save_results(f"{where_save}final.pkl",  trainer.predictor, trainer.encoder, trainer.policy.network, trainer.optimizer,
                 trainer.scheduler, trainer.policy.optimizer, trainer.policy.scheduler)