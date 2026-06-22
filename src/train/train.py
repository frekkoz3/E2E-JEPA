import yaml

from src.jepa.transformers import VisualTransformer, Transformer
from src.policy.policy import Policy, PolicyDQN, PolicyPPO
from src.game.snake import SnakeEnv, TOTAL_HEIGHT, WIDTH, CELL_SIZE
from src.jepa.e2e_jepa import *
from src.utils.utils import *
import time
import cv2
import argparse
import numpy as np
import uuid
import torch

ACTION_DIM = 4
IMG_SIZE = (WIDTH, TOTAL_HEIGHT, 3)
EMBED_DIM = 64

GPU = "cuda"
CPU = "cpu"
XPU = "xpu"

DEFAULT_SAVE_LOCATION = f"models/{time.time()} - {uuid.uuid1()}.pkl"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Active E2E-JEPA Training for Snake Game")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--where", type=str, required=False, help="Where are the stored models' weights.", default=DEFAULT_SAVE_LOCATION)

    parser.add_argument("--total_epochs", type=int, required=False, default=5000, help="Total number of epochs for training")
    parser.add_argument("--steps_per_epoch", type=int, required=False, default=256, help="Number of steps per epoch")
    parser.add_argument("--batch_size", type=int, required=False, default=32, help="Batch size for training")
    parser.add_argument("--refresh_buffer", type=int, required=False, default=8, help="Epoch interval to refresh the buffer")
    
    args = parser.parse_args()

    # Assign parsed arguments to variables
    TOTAL_EPOCHS = args.total_epochs
    STEPS_PER_EPOCH = args.steps_per_epoch
    BATCH_SIZE = args.batch_size
    REFRESH_BUFFER = args.refresh_buffer

    config_path = args.config
    where_save = args.where
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = flat_config(config)

    trainer = ActiveE2EJEPATrainer(
        env=SnakeEnv(**config),
        encoder=VisualTransformer(img_size=IMG_SIZE, embed_dim=EMBED_DIM, patch_size=CELL_SIZE, mlp_dim=256, num_heads=4, depth=3).to(device=GPU), 
        predictor=Transformer(input_dim=EMBED_DIM, hidden_dim=EMBED_DIM, cond_dim=1, output_dim=EMBED_DIM, depth=3, num_heads=4, mlp_dim=256, use_adaLN=True).to(device=GPU),
        policy=eval(config["name"])(**config),
        action_dim=ACTION_DIM,
        embed_dim=EMBED_DIM
    )
    
    env = SnakeEnv(render_mode="rgb_array", observation_type="image")
    x_t, _ = env.reset()
    x_t = torch.tensor(np.expand_dims(x_t, 0)).float().to(device=GPU)

    for epoch in range(TOTAL_EPOCHS):
        if epoch % REFRESH_BUFFER == 0:
            trainer.buffer.refresh()

        trainer.encoder.eval()
        trainer.predictor.eval()
        
        with torch.no_grad():

            for step in range(STEPS_PER_EPOCH):
                z_t = trainer.encoder(x_t)[:, 0, :]
                
                # Choose action actively using current model state
                a_t, _ = trainer.get_action(z_t.detach().unsqueeze(0))
                
                # Step the real environment
                x_tp1, r_t, done, _, info = env.step(a_t)

                # Check if the action is actually legit
                if info:
                    a_t = info["act"]

                x_tp1 = torch.tensor(np.expand_dims(x_tp1, 0)).float().to(device=GPU)

                z_tp1 = trainer.encoder(x_tp1)[:, 0, :]

                # Stream data into the experience buffer seamlessly
                trainer.buffer.push(x_t.squeeze(0).to(device=CPU), torch.tensor(a_t).float().to(device=CPU), r_t, x_tp1.squeeze(0).to(device=CPU), float(done))
                
                if done:
                    # Reset        
                    x_t, _ = env.reset()
                    x_t = torch.tensor(np.expand_dims(x_t, 0)).float().to(device=GPU)
                else:
                    # Move forward
                    x_t = x_tp1
            
        # Optimize over collected transitions at the end of the epoch step block
        metrics = trainer.update_parameters(BATCH_SIZE, epoch, TOTAL_EPOCHS, device=GPU)
        if metrics:
            print(f"Epoch {epoch} Metrics -> Loss: {metrics['total_loss']:.8f} | Pred: {metrics['pred_loss']:.8f} | Policy : {metrics['policy_loss']:.8f}")

        save_results(where_save, trainer.predictor, trainer.encoder, trainer.policy.network)