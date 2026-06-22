r"""
  _____ ____  _____          _ _____ ____   _    
 | ____|___ \| ____|        | | ____|  _ \ / \   
 |  _|   __) |  _| _____ _  | |  _| | |_) / _ \  
 | |___ / __/| |__|_____| |_| | |___|  __/ ___ \ 
 |_____|_____|_____|     \___/|_____|_| /_/   \_\
"""
import yaml

from src.jepa.transformers import VisualTransformer, Transformer
from src.policy.policy import Policy, PolicyDQN, PolicyPPO
from src.game.snake import SnakeEnv, TOTAL_HEIGHT, WIDTH, CELL_SIZE
from src.jepa.e2e_jepa import *
from src.utils.utils import *
import cv2
import argparse
import numpy as np

TOTAL_EPOCHS = 16
STEPS_PER_EPOCH = 256
BATCH_SIZE = 32
ACTION_DIM = 4
REFRESH_BUFFER = 8
IMG_SIZE = (WIDTH, TOTAL_HEIGHT,3)
EMBED_DIM = 64

GPU = "cuda"
CPU = "cpu"
XPU = "xpu"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Active E2E-JEPA Training for Snake Game")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = flat_config(config)

    
    trainer = ActiveE2EJEPATrainer(
        env=SnakeEnv(**config),
        encoder=VisualTransformer(img_size=IMG_SIZE, embed_dim=EMBED_DIM, patch_size=CELL_SIZE, mlp_dim=256, num_heads=4, depth = 3).to(device=GPU), 
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
                print(f"STEP : {step}")

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

                print(f"z_t dim : {z_t.shape} ; z_tp1 dim : {z_tp1.shape}")

                # Stream data into the experience buffer seamlessly
                trainer.buffer.push(x_t.squeeze(0), z_t.detach().squeeze(0), torch.tensor(a_t).float().to(device=GPU), r_t, x_tp1.squeeze(0), z_tp1.detach().squeeze(0), float(done))
                
                if done:
                    print("Dead")        
                    # Reset        
                    x_t, _ = env.reset()
                    x_t = torch.tensor(np.expand_dims(x_t, 0)).float().to(device=GPU)
                else:
                    print(f"action : {a_t} , reward : {r_t}")
                    # Move forward
                    x_t = x_tp1
            
            print(f"EPOCH : {epoch}")

        # Optimize over collected transitions at the end of the epoch step block
        metrics = trainer.update_parameters(BATCH_SIZE, epoch, TOTAL_EPOCHS, device = GPU)
        print(epoch)
        if metrics:
            print(f"Epoch {epoch} Metrics -> Loss: {metrics['total_loss']:.4f} | Pred: {metrics['pred_loss']:.4f}")
