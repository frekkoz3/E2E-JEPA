r"""
  _____ ____  _____          _ _____ ____   _    
 | ____|___ \| ____|        | | ____|  _ \ / \   
 |  _|   __) |  _| _____ _  | |  _| | |_) / _ \  
 | |___ / __/| |__|_____| |_| | |___|  __/ ___ \ 
 |_____|_____|_____|     \___/|_____|_| /_/   \_\
"""
import yaml
import cv2
import torch
import argparse
import numpy as np

from src.jepa.transformers import VisualTransformer, Transformer
from src.jepa.e2e_jepa import *
from src.game.snake import SnakeEnv, TOTAL_HEIGHT, GRID_HEIGHT, WIDTH, CELL_SIZE, BAR_HEIGHT
from src.policy.policy import Policy, PolicyDQN, PolicyPPO
from src.utils.utils import flat_config

if __name__ == '__main__': 
    """
        Quick usage (from the root of the project)

        py -m src.validation.vis --config PATH-TO-CONFIG --weights PATH-TO-WEIGHTS

        make sure that the config is referring to the config used to train the weights you're loading
    """   
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--weights", required=True, help="Path to final.pkl")
    parser.add_argument("--episodes", type=int, default=10)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config = flat_config(config)

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Environment parameters
    action_dim = config.get("action_dim", 4)
    cell_size = CELL_SIZE
    grid_width, grid_height = WIDTH, GRID_HEIGHT
    width = WIDTH
    total_height = TOTAL_HEIGHT
    img_size = (width, total_height, 3)
    n_obstacles = config.get("n_obstacles", 10)
    fps = config.get("fps", 10)

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

    model = E2EJEPA(
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

    load_results(args.weights, model.predictor, model.encoder, model.policy.network)

    env = SnakeEnv(render_mode="human", observation_type="image")

    for episode in range(args.episodes):

        x_t, _ = env.reset()
        x_t = torch.tensor(np.expand_dims(x_t, 0)).float().to(device=device)

        done, trunc = False, False

        while not done and not trunc:

            model.encoder.eval()
            model.predictor.eval()

            with torch.no_grad():

                z_t = model.encoder(x_t)[:, 0, :]

                # Choose the action with the greedy policy
                a_t, _ = model.get_action(z_t.detach().unsqueeze(0), greedy=True)
                
                # Step the real environment
                x_tp1, r_t, done, _, info = env.step(a_t)
                x_tp1 = torch.tensor(np.expand_dims(x_tp1, 0)).float().to(device=device)

                # Check if the action is actually legit
                if info:
                    a_t = info["act"]

                # Move forward
                x_t = x_tp1

                env.render()
            