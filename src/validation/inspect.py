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
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

from src.jepa.transformers import VisualTransformer, Predictor
from src.jepa.e2e_jepa import *
from src.game.snake import SnakeEnv, TOTAL_HEIGHT, GRID_HEIGHT, WIDTH, CELL_SIZE, BAR_HEIGHT, GRID_WIDTH
from src.policy.policy import Policy, PolicyDQN, PolicyPPO
from src.utils.utils import flat_config
from src.validation.clustering import plot_clusters

from torch.optim import lr_scheduler, Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR

if __name__ == '__main__': 
    """
        Quick usage (from the root of the project)

        py -m src.validation.inspect --config PATH-TO-CONFIG --weights PATH-TO-WEIGHTS

        make sure that the config is referring to the config used to train the weights you're loading
    """   
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--weights", required=True, help="Path to final.pkl")
    parser.add_argument("--n_samples", type=int, default=1000)

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

    # Optimizer
    optimizer = config.get("optimizer", "Adam")
    lr_init = config.get("lr_init", 1e-4)
    lr_scheduler = config.get("lr_scheduler", "ExponentialLR")
    lr_step_size = config.get("lr_step_size", 10)
    lr_gamma = config.get("lr_gamma", 0.9)

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
    horizon = config.get("horizon", 1)

    model = E2EJEPA(
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

    load_results(args.weights, 
                 model.predictor,
                 model.encoder,
                 model.policy.network,
                 model.optimizer,
                 model.scheduler,
                 model.policy.optimizer,
                 model.policy.scheduler,
                 model.policy.epsilon_strategy)

    # Experiment 1: Visualize the distribution of selected actions
    selected_actions = []

    for sample in range(args.n_samples):
        with torch.no_grad():

            z_t = torch.rand(1, embed_dim).to(device=device)  # model.encoder(x_t)[:, 0, :]

            # Choose the action with the greedy policy
            a_t, _ = model.get_action(z_t.detach().unsqueeze(0), greedy=True)

            selected_actions.append(a_t)

    # Compute statistics on selected actions
    action_counts = np.bincount(selected_actions, minlength=action_dim)

    print("Action distribution:")
    for action, count in enumerate(action_counts):
        print(f"Action {action}: {count} times ({(count / args.n_samples) * 100:.2f}%)")

    # Experiment 2: Cluster embeddings from random states and visualize the clusters
    env = SnakeEnv(render_mode="rgb_array", observation_type="image", difficulty=config.get("difficulty"))
    
    all_embeddings = []
    all_frames = []

    for _ in range(4):  # Generate 25 random states
        # Generate two random ints for the apple position
        apple_x = np.random.randint(0, GRID_WIDTH)
        apple_y = np.random.randint(0, GRID_HEIGHT)

        for _ in range(25):
            x_t = env._generate_random_frame((apple_x, apple_y))
            all_frames.append(np.transpose(x_t, (1, 2, 0)))
            x_t_tensor = torch.from_numpy(x_t).unsqueeze(0).float().to(device)
            z_t = model.encoder(x_t_tensor)
            all_embeddings.append(z_t[:, 0, :].detach().cpu().numpy())

    # Clustering embeddings
    embeddings = np.vstack(all_embeddings)
    n_clusters = min(4, len(embeddings))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)

    plot_clusters(all_frames, 4, 4, labels, "imgs/cluster_visualization_apples.png")

    all_embeddings = []
    all_frames = []
    location_labels = []

    for loc_idx in range(4):  # Generate 25 random states
        # Generate two random ints for the apple position
        snake_x = np.random.randint(0, GRID_WIDTH)
        snake_y = np.random.randint(0, GRID_HEIGHT)

        for _ in range(25):
            x_t = env._generate_random_frame(snake_pos=(snake_x, snake_y), snake_dir=(0,1))
            all_frames.append(np.transpose(x_t, (1, 2, 0)))
            x_t_tensor = torch.from_numpy(x_t).unsqueeze(0).float().to(device)
            z_t = model.encoder(x_t_tensor)
            all_embeddings.append(z_t[:, 0, :].detach().cpu().numpy())
            location_labels.append(loc_idx)

    # Clustering embeddings
    embeddings = np.vstack(all_embeddings)
    n_clusters = min(4, len(embeddings))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)

    plot_clusters(all_frames, 4, 4, labels, "imgs/cluster_visualization_snakes.png")

    cm = confusion_matrix(location_labels, labels)
    print("\nConfusion matrix (rows=snake location, cols=cluster):")
    print(f"{'':>12}" + "".join(f"  C{c}" for c in range(n_clusters)))
    for loc_idx, row in enumerate(cm):
        print(f"  Location {loc_idx}  " + "".join(f"{v:4d}" for v in row))
