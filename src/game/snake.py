r"""
  _____ ____  _____          _ _____ ____   _    
 | ____|___ \| ____|        | | ____|  _ \ / \   
 |  _|   __) |  _| _____ _  | |  _| | |_) / _ \  
 | |___ / __/| |__|_____| |_| | |___|  __/ ___ \ 
 |_____|_____|_____|     \___/|_____|_| /_/   \_\
"""
import os # needed for hpc cluster 
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

CELL_SIZE = 35
GRID_WIDTH, GRID_HEIGHT = 20, 20
WIDTH = GRID_WIDTH * CELL_SIZE
GAME_HEIGHT = GRID_HEIGHT * CELL_SIZE
BAR_HEIGHT = 2*CELL_SIZE  # Height of the top status bar
TOTAL_HEIGHT = GAME_HEIGHT + BAR_HEIGHT
N_OBSTACLES = 10
FPS = 10

RESOURCES_PATH = "src/game/resources/"

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None, max_step=100, observation_type="grid", difficulty=0, rescale_frames : bool = False, **kwargs):
        """
        Snake Environment

        render_mode (str) : "human" (via Pygame), "rgb_array" (headless)
        observation_type (str) : "grid", "image"
        difficulty (int) :  
            0 means that the snake does not grow
            1 means the standard game
            2 means that there are some obstacles
            3 means that the obstacle can change positions and there are fake apples
        """
        assert GRID_HEIGHT >= 10
        assert GRID_WIDTH >= 10
        self.action_space = spaces.Discrete(4)  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.render_mode = render_mode
        self.observation_type = observation_type
        self.rescale_frames = rescale_frames
        self.difficulty = max(0, min(difficulty, 3)) # Clamp difficulty between 0 and 3
        # Difficulty 0 means that the snake does not grow
        # Difficulty 1 means the standard game
        # Difficulty 2 means that there are some obstacles
        # Difficulty 3 means that the obstacle can change positions and there are fake apples

        if self.observation_type == "image":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(TOTAL_HEIGHT, WIDTH, 3), dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=4, shape=(GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8
            )
            
        if kwargs == {}:
            self.reward_food = 20
            self.reward_death = -5
            self.reward_step = -0.5
        else:
            self.reward_food = kwargs["reward_food"]
            self.reward_death = kwargs["reward_death"]
            self.reward_step = kwargs["reward_step"]

        self.max_step = max_step

        self.reset()        

    def _load_and_scale_sprites(self, update = False):
        """Loads assets and resizes them to match your environment's CELL_SIZE."""
        def load_sp(path):
            img = pygame.image.load(path).convert_alpha()
            return pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE))
        
        self.sprites = {}
        colors = ["red", "green", "yellow"]
        components = ["head", "body", "tail", "ul", "ur", "dl", "dr", "apple"]
        
        for color in colors:
            for comp in components:
                self.sprites[f"{color}_{comp}"] = load_sp(f"{RESOURCES_PATH}{color}/{comp}.png")

        # Obstacles
        obstacles = ["bomb", "rock", "skull"]
        if self.difficulty > 2:
            obstacles.append("purple_apple")
            obstacles.append("violet_apple")

        for i, obs in enumerate(obstacles):
            self.sprites[f"obstacle_{i}"] = load_sp(f"{RESOURCES_PATH}obstacles/{obs}.png")

        grass_categories = ["neutral", "flower", "white"]

        for piece in grass_categories:
            i = 1
            while os.path.exists(f"{RESOURCES_PATH}grass/{piece}_{i}.png"):
                self.sprites[f"{piece}_{i}"] = load_sp(f"{RESOURCES_PATH}grass/{piece}_{i}.png")
                i += 1

        self._blit_background(update = update)
        self.sprites_loaded = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.window = None
        self.canvas = None  
        self.clock = None
        self.font = None
        self.total_step = 0
        self.score = 0         
        
        self.sprites_loaded = False

        self.snake = [(random.randrange(2, GRID_WIDTH - 2), random.randrange(2, GRID_HEIGHT-2))]
        start_x = self.snake[0][0]
        start_y = self.snake[0][1]

        center_x, center_y = GRID_WIDTH // 2, GRID_HEIGHT // 2
        
        if abs(start_x - center_x) > abs(start_y - center_y):
            self.direction = (1, 0) if start_x < center_x else (-1, 0)
        else:
            self.direction = (0, 1) if start_y < center_y else (0, -1)

        self.current_snake_color = self._random_color()
        self.current_apple_type = self._random_apple()
        
        self.obstacles = []

        self._place_food()
        if self.difficulty > 1:
            self._place_obstacles(update = False)

        self.done = False
        self.score = 0
        self.total_step = 0
        self.info = {}
        return self._get_obs(), self.info
    
    def death_state(self):
        if self.observation_type == "image":
            all_black = np.zeros((TOTAL_HEIGHT, WIDTH, 3), dtype=np.uint8)
            all_black_t = np.transpose(all_black, (2, 0, 1)) # format for the ViT 
            return all_black_t
        else:
            return np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
    
    def _random_color(self):
        return random.choice(["red", "green", "yellow"])
    
    def _random_apple(self):
        return f"{self._random_color()}_apple"

    def _place_food(self):
        while True:
            self.food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if self.food not in self.snake and self.food not in self.obstacles:
                break

    def _place_obstacles(self, update = False):
        if not update:
            while len(self.obstacles) < N_OBSTACLES:
                obs = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
                if self.food != obs and obs not in self.snake and obs not in self.obstacles:
                    self.obstacles.append(obs)
        else:
            self.obstacles.pop()
            while True:
                obs = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
                if self.food != obs and obs not in self.snake and obs not in self.obstacles:
                    self.obstacles.append(obs)
                    self._blit_background(update=True)
                    break

    def _blit_background(self, update = False):
        # Grass
        grass_categories = ["neutral", "flower", "white"]
        obstacles_tiles = [k for k in self.sprites.keys() if "obstacle" in k]

        if not update:
            self.grass_background = pygame.Surface((WIDTH, GAME_HEIGHT))
            self.grass_tiles = [['' for j in range (GRID_WIDTH)] for i in range (GRID_HEIGHT)]

            neutral_tiles = [k for k in self.sprites.keys() if "neutral" in k]
            flower_tiles = [k for k in self.sprites.keys() if "flower" in k]
            white_tiles = [k for k in self.sprites.keys() if "white" in k]

            weights = [2, 6, 2]

            for row in range(GRID_HEIGHT):
                for col in range(GRID_WIDTH):
                    chosen_category = random.choices(grass_categories, weights=weights, k=1)[0]
                    
                    if chosen_category == "neutral":
                        tile = random.choice(neutral_tiles)
                    elif chosen_category == "flower":
                        tile = random.choice(flower_tiles)
                    else:
                        tile = random.choice(white_tiles)
                    
                    self.grass_tiles[row][col] = tile
                    self.grass_background.blit(self.sprites[tile], (col * CELL_SIZE, row * CELL_SIZE))

            # Obastacles
            if self.difficulty > 1:
                self.obstacles_tiles = []
                for obstacle in self.obstacles:
                    x, y = obstacle
                    tile = random.choice(obstacles_tiles)
                    self.obstacles_tiles.append(tile)
                    self.grass_background.blit(self.sprites[tile], (x * CELL_SIZE, y * CELL_SIZE) )

        else:
            # Grass
            for row in range(GRID_HEIGHT):
                for col in range(GRID_WIDTH):                    
                    tile = self.grass_tiles[row][col]
                    self.grass_background.blit(self.sprites[tile], (col * CELL_SIZE, row * CELL_SIZE))

            if self.difficulty > 1:
                # New Obstacle Tile
                self.obstacles_tiles.pop()
                tile = random.choice(obstacles_tiles)
                self.obstacles_tiles.append(tile)

                for i, obstacle in enumerate(self.obstacles):
                    x, y = obstacle
                    tile = self.obstacles_tiles[i]
                    self.grass_background.blit(self.sprites[tile], (x * CELL_SIZE, y * CELL_SIZE) )


    def _get_obs(self, done = False):
        if done:
            return self.death_state()
        if self.observation_type == "image":
            frame = self._render_frame()
            if self.render_mode == "rgb_array" and self.rescale_frames:
                # rescale pixels in [0, 1]
                frame = frame.astype(np.float32) / 255.0
            return frame
            
        grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        for x, y in self.snake:
            grid[y, x] = 3 
        head_x, head_y = self.snake[0]
        grid[head_y, head_x] = 2 
        food_x, food_y = self.food
        grid[food_y, food_x] = 1 
        if self.difficulty > 1:
            for x, y in self.obstacles:
                grid[y, x] = 4
        return grid

    def get_possible_actions(self, action):
        if action is None:
            return list(range(self.action_space.n))
        forbidden_action = {0:1, 1:0, 2:3, 3:2}[action]  
        possible_actions = [i for i in range(self.action_space.n) if i != forbidden_action]
        return possible_actions

    def get_score(self):
        return self.score

    def step(self, action):
        self.info = None
        if self.done:
            return self._get_obs(), 0.0, True, False, self.info
        
        self.total_step += 1

        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        new_dir = directions[action]

        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.direction = new_dir
        else:
            self.info = {"act" : directions.index(self.direction)}

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        if (
            new_head in self.snake or
            not (0 <= new_head[0] < GRID_WIDTH) or
            not (0 <= new_head[1] < GRID_HEIGHT) or
            new_head in self.obstacles
        ):
            self.done = True
            reward = self.reward_death
            return self._get_obs(), reward, True, False, self.info

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            reward = self.reward_food
            self.current_snake_color = self.current_apple_type.split("_")[0]
            self.current_apple_type = self._random_apple()
            self._place_food()
            if self.difficulty>2:
                self._place_obstacles(update = True)
            if self.difficulty == 0: # it does not increse the size
                self.snake.pop()
        else:
            self.snake.pop()
            reward = self.reward_step

        if self.total_step > self.max_step:
            self.done = True 
            truncated = True
            reward = 0
            return self._get_obs(), reward, self.done, truncated, self.info

        return self._get_obs(), reward, self.done, False, self.info

    def _get_rotation_angle(self, vector):
        mapping = {(0, -1): 0, (-1, 0): 90, (0, 1): 180, (1, 0): 270}
        return mapping.get(vector, 0)

    def _render_frame(self):
        """Internal worker function that draws the frame onto the canvas."""
        if self.window is None and self.canvas is None:
            pygame.init()
            pygame.font.init()

            font_path = os.path.join(RESOURCES_PATH, "font", "minecraft", "Minecraft.ttf")
            
            try:
                self.font = pygame.font.Font(font_path, 16)
            except FileNotFoundError:
                print(f"Warning: Custom font not found at {font_path}. Falling back to Arial.")
                self.font = pygame.font.SysFont("Arial", 24, bold=True)
            
            if self.render_mode == "human":
                self.window = pygame.display.set_mode((WIDTH, TOTAL_HEIGHT))
                pygame.display.set_caption("Snake Environment")
                self.clock = pygame.time.Clock()
            else:
                os.environ["SDL_VIDEODRIVER"] = "dummy"
                try:
                    self.canvas = pygame.display.set_mode((WIDTH, TOTAL_HEIGHT), pygame.HIDDEN)
                except pygame.error:
                    self.canvas = pygame.display.set_mode((WIDTH, TOTAL_HEIGHT))

        if not self.sprites_loaded:
            self._load_and_scale_sprites(update = False)

        paint_surface = self.window if self.render_mode == "human" else self.canvas
        
        paint_surface.fill((40, 40, 40))
        
        total_seconds = self.total_step // self.metadata["render_fps"]
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        time_str = f"{minutes:02d}.{seconds:02d}"

        diff_text = self.font.render(f"Difficulty {self.difficulty}", True, (255, 255, 255))
        score_text = self.font.render(f"Score {self.score}", True, (255, 215, 0))
        time_text = self.font.render(f"Time {time_str}", True, (255, 255, 255))
        
        text_y = (BAR_HEIGHT - diff_text.get_height()) // 2
        paint_surface.blit(diff_text, (20, text_y))
        paint_surface.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, text_y))
        paint_surface.blit(time_text, (WIDTH - time_text.get_width() - 20, text_y))

        # Grass + Obstacles
        paint_surface.blit(self.grass_background, (0, BAR_HEIGHT))

        # Food
        food_x, food_y = self.food
        paint_surface.blit(self.sprites[self.current_apple_type], (food_x * CELL_SIZE, food_y * CELL_SIZE + BAR_HEIGHT))

        # Snake
        for i, segment in enumerate(self.snake):
            x, y = segment
            screen_pos = (x * CELL_SIZE, y * CELL_SIZE + BAR_HEIGHT)

            # Head
            if i == 0:
                angle = self._get_rotation_angle(self.direction)
                rotated_head = pygame.transform.rotate(self.sprites[f"{self.current_snake_color}_head"], angle)
                paint_surface.blit(rotated_head, screen_pos)

            # Tail
            elif i == len(self.snake) - 1:
                prev_x, prev_y = self.snake[i - 1]
                tail_dir = (prev_x - x, prev_y - y)
                angle = self._get_rotation_angle(tail_dir)
                rotated_tail = pygame.transform.rotate(self.sprites[f"{self.current_snake_color}_tail"], angle)
                paint_surface.blit(rotated_tail, screen_pos)

            # Body (straight or corner)
            else:
                next_x, next_y = self.snake[i - 1]
                prev_x, prev_y = self.snake[i + 1]

                to_prev = (prev_x - x, prev_y - y)
                to_next = (next_x - x, next_y - y)

                if to_prev[0] + to_next[0] == 0 and to_prev[1] + to_next[1] == 0:
                    angle = self._get_rotation_angle(to_next)
                    rotated_straight = pygame.transform.rotate(self.sprites[f"{self.current_snake_color}_body"], angle)
                    paint_surface.blit(rotated_straight, screen_pos)
                else:
                    combined = (to_prev[0] + to_next[0], to_prev[1] + to_next[1])
                    
                    if combined == (1, 1):
                        paint_surface.blit(self.sprites[f"{self.current_snake_color}_dr"], screen_pos)
                    elif combined == (-1, 1):
                        paint_surface.blit(self.sprites[f"{self.current_snake_color}_dl"], screen_pos)
                    elif combined == (1, -1):
                        paint_surface.blit(self.sprites[f"{self.current_snake_color}_ur"], screen_pos)
                    elif combined == (-1, -1):
                        paint_surface.blit(self.sprites[f"{self.current_snake_color}_ul"], screen_pos)

        img_array = pygame.surfarray.array3d(paint_surface)

        return np.transpose(img_array.copy(), (2, 0, 1)) # some format

    def render(self):
        frame = self._render_frame()

        if self.render_mode == "rgb_array":
            return frame

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return False

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return True

    def close(self):
        if self.window or self.canvas:
            pygame.quit()
            self.window = None
            self.canvas = None

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--registration", action="store_true")
    parser.add_argument("--obs_mode", type=str, default="image")
    parser.add_argument("--render_mode", type=str, default="human") # otherwise rgb_array
    parser.add_argument("--max_step", type=int, default=500)

    args = parser.parse_args()

    difficulty = args.difficulty
    registration = args.registration
    obs_mode = args.obs_mode
    render_mode = args.render_mode
    max_step = args.max_step

    if registration and obs_mode != "image":
        print("Warning: registration can happens only if the observation mode is set to image. Registration will be set off.")
        registration = False

    if registration:
        import cv2

        video = cv2.VideoWriter(
            f"video/game_difficulty_{difficulty}.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (WIDTH, TOTAL_HEIGHT)
        )

    env = SnakeEnv(
        render_mode=render_mode,
        observation_type=obs_mode,
        difficulty=difficulty,
        max_step=max_step
    )

    obs, info = env.reset()

    if registration:
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        video.write(frame)

    done = False
    trunc = False
    current_action = 0

    while not done and not trunc:

        if env.window is not None:
            for event in pygame.event.get(pygame.KEYDOWN):
                if event.key == pygame.K_w:
                    current_action = 0
                elif event.key == pygame.K_s:
                    current_action = 1
                elif event.key == pygame.K_a:
                    current_action = 2
                elif event.key == pygame.K_d:
                    current_action = 3

        obs, rew, done, trunc, info = env.step(current_action)

        if registration:
            frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            video.write(frame)

        if render_mode == "human":
            env.render()

    env.close()

    if registration:
        video.release()