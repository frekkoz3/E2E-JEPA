r"""
 ██████████  ████████  ██████████                  █████ ██████████ ███████████    █████████  
▒▒███▒▒▒▒▒█ ███▒▒▒▒███▒▒███▒▒▒▒▒█                 ▒▒███ ▒▒███▒▒▒▒▒█▒▒███▒▒▒▒▒███  ███▒▒▒▒▒███ 
 ▒███  █ ▒ ▒▒▒    ▒███ ▒███  █ ▒                   ▒███  ▒███  █ ▒  ▒███    ▒███ ▒███    ▒███ 
 ▒██████      ███████  ▒██████    ██████████       ▒███  ▒██████    ▒██████████  ▒███████████ 
 ▒███▒▒█     ███▒▒▒▒   ▒███▒▒█   ▒▒▒▒▒▒▒▒▒▒        ▒███  ▒███▒▒█    ▒███▒▒▒▒▒▒   ▒███▒▒▒▒▒███ 
 ▒███ ▒   █ ███      █ ▒███ ▒   █            ███   ▒███  ▒███ ▒   █ ▒███         ▒███    ▒███ 
 ██████████▒██████████ ██████████           ▒▒████████   ██████████ █████        █████   █████
▒▒▒▒▒▒▒▒▒▒ ▒▒▒▒▒▒▒▒▒▒ ▒▒▒▒▒▒▒▒▒▒             ▒▒▒▒▒▒▒▒   ▒▒▒▒▒▒▒▒▒▒ ▒▒▒▒▒        ▒▒▒▒▒   ▒▒▒▒▒ 
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import os

CELL_SIZE = 50
GRID_WIDTH, GRID_HEIGHT = 20, 20  
WIDTH, HEIGHT = GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE
RESOURCES_PATH = "src/game/resources/"

class SnakeEnv(gym.Env):
    """
    Environemnt for the snake game.
    Follows gym protocol.
    Possible observation format: 
        - raw image
        - grid
    For now there is only 1 difficulty.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, max_step=100, observation_type="grid", **kwargs):
        """
        Args:
            render_mode: "human" (visible window) or "rgb_array" (headless frame rendering).
            observation_type: "grid" for the 2D numeric grid, or "image" to return the raw 
                              visual RGB frame directly inside get_obs() / step() / reset().
        """
        self.action_space = spaces.Discrete(4)  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.render_mode = render_mode
        self.observation_type = observation_type
        
        if self.observation_type == "image":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(HEIGHT, WIDTH, 3), dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=3, shape=(GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8
            )
            
        self.window = None
        self.canvas = None  
        self.clock = None
        self.total_step = 0
        self.score = 0         
        self.max_step = max_step
        
        self.sprites_loaded = False
        self.reset()
        
        if kwargs == {}:
            self.reward_food = 20
            self.reward_death = -5
            self.reward_step = -0.5
        else:
            self.reward_food = kwargs["reward_food"]
            self.reward_death = kwargs["reward_death"]
            self.reward_step = kwargs["reward_step"]

    def _load_and_scale_sprites(self):
        """Loads assets and resizes them to match your environment's CELL_SIZE."""
        def load_sp(path):
            img = pygame.image.load(path).convert_alpha()
            return pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE))

        self.sprites = {
            "red_head": load_sp(f"{RESOURCES_PATH}red head.png"),
            "red_straight": load_sp(f"{RESOURCES_PATH}red body straight.png"),
            "red_tail": load_sp(f"{RESOURCES_PATH}red tail.png"),

            "red_curve_up_left": load_sp(f"{RESOURCES_PATH}red up left.png"),
            "red_curve_up_right": load_sp(f"{RESOURCES_PATH}red up right.png"),
            "red_curve_down_left": load_sp(f"{RESOURCES_PATH}red down left.png"),
            "red_curve_down_right": load_sp(f"{RESOURCES_PATH}red down right.png"),

            "green_head": load_sp(f"{RESOURCES_PATH}green head.png"),
            "green_straight": load_sp(f"{RESOURCES_PATH}green body straight.png"),
            "green_tail": load_sp(f"{RESOURCES_PATH}green tail.png"),

            "green_curve_up_left": load_sp(f"{RESOURCES_PATH}green up left.png"),
            "green_curve_up_right": load_sp(f"{RESOURCES_PATH}green up right.png"),
            "green_curve_down_left": load_sp(f"{RESOURCES_PATH}green down left.png"),
            "green_curve_down_right": load_sp(f"{RESOURCES_PATH}green down right.png"),

            "yellow_head": load_sp(f"{RESOURCES_PATH}yellow head.png"),
            "yellow_straight": load_sp(f"{RESOURCES_PATH}yellow body straight.png"),
            "yellow_tail": load_sp(f"{RESOURCES_PATH}yellow tail.png"),

            "yellow_curve_up_left": load_sp(f"{RESOURCES_PATH}yellow up left.png"),
            "yellow_curve_up_right": load_sp(f"{RESOURCES_PATH}yellow up right.png"),
            "yellow_curve_down_left": load_sp(f"{RESOURCES_PATH}yellow down left.png"),
            "yellow_curve_down_right": load_sp(f"{RESOURCES_PATH}yellow down right.png"),

            "apple_red": load_sp(f"{RESOURCES_PATH}red apple.png"),
            "apple_green": load_sp(f"{RESOURCES_PATH}green apple.png"),
            "apple_yellow": load_sp(f"{RESOURCES_PATH}yellow apple.png")
        }
        self.sprites_loaded = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = [(random.randrange(GRID_WIDTH), random.randrange(GRID_HEIGHT))]
        self.direction = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)]) # UP, DOWN, LEFT, RIGHT

        self.current_snake_color = self._random_color()
        self.current_apple_type = self._random_apple()
        
        self._place_food()
        self.done = False
        self.score = 0
        self.total_step = 0
        self.info = {}
        return self._get_obs(), self.info
    
    def _random_color(self):
        return random.choice(["red", "green", "yellow"])
    
    def _random_apple(self):
        return f"apple_{self._random_color()}"

    def _place_food(self):
        while True:
            self.food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if self.food not in self.snake:
                break

    def _get_obs(self):
        if self.observation_type == "image":
            return self._render_frame()
            
        grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        for x, y in self.snake:
            grid[y, x] = 3 
        head_x, head_y = self.snake[0]
        grid[head_y, head_x] = 2 
        food_x, food_y = self.food
        grid[food_y, food_x] = 1 
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
        self.info = {}
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
            not (0 <= new_head[1] < GRID_HEIGHT)
        ):
            self.done = True
            reward = self.reward_death
            return self._get_obs(), reward, True, False, self.info

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            reward = self.reward_food
            self.current_snake_color = self.current_apple_type.split("_")[1]
            self.current_apple_type = self._random_apple()
            self._place_food()
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
        # Ensure a Video Mode is initialized BEFORE loading sprites to avoid convert_alpha() crashing
        if self.window is None and self.canvas is None:
            pygame.init()
            if self.render_mode == "human":
                self.window = pygame.display.set_mode((WIDTH, HEIGHT))
                pygame.display.set_caption("Snake Environment")
                self.clock = pygame.time.Clock()
            else:
                # Setup a hidden video mode back-buffer if running headless for a model
                os.environ["SDL_VIDEODRIVER"] = "dummy"
                try:
                    self.canvas = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HIDDEN)
                except pygame.error:
                    # Fallback
                    self.canvas = pygame.display.set_mode((WIDTH, HEIGHT))

        if not self.sprites_loaded:
            self._load_and_scale_sprites()

        paint_surface = self.window if self.render_mode == "human" else self.canvas
        paint_surface.fill((0, 0, 0))

        # Food
        food_x, food_y = self.food
        paint_surface.blit(self.sprites[self.current_apple_type], (food_x * CELL_SIZE, food_y * CELL_SIZE))

        # Snake
        for i, segment in enumerate(self.snake):
            x, y = segment
            screen_pos = (x * CELL_SIZE, y * CELL_SIZE)

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
                    rotated_straight = pygame.transform.rotate(self.sprites[f"{self.current_snake_color}_straight"], angle)
                    paint_surface.blit(rotated_straight, screen_pos)
                else:
                    combined = (to_prev[0] + to_next[0], to_prev[1] + to_next[1])
                    
                    if combined == (1, 1):
                        paint_surface.blit(self.sprites[f"{self.current_snake_color}_curve_down_right"], screen_pos)
                    elif combined == (-1, 1):
                        paint_surface.blit(self.sprites[f"{self.current_snake_color}_curve_down_left"], screen_pos)
                    elif combined == (1, -1):
                        paint_surface.blit(self.sprites[f"{self.current_snake_color}_curve_up_right"], screen_pos)
                    elif combined == (-1, -1):
                        paint_surface.blit(self.sprites[f"{self.current_snake_color}_curve_up_left"], screen_pos)

        img_array = pygame.surfarray.array3d(paint_surface)
        return np.transpose(img_array, (1, 0, 2))

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


if __name__ == "__main__":
    
    # Usage with human interface
    env = SnakeEnv(render_mode="human", observation_type="grid", max_step=500)
    obs, info = env.reset()
    
    dir_mapping = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
    current_action = dir_mapping.get(env.direction, 0)

    done = False
    trunc = False
    while not done and not trunc:
        if env.window is not None:
            for event in pygame.event.get(pygame.KEYDOWN):
                if event.key == pygame.K_w: current_action = 0
                elif event.key == pygame.K_s: current_action = 1
                elif event.key == pygame.K_a: current_action = 2
                elif event.key == pygame.K_d: current_action = 3
        
        obs, rew, done, trunc, info = env.step(current_action)
        if not env.render():
            break
    env.close()

    # Usage with headless interface
    headless_env = SnakeEnv(render_mode=None, observation_type="image", max_step=100)
    
    img_obs, info = headless_env.reset()
    
    for _ in range(10):
        random_action = headless_env.action_space.sample()
        img_obs, rew, done, trunc, info = headless_env.step(random_action)
    
    headless_env.close()