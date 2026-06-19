# SnakeEnv

Custom Snake Environment which follows the Gymnasium protocol.
All the sprites are designed by us.

---

## Requirements & Installation

To run this environment, you need Python 3.8+ along with the following packages:

```bash
pip install -r requirements.txt
```

> **Note** For video recording capabilities or frame color conversion, opencv-python (cv2) is also required.

---

## Environment Initialization

The environment can be instantiated directly via the SnakeEnv class

```python
from snake import SnakeEnv

env = SnakeEnv(
    render_mode="human",          # "human" or "rgb_array"
    observation_type="grid",       # "grid" or "image"
    difficulty=1,                 # 0, 1, 2, or 3
    max_step=100,                 # Maximum episode duration
    reward_food=20,               # Reward for eating food
    reward_death=-5,              # Reward for dying
    reward_step=-0.5              # Step penalty
)
```

### Keyword Arguments (`kwargs`)

If no custom rewards are provided in `kwargs`, the following defaults are used:

* reward_food: 20
* reward_death: -5
* reward_step: -0.5

## Core Specification

### Action Space

The action space is `spaces.Discrete(4)`, representing four cardinal directions:

* 0: Move UP
* 1: Move DOWN
* 2: Move LEFT
* 3: Move RIGHT

> **Note** If an action opposite to the current direction is passed, the snake will continue moving in its current heading to prevent self-collision via immediate reversal

### Observation Space

Controlled by the observation_type parameter:

1. **grid mode** (`spaces.Box(0, 4, shape=(GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)`):

    * 0: Empty Cell / Grass
    * 1: Food / Apple
    * 2: Snake Head
    * 3: Snake Body/Tail Segment
    * 4: Obstacle (Bombs, rocks, skulls)

2. **image mode** (`spaces.Box(0, 255, shape=(TOTAL_HEIGHT, WIDTH, 3), dtype=np.uint8)`):

    * Returns a fully rendered RGB frame matrix (OpenCV compatible width/height formatting) containing the gameplay area and a top status bar showing difficulty, current score, and elapsed time.

### Difficulty Levels

The environment offers 4 difficulty configurations (0 to 3):

|Level|Name|Description|
|---|---|---|
|0|No-Grow Standard|The snake can eat food and score points, but its length remains fixed at 1 segment.  1Classic SnakeStandard gameplay. The snake grows by 1 segment each time it eats an apple.|
|2|Obstacle Challenge|N static obstacles (bombs, rocks, skulls) are randomly placed across the grid at initialization. Colliding with them causes death.|
|3|Dynamic Chaos|Includes static obstacles, fake apples, and whenever an apple is eaten, a new obstacle replaces an old one on a random cell, dynamically shifting the grid layout.|

### Asset & Rendering Structure

The environment relies on graphical assets located under `src/game/resources/`. Here we provide the default settings:

* **Snake Colors**: Variations for red, green, and yellow. When an apple is consumed, the snake changes color to match the color of the apple eaten.
* **Snake Components**: Includes context-aware textures for heads, body lengths, tails, and corner turns (ul, ur, dl, dr).
* **Grid Specs**: CELL_SIZE = 35, GRID_WIDTH = 20, and GRID_HEIGHT = 20
* **Dimensions**: Gameplay canvas is 700x700 pixels, plus a top status bar of BAR_HEIGHT = 50, resulting in a final frame resolution of 700x750 pixels

## Quick Start Example

Here we include an example importing the module with `absolute path`.

```python
from src.game.snake import SnakeEnv

# Initialize the environment
env = SnakeEnv(render_mode="human", observation_type="grid", difficulty=1)
obs, info = env.reset()

done = False
truncated = False

while not (done or truncated):
    # Sample a random action
    action = env.action_space.sample()
    
    # Step environment
    obs, reward, done, truncated, info = env.step(action)
    
    # Render the frame
    env.render()

env.close()
```
