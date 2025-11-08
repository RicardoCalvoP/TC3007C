from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava, Floor
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import RGBImgObsWrapper, RGBImgPartialObsWrapper
from scipy.special import softmax
from tqdm import trange
import matplotlib.pyplot as plt
import random
import numpy as np
import os

class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=19,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.size = size
        self.key_positions = []
        self.lava_positions = []

        self.start_agent_pos=(1,1)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Reach the goal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Place walls in straight lines
        # Vertical walls
        # for y in range(1, height-1):
        #    self.put_obj(Wall(), width // 2, y)

        # # Horizontal walls
        # for x in range(1, width-1):
        #    self.put_obj(Wall(), x, height//2)

        # # Create openings in the walls
        # openings = [(width//2,5),(width//2,15),(5,height//2),(15,height//2),]

        # for x, y in openings:
        #    self.grid.set(x, y, None)


        # Place a goal square in the bottom-right corner
        self.goal_pos = (width - 2, height - 2)
        self.put_obj(Goal(), *self.goal_pos)

        self._place_agent()

        self.mission = "Reach the goal"

    def _place_agent(self):

        while True:
            x = random.randint(1, self.size - 2)
            y = random.randint(1, self.size - 2)

            pos = (x, y)

            if(x < self.size//2 + 1 or y < self.size//2 + 1):
                continue

            # Check if the position is empty (not wall, lava, floor, or goal)
            if (self.grid.get(*pos) is None and
                pos != self.goal_pos):
                self.agent_pos = pos
                self.agent_dir = random.randint(0, 3)  # Random direction
                break

    def reset(self, **kwargs):
        self.stepped_floors = set()
        obs = super().reset(**kwargs)
        # self._place_agent()  # Place the agent in a new random position
        return obs

    def step(self, action):
        prev_pos=self.agent_pos
        prev_dir=self.agent_dir
        obs, reward, terminated, truncated, info = super().step(action)

        if(self.grid.get(*self.agent_pos) is None):
            reward = reward - 1

        if(prev_dir==self.agent_dir and prev_pos ==  self.agent_pos):
            reward = reward - 3

        if isinstance(self.grid.get(*self.agent_pos), Goal):
            reward = reward + 100
            terminated = True
            self._place_agent()  # Place the agent in a new random position
        return obs, reward, terminated, truncated, info

def policy_gradient_learning(lr=0.1, gamma=0.99, max_trainings = 10000, max_steps=50, eps_start=1.0, eps_end=0.05, eps_decay=500, num_shown_trainings=0):

    env = SimpleEnv(render_mode=None)   # train fast without rendering
    size = env.size                      # Get the dimension of the gird


    if os.path.exists("params.npy"):
        params = np.load("params.npy")
    else:
        params = np.zeros((size, size, 4, 3), dtype=float)   # logits per [x,y,dir,action]

    if os.path.exists("evs.npy"):
        evs = np.load("evs.npy")
    else:
        evs = np.zeros((size, size, 4), dtype=float)      # V(s) baseline per [x,y,dir]

    for j in trange(max_trainings, desc="Training",  colour="cyan", ncols=100):

        if max_trainings - j == num_shown_trainings:
            env.close()
            env = SimpleEnv(render_mode="human")  # show the environment

        step = 0
        terminated = False              # Initialize the terminated condition to False
        truncated = False               # Initialize the truncated condition to False
        _, _ = env.reset()         # Reset the environment to start a new episode

        while step < (max_steps) and not (terminated or truncated):

            # Get the agent position
            x, y = env.agent_pos
            # Get the current direction of the agent
            direction = env.agent_dir

            probs  = softmax(params[x, y, direction])          # π(a|s)
            action = np.random.choice(3, p=probs)      # 0=left,1=right,2=forward

            _obs, reward, terminated, truncated, _info = env.step(action)

            # Get the agent position
            new_x, new_y = env.agent_pos
            # Get the current direction of the agent
            new_direction = env.agent_dir

            # critic: TD error δ = r + γ V(s') - V(s)
            v_s  = evs[x, y, direction]
            v_sp = 0.0 if (terminated or truncated) else evs[new_x, new_y, new_direction]
            delta = reward + gamma * v_sp - v_s

            # update critic V(s)
            evs[x, y, direction] += lr * delta

            # actor gradient: (one_hot(a) - probs)
            grad = np.zeros(3, dtype=float)
            grad[action] = 1.0
            params[x, y, direction, :] += lr * (grad - probs) * delta


            step += 1  # count total steps


    return params, evs

def test(params, human = False):
    mode = "human" if human else None
    env = SimpleEnv(render_mode=mode) # Create the environment

    max_steps = 50
    num_tests = 100
    arrived_goals = 0
    pbar = trange(num_tests, desc="Testing", ncols=100, colour="green")
    for j in pbar:
        step = 0
        terminated = False              # Initialize the terminated condition to False
        truncated = False               # Initialize the truncated condition to False
        _, _ = env.reset()         # Reset the environment to start a new episode
        while step < (max_steps) and not (terminated or truncated):
            step += 1  # count total steps

            # Get the agent position
            x, y = env.agent_pos
            # Get the current direction of the agent
            direction = env.agent_dir

            logits = params[x, y, direction]                 # shape (3,)
            action = np.argmax(logits)               # greedy eval (= argmax softmax)

            _obs, r, terminated, truncated, _info = env.step(action)

            if terminated:
                arrived_goals += 1
        pbar.set_postfix(goals=arrived_goals, success=f"{arrived_goals/(j+1):.2%}")

if __name__ == "__main__":
  params, evs = policy_gradient_learning(max_trainings=1)
  np.save('params.npy', params)
  np.save('evs.npy', evs)
  test(params, human=True)