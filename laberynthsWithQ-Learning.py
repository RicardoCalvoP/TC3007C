from __future__ import annotations
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava, Floor
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import RGBImgObsWrapper,RGBImgPartialObsWrapper
import matplotlib.pyplot as plt
import random
import numpy as np
from tqdm import trange
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

def calc_q(Q_table, x, y, direction, action, reward, next_x, next_y, next_direction, alpha, gamma):
    new_q = Q_table[x][y][direction][action] + alpha * (reward + gamma * np.max(Q_table[next_x][next_y][next_direction]) - Q_table[x][y][direction][action])
    return new_q

def exploration_dilema(eps_end, eps_start, eps_decay, step, episode, max_steps):
    eps_threshold = eps_end + (eps_start - eps_end) * np.exp(-1.0 * (step + (episode * max_steps)) / eps_decay)
    return eps_threshold

def q_learning(lr=0.1, gamma=0.99, max_trainings = 10000, max_steps=50, eps_start=1.0, eps_end=0.05, eps_decay=500, num_shown_trainings=0):

    env = SimpleEnv(render_mode=None)   # train fast without rendering
    size = env.size                      # Get the dimension of the gird

    # Initialize Q-table: Q[x][y][direction][action]
    if os.path.exists("Q_table.npy"):
        Q_table = np.load("Q_table.npy")
    else:
        Q_table = np.zeros((size, size, 4, 3))
    # Agent pos x,y are in [1, size-2], so we use size x size array for easy indexing
    # Direction is 0-3 (4 values), Action is 0-2 (3 values)

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

            step += 1  # count total steps

            eps_threshold = exploration_dilema(eps_end, eps_start, eps_decay, step, j, max_steps)
            if random.random() > eps_threshold:
                action = np.argmax(Q_table[x][y][direction])  # exploit
            else:
                # limit acctions to forward, left, right
                action = random.randint(0,2)                  # explore (left/right/forward)

            # env.step(action) returns 5 values:
            # 1) obs        -> Observation of the environment after executing the action.
            #                  In MiniGrid this is usually a dictionary (e.g., image of the grid, agent direction, mission).
            # 2) reward     -> Numeric reward signal.
            #                  In this custom env: -1 per step, -3 if bumping into a wall, +30 when reaching the goal.
            # 3) terminated -> Boolean flag that indicates if the episode ended naturally
            #                  (e.g., the agent reached the Goal).
            # 4) truncated  -> Boolean flag that indicates if the episode ended due to an external limit
            #                  (e.g., max_steps reached).
            # 5) info       -> Dictionary with additional environment information (for debugging or logging),
            #                  not part of the agent’s input state.
            _, reward, terminated, truncated, _ = env.step(action)

            # Get the agent position
            new_x, new_y = env.agent_pos
            # Get the current direction of the agent
            new_direction = env.agent_dir

            # Update Q-value using the Q-learning formula
            Q_table[x][y][direction][action] = calc_q(
                Q_table, x, y, direction, action, reward,
                new_x, new_y, new_direction,
                lr, gamma
            )

    return Q_table



def test(Q_table, human = False):
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

            action = np.argmax(Q_table[x][y][direction])

            _, _, terminated, truncated, _ = env.step(action)

            if terminated:
                arrived_goals += 1
        pbar.set_postfix(goals=arrived_goals, success=f"{arrived_goals/(j+1):.2%}")


if __name__ == "__main__":
    Q_table = q_learning(max_trainings=1000 ,num_shown_trainings=50)
    np.save('Q_table.npy', Q_table)
    test(Q_table, human=True)
