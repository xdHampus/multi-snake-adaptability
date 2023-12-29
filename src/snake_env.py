import functools

import gymnasium
import random
from gymnasium.spaces import Discrete, MultiDiscrete, Box, Dict

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from collections import deque
import numpy as np

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTIONS_STR = ["UP", "DOWN", "LEFT", "RIGHT"]

EMPTY_CELL = 0
SNAKE_CELL = 1
FOOD = 2
WALL = 3
CELL_TYPES = [EMPTY_CELL, SNAKE_CELL, FOOD, WALL]
CELL_TYPES_STR = ["EMPTY_CELL", "SNAKE_CELL", "FOOD", "WALL"]



def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human", "disabled"], "name": "rps_v2"}

    def __init__(self, render_mode=None,
                num_iters=1000,
                map_width=10,
                map_height=10,
                agent_count=2,
                food_chance=0.10,
                food_gen_min=1,
                food_gen_max=2,
                snake_start_len=1,
                food_total_max=15,
                debug_print=False,
                debug_aop=False,
                render_map=True,
                food_rewards=True,
                food_reward=15,
                death_rewards=True,
                death_reward=-1,
                move_rewards=False,
                move_rewards_length=False,
                move_reward=1,
                render_snake_body=True):
        self.num_iters = num_iters
        self.map_width = map_width
        self.map_height = map_height
        self.map_product = self.map_width * self.map_height
        self.agent_count = agent_count
        self.food_chance = food_chance
        self.food_gen_min = food_gen_min
        self.food_gen_max = food_gen_max
        self.food_total_max = food_total_max
        self.debug_print = debug_print
        self.debug_aop = debug_aop
        self.render_map = render_map
        self.render_snake_body = render_snake_body
        self.render_mode = render_mode
        self.snake_start_len = snake_start_len
        self.food_rewards = food_rewards
        self.food_reward = food_reward
        self.death_rewards = death_rewards
        self.death_reward = death_reward
        self.move_rewards = move_rewards
        self.move_rewards_length = move_rewards_length
        self.move_reward = move_reward


        if self.debug_aop:
            print("CALLED: parallel_env()")

        self.possible_agents = ["player_" + str(r) for r in range(self.agent_count)]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if self.debug_aop:
            print("CALLED: observation_space()")
        #return Dict({
        #    "map": Box(low=0, high=5, shape=(map_product,), dtype=int),
        #    "agent": Box(low=0, high=map_product, shape=(map_product,), dtype=int)
        #})
        return Box(low=0, high=self.map_product+5, shape=(self.map_product*2,), dtype=int)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if self.debug_aop:
            print("CALLED: action_space()")
        return Discrete(4) # up, down, left, right

    def render(self):
        if self.debug_aop:
            print("CALLED: render()")
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.render_mode == "disabled":
            return
        
        string = ""
        if len(self.agents) > 0:
            if self.render_snake_body:
                print(self.state["agents"])
            if self.render_map:
                # print border top wall abd line break
                string += "#" * (self.map_width + 2) + "\n"
                # print the map as a grid with different symbols for each cell type do so by width and height
                for i in range(self.map_height):
                    string += "#"
                    for j in range(self.map_width):
                        position = i * self.map_width + j
                        if self.state["map"][position] == EMPTY_CELL:
                            string += " "
                        elif self.state["map"][position] == SNAKE_CELL:
                            string += "o"
                        elif self.state["map"][position] == FOOD:
                            string += "*"
                        elif self.state["map"][position] == WALL:
                            string += "#"
                    string += "#"
                    string += "\n"
                # print border bottom wall and line break
                string += "#" * (self.map_width + 2) + "\n"

        else:
            string = "Game over"
        print(string)



    def close(self):
        if self.debug_aop:
            print("CALLED: close()")
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        if self.debug_aop:
            print("CALLED: reset()")
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        if seed is not None:
            random.seed(seed)

        self.map = [EMPTY_CELL for _ in range(self.map_product)]

        self.agents = self.possible_agents[:]
        self.snake_bodies = {agent: deque() for agent in self.agents}

        used_positions = set()
        for agent in self.agents:
            
            new_position = random.randint(0, self.map_product-1)
            while new_position in used_positions:
                new_position = random.randint(0, self.map_product-1)
            used_positions.add(new_position)

            self.map[new_position] = SNAKE_CELL
            self.snake_bodies[agent].append(new_position)
            # extend the snake body by the snake_start_len nearby positions
            for _ in range(self.snake_start_len):
                new_position = self.snake_bodies[agent][-1]
                if new_position >= self.map_width:
                    new_position -= self.map_width
                elif new_position + self.map_width < self.map_width * self.map_height:
                    new_position += self.map_width
                elif new_position % self.map_width != 0:
                    new_position -= 1
                elif (new_position + 1) % self.map_width != 0:
                    new_position += 1
                else:
                    new_position = None
                if new_position is not None:
                    self.map[new_position] = SNAKE_CELL
                    self.snake_bodies[agent].append(new_position)

                


        self.terminations = {agent: False for agent in self.agents}
        self.num_moves = 0
        self.state = {
            "map": self.map,
            "agents": self.snake_bodies
        }

        #observations = {agent: {
        #    "map": self.state["map"],
        #    "agent":  list(self.state["agents"][agent]) + [0] * (map_product - len(self.state["agents"][agent]))
        #} for agent in self.agents}
        observations = {agent: self.map + self.map for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos
    

    def step(self, actions):
        if self.debug_aop:
            print("CALLED: step()")
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        rewards = {agent: 0 for agent in self.agents}
        
        if self.debug_print:
            print(actions)

        for agent in self.agents:
            if self.terminations[agent]:
                continue

            # get the current head position
            head_position = self.state["agents"][agent][-1]

            # get the next position based on the action, however if the action causes the snake to go into the borders then terminate the snake

            next_position = None
            agent_action = actions[agent]
            if agent_action == UP and head_position >= self.map_width:
                next_position = head_position - self.map_width
            elif agent_action == DOWN and head_position + self.map_width < self.map_width * self.map_height:
                next_position = head_position + self.map_width
            elif agent_action == LEFT and head_position % self.map_width != 0:
                next_position = head_position - 1
            elif agent_action == RIGHT and (head_position + 1) % self.map_width != 0:
                next_position = head_position + 1
            else:
                if self.debug_print:
                    print(f'{agent} hit the border wall')
                next_position = None

            # check if the next position is valid
            if (
                next_position is None
                or next_position < 0 
                or next_position >= self.map_product 
                or self.state["map"][next_position] == WALL 
                or self.state["map"][next_position] == SNAKE_CELL
            ):                
                if self.debug_print:
                    if next_position is None or next_position < 0 or next_position >= self.map_product:
                        print(f'{agent} hit at {next_position} is not valid')
                    elif self.state["map"][next_position] == WALL:
                        print(f'{agent} hit at {next_position} is wall')
                    elif self.state["map"][next_position] == SNAKE_CELL:
                        print(f'{agent} hit at {next_position} is snake')
                    else:
                        print(f'{agent} hit at {next_position} is unknown')
                self.terminations[agent] = True
                # clear map of snake body
                for position in self.snake_bodies[agent]:
                    self.state["map"][position] = EMPTY_CELL
                # clear snake body
                self.snake_bodies[agent] = deque()
                # set rewards to -1
                rewards[agent] = -1

                continue

            # check if the snake ate food
            if self.state["map"][next_position] == FOOD:
                if self.debug_print:
                    print(f'{agent} ate food') 
                self.state["map"][next_position] = EMPTY_CELL
                rewards[agent] = 15
            else:
                tail_position = self.snake_bodies[agent].popleft()
                self.state["map"][tail_position] = EMPTY_CELL
                if self.move_rewards:
                    rewards[agent] = len(self.snake_bodies[agent]) if self.move_rewards_length else self.move

            # move the snake
            self.snake_bodies[agent].append(next_position)
            self.state["map"][next_position] = SNAKE_CELL

        # add food X amount of food to the map with a 10% chance if there is less than 15 food on the map
        if random.random() < self.food_chance and self.state["map"].count(FOOD) < self.food_total_max:    
            food_generated = random.randint(self.food_gen_min, self.food_gen_max)
            for _ in range(food_generated):
                food_position = random.randint(0, self.map_product-1)
                if self.state["map"][food_position] == EMPTY_CELL:
                    self.state["map"][food_position] = FOOD
        


        terminations = self.terminations



        self.num_moves += 1
        env_truncation = self.num_moves >= self.num_iters or all(self.terminations.values())
        truncations = {agent: env_truncation for agent in self.agents}

        #observations = {agent: {
        #    "map": self.state["map"],
        #    "agent": list(self.state["agents"][agent]) + [0] * (map_product - len(self.state["agents"][agent]))
        #} for agent in self.agents}
        observations = {agent: 
                        np.concatenate(
                            (self.state["map"], 
                            list(self.state["agents"][agent]) + [0] * (self.map_product - len(self.state["agents"][agent])))
                        )
            for agent in self.agents
        }
        # rewards for all agents are placed in the rewards dictionary to be returned
        infos = {agent: {} for agent in self.agents}

        # remove agents that have terminated
        for agent in self.agents:
            if self.terminations[agent]:
                self.agents.remove(agent)

        if env_truncation:
            self.agents = []

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos