import functools

import gymnasium
import random
from gymnasium.spaces import Discrete, MultiDiscrete, Box, Dict

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from collections import deque

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

NUM_ITERS = 1000
MAP_WIDTH = 24
MAP_HEIGHT = 24
MAP_PRODUCT = MAP_WIDTH * MAP_HEIGHT
AGENT_COUNT = 2
FOOD_CHANCE = 0.10
FOOD_GEN_MIN = 1
FOOD_GEN_MAX = 2
FOOD_TOTAL_MAX = 15

DEBUG_PRINT = False
DEBUG_AOP = False
RENDER_MAP = True
RENDER_SNAKE_BODY = True 


def env(render_mode=None):
    if DEBUG_AOP:
        print("CALLED: env()")
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
    if DEBUG_AOP:
        print("CALLED: raw_env()")
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human", "disabled"], "name": "rps_v2"}

    def __init__(self, render_mode=None):
        if DEBUG_AOP:
            print("CALLED: parallel_env()")
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.possible_agents = ["player_" + str(r) for r in range(AGENT_COUNT)]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if DEBUG_AOP:
            print("CALLED: observation_space()")
        #return Dict({
        #    "map": Box(low=0, high=5, shape=(MAP_PRODUCT,), dtype=int),
        #    "agent": Box(low=0, high=MAP_PRODUCT, shape=(MAP_PRODUCT,), dtype=int)
        #})
        return Box(low=0, high=5, shape=(MAP_PRODUCT,), dtype=int)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if DEBUG_AOP:
            print("CALLED: action_space()")
        return Discrete(4) # up, down, left, right

    def render(self):
        if DEBUG_AOP:
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
            if RENDER_SNAKE_BODY:
                print(self.state["agents"])
            if RENDER_MAP:
                # print border top wall abd line break
                string += "#" * (MAP_WIDTH + 2) + "\n"
                # print the map as a grid with different symbols for each cell type do so by width and height
                for i in range(MAP_HEIGHT):
                    string += "#"
                    for j in range(MAP_WIDTH):
                        position = i * MAP_WIDTH + j
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
                string += "#" * (MAP_WIDTH + 2) + "\n"

        else:
            string = "Game over"
        print(string)



    def close(self):
        if DEBUG_AOP:
            print("CALLED: close()")
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        if DEBUG_AOP:
            print("CALLED: reset()")
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.map = [EMPTY_CELL for _ in range(MAP_PRODUCT)]

        self.agents = self.possible_agents[:]
        self.snake_bodies = {agent: deque() for agent in self.agents}

        used_positions = set()
        for agent in self.agents:
            
            new_position = random.randint(0, MAP_PRODUCT-1)
            while new_position in used_positions:
                new_position = random.randint(0, MAP_PRODUCT-1)
            used_positions.add(new_position)

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
        #    "agent":  list(self.state["agents"][agent]) + [0] * (MAP_PRODUCT - len(self.state["agents"][agent]))
        #} for agent in self.agents}
        observations = {agent: self.state["map"] for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos
    

    def step(self, actions):
        if DEBUG_AOP:
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

        for agent in self.agents:
            if self.terminations[agent]:
                continue

            # get the current head position
            head_position = self.state["agents"][agent][-1]

            # get the next position based on the action, however if the action causes the snake to go into the borders then terminate the snake

            next_position = None
            agent_action = actions[agent]
            if agent_action == UP and head_position >= MAP_WIDTH:
                next_position = head_position - MAP_WIDTH
            elif agent_action == DOWN and head_position + MAP_WIDTH < MAP_WIDTH * MAP_HEIGHT:
                next_position = head_position + MAP_WIDTH
            elif agent_action == LEFT and head_position % MAP_WIDTH != 0:
                next_position = head_position - 1
            elif agent_action == RIGHT and (head_position + 1) % MAP_WIDTH != 0:
                next_position = head_position + 1
            else:
                if DEBUG_PRINT:
                    print(f'{agent} hit the border wall')
                next_position = None

            # check if the next position is valid
            if next_position is None or next_position < 0 or next_position >= MAP_PRODUCT:
                if DEBUG_PRINT:
                    print(f'{agent} hit the border wall')
                self.terminations[agent] = True
                # clear map of snake body
                for position in self.snake_bodies[agent]:
                    self.state["map"][position] = EMPTY_CELL
                # clear snake body
                self.snake_bodies[agent] = deque()

                continue

            # check if the snake ate food
            if self.state["map"][next_position] == FOOD:
                if DEBUG_PRINT:
                    print(f'{agent} ate food') 
                self.state["map"][next_position] = EMPTY_CELL
            else:
                tail_position = self.snake_bodies[agent].popleft()
                self.state["map"][tail_position] = EMPTY_CELL

            # check if the next position is empty
            if self.state["map"][next_position] != EMPTY_CELL:
                if DEBUG_PRINT:
                    print(f'{agent} collided with {CELL_TYPES_STR[self.state["map"][next_position]]}')
                self.terminations[agent] = True
                # clear map of snake body
                for position in self.snake_bodies[agent]:
                    self.state["map"][position] = EMPTY_CELL
                # clear snake body
                self.snake_bodies[agent] = deque()
                continue


            # move the snake
            self.snake_bodies[agent].append(next_position)
            self.state["map"][next_position] = SNAKE_CELL

        # add food X amount of food to the map with a 10% chance if there is less than 15 food on the map
        if random.random() < FOOD_CHANCE and self.state["map"].count(FOOD) < FOOD_TOTAL_MAX:    
            food_generated = random.randint(FOOD_GEN_MIN, FOOD_GEN_MAX)
            for _ in range(food_generated):
                food_position = random.randint(0, MAP_PRODUCT-1)
                if self.state["map"][food_position] == EMPTY_CELL:
                    self.state["map"][food_position] = FOOD
        


        terminations = self.terminations



        self.num_moves += 1
        env_truncation = self.num_moves >= NUM_ITERS or all(self.terminations.values())
        truncations = {agent: env_truncation for agent in self.agents}

        #observations = {agent: {
        #    "map": self.state["map"],
        #    "agent": list(self.state["agents"][agent]) + [0] * (MAP_PRODUCT - len(self.state["agents"][agent]))
        #} for agent in self.agents}
        observations = {agent: self.state["map"] for agent in self.agents}

        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {agent: len(self.state["agents"][agent]) for agent in self.agents}
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