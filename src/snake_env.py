import functools

import gymnasium
import random
from gymnasium.spaces import Discrete, MultiDiscrete, Box

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from collections import deque
import numpy as np
import supersuit as ss

from typing import List, Union, Tuple

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
MAX_LIMIT_WIDTH = 64
MAX_LIMIT_HEIGHT = 64


def create_env(render_mode="human", num_vec_envs=1, num_cpus=4, debug_print=False, map_width=5, map_height=5, seed=None):
    env = parallel_env(
        render_mode=render_mode, 
        map_width=map_width, 
        map_height=map_height, 
        agent_count=2, 
        snake_start_len=2, 
        food_gen_max=1, 
        food_total_max=5, 
        move_rewards=True, 
        move_rewards_length=False,
        move_reward=-0.3, 
        food_rewards=True, 
        food_reward=27, 
        food_rewards_length_multiplier=False, 
        death_reward=-29, 
        debug_print=debug_print)
    observations, infos = env.reset(seed=seed)
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=num_vec_envs, num_cpus=num_cpus, base_class="stable_baselines3")
    return env



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

def dist_to_edge(width, height, point):
    x = point[0]
    y = point[1]
    return min(x, y, width - x - 1, height - y - 1) + 1

class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human", "disabled"], "name": "rps_v2"}

    def __init__(self, render_mode=None,
                num_iters=5000,
                map_width=11,
                map_height=11,
                agent_count=2,
                food_chance=0.10,
                food_gen_min=1,
                food_gen_max=2,
                snake_start_len=0,
                food_total_max=15,
                debug_print=False,
                debug_aop=False,
                render_map=True,
                food_rewards=True,
                food_rewards_length_multiplier=False,
                food_reward=15,
                death_rewards=True,
                death_reward=-1,
                move_rewards=False,
                move_rewards_length=False,
                move_reward=1,
                walls_enabled=False,
                walls_max=10,
                walls_chance=0.10,
                walls_replace=True,
                walls_gen_min=1,
                walls_gen_max=1,
                kill_idler=True,
                kill_idler_after=15,
                kill_idler_reward=-50,
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
        self.food_rewards_length_multiplier = food_rewards_length_multiplier
        self.food_reward = food_reward
        self.death_rewards = death_rewards
        self.death_reward = death_reward
        self.move_rewards = move_rewards
        self.move_rewards_length = move_rewards_length
        self.move_reward = move_reward
        self.walls_enabled = walls_enabled
        self.walls_max = walls_max
        self.walls_chance = walls_chance
        self.walls_replace = walls_replace
        self.walls_gen_min = walls_gen_min
        self.walls_gen_max = walls_gen_max
        self.kill_idler = kill_idler
        self.kill_idler_after = kill_idler_after
        self.kill_idler_reward = kill_idler_reward

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
        #  
        #   "agent": Box(low=0, high=map_product, shape=(map_product,), dtype=int)
        #})

        # food pos closest to snake head
        # data[0] = 1 if pos is snake
        # data[1] = 1 if pos is food
        # data[2] = distance to its closest body part
        # data[3] = distance to its closest food
        # data[4] = distance to its closest wall or edge map
        return Box(low=0, high=MAX_LIMIT_WIDTH*MAX_LIMIT_HEIGHT, shape=(16,5), dtype=int)
        #return Box(low=0, high=self.map_product+5, shape=(self.map_product*2,), dtype=int)

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

                
        self.idler = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.num_moves = 0
        self.state = {
            "map": self.map,
            "agents": self.snake_bodies,
            "food_pos": [],
            "walls_pos": [],
        }
        self.state["last_observation"] = {agent: self.get_observation(agent) for agent in self.agents}

        #observations = {agent: {
        #    "map": self.state["map"],
        #    "agent":  list(self.state["agents"][agent]) + [0] * (map_product - len(self.state["agents"][agent]))
        #} for agent in self.agents}


        observations = self.state["last_observation"]
        infos = {agent: {
            "snake_size": len(self.snake_bodies[agent])
        } for agent in self.agents}

        return observations, infos
    
    def out_of_bounds(self, pos: Tuple[int,int]) -> bool:
        """If the 2d coord is out of bounds then return False. Else True"""
        x = pos[0]
        y = pos[1]

        if (x < 0 or self.map_width <= x):
             return True
        if (y < 0 or self.map_height <= y):
             return True

        return False

    def convert_point_to_xy(self, point):
        assert  0 <= point and point < self.map_product

        return (point % self.map_width), (point // self.map_width)
    
    def convert_xy_to_1d(self, pos: Tuple[int,int]) -> int:
        assert 0 <= pos[0] and pos[0] < self.map_width
        assert 0 <= pos[1] and pos[1] < self.map_height

        return pos[0] + pos[1] * self.map_width

    def get_observation(self, agent):
        if self.debug_aop:
            print("CALLED: get_observation()")
        """
        Returns the observation for agent
        """
        if len(self.state["agents"][agent]) < 1:
            return np.zeros((16, 5), dtype=int) # TODO: Body length is NULL dno how to handle
        
        head_pos = self.convert_point_to_xy(self.state["agents"][agent][-1])

        check_pos = [
            (head_pos[0] + 1, head_pos[1]), # right 2
            (head_pos[0] + 2, head_pos[1]), # right 2
            (head_pos[0] - 1, head_pos[1]), # left 1
            (head_pos[0] - 2, head_pos[1]), # left 1
            # up down
            (head_pos[0], head_pos[1] + 1), # down 1
            (head_pos[0], head_pos[1] + 2), # down 2
            (head_pos[0], head_pos[1] - 1), # up 1
            (head_pos[0], head_pos[1] - 2), # up 2
            # corners
            (head_pos[0] + 1, head_pos[1] + 1), # bottom right
            (head_pos[0] - 1, head_pos[1] + 1), # bottom left
            (head_pos[0] + 1, head_pos[1] - 1), # top right
            (head_pos[0] - 1, head_pos[1] - 1), # top left
            # corners 2
            (head_pos[0] + 2, head_pos[1] + 2), # bottom right
            (head_pos[0] - 2, head_pos[1] + 2), # bottom left
            (head_pos[0] + 2, head_pos[1] - 2), # top right
            (head_pos[0] - 2, head_pos[1] - 2), # top left
        ]



        # Init array of of 0s 5 long in arrays 16 long
        checked_pos = np.zeros((16, 5), dtype=int)


        cur_body = list(map(lambda x: self.convert_point_to_xy(x), list(self.state["agents"][agent])))


        # iterate over and fill in checked_pos using pos from check_pos
        if self.debug_print:
            print(check_pos)

        for i, pos in enumerate(check_pos):
            # TODO: Should we check out of map at the beginning?
            if not self.out_of_bounds(pos):
                # if pos is snake set data[0] to 1
                if pos in cur_body:
                    checked_pos[i][0] = 1
                
                # if pos is food set data[1] to 1
                xd = self.convert_xy_to_1d(pos)
                if self.state["map"][xd] == FOOD:
                    checked_pos[i][1] = 1

            # distance to its closest body part
            checked_pos[i][2] = min([self.distance(pos, body_part) for body_part in cur_body])

            # distance to its closest food
            checked_pos[i][3] = min([self.distance(pos, self.convert_point_to_xy(food_part)) for food_part in self.state["food_pos"]]) if len(self.state["food_pos"]) > 0 else self.map_product

            # distance to its closest wall or distance to being outside map
            if self.out_of_bounds(pos):
                checked_pos[i][4] = 0
            else:
                edge_dist = dist_to_edge(self.map_width, self.map_height, pos)
                wall_dist = min([self.distance(pos, self.convert_point_to_xy(wall_part)) for wall_part in self.state["walls_pos"]]) if len(self.state["walls_pos"]) > 0 else self.map_product
                min_obstacle_dist = min(edge_dist, wall_dist)
                # calculate dist to snake body but do not calculate for own body, only other agents - if hostiles len greater than 0 then this should be set to MAP_PRODUCT using ternary. Finally min this with min_obstacle_dist
                hostiles = [agent for agent in self.agents if agent != agent]
                hostiles_min_dist = min([self.distance(pos, self.convert_point_to_xy(body_part)) for body_part in self.state["agents"][hostiles]]) if len(hostiles) > 0 else self.map_product
                checked_pos[i][4] = min(min_obstacle_dist, hostiles_min_dist)


        return checked_pos
    
    def distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def kill_agent(self, agent, rewards, death_reward=None):
        self.terminations[agent] = True
        # clear map of snake body
        for position in self.snake_bodies[agent]:
            self.state["map"][position] = EMPTY_CELL
        # clear snake body
        self.snake_bodies[agent] = deque()
        # set rewards to death reward
        rewards[agent] = self.death_reward if death_reward is None else death_reward

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
        
        observations = {agent: self.get_observation(agent) for agent in self.agents}
        
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
                self.kill_agent(agent, rewards)
                continue


            # check if the snake ate food
            if self.state["map"][next_position] == FOOD:
                if self.debug_print:
                    print(f'{agent} ate food') 
                self.state["map"][next_position] = EMPTY_CELL
                if self.food_rewards:
                    rewards[agent] += (len(self.snake_bodies[agent]) * self.food_reward) if self.food_rewards_length_multiplier else self.food_reward
                    self.state["food_pos"].remove(next_position)
                # Idler reset
                if self.kill_idler:
                    self.idler[agent] = 0
            else:
                tail_position = self.snake_bodies[agent].popleft()
                self.state["map"][tail_position] = EMPTY_CELL
                if self.move_rewards:
                    rewards[agent] = len(self.snake_bodies[agent]) if self.move_rewards_length else self.move_reward
                # Idler add
                if self.kill_idler:
                    self.idler[agent] += 1
                    if self.idler[agent] >= self.kill_idler_after:
                        self.kill_agent(agent, rewards, self.kill_idler_reward)
                        continue
                        
            # Reward if it came closer to food
            prev_min_food_distance = np.min(self.state["last_observation"][agent][:,3])
            current_min_food_distance = np.min(observations[agent][:,3])

            if current_min_food_distance < prev_min_food_distance:
                rewards[agent] += 0.3
                
            # move the snake
            self.snake_bodies[agent].append(next_position)
            self.state["map"][next_position] = SNAKE_CELL

        # add food X amount of food to the map with a Y% chance if there is less than Z food on the map
        if random.random() < self.food_chance and self.state["map"].count(FOOD) < self.food_total_max:    
            food_generated = random.randint(self.food_gen_min, self.food_gen_max)
            for _ in range(food_generated):
                food_position = random.randint(0, self.map_product-1)
                if self.state["map"][food_position] == EMPTY_CELL:
                    self.state["map"][food_position] = FOOD
                    self.state["food_pos"].append(food_position)
        
        # if Y% chance if there is less than Z walls on the map add a new one else find an existing wall, set it to empty and place a wall elsewhere - do this X times
        if self.walls_enabled:
            wall_chance = random.random()
            if wall_chance < self.walls_chance and self.state["map"].count(WALL) < self.walls_max:
                walls_generated = random.randint(self.walls_gen_min, self.walls_gen_max)
                for _ in range(walls_generated):
                    wall_position = random.randint(0, self.map_product-1)
                    if self.state["map"][wall_position] == EMPTY_CELL:
                        self.state["map"][wall_position] = WALL
                        self.state["walls_pos"].append(wall_position)
            elif wall_chance < self.walls_chance and self.state["map"].count(WALL) >= self.walls_max:
                for _ in range(walls_generated):
                    wall_position = random.randint(0, self.map_product-1)
                    if self.state["map"][wall_position] == WALL:
                        self.state["map"][wall_position] = EMPTY_CELL
                        self.state["walls_pos"].remove(wall_position)
                        new_wall_position = random.randint(0, self.map_product-1)
                        while self.state["map"][new_wall_position] != EMPTY_CELL:
                            new_wall_position = random.randint(0, self.map_product-1)
                        self.state["map"][new_wall_position] = WALL
                        self.state["walls_pos"].append(new_wall_position)


        terminations = self.terminations

        self.num_moves += 1
        env_truncation = self.num_moves >= self.num_iters or all(self.terminations.values())
        truncations = {agent: env_truncation for agent in self.agents}

        #observations = {agent: {
        #    "map": self.state["map"],
        #    "agent": list(self.state["agents"][agent]) + [0] * (map_product - len(self.state["agents"][agent]))
        #} for agent in self.agents}
        if self.debug_print:
            print(f"observations: {observations}")

        infos = {agent: {
            "snake_size": len(self.snake_bodies[agent])
        } for agent in self.agents}

        # remove agents that have terminated
        for agent in self.agents:
            if self.terminations[agent]:
                self.agents.remove(agent)

        if env_truncation:
            self.agents = []

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos