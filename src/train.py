import snake_env
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import supersuit as ss
import datetime
import eval
from utils import human_format, game_parameter_combinations, game_parameter_difficulty_estimator
import os
import math
import numpy as np
# get current timedate as string
now = datetime.datetime.now()
now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
training_version = "v1_1"
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


class SummaryWriterCallback(BaseCallback):

    def __init__(self, verbose=0):
        super(SummaryWriterCallback, self).__init__(verbose)


    def _on_training_start(self):
        self._log_freq = 10  # log every 10 calls
        self.avg_freq = 5000
        self.avg_size = 5000

        self.highest_snake_size = 0
        self.highest_food_eaten = 0
        self.highest_moves = 0
        self.highest_total_reward = 0
        self.old_rewards = []
        self.old_snake_size = []
        self.old_food_eaten = []
        self.old_moves = []
        self.old_total_reward = []


        


        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        '''
        Log my_custom_reward every _log_freq(th) to tensorboard for each environment
        '''

        # Snippet from snake_env.py
        #self.state = {
        #    "map": self.map,
        #    "agents": self.snake_bodies,
        #    "food_pos": [],
        #    "walls_pos": [],
        #    "last_observation": {},
        #    "agent_metrics": {agent: {
        #        "snake_size": len(self.snake_bodies[agent]),
        #        "food_eaten": 0,
        #        "moves": 0,
        #        "total_reward": 0,
        #     } for agent in self.agents},
        #}
        #self.state["last_observation"] = {agent: self.get_observation(agent) for agent in self.agents}
        #observations = self.state["last_observation"]
        ## Metrics for each agent to evaluate performance
        #infos = self.state["agent_metrics"]


        if self.n_calls % self._log_freq == 0:
            # log snake_size cur max from infos to tensorboard scalar, it might not exist in all infos
            snake_size = np.max([info['snake_size'] for info in self.locals['infos'] if 'snake_size' in info])
            self.tb_formatter.writer.add_scalar("performance_cur/snake_size_cur_max", snake_size, self.num_timesteps)

            # log snake_size max all time from infos to tensorboard scalar, it might not exist in all infos
            self.highest_snake_size = max(self.highest_snake_size, snake_size)
            self.tb_formatter.writer.add_scalar("performance_max/snake_size_max", self.highest_snake_size, self.num_timesteps)


            # log food_eaten cur max from infos to tensorboard scalar, it might not exist in all infos
            food_eaten = np.max([info['food_eaten'] for info in self.locals['infos'] if 'food_eaten' in info])
            self.tb_formatter.writer.add_scalar("performance_cur/food_eaten_cur_max", food_eaten, self.num_timesteps)

            # log food_eaten max all time from infos to tensorboard scalar, it might not exist in all infos
            self.highest_food_eaten = max(self.highest_food_eaten, food_eaten)
            self.tb_formatter.writer.add_scalar("performance_max/food_eaten_max", self.highest_food_eaten, self.num_timesteps)

            # log moves cur max from infos to tensorboard scalar, it might not exist in all infos
            moves = np.max([info['moves'] for info in self.locals['infos'] if 'moves' in info])
            self.tb_formatter.writer.add_scalar("performance_cur/moves_cur_max", moves, self.num_timesteps)

            # log moves max all time from infos to tensorboard scalar, it might not exist in all infos
            self.highest_moves = max(self.highest_moves, moves)
            self.tb_formatter.writer.add_scalar("performance_max/moves_max", self.highest_moves, self.num_timesteps)

            # log total_reward cur max from infos to tensorboard scalar, it might not exist in all infos
            total_reward = np.max([info['total_reward'] for info in self.locals['infos'] if 'total_reward' in info])
            self.tb_formatter   .writer.add_scalar("performance_cur/total_reward__max", total_reward, self.num_timesteps)

            # log total_reward max all time from infos to tensorboard scalar, it might not exist in all infos
            self.highest_total_reward = max(self.highest_total_reward, total_reward)
            self.tb_formatter.writer.add_scalar("performance_max/total_reward_max", self.highest_total_reward, self.num_timesteps)
            
            # log mean reward to tensorboard scalar
            self.tb_formatter.writer.add_scalar("performance_cur/mean_reward", np.mean(self.locals['rewards']), self.num_timesteps)

            # log std reward to tensorboard scalar
            self.tb_formatter.writer.add_scalar("performance_cur/std_reward", np.std(self.locals['rewards']), self.num_timesteps)

            # log sem reward to tensorboard scalar
            self.tb_formatter.writer.add_scalar("performance_cur/sem_reward", np.std(self.locals['rewards']) / np.sqrt(len(self.locals['rewards'])), self.num_timesteps)

            # log min reward to tensorboard scalar
            self.tb_formatter.writer.add_scalar("performance_cur/min_reward", np.min(self.locals['rewards']), self.num_timesteps)

            # log max reward to tensorboard scalar
            self.tb_formatter.writer.add_scalar("performance_cur/max_reward", np.max(self.locals['rewards']), self.num_timesteps)


            # First save the new values and remove the old values, remember self.locals['rewards'] contains all rewards for the current episode so
            if(len(self.old_rewards) == 0):
                self.old_rewards = self.locals['rewards']
            else:
                self.old_rewards = self.old_rewards + self.locals['rewards']
                self.old_rewards = self.old_rewards[-self.avg_size:]


            # Do the same for other interesting metrics like snake_size, food_eaten, moves, total_reward
            if (len(self.old_snake_size) == 0):
                self.old_snake_size = [info['snake_size'] for info in self.locals['infos'] if 'snake_size' in info]
            else:
                self.old_snake_size = self.old_snake_size + [info['snake_size'] for info in self.locals['infos'] if 'snake_size' in info]
                self.old_snake_size = self.old_snake_size[-self.avg_size:]
            
            if (len(self.old_food_eaten) == 0):
                self.old_food_eaten = [info['food_eaten'] for info in self.locals['infos'] if 'food_eaten' in info]
            else:
                self.old_food_eaten = self.old_food_eaten + [info['food_eaten'] for info in self.locals['infos'] if 'food_eaten' in info]
                self.old_food_eaten = self.old_food_eaten[-self.avg_size:]

            if (len(self.old_moves) == 0):
                self.old_moves = [info['moves'] for info in self.locals['infos'] if 'moves' in info]
            else:
                self.old_moves = self.old_moves + [info['moves'] for info in self.locals['infos'] if 'moves' in info]
                self.old_moves = self.old_moves[-self.avg_size:]

            if (len(self.old_total_reward) == 0):
                self.old_total_reward = [info['total_reward'] for info in self.locals['infos'] if 'total_reward' in info]
            else:
                self.old_total_reward = self.old_total_reward + [info['total_reward'] for info in self.locals['infos'] if 'total_reward' in info]
                self.old_total_reward = self.old_total_reward[-self.avg_size:]
                

            if self.num_timesteps > self.avg_size and self.num_timesteps % self.avg_freq == 0:


                # Write the average values to tensorboard
                self.tb_formatter.writer.add_scalar("performance_avg_reward/mean_reward", np.mean(self.old_rewards), self.num_timesteps)
                self.tb_formatter.writer.add_scalar("performance_avg_reward/std_reward", np.std(self.old_rewards), self.num_timesteps)
                self.tb_formatter.writer.add_scalar("performance_avg_reward/sem_reward", np.std(self.old_rewards) / np.sqrt(len(self.old_rewards)), self.num_timesteps)
                self.tb_formatter.writer.add_scalar("performance_avg_reward/min_reward", np.min(self.old_rewards), self.num_timesteps)
                self.tb_formatter.writer.add_scalar("performance_avg_reward/max_reward", np.max(self.old_rewards), self.num_timesteps)

                # Snake size
                self.tb_formatter.writer.add_scalar("performance_avg_snake_size/snake_size_avg", np.mean(self.old_snake_size), self.num_timesteps)
                self.tb_formatter.writer.add_scalar("performance_avg_snake_size/snake_size_std", np.std(self.old_snake_size), self.num_timesteps)
                self.tb_formatter.writer.add_scalar("performance_avg_snake_size/snake_size_sem", np.std(self.old_snake_size) / np.sqrt(len(self.old_snake_size)), self.num_timesteps)
                self.tb_formatter.writer.add_scalar("performance_avg_snake_size/snake_size_min", np.min(self.old_snake_size), self.num_timesteps)
                self.tb_formatter.writer.add_scalar("performance_avg_snake_size/snake_size_max", np.max(self.old_snake_size), self.num_timesteps)

                # Food eaten
                self.tb_formatter.writer.add_scalar("performance_avg_food_eaten/food_eaten_avg", np.mean(self.old_food_eaten), self.num_timesteps)
                self.tb_formatter.writer.add_scalar("performance_avg_food_eaten/food_eaten_std", np.std(self.old_food_eaten), self.num_timesteps)
                self.tb_formatter.writer.add_scalar("performance_avg_food_eaten/food_eaten_sem", np.std(self.old_food_eaten) / np.sqrt(len(self.old_food_eaten)), self.num_timesteps)
                self.tb_formatter.writer.add_scalar("performance_avg_food_eaten/food_eaten_min", np.min(self.old_food_eaten), self.num_timesteps)
                self.tb_formatter.writer.add_scalar("performance_avg_food_eaten/food_eaten_max", np.max(self.old_food_eaten), self.num_timesteps)

                # Moves
                self.tb_formatter.writer.add_scalar("performance_avg_moves/moves_avg", np.mean(self.old_moves), self.num_timesteps)
                self.tb_formatter.writer.add_scalar("performance_avg_moves/moves_std", np.std(self.old_moves), self.num_timesteps)
                self.tb_formatter.writer.add_scalar("performance_avg_moves/moves_sem", np.std(self.old_moves) / np.sqrt(len(self.old_moves)), self.num_timesteps)
                self.tb_formatter.writer.add_scalar("performance_avg_moves/moves_min", np.min(self.old_moves), self.num_timesteps)
                self.tb_formatter.writer.add_scalar("performance_avg_moves/moves_max", np.max(self.old_moves), self.num_timesteps)

                # Total reward
                self.tb_formatter.writer.add_scalar("performance_avg_total_reward/total_reward_avg", np.mean(self.old_total_reward), self.num_timesteps)
                self.tb_formatter.writer.add_scalar("performance_avg_total_reward/total_reward_std", np.std(self.old_total_reward), self.num_timesteps)
                self.tb_formatter.writer.add_scalar("performance_avg_total_reward/total_reward_sem", np.std(self.old_total_reward) / np.sqrt(len(self.old_total_reward)), self.num_timesteps)
                self.tb_formatter.writer.add_scalar("performance_avg_total_reward/total_reward_min", np.min(self.old_total_reward), self.num_timesteps)
                self.tb_formatter.writer.add_scalar("performance_avg_total_reward/total_reward_max", np.max(self.old_total_reward), self.num_timesteps)



            

        return True
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record('random_value', value)
        return True                

def train(training_goal = 100_000, n_steps = 512, batch_size = 64, num_vec_envs=1, num_cpus=8, map_width=11, map_height=11):

    training_jumps = [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000, 60_000_000, 70_000_000, 80_000_000, 90_000_000, 100_000_000]
    trained_models = []
    print(f'training {training_version} to {human_format(training_goal)} steps')
    print('n_steps', n_steps, 'batch_size', batch_size)

    env = snake_env.create_env(render_mode="disabled", num_vec_envs=num_vec_envs, num_cpus=num_cpus)
    model = PPO(
        MlpPolicy,
        env,    
        verbose=3,
        n_steps=n_steps,
        batch_size=batch_size,
    )

    trained_so_far = 0
    for jump in training_jumps:
        if training_goal >= jump:
            dir_name = f"models/{now_str}"
            os.makedirs(f'{dir_name}', exist_ok=True)
            cur_model_name = f'{dir_name}/pz_snake_{training_version}_{map_width}x{map_height}_{now_str}_{human_format(jump)}'
            print(f'training {cur_model_name}')

            model.learn(total_timesteps=(jump - trained_so_far))
            model.save(cur_model_name)

            trained_models.append(cur_model_name)
            print(f'trained {cur_model_name}')
            trained_so_far += jump
        else:
            print(f'skipping {human_format(jump)} steps')
            break
    env.close()

    env = snake_env.create_env(render_mode="disabled", num_vec_envs=1, num_cpus=num_cpus)
    print()
    for model_name in trained_models:
        print(f'evaluating {model_name}')
        eval.evaluate(model_name, env, 1000)
        print()
    env.close()

def matrix_trainer(combinations, n_steps = 512, batch_size = 64, num_vec_envs=1, num_cpus=8, from_combo=0, to_combo=99999999999):
    to_combo = min(to_combo, len(combinations))
    assert from_combo < to_combo, f'from_combo {from_combo} must be less than to_combo {to_combo}'
    print(f'all combinations: {len(combinations)}, training {from_combo} to {to_combo} for a total of {to_combo - from_combo} combinations')

    training_jumps = [25_000_000]
    for i, combo in enumerate(combinations):
        if i < from_combo:
            continue
        if i > to_combo:
            break
        print(f'combo {i} of {to_combo}, left till completion: {to_combo - i}')
        print(f'Difficulty: {game_parameter_difficulty_estimator(combo)}, combo: {combo}')
        print()
        env = snake_env.create_env_from_combo(combo, render_mode="disabled", num_vec_envs=num_vec_envs, num_cpus=num_cpus)

        os.makedirs(f'./logs/{now_str}', exist_ok=True)
        model = PPO(
            MlpPolicy,
            env,    
            verbose=3,
            n_steps=n_steps,
            batch_size=batch_size,
            tensorboard_log=f"./logs/{now_str}",

        )        

        trained_so_far = 0
        for training_goal in training_jumps:
            print(f'training {training_version} to {human_format(training_goal)} steps')
            print('n_steps', n_steps, 'batch_size', batch_size)

            model.learn(
                total_timesteps=(training_goal - trained_so_far), 
                callback=SummaryWriterCallback(),
                tb_log_name=f'pz_snake_{training_version}_{now_str}_{human_format(training_goal)}_{i}')
            trained_so_far += training_goal

            dir_name = f"models/{now_str}"
            os.makedirs(f'{dir_name}', exist_ok=True)
            cur_model_name = f'{dir_name}/pz_snake_{training_version}_{now_str}_{human_format(training_goal)}_{i}'
            print(f'saving {cur_model_name}')
            model.save(cur_model_name)
            print(f'saved {cur_model_name}')
            print()

        env.close()

limits = {
    'map_size': [5, 11, 19],
    'food_chance': [0.20],
    'snake_start_len': [0],
    'food_total_max': [2, 10, 15],
    'walls_enabled': [False, True],
    'walls_max': [2, 10, 15],
    'walls_chance': [0.20]
}
combinations = game_parameter_combinations(limits)

steps = 32_000
batch = 8_000

# Models trained so far  [0 - 11] [18 - 21]
matrix_trainer(combinations, num_vec_envs=8, num_cpus=8, n_steps=steps, batch_size=batch, from_combo=32, to_combo=len(combinations))
# Full
#matrix_trainer(combinations, num_vec_envs=8, num_cpus=8, n_steps=steps, batch_size=batch)
# Part 1
#matrix_trainer(combinations, num_vec_envs=8, num_cpus=8, n_steps=steps, batch_size=batch, from_combo=0, to_combo=len(combinations)//2)
# Part 2
#matrix_trainer(combinations, num_vec_envs=8, num_cpus=8, n_steps=steps, batch_size=batch, from_combo=len(combinations)//2, to_combo=len(combinations))



#train(training_goal=50_000_000, num_vec_envs=1, n_steps=100_000, batch_size=200)



#model = PPO(
#    "MlpPolicy",  # Use MlpPolicy for simplicity, you can change it based on your requirements
#    env,
#    verbose=3,
#    learning_rate=0.00025,  # Adjust based on your problem
#    n_steps=2048,  # Adjust based on your problem
#    batch_size=64,  # Adjust based on your problem
#    n_epochs=10,  # Adjust based on your problem
#    gamma=0.99,  # Adjust based on your problem
#    gae_lambda=0.95,  # Adjust based on your problem
#    clip_range=0.2,  # Adjust based on your problem
#    clip_range_vf=1.0,  # Adjust based on your problem
#    ent_coef=0.0,  # Adjust based on your problem
#    vf_coef=0.5,  # Adjust based on your problem
#    max_grad_norm=0.5,  # Adjust based on your problem
#    tensorboard_log="./logs",  # Adjust based on your preferences
#    create_eval_env=True,  # Adjust based on your preferences
#    policy_kwargs=dict(net_arch=[64, 64]),  # Adjust based on your problem
#)
