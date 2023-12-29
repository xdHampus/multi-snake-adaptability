import snake_env
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import supersuit as ss
import datetime
import eval
from utils import human_format
# get current timedate as string
training_version = "v1_1"


env = snake_env.parallel_env(render_mode="disabled", map_width=16, map_height=16, agent_count=2, snake_start_len=2, food_gen_max=1, food_total_max=5, move_rewards=True, move_rewards_length=True, food_reward=200, death_reward=-50, debug_print=False)
observations, infos = env.reset()
env = ss.black_death_v3(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=4, base_class="stable_baselines3")


time_version = "2021-06-17_17-39-17"
steps_version = "10k"
model_name = f'pz_snake_{training_version}_{time_version}_{steps_version}'
model_name = "pz_snake_v1_1_2023-12-29_15-32-52_100K"
eval.evaluate(model_name, env, 1000)

