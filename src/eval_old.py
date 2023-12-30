import snake_env
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import supersuit as ss
import datetime
import eval
from utils import human_format

env = snake_env.create_env(render_mode="disabled")

training_version = "v1_1"
time_version = "2021-06-17_17-39-17"
steps_version = "10k"

model_name = f'pz_snake_{training_version}_{time_version}_{steps_version}'
model_name = "pz_snake_v1_1_2023-12-29_22-07-16_200M"
eval.evaluate(model_name, env, 1000)

