import snake_env
import supersuit as ss
from stable_baselines3.common.env_checker import check_env
from pettingzoo.test import parallel_api_test


env = snake_env.parallel_env(render_mode="human")

# check parallel api
observations, infos = env.reset()
parallel_api_test(env)
