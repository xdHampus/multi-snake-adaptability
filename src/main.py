import snake_env
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import supersuit as ss
import numpy as np

env = snake_env.parallel_env(render_mode="human", map_width=16, map_height=16, agent_count=2, snake_start_len=2, food_gen_max=5, food_total_max=25, debug_print=False)
observations, infos = env.reset(seed=113)
env = ss.black_death_v3(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=4, base_class="stable_baselines3")



# Set hyperparameters
model = PPO(
    MlpPolicy,
    env,    
    verbose=3,
    n_steps=2048,
    batch_size=128,
    tensorboard_log="./logs",
)

model.learn(total_timesteps=1_000_000)
model.save("pz_snake_v1_1")
 
##env.close()
#
#while env.agents:
#    # this is where you would insert your policy
#    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
#    actions_as_str = [snake_env.ACTIONS_STR[actions[agent]] for agent in env.agents]
#    print(actions_as_str)
#
#    observations, rewards, terminations, truncations, infos = env.step(actions)
#    # print all the above with headers above
#    print("observations ", observations)
#    print("rewards ", rewards)
#    print("terminations ", terminations)
#    print("truncations ", truncations)
#    print("infos ", infos)
#    print()
#
env.close()