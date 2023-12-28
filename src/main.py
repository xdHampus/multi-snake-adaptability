import snake_env
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import supersuit as ss


env = snake_env.parallel_env(render_mode="disabled")
observations, infos = env.reset()
env = ss.black_death_v3(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, num_vec_envs=4, num_cpus=4, base_class="stable_baselines3")



model = PPO(
    MlpPolicy,
    env,
    verbose=3,
    batch_size=250,
)

model.learn(total_timesteps=10_000_000)
model.save("pz_snake_v1_0")

#from pettingzoo.test import parallel_api_test
#parallel_api_test(env, num_cycles=1000)





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

#
#seed = 11
#env_steps = 1000  # 2 * env.width * env.height  # Code uses 1.5 to calculate max_steps
#rollout_fragment_length = 50
#model = PPO(MlpPolicy, env, tensorboard_log=f"/tmp/uwa", verbose=3, gamma=0.95, 
#    n_steps=rollout_fragment_length, ent_coef=0.01, 
#    learning_rate=5e-5, vf_coef=1, max_grad_norm=0.9, gae_lambda=1.0, n_epochs=30, clip_range=0.3,
#    batch_size=150)
#
#while env.agents:
#    # this is where you would insert your policy
#    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
#    #print(actions)
#    #actions, _states = model.predict(observations, deterministic=True)
#    observations, rewards, terminations, truncations, infos = env.step(actions)
#
#env.close()


#test_run(env)

#env = gymnasium.make("CartPole-v1", render_mode="human")

#env = make_vec_env('CartPole-v1', n_envs=4)
#
#model = PPO("MlpPolicy", env, verbose=1)
#model.learn(total_timesteps=25000)
#model.save("ppo2_cartpole")

#del model # remove to demonstrate saving and loading

#model = PPO2.load("ppo2_cartpole")

# Enjoy trained agent
#obs = env.reset()
#while True:
#    action, _states = model.predict(obs)
#    obs, rewards, dones, info = env.step(action)
#    env.render()
