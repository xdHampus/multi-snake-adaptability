import snake_env
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import supersuit as ss
import time
import numpy as np

def evaluate(model, env, num_episodes=50, render=False):
    model = PPO.load(model)
    all_episode_rewards = [] # Array of dicts with rewards for each agent

    for episode in range(num_episodes):
        obs = env.reset()
        episode_rewards = {agent: 0 for agent in range(len(obs))}

        while True:
            actions, _states = model.predict(obs)

            # Step through the environment
            observations, rewards, terminations, infos = env.step(actions)

            # Accumulate rewards for each agent 
            for agent in range(len(rewards)):
                episode_rewards[agent] += rewards[agent]

            # Render the environment
            if render:
                env.render()

            # Check if all agents have terminated
            if all(value == 1 for value in terminations):
                break

        # Store the episode rewards for later analysis
        all_episode_rewards.append(episode_rewards) 

    # Analyze episode rewards
    
    # Get mean rewards for all actors across all episodes and all agents from the array of dicts 
    mean_rewards = np.mean([np.mean(list(episode_rewards.values())) for episode_rewards in all_episode_rewards])
    # Get the best mean rewards for all actors across all episodes and all agents from the array of dicts
    best_rewards = np.max([np.mean(list(episode_rewards.values())) for episode_rewards in all_episode_rewards])
    # Get the worst mean rewards for all actors across all episodes and all agents from the array of dicts
    worst_rewards = np.min([np.mean(list(episode_rewards.values())) for episode_rewards in all_episode_rewards])
    # Get the standard deviation of the mean rewards for all actors across all episodes and all agents from the array of dicts
    std_rewards = np.std([np.mean(list(episode_rewards.values())) for episode_rewards in all_episode_rewards])
    # Get the standard error of the mean rewards for all actors across all episodes and all agents from the array of dicts
    sem_rewards = std_rewards / np.sqrt(num_episodes)

    # Get best reward for a single agent across all episodes
    best_agent_reward = np.max([np.max(list(episode_rewards.values())) for episode_rewards in all_episode_rewards])
    # Get best for both agents across all episodes
    best_agent_rewards = [np.max([episode_rewards[agent] for episode_rewards in all_episode_rewards]) for agent in range(len(obs))]

    # Print results
    print("Mean Episode Rewards:", mean_rewards)
    print("Best Mean:", best_rewards, "Worst Mean:", worst_rewards)
    print("Standard Deviation:", std_rewards, "Standard Error:", sem_rewards)
    print("Best Agent Reward:", best_agent_reward, "Best Agent Rewards:", best_agent_rewards)




    return mean_rewards



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
