import snake_env
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import supersuit as ss
import time
import numpy as np
import random
import os


def evaluate(model, env, num_episodes=50, render=False, seed=None):
    if seed is not None:
        random.seed(seed)

    # Wrap the environment in a Monitor wrapper

    model = PPO.load(model)
    all_episode_rewards = [] # Array of dicts with rewards for each agent
    average_longest_snake = 0
    for episode in range(num_episodes):
        obs = env.reset() if seed is None else env.reset(seed=random.randint(0, 100000))
        episode_rewards = {agent: 0 for agent in range(len(obs))}
        longest_snake = 0

        while True:
            actions, _states = model.predict(obs)

            # Step through the environment
            observations, rewards, terminations, infos = env.step(actions)

            # Accumulate rewards for each agent 
            for agent in range(len(rewards)):
                episode_rewards[agent] += rewards[agent]

            # Get the longest snake from snake_size in infos
            for info in infos:
                if 'snake_size' in info:
                    longest_snake = max(longest_snake, info['snake_size'])
                

            if render:
                env.render()

            # Check if all agents have terminated
            if all(value == 1 for value in terminations):
                break

        # Store the episode rewards for later analysis
        all_episode_rewards.append(episode_rewards) 
        average_longest_snake += longest_snake

    average_longest_snake /= num_episodes
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
    print("Mean Episode Rewards:", mean_rewards, "Average Longest Snake:", average_longest_snake)
    print("Best Mean:", best_rewards, "Worst Mean:", worst_rewards)
    print("Standard Deviation:", std_rewards, "Standard Error:", sem_rewards)
    print("Best Agent Reward:", best_agent_reward, "Best Agent Rewards:", best_agent_rewards)
    
    res = evaluate_policy(model, env, n_eval_episodes=num_episodes, render=render)
    print('Evaluation results as mean reward per episode & std of reward per episode: ', res)

    return mean_rewards


if __name__ == "__main__":
    files = os.listdir()
    files = [file for file in files if file.endswith('.zip')]
    files.sort()

    print("Available models:")
    for i, file in enumerate(files):
        print(i, file)
        
    model = input("Enter model number: ")
    model = files[int(model)]
    env =  snake_env.create_env(render_mode="disabled", num_vec_envs=1, num_cpus=os.cpu_count())
    evaluate(model, env, num_episodes=100, render=True)