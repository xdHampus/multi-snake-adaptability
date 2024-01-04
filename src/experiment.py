from snake_env import create_env_from_combo
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from utils import game_parameter_combinations
from train import training_limits
import supersuit as ss
import time
import numpy as np
import random
import os
import pandas as pd


def evaluate(model, env, num_episodes=50, render=False):
    model = PPO.load(model)
    all_episode_rewards = []
    all_episode_infos_total = []
    for episode in range(num_episodes):
        obs = env.reset()

        episode_info = {}
        episode_info_final = {}
        actions_all = []

        while True:
            actions, _states = model.predict(obs)
            actions_all.append(actions)

            # Step through the environment
            observations, rewards, terminations, infos = env.step(actions)


            for info in infos:
                if info['agent_id'] not in episode_info:
                    episode_info[info['agent_id']] = []

                episode_info[info['agent_id']].append(info)

            # Check if all agents have terminated
            if all(value == 1 for value in terminations):
                # Take the second last element of each list in episode_info if it exists otherwise take the last element
                for key in episode_info:
                    episode_info_final[key] = episode_info[key][-2] if len(episode_info[key]) > 1 else episode_info[key][-1]
                break
        # episode info final dict to list
        all_episode_rewards.append(list(episode_info_final.values()))
    
    # Get average total_reward for every dict in all_episode_rewards
    avg_total_reward = sum([sum([episode_info['total_reward'] for episode_info in episode_infos]) / len(episode_infos) for episode_infos in all_episode_rewards] ) / len(all_episode_rewards)
    avg_snake_size = sum([sum([episode_info['snake_size'] for episode_info in episode_infos]) / len(episode_infos) for episode_infos in all_episode_rewards]) / len(all_episode_rewards)
    avg_food_eaten = sum([sum([episode_info['food_eaten'] for episode_info in episode_infos]) / len(episode_infos) for episode_infos in all_episode_rewards]) / len(all_episode_rewards)
    avg_moves = sum([sum([episode_info['moves'] for episode_info in episode_infos]) / len(episode_infos) for episode_infos in all_episode_rewards]) / len(all_episode_rewards)
    max_snake_size = max([max([episode_info['snake_size'] for episode_info in episode_infos]) for episode_infos in all_episode_rewards])
    max_food_eaten = max([max([episode_info['food_eaten'] for episode_info in episode_infos]) for episode_infos in all_episode_rewards])
    max_moves = max([max([episode_info['moves'] for episode_info in episode_infos]) for episode_infos in all_episode_rewards])
    max_total_reward = max([max([episode_info['total_reward'] for episode_info in episode_infos]) for episode_infos in all_episode_rewards])

    # Return as dict
    all_episode_rewards = {
        'avg_total_reward': avg_total_reward,
        'avg_snake_size': avg_snake_size,
        'avg_food_eaten': avg_food_eaten,
        'avg_moves': avg_moves,
        'max_snake_size': max_snake_size,
        'max_food_eaten': max_food_eaten,
        'max_moves': max_moves,
        'max_total_reward': max_total_reward
    }    

    return all_episode_rewards



if __name__ == "__main__":
    model_files = []

    models_dir = "models"

    # Use os.walk to traverse all subdirectories
    for root, dirs, files in os.walk(models_dir):
        # Filter files that end with '.zip'
        model_files.extend([os.path.join(root, file) for file in files if file.endswith('.zip')])
    
    # Remove .zip extension from file names
    model_files = [file[:-4] for file in model_files]

    # Sort based on the last number in the file name separated by '_'
    model_files.sort(key=lambda file: int(file.split('_')[-1]))

    # Get combinations of environments get_game_parameters() and training_limits() and create environments using create_env_from_combo()
    seed = 42
    combinations = game_parameter_combinations(limits=training_limits())

    #environments = [create_env_from_combo(combo, render_mode="disabled", seed=seed) for combo in combinations]

    # Evaluate all environments with all models
    all_results = []
    for i, combo in enumerate(combinations):
        model_results = []
        for j, model in enumerate(model_files):
            env = create_env_from_combo(combo, render_mode="disabled", seed=seed)
            result = evaluate(model, env, num_episodes=500, render=False)
            result['model'] = model.split('_')[-1]
            env.close()
            model_results.append(result)
            print(f"{i + 1}/{len(combinations)} Finished evaluating model {j + 1}/{len(model_files)}")
        for result in model_results:
            result['combo'] = i
        all_results.extend(model_results)

    df = pd.DataFrame(all_results)
    df.to_csv("results.csv")
        
