import snake_env
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import supersuit as ss
import datetime
import eval
from utils import human_format, game_parameter_combinations, game_parameter_difficulty_estimator
import os
import math
# get current timedate as string
now = datetime.datetime.now()
now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
training_version = "v1_1"



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

    training_jumps = [5_000_000, 10_000_000, 15_000_000, 20_000_000, 25_000_000]
    for i, combo in enumerate(combinations):
        if i < from_combo:
            continue
        if i > to_combo:
            break
        print(f'combo {i} of {to_combo}, left till completion: {to_combo - i}')
        print(f'Difficulty: {game_parameter_difficulty_estimator(combo)}, combo: {combo}')
        print()
        env = snake_env.create_env_from_combo(combo, render_mode="disabled", num_vec_envs=num_vec_envs, num_cpus=num_cpus)

        os.makedirs(f'./.logs/{now_str}', exist_ok=True)
        model = PPO(
            MlpPolicy,
            env,    
            verbose=3,
            n_steps=n_steps,
            batch_size=batch_size,
            tensorboard_log=f"./.logs/{now_str}",
        )        

        trained_so_far = 0
        for training_goal in training_jumps:
            print(f'training {training_version} to {human_format(training_goal)} steps')
            print('n_steps', n_steps, 'batch_size', batch_size)
            model.learn(total_timesteps=(training_goal - trained_so_far))
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

steps = 80_000
batch = 8000
# Full
matrix_trainer(combinations, num_vec_envs=6, num_cpus=8, n_steps=steps, batch_size=batch)
# Part 1
#matrix_trainer(combinations, num_vec_envs=6, num_cpus=8, n_steps=steps, batch_size=batch, from_combo=0, to_combo=len(combinations)//2)
# Part 2
#matrix_trainer(combinations, num_vec_envs=6, num_cpus=8, n_steps=steps, batch_size=batch, from_combo=len(combinations)//2, to_combo=len(combinations))



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
