import snake_env
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import supersuit as ss
import datetime
import eval
from utils import human_format
# get current timedate as string
now = datetime.datetime.now()
now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
training_version = "v1_1"


env = snake_env.parallel_env(render_mode="disabled", map_width=16, map_height=16, agent_count=2, snake_start_len=2, food_gen_max=1, food_total_max=5, move_rewards=True, move_rewards_length=True, food_reward=200, death_reward=-50, debug_print=False)
observations, infos = env.reset()
env = ss.black_death_v3(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=4, base_class="stable_baselines3")




def train(training_goal = 100_000):
    training_jumps = [10_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000, 100_000_000]
    trained_models = []
    print(f'training {training_version} to {training_goal} steps')
    model = PPO(
        MlpPolicy,
        env,    
        verbose=3,
        n_steps=2048,
        batch_size=128,
    )

    trained_so_far = 0
    for jump in training_jumps:
        if training_goal >= jump:
            cur_model_name = f'pz_snake_{training_version}_{now_str}_{human_format(jump)}'
            print(f'training {cur_model_name}')

            model.learn(total_timesteps=(jump - trained_so_far))
            model.save(cur_model_name)

            trained_models.append(cur_model_name)
            print(f'trained {cur_model_name}')
            trained_so_far += jump
        else:
            print(f'skipping {human_format(jump)} steps')
            break
    
    for model_name in trained_models:
        print(f'evaluating {model_name}')
        eval.evaluate(model_name, env, 1000)


train(2_000_000)
env.close()





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
