import datetime
import os
import sys
sys.path.append("/home/docker/src/src/carla-0.9.10-py3.7-linux-x86_64.egg")

import yaml
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor

import carla_env_custom
import train_config


if __name__ == "__main__":

    # parameters for the network
    NEURONS = (400, 300)
    LEARNING_RATE = 1e-4
    BUFFER_SIZE = 100000
    LEARNING_STARTS = 1000
    GAMMA = 0.98
    TRAIN_FREQ = (1, "episode")
    GRADIENT_STEPS = -1
    VERBOSE = 1
    TRAINING_TIMESTEPS = 2e6

    # Setup logging and callback
    model_dir = os.path.join("Training", "Models")
    model_save_path = os.path.join(model_dir, f"ddpg_carla_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    callback = train_config.SaveOnBestTrainingRewardCallback(check_freq=20, log_dir=model_dir, save_path=model_save_path, verbose=0)

    # Setup environment
    cfg = yaml.safe_load(open("config.yaml", "r"))
    env = carla_env_custom.CarlaEnv(cfg=cfg)
    env = Monitor(env, model_dir)

    model = DDPG(
        "MlpPolicy",
        env,
        verbose=VERBOSE,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        gamma=GAMMA,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        policy_kwargs={"net_arch": [400, 300]},
        tensorboard_log="./logs/"
    )

    model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=callback, progress_bar=True)
    model.save(os.path.join(model_dir, "last"))
