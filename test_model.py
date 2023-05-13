import sys
sys.path.append("/home/docker/src/src/carla-0.9.10-py3.7-linux-x86_64.egg")

import yaml
import numpy as np
import pygame
from stable_baselines3 import PPO

import carla_env_custom

if __name__ == "__main__":
    # Setup environment
    cfg = yaml.safe_load(open("config.yaml", "r"))
    env = carla_env_custom.CarlaEnv(cfg=cfg)
    
    # Load model
    model = PPO.load("/home/docker/repos/test_env/Training/Models/PPO/ppo_carla_2023-05-12_19-22-56.zip")

    # Make pygame display
    pygame.init()
    display = pygame.display.set_mode(
    (1024, 1024),
    pygame.HWSURFACE | pygame.DOUBLEBUF)

    obs, info = env.reset()
    
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            print("Action: ", action)
            obs, reward, done, done, info = env.step(action)
            if done:
                print("Reward: ", reward)
                obs, info = env.reset()
            
            env.display(display=display)
            pygame.display.flip()
    except KeyboardInterrupt:
        pygame.display.quit()