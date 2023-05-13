
import sys
sys.path.append("/home/docker/src/src/carla-0.9.10-py3.7-linux-x86_64.egg")

import stable_baselines3.common.env_checker
import yaml

import carla_env_custom

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml", "r"))
    env = carla_env_custom.CarlaEnv(cfg=cfg)

    stable_baselines3.common.env_checker.check_env(env, warn=False)