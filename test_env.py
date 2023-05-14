
import stable_baselines3.common.env_checker
import yaml

import carla_env_custom

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml", "r"))
    env = carla_env_custom.CarlaEnv(cfg=cfg)

    stable_baselines3.common.env_checker.check_env(env, warn=False)