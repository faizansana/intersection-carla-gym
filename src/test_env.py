import stable_baselines3.common.env_checker
import yaml

import carla_env_custom

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config_discrete.yaml", "r"))
    env = carla_env_custom.CarlaEnv(cfg=cfg, host="carla_server", port=2000, tm_port=9002)

    stable_baselines3.common.env_checker.check_env(env, warn=True)