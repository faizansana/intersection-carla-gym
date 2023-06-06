# Intersection Environment for Training Reinforcement Learning Algorithms

[OpenAI gym](https://github.com/openai/gym) and [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environment for [CARLA Simulator](https://carla.org/), particularly for a 4-way unsignalized intersection environment.

# Getting Started

## Prerequisites

- [CARLA v0.9.10.1](https://github.com/carla-simulator/carla/releases/tag/0.9.10.1)
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

## Setup

1. Clone the repository

    ```
    git clone https://github.com/faizansana/intersection-carla-gym.git
    ```

2. (Optional) If you want to use the gymnasium environment, then use the main branch. To use gym v0.21, checkout the following branch.

    ```
    git checkout gym-v0.21
    ```

3. From within the working directory, create the python environment.

    ```
    conda create -f environment.yml -n NAME
    ```

# Usage

1. Setup a configuration file based on your requirements:

    ```yaml
    exp_name: "first_test" # Name of experiment
    output_dir: "output" # Name of output directory for logs
    env:
        obs_space: "dict" # Choose from "dict" or "normal"
        continuous: True # If False then discrete mode is used
        target_speeds: [0, 3, 6, 9, 12] # For discrete speed control
        desired_speed: 12 # For continuous speed control
        dt: 0.05 
        port: 2000 # Change this to match your server port
        render: false
        ego_vehicle_filter: "vehicle.lincoln*" # Vehicle to use for ego vehicle
        num_veh: 1 # Number of vehicles in each intersection except ego vehicle
        num_ped: 1 # Number of pedestrians at each crosswalk
        max_steps: 500 # Maximum number of steps per episode
        CAM_RES: 1024 # Camera resolution for rendering
        max_waypt: 200 # Maximum number of waypoints
        pedestrian_proximity_threshold: 2.0 # Threshold to give negative reward when vehicle distance to pedestrian is less than this value
        vehicle_proximity_threshold: 2.5 # Threshold to give negative reward when vehicle distance to other vehicle is less than this value
        reward_weights:
            # Route completion reward
            c_completion: 100.0
            # Collision penalty with vehicle
            c_terminal_collision: -100.0
            # Collision penalty with pedestrian
            c_terminal_pedestrian_collision: -200.0
            # Timeout penalty
            c_terminal_timeout: -10.0
            # Velocity reward constants
            c_v_eff_under_limit: 1.0
            c_v_eff_over_limit: -2.0
            # Penalty for needing another step
            r_step: -0.0
            # Penalty for non-smooth actions
            c_action_reg: -0.0
            # Penalty for yaw delta w.r.t. road heading
            c_yaw_delta: -0.0
            # Penalty for lateral deviation
            c_lat_dev: -0.0
            # Distance from goal penalty
            c_dist_from_goal: 3.5
            # Progress reward
            c_progress: 0.0
            # Penalty for being close to pedestrians
            c_pedestrian_proximity: -10.0
            # Penalty for being close to vehicles
            c_vehicle_proximity: -5.0
    ```

2. Save the config file as `config_name.yaml`.

3. Setup the environment using the following code snippet. This is also found in `test_env.py`.


    ```python
    import yaml

    import carla_env_custom

    if __name__ == "__main__":
        cfg = yaml.safe_load(open("config_name.yaml", "r"))
        env = carla_env_custom.CarlaEnv(cfg=cfg, host="HOST", tm_port=9000)

        obs, info = env.reset()

        while True:
                obs, reward, done, _, info = env.step(np.array([1.0], dtype=np.float32))
                if done:
                    obs, info = env.reset()
    ```





