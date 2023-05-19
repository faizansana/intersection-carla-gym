#!/usr/bin/env python

from __future__ import division

import copy
import random
import sys
import time
from typing import Dict, Union

sys.path.append("../carla-0.9.10-py3.7-linux-x86_64.egg")
import carla
import gym
import numpy as np
import pygame
from carla import ColorConverter as cc
from gym import spaces

import util.carla_logger as carla_logger
import util.misc as helper
from local_carla_agents.navigation.global_route_planner import GlobalRoutePlanner
from local_carla_agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from local_carla_agents.navigation.local_planner import LocalPlanner


class CarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""

    ACTIONS: Dict[int, str] = {
        0: "SLOWER",
        1: "IDLE",
        2: "FASTER"
    }

    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    def __init__(self, cfg: Dict, host: str, tm_port: int = 8000):
        for k, v in cfg["env"].items():
            setattr(self, k, v)
        self.tm_port = tm_port
        host_num = host.split("_")[-1]
        exp_name = cfg["exp_name"]
        outdir = cfg["output_dir"]
        self.logger = carla_logger.setup_carla_logger(output_dir=outdir, exp_name=exp_name, rank=host_num)
        self.logger.info(f"Env running on server {host}")

        # action and observation space
        if self.continuous:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(3)
            self.speed_index = 0
        num_vehicles = 3 * self.num_veh
        num_pedestrians = 4 * self.num_ped
        self.observation_space = spaces.Box(low=-50.0, high=50.0, shape=(7 + 3*num_vehicles + 3*num_pedestrians, ), dtype=np.float32)

        # Connect to carla server and get world object
        self._make_carla_client(host, self.port)

        # Create the ego vehicle blueprint
        self.ego_bp = self._create_vehicle_bluepprint(self.ego_vehicle_filter, color="49,8,8")

        # Collision sensor
        self.collision_occured = False
        self.collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")

        # Add Top down Camera sensor
        self.camera_img = np.zeros((self.CAM_RES, self.CAM_RES, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=50), carla.Rotation(pitch=-90))
        self.camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute("image_size_x", str(self.CAM_RES))
        self.camera_bp.set_attribute("image_size_y", str(self.CAM_RES))
        self.camera_bp.set_attribute("fov", "110")
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute("sensor_tick", "0.02")

        # Record the time of total steps and resetting steps
        self.total_step = 0

        # A dict used for storing state data
        self.state_info = {}

        # A list stores the ids for each episode
        self.actors = []

        # Future distances to get heading
        # self.distances = [1., 5., 10.]
        self.target_vehicles = []
        self.peds = []

        self.og_camera_img = None

        # Make global plan based on start and end points
        global_planner_dao = GlobalRoutePlannerDAO(self.map, sampling_resolution=0.1)
        global_planner = GlobalRoutePlanner(global_planner_dao)
        global_planner.setup()
        # Start and Destination
        self.start = carla.Transform(carla.Location(x=84, y=-110, z=10), carla.Rotation(yaw=270))
        self.dest = carla.Transform(carla.Location(x=49, y=-137), carla.Rotation())
        self.waypoints = global_planner.trace_route(self.start.location, self.dest.location)

        # Setup info variables
        self.isCollided = False
        self.isSuccess = False

    def _populate_state_info(self):
        ego_x, ego_y = self._get_ego_pos()
        self.current_wpt, progress = self._get_waypoint_xyz()

        delta_yaw, wpt_yaw, ego_yaw = self._get_delta_yaw()
        road_heading = np.array(
            [np.cos(wpt_yaw / 180 * np.pi),
             np.sin(wpt_yaw / 180 * np.pi)])
        ego_heading = np.float32(ego_yaw / 180.0 * np.pi)
        ego_heading_vec = np.array((np.cos(ego_heading),
                                    np.sin(ego_heading)))

        # future_angles = self._get_future_wpt_angle(distances=self.distances)

        # Get dynamics info
        velocity = self.ego.get_velocity()
        accel = self.ego.get_acceleration()
        dyaw_dt = self.ego.get_angular_velocity().z
        v_t_absolute = np.array([velocity.x, velocity.y])
        a_t_absolute = np.array([accel.x, accel.y])

        # decompose v and a to tangential and normal in ego coordinates
        v_t = helper.vec_decompose(v_t_absolute, ego_heading_vec)
        a_t = helper.vec_decompose(a_t_absolute, ego_heading_vec)

        pos_err_vec = np.array((ego_x, ego_y)) - self.current_wpt[0:2]

        self.state_info["collision"] = self.isCollided
        self.state_info["success"] = self.isSuccess

        self.state_info["velocity_t"] = v_t
        self.state_info["acceleration_t"] = a_t

        self.state_info["ego_heading"] = ego_heading
        self.state_info["delta_yaw_t"] = delta_yaw
        self.state_info["dyaw_dt_t"] = dyaw_dt

        self.state_info["lateral_dist_t"] = np.linalg.norm(pos_err_vec) * \
            np.sign(pos_err_vec[0] * road_heading[1] -
                    pos_err_vec[1] * road_heading[0])
        self.state_info["action_t_1"] = self.last_action
        # self.state_info["angles_t"] = future_angles
        self.state_info["progress"] = progress

        self.state_info["target_vehicles_dist_y"] = []
        self.state_info["target_vehicles_dist_x"] = []
        self.state_info["target_vehicles_vel"] = []
        for target_veh in self.target_vehicles:
            t_loc = target_veh.get_location()
            e_loc = self.ego.get_location()
            self.state_info["target_vehicles_dist_y"].append(t_loc.y - e_loc.y)
            self.state_info["target_vehicles_dist_x"].append(e_loc.x - t_loc.x)
            self.state_info["target_vehicles_vel"].append(-1*target_veh.get_velocity().y)
        self.state_info["peds_dist_y"] = []
        self.state_info["peds_dist_x"] = []
        self.state_info["peds_vel"] = []
        for target_ped in self.peds:
            t_loc = target_ped.get_location()
            e_loc = target_ped.get_location()
            self.state_info["peds_dist_y"].append(t_loc.y - e_loc.y)
            self.state_info["peds_dist_x"].append(e_loc.x - t_loc.x)
            self.state_info["peds_vel"].append(-1*target_ped.get_velocity().y)

    def _to_display_surface(self, image):
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def _create_vehicle_bluepprint(self,
                                   actor_filter: str,
                                   color=None,
                                   number_of_wheels: list = [4]):
        """Create the blueprint for a specific actor type.

        Args:
            actor_filter: a string indicating the actor type, e.g, "vehicle.lincoln*".

        Returns:
            bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [
                x for x in blueprints
                if int(x.get_attribute("number_of_wheels")) == nw
            ]
        bp = random.choice(blueprint_library)
        if bp.has_attribute("color"):
            if not color:
                color = random.choice(bp.get_attribute("color").recommended_values)
            bp.set_attribute("color", color)
        return bp

    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        # Get ego state
        ego_x, ego_y = self._get_ego_pos()

        # If at destination
        dest = self.dest
        if np.sqrt((ego_x-dest.location.x)**2+(ego_y-dest.location.y)**2) < 2.0:
            self.logger.debug("Get destination!")
            self.isSuccess = True
            return True

        # If collides
        if self.collision_occured:
            self.logger.debug("Collision happened!")
            self.isCollided = True
            return True

        # If reach maximum timestep
        if self.time_step >= self.max_steps:
            self.logger.debug("Time out!")
            self.isTimeOut = True
            return True

        # If out of lane
        if abs(self.state_info["lateral_dist_t"]) > 2.0:
            if self.state_info["lateral_dist_t"] > 0:
                self.logger.debug("Left Lane invasion!")
            else:
                self.logger.debug("Right Lane invasion!")
            self.isOutOfLane = True
            return True

        # If speed is special
        velocity = self.ego.get_velocity()
        v_norm = np.linalg.norm(np.array((velocity.x, velocity.y)))
        if v_norm < 0.0:
            self.logger.debug("Speed too slow!")
            self.isSpecialSpeed = True
            return True
        elif v_norm > (5 * self.desired_speed):
            self.logger.debug("Speed too fast!")
            self.isSpecialSpeed = True
            return True

        return False

    def _get_ego_pos(self):
        """Get the ego vehicle pose (x, y)."""
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        return ego_x, ego_y

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.

        Args:
            transform: the carla transform object.

        Returns:
            Bool indicating whether the spawn is successful.
        """
        vehicle = self.world.spawn_actor(self.ego_bp, transform)
        if vehicle is not None:
            self.ego = vehicle
            return True
        return False

    def _get_obs(self):
        return np.float32(self._info2normalized_state_vector())

    def _get_reward(self, action):
        """
        calculate the reward of current state
        params:
            action: np.array of shape(2,)
        """
        weights = self.reward_weights

        if self.isCollided:
            return weights["c_terminal_collision"]
        if self.isTimeOut:
            return weights["c_terminal_timeout"]
        if self.isSuccess:
            return weights["c_completion"]

        v = self.ego.get_velocity()
        speed_norm = np.linalg.norm(np.array([v.x, v.y]))
        if speed_norm > self.desired_speed:
            r_v_eff = weights["c_v_eff_over_limit"] * abs(self.desired_speed - speed_norm)
        else:
            r_v_eff = weights["c_v_eff_under_limit"] * speed_norm

        delta_yaw, _, _ = self._get_delta_yaw()
        r_delta_yaw = weights["c_yaw_delta"] * delta_yaw

        if self.continuous:
            r_action_regularized = weights["c_action_reg"] * np.linalg.norm(action)**2
        else:
            r_action_regularized = 0

        lateral_dist = self.state_info["lateral_dist_t"]
        r_lateral = weights["c_lat_dev"] * abs(lateral_dist)

        r_dist_from_goal = weights["c_dist_from_goal"] * (-1 + self.state_info["progress"])
        r_progress = weights["c_progress"] * self.state_info["progress"]

        r_tot = r_v_eff + weights["r_step"] + r_delta_yaw + r_action_regularized + r_lateral + r_progress + r_dist_from_goal

        return r_tot

    def _load_world(self):
        self.world = self.client.load_world("Town03")
        self.map = self.world.get_map()
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

    def _make_carla_client(self, host, port):
        self.logger.info("connecting to Carla server...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(100.0)
        self._load_world()

        self.logger.info("Carla server port {} connected!".format(port))

    def _get_delta_yaw(self):
        """
        calculate the delta yaw between ego and current waypoint
        """
        current_wpt, _ = self._get_waypoint(location=self.ego.get_location())
        if not current_wpt:
            self.logger.error("Fail to find a waypoint")
            wpt_yaw = self.current_wpt[2] % 360
        else:
            wpt_yaw = current_wpt.transform.rotation.yaw % 360
        ego_yaw = self.ego.get_transform().rotation.yaw % 360

        delta_yaw = ego_yaw - wpt_yaw
        if 180 <= delta_yaw and delta_yaw <= 360:
            delta_yaw -= 360
        elif -360 <= delta_yaw and delta_yaw <= -180:
            delta_yaw += 360

        return delta_yaw, wpt_yaw, ego_yaw

    def _get_waypoint_xyz(self):
        """
        Get the (x,y) waypoint of current ego position
            if t != 0 and None, return the wpt of last moment
            if t == 0 and None wpt: return self.starts
        """
        waypoint, progress = self._get_waypoint(location=self.ego.get_location())
        waypoint_loc = self._get_waypoint(location=self.ego.get_location())[0].transform.location
        self.world.debug.draw_point(waypoint_loc, life_time=20)
        if waypoint:
            return np.array(
                (waypoint.transform.location.x, waypoint.transform.location.y,
                 waypoint.transform.rotation.yaw)), progress
        else:
            return self.current_wpt, progress

    def _get_waypoint(self, location):
        min_wp = self.waypoints[0]
        index = 0
        for i, wp in enumerate(self.waypoints):
            if location.distance(wp[0].transform.location) < location.distance(min_wp[0].transform.location):
                min_wp = wp
                index = i
        return min_wp[0], float(index) / len(self.waypoints)

    def _info2normalized_state_vector(self):
        """
        params: dict of ego state(velocity_t, accelearation_t, dist, command, delta_yaw_t, dyaw_dt_t)
        type: np.array
        return: array of size[9,], torch.Tensor (v_x, v_y, a_x, a_y
                                                 delta_yaw, dyaw, d_lateral, action_last,
                                                 future_angles)
        """
        velocity_t = self.state_info["velocity_t"] / (self.desired_speed * 1.5)
        accel_t = self.state_info["acceleration_t"] / 40
        delta_yaw_t = np.array(self.state_info["delta_yaw_t"]).reshape((1, )) / 180
        dyaw_dt_t = np.array(self.state_info["dyaw_dt_t"]).reshape((1, )) / 30.0
        lateral_dist_t = self.state_info["lateral_dist_t"].reshape((1, )) / 5
        action_last = self.state_info["action_t_1"] / 3

        # future_angles = self.state_info["angles_t"] / 90
        target_dist_y = np.array(self.state_info["target_vehicles_dist_y"]).reshape((len(self.state_info["target_vehicles_dist_y"]), )) / 40
        target_dist_x = np.array(self.state_info["target_vehicles_dist_x"]).reshape((len(self.state_info["target_vehicles_dist_y"]), )) / 30
        target_vel = np.array(self.state_info["target_vehicles_vel"]).reshape((len(self.state_info["target_vehicles_dist_y"]), )) / 7
        ped_dist_y = np.array(self.state_info["peds_dist_y"]).reshape((len(self.state_info["peds_dist_y"]), )) / 40
        ped_dist_x = np.array(self.state_info["peds_dist_x"]).reshape((len(self.state_info["peds_dist_y"]), )) / 30
        ped_vel = np.array(self.state_info["peds_vel"]).reshape((len(self.state_info["peds_dist_y"]), )) / 7

        info_vec = np.concatenate([
            velocity_t, accel_t, delta_yaw_t, dyaw_dt_t, lateral_dist_t,
            target_dist_y, target_dist_x, target_vel, ped_dist_y, ped_dist_x, ped_vel
        ], axis=0)
        info_vec = info_vec.squeeze()

        return info_vec

    def _transform(self, x, y, yaw):
        veh1_t = carla.Transform()
        veh1_t.location.x = x
        veh1_t.location.y = y
        veh1_t.location.z = 0.2
        veh1_t.rotation.yaw += yaw
        return veh1_t

    def reset(self, seed=None):
        self.ego_collision_sensor = None
        self.camera_sensor = None
        self.collision_occured = False

        self._set_synchronous_mode(False)
        # Delete sensors, vehicles and walkers
        self._clear_all_actors(['sensor.other.collision', 'sensor.camera.rgb', 'vehicle.*', 'controller.ai.walker', 'walker.*'])

        self.target_vehicles = []
        self.peds = []
        assert (self.num_veh <= 3)
        # assert(self.num_ped <= 2)
        if self._try_spawn_ego_vehicle_at(self.start) is False:
            raise Exception("Error: Cannot spawn ego vehicle")
        self._spawn_surrounding_close_proximity_vehicles()
        # Spawn pedestrians
        self._spawn_surrounding_pedestrians()

        # self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
        self.routeplanner = LocalPlanner(self.ego)
        self.routeplanner.set_global_plan(self.waypoints)
        # self.waypoints, _, _ = self.routeplanner.run_step()

        # Add collision sensor
        self.ego_collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
        self.ego_collision_sensor.listen(lambda event: collision_event(event))

        def collision_event(event):
            self.collision_occured = True

        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
        self.camera_sensor.listen(lambda data: get_camera_img(data))

        def get_camera_img(data):
            self.og_camera_img = data

        # Update timesteps
        self.time_step = 1

        # Enable sync mode
        self._set_synchronous_mode(True)

        self._enable_adv_vehicles_autopilot_mode()

        # Reset action of last time step
        # TODO:[another kind of action]
        self.last_action = np.array([0.0], dtype=np.float32)

        self._populate_state_info()

        # End State variable initialized
        self.isCollided = False
        self.isTimeOut = False
        self.isSuccess = False
        self.isOutOfLane = False
        self.isSpecialSpeed = False

        return self._get_obs()

    def _spawn_surrounding_pedestrians(self):
        x = 92.7
        for _ in range(self.num_ped):
            pedestrian = self._try_spawn_random_walker_at(carla.Transform(carla.Location(x=x, y=-144, z=10), carla.Rotation(yaw=random.randint(0, 360))))
            x += 1
            self.peds.append(pedestrian)
        x = 74.6
        for _ in range(self.num_ped):
            self._try_spawn_random_walker_at(carla.Transform(carla.Location(x=x, y=-144, z=10), carla.Rotation(yaw=random.randint(0, 360))))
            x += 1
            self.peds.append(pedestrian)
        x = 92.7
        for _ in range(self.num_ped):
            pedestrian = self._try_spawn_random_walker_at(carla.Transform(carla.Location(x=x, y=-125, z=10), carla.Rotation(yaw=random.randint(0, 360))))
            x += 1
            self.peds.append(pedestrian)
        x = 74.6
        for _ in range(self.num_ped):
            pedestrian = self._try_spawn_random_walker_at(carla.Transform(carla.Location(x=x, y=-125, z=10), carla.Rotation(yaw=random.randint(0, 360))))
            x += 1
            self.peds.append(pedestrian)

    def _enable_adv_vehicles_autopilot_mode(self):
        traffic_manager = self.client.get_trafficmanager(self.tm_port)
        traffic_manager.global_percentage_speed_difference(10)

        for vehicle in self.target_vehicles:
            vehicle.set_autopilot(True, self.tm_port)
            traffic_manager.ignore_lights_percentage(vehicle, 100)
            traffic_manager.distance_to_leading_vehicle(vehicle, 5)

    def _spawn_surrounding_close_proximity_vehicles(self):

        adversary_bp = self._create_vehicle_bluepprint("vehicle.tesla.model3")

        # Vehicle in same lane as ego vehicle
        # adversary_transform = carla.Transform(carla.Location(x=84, y=-100 + random.randint(-10, 10), z=10), carla.Rotation(yaw=270))
        # actor = self.world.try_spawn_actor(adversary_bp, adversary_transform)
        # self.target_vehicles.append(actor)
        # actor.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))

        y = -133.6
        x = 75
        for _ in range(self.num_veh):
            x = x - 15
            adversary_transform = carla.Transform(carla.Location(x=x + random.randint(-5, 5), y=y, z=8), carla.Rotation(yaw=0))
            actor = self.world.try_spawn_actor(adversary_bp, adversary_transform)
            self.target_vehicles.append(actor)
            actor.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))
        y = -135.5
        x = 90
        for _ in range(self.num_veh):
            x = x + 15
            adversary_transform = carla.Transform(carla.Location(x=x + random.randint(-5, 5), y=y, z=10), carla.Rotation(yaw=180))
            actor = self.world.try_spawn_actor(adversary_bp, adversary_transform)
            self.target_vehicles.append(actor)
            actor.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))

        x = 82.5
        y = -135
        for _ in range(self.num_veh):
            y = y - 15
            adversary_transform = carla.Transform(carla.Location(x=x, y=y + random.randint(-5, 5), z=10), carla.Rotation(yaw=90))
            actor = self.world.try_spawn_actor(adversary_bp, adversary_transform)
            self.target_vehicles.append(actor)
            actor.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))

    def _try_spawn_random_walker_at(self, transform):
        """Try to spawn a walker at specific transform with random bluprint.

        Args:
        transform: the carla transform object.

        Returns:
        Bool indicating whether the spawn is successful.
        """
        walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
        # set as not invencible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)

        if walker_actor is not None:
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
            # start walker
            walker_controller_actor.start()
            # set walk to random point
            walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
            # random max speed
            walker_controller_actor.set_max_speed(1 + random.random())    # max speed between 1 and 2 (default is 1.4 m/s)
            return walker_actor
        return None

    def _get_action_speed(self, action: Union[np.ndarray, int]) -> float:
        """Map action to speed value in m/s."""
        if self.continuous:
            # Map action to [-self.desired_speed, self.desired_speed]
            speed = action * self.desired_speed
            return speed[0]
        else:
            if self.ACTIONS[action] == "SLOWER":
                # Shift list to the left
                self.speed_index = max(0, self.speed_index - 1)
            elif self.ACTIONS[action] == "FASTER":
                # Shift list to the right
                self.speed_index = min(len(self.target_speeds) - 1, self.speed_index + 1)
            elif self.ACTIONS[action] == "IDLE":
                self.speed_index = self.speed_index

            return self.target_speeds[self.speed_index]

    def step(self, action: Union[np.ndarray, int]):
        speed_in_vms = self._get_action_speed(action)
        # Convert m/s to km/h since the planner takes km/h as input
        speed_in_km_h = speed_in_vms * 3.6

        self.routeplanner.set_speed(speed_in_km_h)
        control = self.routeplanner.run_step(debug=True)
        self.ego.apply_control(control)

        self.world.tick()

        # Update timesteps
        self.time_step += 1
        self.total_step += 1
        self.last_action = action

        # calculate reward
        isDone = self._terminal()
        current_reward = self._get_reward(np.array(action))

        self._populate_state_info()

        return (self._get_obs(), current_reward, isDone, copy.deepcopy(self.state_info))

    def display(self, display):
        if not self.og_camera_img:
            return
        camera_surface = self._to_display_surface(self.og_camera_img)
        display.blit(camera_surface, (0, 0))

    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == 'controller.ai.walker':
                        actor.stop()
                    actor.destroy()

    def _set_synchronous_mode(self, synchronous=True):
        """Set whether to use the synchronous mode.
        """
        settings = self.world.get_settings()
        if synchronous:
            settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = synchronous
        self.world.apply_settings(settings)


if __name__ == "__main__":
    import yaml

    # Make pygame display
    pygame.init()
    display = pygame.display.set_mode(
        (1024, 1024),
        pygame.HWSURFACE | pygame.DOUBLEBUF)

    cfg = yaml.safe_load(open("config.yaml", "r"))
    env = CarlaEnv(cfg=cfg, host="intersection-driving-carla_server_high-2", tm_port=8010)
    obs, info = env.reset()

    try:
        while True:
            obs, reward, done, done, info = env.step(np.array([1.0], dtype=np.float32))
            if done:
                obs, info = env.reset()

            env.display(display=display)
            pygame.display.flip()
    except KeyboardInterrupt:
        pygame.display.quit()
