#!/usr/bin/env python

from __future__ import division

import copy
import random
import time
import sys
from typing import Dict

sys.path.append("/home/docker/src/carla-0.9.6-py3.5-linux-x86_64.egg")
import carla
import gymnasium as gym
import numpy as np
import pygame
from carla import ColorConverter as cc
from gymnasium import spaces

import util.carla_logger as carla_logger
from util.misc import _vec_decompose, delta_angle_between


class CarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""

    def __init__(self, cfg: Dict):
        for k, v in cfg["env"].items():
            setattr(self, k, v)
        host_num = self.host.split("_")[-1]
        exp_name = cfg["exp_name"]
        outdir = cfg["output_dir"]
        self.logger = carla_logger.setup_carla_logger(output_dir=outdir, exp_name=exp_name, rank=host_num)
        self.logger.info(f"Env running on server {self.host}")

        # action and observation space
        self.action_space = spaces.Box(np.array([-2.0, -2.0]), np.array([2.0, 2.0]), dtype=np.float32)
        self.state_space = spaces.Box(low=-50.0, high=50.0, shape=(12 + 3*self.num_veh + 3*self.num_ped, ), dtype=np.float32)

        # Connect to carla server and get world object
        self._make_carla_client(self.host, self.port)

        # Create the ego vehicle blueprint
        self.ego_bp = self._create_vehicle_bluepprint(self.ego_vehicle_filter, color="49,8,8")

        # Collision sensor
        self.collision_occured = False
        self.collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")

        # Add camera sensor
        self.camera_img = np.zeros((self.CAM_RES, self.CAM_RES, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=60), carla.Rotation(pitch=-90))
        self.camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute("image_size_x", str(self.CAM_RES))
        self.camera_bp.set_attribute("image_size_y", str(self.CAM_RES))
        self.camera_bp.set_attribute("fov", "110")
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute("sensor_tick", "0.02")

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt

        # Record the time of total steps and resetting steps
        self.total_step = 0

        # A dict used for storing state data
        self.state_info = {}

        # A list stores the ids for each episode
        self.actors = []

        # Future distances to get heading
        self.distances = [1., 5., 10.]
        self.target_vehicles = []
        self.peds = []

        self.og_camera_img = None

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

        future_angles = self._get_future_wpt_angle(distances=self.distances)

        # Get dynamics info
        velocity = self.ego.get_velocity()
        accel = self.ego.get_acceleration()
        dyaw_dt = self.ego.get_angular_velocity().z
        v_t_absolute = np.array([velocity.x, velocity.y])
        a_t_absolute = np.array([accel.x, accel.y])

        # decompose v and a to tangential and normal in ego coordinates
        v_t = _vec_decompose(v_t_absolute, ego_heading_vec)
        a_t = _vec_decompose(a_t_absolute, ego_heading_vec)

        pos_err_vec = np.array((ego_x, ego_y)) - self.current_wpt[0:2]

        self.state_info["velocity_t"] = v_t
        self.state_info["acceleration_t"] = a_t

        # self.state_info["ego_heading"] = ego_heading
        self.state_info["delta_yaw_t"] = delta_yaw
        self.state_info["dyaw_dt_t"] = dyaw_dt

        self.state_info["lateral_dist_t"] = np.linalg.norm(pos_err_vec) * \
                                            np.sign(pos_err_vec[0] * road_heading[1] - \
                                                    pos_err_vec[1] * road_heading[0])
        self.state_info["action_t_1"] = self.last_action
        self.state_info["angles_t"] = future_angles
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

    def _collision_event(self, event):
        y_loc = event.transform.location.y
        # Make sure the collision is in the intersection
        if y_loc > -30 and y_loc < 30:
            self.collision_occured = True

    def _to_display_surface(self, image):
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def _create_vehicle_bluepprint(self,
                                   actor_filter,
                                   color=None,
                                   number_of_wheels=[4]):
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

    def _try_spawn_random_vehicle(self, transform, vel):
        vehicle = self.world.try_spawn_actor(self.ego_bp, transform)
        if vehicle is not None:
            vehicle.set_simulate_physics(True)
            vehicle.set_target_velocity(vel)
            collision_sensor = self.world.try_spawn_actor(
                self.collision_bp, carla.Transform(), attach_to=vehicle)
            self.actors.append(collision_sensor)
            collision_sensor.listen(
                lambda event: self._collision_event(event))
            return vehicle
        raise Exception("Failed to spawn target vehicle")

    def _try_spawn_random_ped(self):
        blueprintsWalkers = self.world.get_blueprint_library().filter("vehicle.yamaha.yzf")
        walker_bp = random.choice(blueprintsWalkers)
        transform = carla.Transform()
        transform.location.x = 112
        transform.location.z = 0.5
        transform.location.y = 1.5
        # transform.location.y = 5 + -np.random.randint(0, 10)
        transform.rotation.yaw += -90
        ped = self.world.try_spawn_actor(walker_bp, transform)
        if ped is not None:
            ped.set_target_velocity(carla.Vector3D(0, -2, 0))
            ped.set_simulate_physics(True)
            collision_sensor = self.world.try_spawn_actor(
                self.collision_bp, carla.Transform(), attach_to=ped)
            self.actors.append(collision_sensor)
            collision_sensor.listen(
                lambda event: self._collision_event(event))
            return ped
        raise Exception("Failed to spawn ped")

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
            self.logger.debug(
                "Speed too fast!")
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
        time.sleep(1)
        if vehicle is not None:
            self.actors.append(vehicle)
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

        if self.isCollided or self.isTimeOut:
            return weights["c_terminal"]
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

        r_action_regularized = weights["c_action_reg"] * np.linalg.norm(action)**2

        lateral_dist = self.state_info["lateral_dist_t"]
        r_lateral = weights["c_lat_dev"] * abs(lateral_dist)

        r_dist_from_goal = weights["c_dist_from_goal"] * (-1 + self.state_info["progress"])
        r_progress = weights["c_progress"] * self.state_info["progress"]

        r_tot = r_v_eff + weights["r_step"] + r_delta_yaw + r_action_regularized + r_lateral + r_progress + r_dist_from_goal

        return r_tot

    def _load_world(self):
        self.world = self.client.load_world("Town05")
        self.map = self.world.get_map()
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

    def _make_carla_client(self, host, port):
        self.logger.info("connecting to Carla server...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(100.0)
        self.left_turn_wpts = None
        self._load_world()
        self.logger.info("Carla server port {} connected!".format(port))
        self.left_turn_wpts = []
        r1 = [wpt for wpt in self.map.generate_waypoints(0.1) if wpt.road_id == 48 and wpt.lane_id > 0]
        r1.reverse()
        self.left_turn_wpts += r1
        r2 = [wpt for wpt in self.map.generate_waypoints(0.1) if wpt.road_id == 744 and wpt.lane_id > 0]
        r2.reverse()
        self.left_turn_wpts += r2
        r3 = [wpt for wpt in self.map.generate_waypoints(0.1) if wpt.road_id == 30 and wpt.lane_id == 1]
        r3.reverse()
        self.left_turn_wpts += r3
        self.left_turn_wpts = self.left_turn_wpts[280:550]

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
        min_wp = self.left_turn_wpts[0]
        index = 0
        for i, wp in enumerate(self.left_turn_wpts):
            if location.distance(wp.transform.location) < location.distance(min_wp.transform.location):
                min_wp = wp
                index = i
        return min_wp, float(index) / len(self.left_turn_wpts)

    def _get_future_wpt_angle(self, distances):
        """
        Get next wpts in distances
        params:
            distances: list of int/float, the dist of wpt which user wants to get
        return:
            future_angles: np.array, <current_wpt, wpt(dist_i)> correspond to the dist in distances
        """
        angles = []
        current_wpt, _ = self._get_waypoint(location=self.ego.get_location())
        if not current_wpt:
            self.logger.error("Fail to find a waypoint")
            current_road_heading = self.current_wpt[3]
        else:
            current_road_heading = current_wpt.transform.rotation.yaw

        for d in distances:
            wpts = current_wpt.next(d)
            # if len(wpts) > 1:
            #     wpt = [w for w in wpts if w.road_id == 744][0]
            # else:
            #     wpt = wpts[0]
            wpt = wpts[0]
            # self.world.debug.draw_point(wpt.transform.location, life_time=1, color=carla.Color(r=0,b=255,g=0))
            wpt_heading = wpt.transform.rotation.yaw
            delta_heading = delta_angle_between(current_road_heading,
                                                wpt_heading)
            angles.append(delta_heading)

        return np.array(angles, dtype=np.float32)

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
        delta_yaw_t = np.array(self.state_info["delta_yaw_t"]).reshape(
            (1, )) / 180
        dyaw_dt_t = np.array(self.state_info["dyaw_dt_t"]).reshape((1, )) / 30.0
        lateral_dist_t = self.state_info["lateral_dist_t"].reshape(
            (1, )) / 5      
        action_last = self.state_info["action_t_1"] / 3

        future_angles = self.state_info["angles_t"] / 90
        target_dist_y = np.array(self.state_info["target_vehicles_dist_y"]).reshape((len(self.state_info["target_vehicles_dist_y"]), )) / 40
        target_dist_x = np.array(self.state_info["target_vehicles_dist_x"]).reshape((len(self.state_info["target_vehicles_dist_y"]), )) / 30
        target_vel = np.array(self.state_info["target_vehicles_vel"]).reshape((len(self.state_info["target_vehicles_dist_y"]), )) / 7
        ped_dist_y = np.array(self.state_info["peds_dist_y"]).reshape((len(self.state_info["peds_dist_y"]), )) / 40
        ped_dist_x = np.array(self.state_info["peds_dist_x"]).reshape((len(self.state_info["peds_dist_y"]), )) / 30
        ped_vel = np.array(self.state_info["peds_vel"]).reshape((len(self.state_info["peds_dist_y"]), )) / 7

        
        info_vec = np.concatenate([
            velocity_t, accel_t, delta_yaw_t, dyaw_dt_t, lateral_dist_t,
            action_last, future_angles, target_dist_y, target_dist_x, target_vel, ped_dist_y, ped_dist_x, ped_vel
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

    def reset(self):
        self.ego_collision_sensor = None
        self.lane_sensor = None

        # Delete sensors, vehicles and walkers
        while self.actors:
            (self.actors.pop()).destroy()

        # self._load_world()

        # Spawn the ego vehicle at a random position between start and dest
        # Start and Destination
        self.start = self.left_turn_wpts[0].transform
        self.start.location.z = 10
        self.dest = self.left_turn_wpts[-1].transform

        veh1_t = self._transform(103.5,
            np.random.randint(0, 6) * -8 + 45,
            -90)
        veh1_vel = carla.Vector3D(0, -self.desired_speed, 0)
        veh2_t = self._transform(100.0, 
            np.random.randint(0, 6) * -8 -10,
            90)
        veh2_vel = carla.Vector3D(0, self.desired_speed, 0)
        self.target_vehicles = []
        self.peds = []
        assert(self.num_veh <= 2)
        assert(self.num_ped <= 1)
        if self.num_veh > 0:
            self.target_vehicles.append(self._try_spawn_random_vehicle(veh1_t, veh1_vel))
            self.actors.append(self.target_vehicles[-1])
        if self.num_veh > 1:
            self.target_vehicles.append(self._try_spawn_random_vehicle(veh2_t, veh2_vel))
            self.actors.append(self.target_vehicles[-1])
        if self.num_ped > 0:
            self.peds.append(self._try_spawn_random_ped())
            self.actors.append(self.peds[-1])
        self._try_spawn_ego_vehicle_at(self.start)

        # Add collision sensor
        self.ego_collision_sensor = self.world.try_spawn_actor(
            self.collision_bp, carla.Transform(), attach_to=self.ego)
        self.actors.append(self.ego_collision_sensor)
        self.ego_collision_sensor.listen(
            lambda event: self._collision_event(event))
        self.collision_occured = False

        def get_camera_img(data):
            self.og_camera_img = data
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
        self.actors.append(self.camera_sensor)
        self.camera_sensor.listen(lambda data: get_camera_img(data))

        # Update timesteps
        self.time_step = 1

        # Enable sync mode
        self.settings.synchronous_mode = True
        self.settings.no_rendering_node = not self.render
        self.world.apply_settings(self.settings)

        # Set the initial speed to desired speed
        yaw = (self.start.rotation.yaw) * np.pi / 180.0
        init_fwd_speed = 0
        init_speed = carla.Vector3D(
                    x=init_fwd_speed * np.cos(yaw),
                    y=init_fwd_speed * np.sin(yaw))
        self.ego.set_target_velocity(init_speed)
        physics = self.ego.get_physics_control()
        physics.gear_switch_time *= 0.0
        physics.use_gear_autobox = False
        self.ego.apply_physics_control(physics)
        for _ in range(2):
            self.world.tick()

        # Reset action of last time step
        # TODO:[another kind of action]
        self.last_action = np.array([0.0, 0.0])
        
        self._populate_state_info()

        # End State variable initialized
        self.isCollided = False
        self.isTimeOut = False
        self.isSuccess = False
        self.isOutOfLane = False
        self.isSpecialSpeed = False

        return self._get_obs(), copy.deepcopy(self.state_info)

    def step(self, action):
        current_action = np.array(action) + self.last_action
        current_action = np.clip(
            current_action, -1.0, 1.0, dtype=np.float32)
        throttle_or_brake, steer = current_action
        # throttle_or_brake = 0
        if throttle_or_brake >= 0:
            throttle = throttle_or_brake
            brake = 0
        else:
            throttle = 0
            brake = -throttle_or_brake

        # Apply control
        act = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            gear=1,
            manual_gear_shift=True
            )
        self.ego.apply_control(act)

        for _ in range(1):
            self.world.tick()

        self._populate_state_info()

        # Update timesteps
        self.time_step += 1
        self.total_step += 1
        self.last_action = current_action

        # calculate reward
        isDone = self._terminal()
        current_reward = self._get_reward(np.array(current_action))

        return (self._get_obs(), current_reward, isDone,
                copy.deepcopy(self.state_info))

    def display(self, display):
        if not self.og_camera_img:
            return
        camera_surface = self._to_display_surface(self.og_camera_img)
        display.blit(camera_surface, (0, 0))

if __name__ == "__main__":
    import yaml

    # Make pygame display
    pygame.init()
    display = pygame.display.set_mode(
    (1024, 1024),
    pygame.HWSURFACE | pygame.DOUBLEBUF)

    cfg = yaml.safe_load(open("config.yaml", "r"))
    env = CarlaEnv(cfg=cfg)
    obs, info = env.reset()
    
    try:
        while True:
            obs, reward, done, info = env.step([0.5, 0.0])
            if done:
                obs, info = env.reset()
            
            env.display(display=display)
            pygame.display.flip()
    except KeyboardInterrupt:
        pygame.display.quit()