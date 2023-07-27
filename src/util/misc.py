#!/usr/bin/env python

import math
import numpy as np


def vec_decompose(vec_to_be_decomposed, direction):
    """
    Decompose the vector along the direction vec
    params:
        vec_to_be_decomposed: np.array, shape:(2,)
        direction: np.array, shape:(2,); |direc| = 1
    return:
        vec_longitudinal
        vec_lateral
            both with sign
    """
    assert vec_to_be_decomposed.shape[0] == 2, direction.shape[0] == 2
    lon_scalar = np.inner(vec_to_be_decomposed, direction)
    lat_vec = vec_to_be_decomposed - lon_scalar * direction
    lat_scalar = np.linalg.norm(lat_vec) * np.sign(lat_vec[0] * direction[1] -
                                                   lat_vec[1] * direction[0])
    return np.array([lon_scalar, lat_scalar], dtype=np.float32)


def delta_angle_between(theta_1, theta_2):
    """
    Compute the delta angle between theta_1 & theta_2(both in degree)
    params:
        theta: float
    return:
        delta_theta: float, in [-pi, pi]
    """
    theta_1 = theta_1 % 360
    theta_2 = theta_2 % 360
    delta_theta = theta_2 - theta_1
    if 180 <= delta_theta and delta_theta <= 360:
        delta_theta -= 360
    elif -360 <= delta_theta and delta_theta <= -180:
        delta_theta += 360
    return delta_theta


def get_speed(vehicle):
  """
  Compute speed of a vehicle in Kmh
  :param vehicle: the vehicle for which speed is calculated
  :return: speed as a float in Kmh
  """
  vel = vehicle.get_velocity()
  return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def get_pos(vehicle):
  """
  Get the position of a vehicle
  :param vehicle: the vehicle whose position is to get
  :return: speed as a float in Kmh
  """
  trans = vehicle.get_transform()
  x = trans.location.x
  y = trans.location.y
  return x, y


def get_info(vehicle):
  """
  Get the full info of a vehicle
  :param vehicle: the vehicle whose info is to get
  :return: a tuple of x, y positon, yaw angle and half length, width of the vehicle
  """
  trans = vehicle.get_transform()
  x = trans.location.x
  y = trans.location.y
  yaw = trans.rotation.yaw / 180 * np.pi
  bb = vehicle.bounding_box
  length = bb.extent.x
  width = bb.extent.y
  info = (x, y, yaw, length, width)
  return info


def get_local_pose(global_pose, ego_pose):
  """
  Transform vehicle to ego coordinate
  :param global_pose: surrounding vehicle's global pose
  :param ego_pose: ego vehicle pose
  :return: tuple of the pose of the surrounding vehicle in ego coordinate
  """
  x, y, yaw = global_pose
  ego_x, ego_y, ego_yaw = ego_pose
  R = np.array([[np.cos(ego_yaw), np.sin(ego_yaw)],
                [-np.sin(ego_yaw), np.cos(ego_yaw)]])
  vec_local = R.dot(np.array([x - ego_x, y - ego_y]))
  yaw_local = yaw - ego_yaw
  local_pose = (vec_local[0], vec_local[1], yaw_local)
  return local_pose


def get_lane_dis(waypoints, x, y):
  """
  Calculate distance from (x, y) to waypoints.
  :param waypoints: a list of list storing waypoints like [[x0, y0], [x1, y1], ...]
  :param x: x position of vehicle
  :param y: y position of vehicle
  :return: a tuple of the distance and the closest waypoint orientation
  """
  dis_min = 1000
  waypt = waypoints[0]
  for pt in waypoints:
    d = np.sqrt((x-pt[0])**2 + (y-pt[1])**2)
    if d < dis_min:
      dis_min = d
      waypt=pt
  vec = np.array([x - waypt[0], y - waypt[1]])
  lv = np.linalg.norm(np.array(vec))
  w = np.array([np.cos(waypt[2]/180*np.pi), np.sin(waypt[2]/180*np.pi)])
  cross = np.cross(w, vec/lv)
  dis = - lv * cross
  return dis, w


def is_within_distance_ahead(target_location, current_location, orientation, max_distance):
  """
  Check if a target object is within a certain distance in front of a reference object.

  :param target_location: location of the target object
  :param current_location: location of the reference object
  :param orientation: orientation of the reference object
  :param max_distance: maximum allowed distance
  :return: True if target object is within max_distance ahead of the reference object
  """
  target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
  norm_target = np.linalg.norm(target_vector)
  if norm_target > max_distance:
    return False

  forward_vector = np.array(
    [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
  d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

  return d_angle < 90.0


def compute_magnitude_angle(target_location, current_location, orientation):
  """
  Compute relative angle and distance between a target_location and a current_location

  :param target_location: location of the target object
  :param current_location: location of the reference object
  :param orientation: orientation of the reference object
  :return: a tuple composed by the distance to the object and the angle between both objects
  """
  target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
  norm_target = np.linalg.norm(target_vector)

  forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
  d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

  return (norm_target, d_angle)


def distance_vehicle(waypoint, vehicle_transform):
  loc = vehicle_transform.location
  dx = waypoint.transform.location.x - loc.x
  dy = waypoint.transform.location.y - loc.y

  return math.sqrt(dx * dx + dy * dy)


def distance_vehicle(waypoint, vehicle_transform):
  loc = vehicle_transform.location
  dx = waypoint.transform.location.x - loc.x
  dy = waypoint.transform.location.y - loc.y

  return math.sqrt(dx * dx + dy * dy)
