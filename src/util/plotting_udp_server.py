# Based on implementation done by Mafumaful
import json
import logging
import socket
import time

import carla


class PlottingUDPServer():
    """
    A class for sending IMU and GNSS data over UDP.

    Attributes:
    - dest_host (str): The destination host for the UDP packets. Defaults to "localhost".
    - dest_port (int): The destination port for the UDP packets. Defaults to 9870.
    - sock (socket.socket): The UDP socket used for sending data.
    - imu_gnss_data (dict): A dictionary containing the latest IMU and GNSS data.

    Methods:
    - __init__(self, host: str = "localhost", port: int = 9870): Initializes the UDPServer object.
    - update_IMU(self, acc_gyro): Updates the IMU data in the imu_gnss_data dictionary.
    - update_GNSS(self, gnss): Updates the GNSS data in the imu_gnss_data dictionary.
    """

    def __init__(self, host: str = "localhost", port: int = 9870):
        # Set destination host and port
        self.dest_host = host
        self.dest_port = port

        # Create UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Initialize IMU and GNSS data dictionary
        self.data = {
            "timestamp": 0.0,
            # IMU data
            "IMU": {
                "accelerometer": [0.0, 0.0, 0.0],
                "gyroscope": [0.0, 0.0, 0.0]
            },
            # GNSS data
            "GNSS": {
                "latitude": 0.0,
                "longitude": 0.0,
                "altitude": 0.0
            },
            # Control
            "control": {
                "steering": 0.0,
                "throttle": 0.0,
                "brake": 0.0
            },
            # Vehicle
            "vehicle": {
                "speed": 0.0,
                "velocity": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0
                },
                "acceleration": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0
                },
                "orientation": {
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0
                },
                "angular_velocity": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0
                }
            }

        }

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def update_IMU(self, acc_gyro: carla.IMUMeasurement):
        # Update IMU data
        self.data['IMU']['accelerometer'] = acc_gyro.accelerometer.x, acc_gyro.accelerometer.y, acc_gyro.accelerometer.z
        self.data['IMU']['gyroscope'] = acc_gyro.gyroscope.x, acc_gyro.gyroscope.y, acc_gyro.gyroscope.z

    def update_GNSS(self, gnss):
        # Update GNSS data
        self.data['GNSS']['latitude'] = gnss.latitude
        self.data['GNSS']['longitude'] = gnss.longitude
        self.data['GNSS']['altitude'] = gnss.altitude

    def update_control(self, control: carla.VehicleControl):
        # Update control data
        self.data['control']['steering'] = control.steer
        self.data['control']['throttle'] = control.throttle
        self.data['control']['brake'] = control.brake

    def update_speed(self, velocity: carla.Vector3D):
        # Calculate speed
        speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        # Update speed data
        self.data["vehicle"]['speed'] = speed

    def update_ego_vehicle(self, ego_vehicle: carla.Vehicle):
        # Update vehicle data
        velocity = ego_vehicle.get_velocity()
        self.data["vehicle"]["velocity"] = {
            "x": velocity.x,
            "y": velocity.y,
            "z": velocity.z
        }
        self.update_speed(ego_vehicle.get_velocity())

        acceleration = ego_vehicle.get_acceleration()
        self.data["vehicle"]["acceleration"] = {
            "x": acceleration.x,
            "y": acceleration.y,
            "z": acceleration.z
        }

        orientation = ego_vehicle.get_transform().rotation
        self.data["vehicle"]["orientation"] = {
            "roll": orientation.roll,
            "pitch": orientation.pitch,
            "yaw": orientation.yaw
        }

        angular_velocity = ego_vehicle.get_angular_velocity()
        self.data["vehicle"]["angular_velocity"] = {
            "x": angular_velocity.x,
            "y": angular_velocity.y,
            "z": angular_velocity.z
        }

    def send_update(self):
        # Update timestamp
        self.data['timestamp'] = time.time()

        # Send data
        try:
            data = json.dumps(self.data).encode()
            self.sock.sendto(data, (self.dest_host, self.dest_port))
            self.logger.debug(f"Data sent to {self.dest_host}:{self.dest_port}")
        except socket.error as e:
            self.logger.error(f"Error sending data: {e}")
