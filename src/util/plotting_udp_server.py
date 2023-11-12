# Based on implementation done by Mafumaful

import json
import logging
import socket
import time


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
        self.imu_gnss_data = {
            "timestamp": 0.0,
            "IMU": {
                "accelerometer": [0.0, 0.0, 0.0],
                "gyroscope": [0.0, 0.0, 0.0]
            },
            "GNSS": {
                "latitude": 0.0,
                "longitude": 0.0,
                "altitude": 0.0
            },
        }

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def update_IMU(self, acc_gyro):
        # Update IMU data
        self.imu_gnss_data["timestamp"] = acc_gyro.timestamp
        self.imu_gnss_data['IMU']['accelerometer'] = acc_gyro.accelerometer.x, acc_gyro.accelerometer.y, acc_gyro.accelerometer.z
        self.imu_gnss_data['IMU']['gyroscope'] = acc_gyro.gyroscope.x, acc_gyro.gyroscope.y, acc_gyro.gyroscope.z

    def update_GNSS(self, gnss):
        # Update GNSS data
        self.imu_gnss_data['GNSS']['latitude'] = gnss.latitude
        self.imu_gnss_data['GNSS']['longitude'] = gnss.longitude
        self.imu_gnss_data['GNSS']['altitude'] = gnss.altitude

    def send_update(self):
        # Update timestamp
        self.imu_gnss_data['timestamp'] = time.time()

        # Send data
        try:
            data = json.dumps(self.imu_gnss_data).encode()
            self.sock.sendto(data, (self.dest_host, self.dest_port))
            self.logger.info(f"Data sent to {self.dest_host}:{self.dest_port}")
        except socket.error as e:
            self.logger.error(f"Error sending data: {e}")
