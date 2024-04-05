import logging


def get_carla_logger():
    return logging.getLogger('carla-debug')


def setup_carla_logger(logger_name='carla-debug'):
    # Logging setup - we need to add a name so that we don't conflict with Tensorboard
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())  # Write logging to STDOUT too
    return logger
