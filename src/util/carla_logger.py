"""
    Logging for debugging our CARLA training code. Eventually, should only use it inside
    train.py since it should not be used in any code that would be open sourced as a standalone
    OpenAI gym env.
"""
import logging
import os


def get_carla_logger():
    return logging.getLogger("carla-debug")


def setup_carla_logger(output_dir, exp_name, rank, logger_name="carla-debug"):
    # Logging setup - we need to add a name so that we don't conflict with Tensorboard
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger.setLevel(logging.DEBUG)
    if exp_name is not None:
        log_dir = os.path.join(output_dir, exp_name, "env_logs")
        os.makedirs(log_dir, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(log_dir, f"env_{rank}.log"))
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
    logger.addHandler(logging.StreamHandler())  # Write logging to STDOUT too
    return logger
