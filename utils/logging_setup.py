"""
Sets up logging for the app - helpful for tracking down issues and keeping an eye on what's happening under the hood.
"""

import logging

def setup_logging():
    """
    Configures logging with a basic setup at INFO level.
    You can adjust the level or add more configurations here if needed.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

# TODO: Consider adding a file handler if we want logs saved to a file for longer-running sessions
# TODO: Experiment with different logging levels (DEBUG, WARNING) depending on how much detail you want