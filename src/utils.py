
import os
import logging

def setup_logging(log_file="device_health.log"):
    """Configures logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def ensure_directories(dirs):
    """Ensures that the specified directories exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
