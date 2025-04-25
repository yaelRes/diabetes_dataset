"""
Logging utilities for diabetes clustering analysis.
"""

import os
import sys
import logging
from datetime import datetime


def setup_logging(log_dir="logs"):
    """Set up logging configuration.
    
    Args:
        log_dir (str): Directory to store log files
        
    Returns:
        str: Path to the log file
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"diabetes_clustering_analysis_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(stream=sys.stdout)
        ]
    )

    logging.info("Logging initialized")
    return log_file
