"""CASS - Campus AI Safety & Surveillance System"""

__version__ = "0.1.0"
__author__ = "katoki-dev"

from .utils.config import load_config
from .utils.logger import setup_logger

__all__ = ["load_config", "setup_logger"]
