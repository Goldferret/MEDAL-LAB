"""
Components package for DOFBOT Pro modular node.

This package contains specialized components for the DOFBOT Pro robot:
- camera_manager: Camera operations and frame capture
- vision_detector: Object detection and computer vision
- movement_controller: Robot movement and servo control
- calibration_manager: Camera and system calibration
- experiment_logger: Data logging and experiment persistence
"""

from .camera_manager import CameraManager
from .vision_detector import VisionDetector
from .movement_controller import MovementController
from .calibration_manager import CalibrationManager
from .experiment_logger import ExperimentLogger

__all__ = [
    'CameraManager',
    'VisionDetector', 
    'MovementController',
    'CalibrationManager',
    'ExperimentLogger'
]
