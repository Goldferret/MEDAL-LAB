"""
Robot Arm Configuration for DOFBOT Pro

Contains configuration classes and settings for the DOFBOT Pro robot.
"""

from typing import Dict, List, Tuple, Any, Union
from madsci.common.types.node_types import RestNodeConfig

# Type aliases for cleaner annotations
HSVRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]
HSVRanges = Union[HSVRange, List[HSVRange]]
PIDParams = Dict[str, float]


class RobotArmConfig(RestNodeConfig):
    """Configuration for DOFBOT Pro Robot Arm Node."""
    
    # Device configuration
    device_number: int = 0
    
    # Data paths (match original field name)
    data_collection_path: str = "./captures"
    
    # Camera settings
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 15
    
    # Detection thresholds
    position_threshold: int = 5  # pixels
    confidence_threshold: float = 0.7
    color_match_threshold: float = 0.6
    
    # Movement settings
    servo_movement_time: int = 500  # milliseconds
    scan_delay: float = 1.0  # seconds
    
    # Vision detection thresholds (centralized from vision_detector)
    vision_thresholds: Dict[str, Any] = {
        "min_area": 100,                    # Minimum contour area
        "min_radius": 10,                   # Minimum circle radius (Yahboom)
        "confidence_weights": (0.6, 0.4),  # Shape vs color confidence weighting
        "canny_thresholds": (50, 150),     # Edge detection thresholds
        "polygon_epsilon": 0.02,           # Polygon approximation factor
        "rectangle_extent": 0.7,           # Minimum rectangle fill ratio
        "aspect_ratio_square": (0.8, 1.2), # Square aspect ratio range
        "morphology_iterations": 2,        # Erosion/dilation iterations
        "gaussian_kernel": (5, 5),         # Gaussian blur kernel size
    }
    
    # HSV color ranges (calibrated)
    color_hsv_ranges: Dict[str, HSVRanges] = {
        "green": ((40, 65, 20), (91, 248, 88)),
        "blue": ((100, 113, 16), (117, 255, 141)),
        "yellow": ((15, 142, 91), (31, 255, 229)),
        "red": [
            ((0, 135, 39), (10, 255, 158)),
            ((170, 135, 39), (179, 255, 178)),
        ]
    }
    
    # Robot movement presets
    starting_scan_position: List[int] = [180, 110, 5, 0, 90]
    scan_positions: List[int] = [180, 135, 90, 45, 0]
    target_center_point: Tuple[int, int] = (325, 290)
    
    # PID parameters for centering
    pid_params: Dict[str, PIDParams] = {
        "x": {"kp": 0.3, "ki": 0.05, "kd": 0.02},  # For servo 1 (base rotation)
        "y": {"kp": 0.3, "ki": 0.05, "kd": 0.02}   # For servo 4 (vertical)
    }
    
    # Add property for backward compatibility
    @property
    def data_path(self) -> str:
        """Backward compatibility property."""
        return self.data_collection_path
