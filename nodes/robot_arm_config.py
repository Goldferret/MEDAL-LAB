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
    confidence_threshold: float = 0.6  # Lowered from 0.7 to accommodate recording workflow operational distances
    color_match_threshold: float = 0.6
    
    # Movement settings
    servo_movement_time: int = 500  # milliseconds
    scan_delay: float = 1.0  # seconds
    
    # Vision detection thresholds (centralized from vision_detector)
    vision_thresholds: Dict[str, Any] = {
        "min_area": 400,                    # Reduced to below largest contour (616 pixels)
        "min_radius": 15,                   # Kept the same - minimum circle radius
        "confidence_weights": (0.6, 0.4),   # Shape vs color confidence weighting
        "canny_thresholds": (15, 45),       # Less sensitive edge detection
        "polygon_epsilon": 0.02,            # More flexible polygon approximation
        "rectangle_extent": 0.4,            # Accept less perfect rectangles
        "aspect_ratio_square": (0.8, 1.2),  # Kept the same - square aspect ratio range
        "morphology_iterations": 2,         # Kept the same - erosion/dilation iterations
        "gaussian_kernel": (3, 3),        # Stronger noise reduction
        "edge_margin": 50,                  # Kept the same - minimum distance from image edges
    }
    
    # HSV color ranges (calibrated with interactive tool across all positions)
    color_hsv_ranges: Dict[str, HSVRanges] = {
        "green": ((27, 87, 22), (99, 255, 139)),
        "blue": ((96, 73, 12), (126, 255, 218)),
        "yellow": ((16, 80, 85), (30, 255, 255)),  # Lowered S min from 134 to 80
        "red": [
            ((0, 137, 62), (10, 255, 182)),         # From calibrated_hsv_ranges_red.json
            ((170, 137, 62), (179, 255, 182)),      # From calibrated_hsv_ranges_red.json
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
