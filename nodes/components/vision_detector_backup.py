"""
Vision Detector Component for DOFBOT Pro

Handles all computer vision and object detection operations including:
- HSV color detection
- Shape detection and analysis
- Object recognition and classification
- Target tracking and centering
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from functools import wraps


def handle_vision_errors(operation_name: str):
    """Decorator for consistent vision error handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                self.logger.log_error(f"Error in {operation_name}: {e}")
                return None
        return wrapper
    return decorator


class VisionDetector:
    """Handles computer vision and object detection for DOFBOT Pro robot."""
    
    def __init__(self, logger, config):
        """Initialize vision detector.
        
        Args:
            logger: Logger instance for logging
            config: Configuration object with HSV ranges and thresholds
        """
        self.logger = logger
        self.config = config
        
        # Get configuration values
        self.color_hsv_ranges = config.color_hsv_ranges
        self.confidence_threshold = config.confidence_threshold
        self.color_match_threshold = config.color_match_threshold
        self.vision_thresholds = config.vision_thresholds
    
    @handle_vision_errors("target object detection")
    def detect_target_object(self, bgr_image: np.ndarray, object_type: str, color: str) -> Optional[Tuple[Tuple[int, int], float]]:
        """Detect target object using shape-first approach for accuracy.
        
        Args:
            bgr_image: Input BGR image
            object_type: Type of object ('cube' or 'rectangular_prism')
            color: Color of object ('red', 'green', 'blue', 'yellow')
            
        Returns:
            ((x, y), confidence) tuple or None if no object detected
        """
        # Step 1: Shape Detection First (more accurate than color-first)
        shape_candidates = self._detect_shapes_in_image(bgr_image, object_type)
        
        self.logger.log_debug(f"Shape detection: Found {len(shape_candidates)} rectangular shapes")
        
        if not shape_candidates:
            self.logger.log_debug("No rectangular shapes found - returning None")
            return None
        
        # Step 2: Color filtering on shape candidates
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        valid_targets = self._filter_shapes_by_color(shape_candidates, color, hsv_image)
        
        if not valid_targets:
            self.logger.log_debug("No valid targets found after color filtering")
            return None
        
        # Step 3: Validate and rank targets
        return self._validate_and_rank_targets(valid_targets)
    
    def _detect_shapes_in_image(self, bgr_image: np.ndarray, object_type: str) -> List[Tuple]:
        """Detect shapes in the image matching the specified object type.
        
        Args:
            bgr_image: Input BGR image
            object_type: Type of object to find
            
        Returns:
            List of (contour, shape_type, confidence, centroid) tuples
        """
        # Convert to grayscale for shape detection
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        kernel_size = self.vision_thresholds["gaussian_kernel"]
        blurred = cv2.GaussianBlur(gray, kernel_size, 0)
        
        # Edge detection
        canny_low, canny_high = self.vision_thresholds["canny_thresholds"]
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shape_candidates = []
        min_area = self.vision_thresholds["min_area"]
        
        for contour in contours:
            # Filter by area (avoid tiny contours)
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Validate shape candidate
            shape_info = self._validate_shape_candidate(contour, object_type)
            if shape_info:
                shape_candidates.append(shape_info)
        
        return shape_candidates
    
    def _validate_shape_candidate(self, contour, object_type: str) -> Optional[Tuple]:
        """Validate if shape candidate meets criteria for the specified object type.
        
        Args:
            contour: OpenCV contour
            object_type: Type of object being searched for
            
        Returns:
            (contour, shape_type, confidence, centroid) tuple or None
        """
        # Approximate contour to polygon
        epsilon = self.vision_thresholds["polygon_epsilon"] * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Look for 4-sided polygons (rectangles/squares)
        if len(approx) != 4:
            return None
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(approx)
        
        # Calculate aspect ratio
        aspect_ratio = float(w) / h
        
        # Calculate how well the contour fits a rectangle
        area = cv2.contourArea(contour)
        rect_area = w * h
        extent = float(area) / rect_area
        
        # Filter for good rectangular shapes
        min_extent = self.vision_thresholds["rectangle_extent"]
        if extent <= min_extent:
            return None
        
        # Determine shape type based on aspect ratio
        square_range = self.vision_thresholds["aspect_ratio_square"]
        if square_range[0] <= aspect_ratio <= square_range[1]:
            shape_type = "square"
            confidence = 0.9 - abs(1.0 - aspect_ratio)  # Closer to 1:1 = higher confidence
        elif aspect_ratio > square_range[1] or aspect_ratio < square_range[0]:
            shape_type = "rectangle"
            confidence = min(0.8, extent)  # Confidence based on how rectangular it is
        else:
            return None
        
        # Check if shape type matches what we're looking for
        if object_type == "cube" and shape_type != "square":
            return None
        elif object_type == "rectangular_prism" and shape_type != "rectangle":
            return None
        
        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = (cx, cy)
            
            return (contour, shape_type, confidence, centroid)
        
        return None
    
    def _filter_shapes_by_color(self, shape_candidates: List[Tuple], color: str, hsv_image: np.ndarray) -> List[Tuple]:
        """Filter shape candidates by color matching.
        
        Args:
            shape_candidates: List of shape candidate tuples
            color: Target color name
            hsv_image: HSV version of the input image
            
        Returns:
            List of valid target tuples (centroid, confidence, area)
        """
        valid_targets = []
        
        for i, candidate in enumerate(shape_candidates):
            contour, shape_type, confidence, centroid = candidate
            
            self.logger.log_debug(f"Checking shape {i + 1}: {shape_type}, confidence={confidence:.2f}")
            
            # Check color within the contour area
            mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            
            # Apply color filtering to the masked region
            color_mask = self._create_color_mask_unified(hsv_image, color)
            combined_mask = cv2.bitwise_and(mask, color_mask)
            
            # Calculate color match percentage
            total_pixels = cv2.countNonZero(mask)
            color_pixels = cv2.countNonZero(combined_mask)
            
            if total_pixels > 0:
                color_match_ratio = color_pixels / total_pixels
                self.logger.log_debug(f"  Color match: {color_pixels}/{total_pixels} = {color_match_ratio:.2f} ({color_match_ratio * 100:.1f}%)")
                
                # Require sufficient color match
                if color_match_ratio > self.color_match_threshold:
                    # Calculate overall confidence using configured weights
                    shape_weight, color_weight = self.vision_thresholds["confidence_weights"]
                    overall_confidence = (confidence * shape_weight) + (color_match_ratio * color_weight)
                    area = cv2.contourArea(contour)
                    valid_targets.append((centroid, overall_confidence, area))
                    self.logger.log_debug(f"  ✅ Valid target! Overall confidence: {overall_confidence:.2f}, area: {area}")
                else:
                    self.logger.log_debug(f"  ❌ Color match too low: {color_match_ratio:.2f} < {self.color_match_threshold}")
            else:
                self.logger.log_debug(f"  ❌ No pixels in shape mask")
        
        return valid_targets
    
    def _validate_and_rank_targets(self, valid_targets: List[Tuple]) -> Optional[Tuple[Tuple[int, int], float]]:
        """Validate and rank targets, returning the best one.
        
        Args:
            valid_targets: List of (centroid, confidence, area) tuples
            
        Returns:
            ((x, y), confidence) tuple or None
        """
        if not valid_targets:
            return None
        
        # Return the largest valid target (highest area)
        best_target = max(valid_targets, key=lambda x: x[2])  # Sort by area
        centroid, confidence, area = best_target
        
        self.logger.log_debug(f"Best target: centroid={centroid}, confidence={confidence:.2f}, area={area}")
        
        # Apply size threshold
        min_area = self.vision_thresholds["min_area"]
        if area > min_area:
            self.logger.log_debug(f"✅ Target passes size threshold (area {area} > {min_area})")
            return (centroid, confidence)
        else:
            self.logger.log_debug(f"❌ Target too small (area {area} < {min_area})")
            return None
    
    def _create_color_mask_unified(self, hsv_image: np.ndarray, color: str) -> np.ndarray:
        """Unified color mask creation handling both single and dual ranges.
        
        Args:
            hsv_image: Input HSV image
            color: Color name
            
        Returns:
            Binary mask for the specified color
        """
        color_ranges = self._get_color_ranges(color)
        
        if isinstance(color_ranges, list):
            # Handle dual ranges (like red)
            masks = []
            for range_tuple in color_ranges:
                hsv_lower, hsv_upper = range_tuple
                mask = cv2.inRange(hsv_image, np.array(hsv_lower), np.array(hsv_upper))
                masks.append(mask)
            return cv2.bitwise_or(masks[0], masks[1])
        else:
            # Handle single range
            hsv_lower, hsv_upper = color_ranges
            return cv2.inRange(hsv_image, np.array(hsv_lower), np.array(hsv_upper))
    
    def _get_color_ranges(self, color: str) -> Union[Tuple, List[Tuple]]:
        """Get HSV ranges for specified color, handling red's dual ranges.
        
        Args:
            color: Color name
            
        Returns:
            HSV range tuple or list of tuples for dual ranges
        """
        return self.color_hsv_ranges[color]
    
    @handle_vision_errors("Yahboom object detection")
    def detect_colored_object_yahboom(self, bgr_image: np.ndarray, hsv_lower: np.ndarray, hsv_upper: np.ndarray, color: str = "") -> Optional[Tuple[int, int]]:
        """Detect colored object using Yahboom's exact approach with calibrated HSV ranges.
        
        Args:
            bgr_image: Input BGR image
            hsv_lower: Lower HSV threshold (numpy array) - unused, kept for compatibility
            hsv_upper: Upper HSV threshold (numpy array) - unused, kept for compatibility
            color: Color name for special handling (e.g., "red")
            
        Returns:
            (x, y) centroid tuple or None if no object detected
        """
        # Yahboom's exact preprocessing steps
        kernel_size = self.vision_thresholds["gaussian_kernel"]
        img_blur = cv2.GaussianBlur(bgr_image, kernel_size, 0)
        hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
        
        # Create mask using unified method
        mask = self._create_color_mask_unified(hsv, color)
        
        # Yahboom's morphological operations
        iterations = self.vision_thresholds["morphology_iterations"]
        mask = cv2.erode(mask, None, iterations=iterations)
        mask = cv2.dilate(mask, None, iterations=iterations)
        mask = cv2.GaussianBlur(mask, kernel_size, 0)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Get largest contour (Yahboom's approach)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get minimum enclosing circle (Yahboom's method)
            (center_x, center_y), radius = cv2.minEnclosingCircle(largest_contour)
            
            # Yahboom's size threshold
            min_radius = self.vision_thresholds["min_radius"]
            if radius > min_radius:
                return (int(center_x), int(center_y))
        
        return None
    
    def create_color_mask(self, hsv_image: np.ndarray, color: str) -> np.ndarray:
        """Create color mask for specified color.
        
        Args:
            hsv_image: Input HSV image
            color: Color name ('red', 'green', 'blue', 'yellow')
            
        Returns:
            Binary mask for the specified color
        """
        return self._create_color_mask_unified(hsv_image, color)
    
    @handle_vision_errors("color coverage calculation")
    def calculate_color_coverage(self, color_mask: np.ndarray) -> Tuple[int, float]:
        """Calculate color coverage statistics.
        
        Args:
            color_mask: Binary color mask
            
        Returns:
            Tuple of (color_pixels, coverage_percentage)
        """
        color_pixels = cv2.countNonZero(color_mask)
        total_pixels = color_mask.shape[0] * color_mask.shape[1]
        coverage_percentage = (color_pixels / total_pixels) * 100
        return color_pixels, coverage_percentage
    
    def update_hsv_ranges(self, new_ranges: Dict[str, Any]):
        """Update HSV color ranges.
        
        Args:
            new_ranges: Dictionary of new HSV ranges
        """
        self.color_hsv_ranges.update(new_ranges)
        self.logger.log_info("HSV color ranges updated")
    
    def get_supported_colors(self) -> List[str]:
        """Get list of supported colors.
        
        Returns:
            List of supported color names
        """
        return list(self.color_hsv_ranges.keys())
