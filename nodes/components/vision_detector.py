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
import time
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
    def detect_target_object(self, bgr_image: np.ndarray, object_type: str, color: str) -> Tuple[Optional[Tuple[Tuple[int, int], float, Tuple[int, int, int, int]]], Dict[str, np.ndarray]]:
        """Detect target object using shape-first approach.
        
        Args:
            bgr_image: Input BGR image
            object_type: Type of object ('cube' or 'rectangular_prism')
            color: Color of object ('red', 'green', 'blue', 'yellow')
            
        Returns:
            Tuple of (detection_result, debug_images):
            - detection_result: ((x, y), confidence, (x, y, w, h)) tuple or None if no object detected
              where (x, y) is the centroid and (x, y, w, h) is the bounding rectangle
            - debug_images: Dictionary containing debug visualization images
        """
        # Create HSV image for color filtering
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        
        # Create initial color mask for debugging purposes
        initial_mask = self._create_color_mask_unified(hsv_image, color)
        
        # Step 1: Shape Detection with color guidance (hybrid approach)
        shape_candidates = self._detect_shapes_in_image(bgr_image, object_type, color)
        
        # If no shape candidates found, return early with debug images
        if not shape_candidates:
            self.logger.log_debug(f"No shape candidates found for {object_type}")
            detection_visualization = self._create_detection_visualization(bgr_image, None, color, object_type, None)
            return None, {"initial_mask": initial_mask, "final_mask": initial_mask, "detection_visualization": detection_visualization}
        
        # Step 2: Color Filtering (Secondary) - Only within detected shapes
        valid_targets = self._filter_shapes_by_color(shape_candidates, color, hsv_image)
        
        # Step 3: Target Validation and Ranking
        detection_result = self._validate_and_rank_targets(valid_targets)
        
        # Get bounding rectangle for visualization if we have a valid detection
        bounding_rect = None
        if detection_result and valid_targets:
            # Find the target with matching centroid
            for target in valid_targets:
                centroid, confidence, area, contour = target
                if centroid == detection_result[0]:  # Match by centroid
                    x, y, w, h = cv2.boundingRect(contour)
                    bounding_rect = (x, y, w, h)
                    # Update detection result to include bounding rectangle
                    detection_result = (detection_result[0], detection_result[1], bounding_rect)
                    break
        
        # Step 4: Create detection visualization with bounding rectangle
        detection_visualization = self._create_detection_visualization(
            bgr_image, detection_result, color, object_type, bounding_rect
        )
        
        # Create final mask for debugging (combine all valid target masks)
        final_mask = np.zeros_like(initial_mask)
        if valid_targets:
            for _, _, _, contour in valid_targets:
                cv2.drawContours(final_mask, [contour], 0, 255, -1)
        
        # Step 5: Create debug images dictionary
        debug_images = {
            "initial_mask": initial_mask,
            "final_mask": final_mask,
            "detection_visualization": detection_visualization
        }
        
        # Step 6: Save debug images if recording
        if self._should_save_debug_images():
            self._request_debug_image_save(bgr_image, initial_mask, final_mask, detection_result, color, object_type)
        
        # Log detection results
        if detection_result:
            centroid, confidence, rect = detection_result
            self.logger.log_debug(f"Shape-first detection successful: {color} {object_type} at {centroid} with confidence {confidence:.3f}, bbox {rect}")
        else:
            self.logger.log_debug(f"Shape-first detection failed: no valid {color} {object_type} found")
        
        return detection_result, debug_images
    
    def _detect_shapes_in_image(self, bgr_image: np.ndarray, object_type: str, color: str = None) -> List[Tuple]:
        """Detect shapes in the image matching the specified object type.
        
        Args:
            bgr_image: Input BGR image
            object_type: Type of object to find
            color: Optional color to guide edge detection
            
        Returns:
            List of (contour, shape_type, confidence, centroid) tuples
        """
        # Convert to grayscale for shape detection
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise - using (3,3) or (5,5) as recommended
        kernel_size = (5, 5)  # Reduced from (11,11) to avoid over-smoothing
        blurred = cv2.GaussianBlur(gray, kernel_size, 0)
        
        # Edge detection with adjusted Canny thresholds (20,60) as recommended
        edges = cv2.Canny(blurred, 20, 60)
        
        # If color is provided, use color information to guide edge detection
        if color:
            # Convert to HSV for color detection
            hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
            
            # Create color mask
            color_mask = self._create_color_mask_unified(hsv_image, color)
            
            # Apply color mask to edges (hybrid color-edge approach)
            masked_edges = cv2.bitwise_and(edges, color_mask)
            
            # Use masked edges for contour detection
            edges = masked_edges
        
        # Add morphological operations to connect edge fragments
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Debug: Log number of contours found
        self.logger.log_info(f"Shape detection: Found {len(contours)} contours in edge image")
        
        shape_candidates = []
        
        for i, contour in enumerate(contours):
            # Get area and bounding rectangle
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Debug: Log contour area
            self.logger.log_info(f"Shape detection: Contour {i} area = {area:.1f} pixels")
            
            # Use bounding rectangle approach instead of strict polygon approximation
            aspect_ratio = w / h if h > 0 else 0
            
            # Validate using aspect ratio and minimum area
            if 0.7 < aspect_ratio < 1.3 and area > self.vision_thresholds["min_area"]:
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroid = (cx, cy)
                    
                    # Determine shape type based on aspect ratio
                    shape_type = "square" if 0.9 < aspect_ratio < 1.1 else "rectangle"
                    
                    # Calculate confidence based on area
                    confidence = min(1.0, area / 10000.0)
                    
                    shape_candidates.append((contour, shape_type, confidence, centroid))
                    
                    # Debug: Log valid shape candidate
                    self.logger.log_info(f"Shape detection: Valid {shape_type} found at {centroid} with confidence {confidence:.2f}")
        
        # Debug: Log total valid shapes found
        self.logger.log_info(f"Shape detection: Found {len(shape_candidates)} valid {object_type} shapes")
        
        return shape_candidates
    
    def _validate_shape_candidate(self, contour, object_type: str) -> Optional[Tuple]:
        """Validate if shape candidate meets criteria for the specified object type.
        
        Args:
            contour: OpenCV contour
            object_type: Type of object being searched for
            
        Returns:
            (contour, shape_type, confidence, centroid) tuple or None
        """
        # DEBUG Checking
        area = cv2.contourArea(contour)
        self.logger.log_debug(f"Validating contour with area {area}")
        
        # Approximate contour to polygon
        epsilon = self.vision_thresholds["polygon_epsilon"] * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # DEBUG Checking
        self.logger.log_debug(f"Polygon approximation: {len(approx)} vertices (need 4)")
        
        # Look for 4-sided polygons (rectangles/squares)
        if len(approx) != 4:
            # DEBUG Checking
            self.logger.log_debug("❌ Rejected: Not 4 vertices")
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
            List of valid target tuples (centroid, confidence, area, contour)
        """
        valid_targets = []
        
        # Debug: Log color filtering start
        self.logger.log_info(f"Color filtering: Starting with {len(shape_candidates)} shape candidates for color '{color}'")
        
        # Debug: Log HSV ranges being used
        color_ranges = self._get_color_ranges(color)
        if isinstance(color_ranges, list):
            self.logger.log_info(f"Color filtering: Using dual HSV ranges for '{color}': {color_ranges}")
        else:
            self.logger.log_info(f"Color filtering: Using HSV range for '{color}': {color_ranges}")
        
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
                
                # Debug: Log color match details
                self.logger.log_info(f"Color filtering: Shape {i} - {color_pixels}/{total_pixels} pixels match '{color}' ({color_match_ratio:.2f} or {color_match_ratio * 100:.1f}%)")
                
                # Require sufficient color match
                if color_match_ratio > self.color_match_threshold:
                    # Calculate overall confidence using configured weights
                    shape_weight, color_weight = self.vision_thresholds["confidence_weights"]
                    overall_confidence = (confidence * shape_weight) + (color_match_ratio * color_weight)
                    area = cv2.contourArea(contour)
                    valid_targets.append((centroid, overall_confidence, area, contour))
                    
                    # Debug: Log valid target
                    self.logger.log_info(f"Color filtering: ✅ Valid target! Shape {i} - {shape_type} at {centroid} with confidence {overall_confidence:.2f}, area {area}")
                else:
                    # Debug: Log color match failure
                    self.logger.log_info(f"Color filtering: ❌ Shape {i} - Color match too low: {color_match_ratio:.2f} < threshold {self.color_match_threshold}")
            else:
                self.logger.log_debug(f"  ❌ No pixels in shape mask")
        
        # Debug: Log final results
        self.logger.log_info(f"Color filtering: Found {len(valid_targets)} valid targets matching '{color}'")
        
        return valid_targets
    
    def _validate_and_rank_targets(self, valid_targets: List[Tuple]) -> Optional[Tuple[Tuple[int, int], float]]:
        """Validate and rank targets, returning the best one.
        
        Args:
            valid_targets: List of (centroid, confidence, area, contour) tuples
            
        Returns:
            ((x, y), confidence) tuple or None
        """
        if not valid_targets:
            # Debug: Log no valid targets
            self.logger.log_info("Target validation: No valid targets to rank")
            return None
        
        # Debug: Log all valid targets before ranking
        self.logger.log_info(f"Target validation: Ranking {len(valid_targets)} valid targets")
        for i, target in enumerate(valid_targets):
            centroid, confidence, area, _ = target
            self.logger.log_info(f"Target validation: Target {i} - centroid={centroid}, confidence={confidence:.2f}, area={area}")
        
        # Return the largest valid target (highest area)
        best_target = max(valid_targets, key=lambda x: x[2])  # Sort by area
        centroid, confidence, area, _ = best_target
        
        # Debug: Log best target
        self.logger.log_info(f"Target validation: Best target selected - centroid={centroid}, confidence={confidence:.2f}, area={area}")
        
        # Apply size threshold
        min_area = self.vision_thresholds["min_area"]
        
        # Debug: Log size threshold check
        self.logger.log_info(f"Target validation: Size threshold check - area {area} vs min_area {min_area}")
        
        if area > min_area:
            # Debug: Log successful validation
            self.logger.log_info(f"Target validation: ✅ Target passes size threshold (area {area} > {min_area})")
            return (centroid, confidence)
        else:
            # Debug: Log size threshold failure
            self.logger.log_info(f"Target validation: ❌ Target too small (area {area} < {min_area})")
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
    
    # ========================================
    # HYBRID APPROACH MODULAR FUNCTIONS
    # ========================================
    
    def _preprocess_image_for_detection(self, bgr_image: np.ndarray, color: str) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess image and create initial color mask.
        
        Args:
            bgr_image: Input BGR image
            color: Target color name
            
        Returns:
            Tuple of (hsv_image, initial_mask)
        """
        # Yahboom's proven preprocessing
        img_blur = cv2.GaussianBlur(bgr_image, (5, 5), 0)
        hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
        
        # Multi-range color detection (from coworkers)
        mask = self._create_color_mask_unified(hsv, color)
        
        return hsv, mask
    
    def _apply_morphological_operations(self, mask: np.ndarray) -> np.ndarray:
        """Apply Yahboom's morphological operations for noise removal.
        
        Args:
            mask: Input binary mask
            
        Returns:
            Processed mask after morphological operations
        """
        # Yahboom's proven morphological sequence
        mask_eroded = cv2.erode(mask, None, iterations=2)
        mask_dilated = cv2.dilate(mask_eroded, None, iterations=2)
        mask_final = cv2.GaussianBlur(mask_dilated, (5, 5), 0)
        
        return mask_final
    
    def _validate_object_shape(self, contour, object_type: str) -> bool:
        """Validate object shape using simplified geometric checks.
        
        Args:
            contour: OpenCV contour
            object_type: Target object type ('cube' or 'rectangular_prism')
            
        Returns:
            True if shape is valid for the object type
        """
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Calculate how well the contour fits its bounding rectangle
        area = cv2.contourArea(contour)
        rect_area = w * h
        extent = float(area) / rect_area if rect_area > 0 else 0
        
        # Check if detection is too close to image edges (likely false positive)
        edge_margin = self.vision_thresholds.get("edge_margin", 50)
        image_width, image_height = 1280, 720  # Standard camera resolution
        
        too_close_to_edge = (x < edge_margin or y < edge_margin or 
                           (x + w) > (image_width - edge_margin) or 
                           (y + h) > (image_height - edge_margin))
        
        if too_close_to_edge:
            return False
        
        # Filter for good rectangular shapes
        min_extent = self.vision_thresholds["rectangle_extent"]
        
        if extent <= min_extent:
            return False
        
        # Check aspect ratio based on object type
        square_range = self.vision_thresholds["aspect_ratio_square"]
        
        if object_type == "cube":
            # Should be approximately square
            is_valid = square_range[0] <= aspect_ratio <= square_range[1]
            return is_valid
        elif object_type == "rectangular_prism":
            # Should be rectangular (not square)
            is_valid = aspect_ratio < square_range[0] or aspect_ratio > square_range[1]
            return is_valid
        
        return True  # Accept any shape if object_type not specified
    
    def _should_save_debug_images(self) -> bool:
        """Check if we should save debug images (during recording).
        
        Returns:
            True if debug images should be saved
        """
        # Use existing camera manager's recording state logic
        return hasattr(self, 'camera_manager') and getattr(self.camera_manager, '_was_recording_before_scan', False)
    
    def _request_debug_image_save(self, bgr_image: np.ndarray, initial_mask: np.ndarray, 
                                 final_mask: np.ndarray, detection_result: Optional[Tuple], 
                                 color: str, object_type: str):
        """Request debug image save (DEPRECATED - now handled by coordinated flow).
        
        This function is deprecated as debug image saving is now handled through
        the coordinated flow in robot_arm_interface using experiment_logger.
        
        Args:
            bgr_image: Original BGR image
            initial_mask: Initial color mask
            final_mask: Final processed mask
            detection_result: Detection result tuple or None
            color: Target color
            object_type: Target object type
        """
        # Debug image saving is now handled through coordinated flow in robot_arm_interface
        # This function is kept for backward compatibility but does nothing
        pass
    
    def _create_detection_visualization(self, bgr_image: np.ndarray, detection_result: Optional[Tuple], 
                                  color: str, object_type: str, bounding_rect: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Create annotated detection visualization image with bounding rectangle.
        
        Args:
            bgr_image: Original BGR image
            detection_result: Detection result tuple or None
            color: Target color
            object_type: Target object type
            bounding_rect: Bounding rectangle (x, y, w, h) or None
            
        Returns:
            Annotated detection image showing what was detected
        """
        detection_image = bgr_image.copy()
        
        if detection_result:
            centroid, confidence = detection_result[:2]
            
            # Draw bounding rectangle if available
            if bounding_rect:
                x, y, w, h = bounding_rect
                # Draw rectangle with green color and 2px thickness
                cv2.rectangle(detection_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(detection_image, centroid, 5, (0, 255, 0), -1)
                
                # Add detection label with bounding box dimensions
                label = f"{color} {object_type}: {confidence:.2f}"
                label += f" ({w}x{h})"
                
                # Position label at top of bounding box
                label_position = (x, y - 10) if y > 20 else (x, y + h + 20)
                cv2.putText(detection_image, label, label_position, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # Fallback to just showing centroid if no bounding rect
                cv2.circle(detection_image, centroid, 10, (0, 255, 0), 2)
                cv2.putText(detection_image, f"{color} {object_type}: {confidence:.2f}", 
                           (centroid[0] + 15, centroid[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Draw "NO DETECTION" indicator
            cv2.putText(detection_image, "NO DETECTION", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(detection_image, f"Target: {color} {object_type}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return detection_image
