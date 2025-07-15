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
    def detect_target_object(self, bgr_image: np.ndarray, object_type: str, color: str) -> Tuple[Optional[Tuple[Tuple[int, int], float]], Dict[str, np.ndarray]]:
        """Detect target object using hybrid approach combining best practices.
        
        Args:
            bgr_image: Input BGR image
            object_type: Type of object ('cube' or 'rectangular_prism')
            color: Color of object ('red', 'green', 'blue', 'yellow')
            
        Returns:
            Tuple of (detection_result, debug_images):
            - detection_result: ((x, y), confidence) tuple or None if no object detected
            - debug_images: Dictionary containing debug visualization images
        """
        # Step 1: Preprocessing
        hsv_image, initial_mask = self._preprocess_image_for_detection(bgr_image, color)
        
        # Step 2: Morphological processing
        final_mask = self._apply_morphological_operations(initial_mask)
        
        # Step 3: Object detection and validation (now returns radius too)
        detection_with_radius = self._detect_and_validate_objects(bgr_image, final_mask, object_type)
        
        # Extract detection result and radius for visualization
        if detection_with_radius:
            centroid, confidence, radius = detection_with_radius
            detection_result = (centroid, confidence)  # Keep original format for compatibility
        else:
            detection_result = None
            radius = None
        
        # Step 4: Create detection visualization (now with actual radius)
        detection_visualization = self._create_detection_visualization(bgr_image, detection_result, color, object_type, radius)
        
        # Step 5: Create debug images dictionary
        debug_images = {
            "initial_mask": initial_mask,
            "final_mask": final_mask,
            "detection_visualization": detection_visualization
        }
        
        # Step 6: Save debug images if recording (delegated to camera manager)
        if self._should_save_debug_images():
            self._request_debug_image_save(bgr_image, initial_mask, final_mask, detection_result, color, object_type)
        
        # Log detection results
        if detection_result:
            centroid, confidence = detection_result
            self.logger.log_debug(f"Hybrid detection successful: {color} {object_type} at {centroid} with confidence {confidence:.3f}, radius {radius:.1f}")
        else:
            self.logger.log_debug(f"Hybrid detection failed: no valid {color} {object_type} found")
        
        return detection_result, debug_images
    
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
    
    def _detect_and_validate_objects(self, bgr_image: np.ndarray, mask: np.ndarray, object_type: str) -> Optional[Tuple[Tuple[int, int], float, float]]:
        """Detect and validate objects using hybrid approach.
        
        Args:
            bgr_image: Original BGR image
            mask: Processed binary mask
            object_type: Target object type
            
        Returns:
            ((x, y), confidence, radius) tuple or None if no valid object found
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # DEBUG: Log contour information - REMOVE AFTER DEBUGGING
        self.logger.log_info(f"DEBUG: Found {len(contours)} contours for {object_type}")
        
        if not contours:
            # DEBUG: Log when no contours found - REMOVE AFTER DEBUGGING
            self.logger.log_info("DEBUG: No contours found in mask")
            return None
        
        best_candidate = None
        highest_confidence = 0.0
        best_radius = 0.0
        
        # DEBUG: Track all candidates for analysis - REMOVE AFTER DEBUGGING
        debug_candidates = []
        
        for i, contour in enumerate(contours):
            # Yahboom's circular method
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            area = cv2.contourArea(contour)
            
            # DEBUG: Log detailed contour information - REMOVE AFTER DEBUGGING
            self.logger.log_info(f"DEBUG: Contour {i}: area={area:.1f}, radius={radius:.1f}, center=({cx:.1f},{cy:.1f})")
            
            if radius > self.vision_thresholds["min_radius"]:
                # DEBUG: Log radius check pass - REMOVE AFTER DEBUGGING
                self.logger.log_info(f"DEBUG: Contour {i} passed radius check (radius={radius:.1f} > {self.vision_thresholds['min_radius']})")
                
                # MEDAL-LAB's shape validation for precision
                shape_valid = self._validate_object_shape(contour, object_type)
                
                # DEBUG: Log shape validation result - REMOVE AFTER DEBUGGING
                self.logger.log_info(f"DEBUG: Contour {i} shape validation: {shape_valid}")
                
                if shape_valid:
                    # Calculate confidence
                    confidence = min(1.0, area / 1000.0)
                    
                    # DEBUG: Log confidence calculation details - REMOVE AFTER DEBUGGING
                    self.logger.log_info(f"DEBUG: Contour {i} confidence calculation: area={area:.1f} -> confidence={confidence:.3f} (threshold={self.confidence_threshold})")
                    
                    debug_candidates.append({
                        "contour_id": i,
                        "area": area,
                        "radius": radius,
                        "center": (cx, cy),
                        "confidence": confidence,
                        "passes_threshold": confidence > self.confidence_threshold
                    })
                    
                    if confidence > highest_confidence and confidence > self.confidence_threshold:
                        highest_confidence = confidence
                        best_candidate = (int(cx), int(cy))
                        best_radius = radius
                        
                        # DEBUG: Log new best candidate - REMOVE AFTER DEBUGGING
                        self.logger.log_info(f"DEBUG: New best candidate: contour {i} with confidence {confidence:.3f}, radius {radius:.1f}")
                else:
                    # DEBUG: Log shape validation failure - REMOVE AFTER DEBUGGING
                    self.logger.log_info(f"DEBUG: Contour {i} failed shape validation")
            else:
                # DEBUG: Log radius check failure - REMOVE AFTER DEBUGGING
                self.logger.log_info(f"DEBUG: Contour {i} failed radius check (radius={radius:.1f} <= {self.vision_thresholds['min_radius']})")
        
        # DEBUG: Log final summary - REMOVE AFTER DEBUGGING
        self.logger.log_info(f"DEBUG: Detection summary - Found {len(debug_candidates)} valid candidates, best_confidence={highest_confidence:.3f}")
        for candidate in debug_candidates:
            self.logger.log_info(f"DEBUG: Candidate {candidate['contour_id']}: area={candidate['area']:.1f}, confidence={candidate['confidence']:.3f}, passes_threshold={candidate['passes_threshold']}")
        
        if best_candidate:
            # DEBUG: Log successful detection - REMOVE AFTER DEBUGGING
            self.logger.log_info(f"DEBUG: Returning successful detection: {best_candidate} with confidence {highest_confidence:.3f}, radius {best_radius:.1f}")
            return best_candidate, highest_confidence, best_radius
        else:
            # DEBUG: Log detection failure - REMOVE AFTER DEBUGGING
            self.logger.log_info(f"DEBUG: No valid candidates found (highest_confidence={highest_confidence:.3f} < threshold={self.confidence_threshold})")
        
        return None
    
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
        
        # DEBUG: Log shape validation details - REMOVE AFTER DEBUGGING
        self.logger.log_info(f"DEBUG: Shape validation for {object_type}: w={w}, h={h}, aspect_ratio={aspect_ratio:.3f}")
        self.logger.log_info(f"DEBUG: Shape validation: area={area:.1f}, rect_area={rect_area}, extent={extent:.3f}")
        self.logger.log_info(f"DEBUG: Bounding box: x={x}, y={y}, w={w}, h={h}")
        
        # Check if detection is too close to image edges (likely false positive)
        edge_margin = self.vision_thresholds.get("edge_margin", 50)
        image_width, image_height = 1280, 720  # Standard camera resolution
        
        too_close_to_edge = (x < edge_margin or y < edge_margin or 
                           (x + w) > (image_width - edge_margin) or 
                           (y + h) > (image_height - edge_margin))
        
        # DEBUG: Log edge check - REMOVE AFTER DEBUGGING
        self.logger.log_info(f"DEBUG: Edge check: margin={edge_margin}, too_close={too_close_to_edge}")
        
        if too_close_to_edge:
            # DEBUG: Log edge rejection - REMOVE AFTER DEBUGGING
            self.logger.log_info(f"DEBUG: Shape validation FAILED: detection too close to image edge")
            return False
        
        # Filter for good rectangular shapes
        min_extent = self.vision_thresholds["rectangle_extent"]
        
        # DEBUG: Log extent check - REMOVE AFTER DEBUGGING
        self.logger.log_info(f"DEBUG: Extent check: {extent:.3f} > {min_extent} = {extent > min_extent}")
        
        if extent <= min_extent:
            # DEBUG: Log extent failure - REMOVE AFTER DEBUGGING
            self.logger.log_info(f"DEBUG: Shape validation FAILED: extent {extent:.3f} <= {min_extent}")
            return False
        
        # Check aspect ratio based on object type
        square_range = self.vision_thresholds["aspect_ratio_square"]
        
        # DEBUG: Log aspect ratio check - REMOVE AFTER DEBUGGING
        self.logger.log_info(f"DEBUG: Aspect ratio check for {object_type}: {aspect_ratio:.3f} in range {square_range}")
        
        if object_type == "cube":
            # Should be approximately square
            is_valid = square_range[0] <= aspect_ratio <= square_range[1]
            # DEBUG: Log cube validation result - REMOVE AFTER DEBUGGING
            self.logger.log_info(f"DEBUG: Cube aspect ratio validation: {square_range[0]} <= {aspect_ratio:.3f} <= {square_range[1]} = {is_valid}")
            return is_valid
        elif object_type == "rectangular_prism":
            # Should be rectangular (not square)
            is_valid = aspect_ratio < square_range[0] or aspect_ratio > square_range[1]
            # DEBUG: Log rectangular validation result - REMOVE AFTER DEBUGGING
            self.logger.log_info(f"DEBUG: Rectangular aspect ratio validation: {aspect_ratio:.3f} outside {square_range} = {is_valid}")
            return is_valid
        
        # DEBUG: Log default acceptance - REMOVE AFTER DEBUGGING
        self.logger.log_info(f"DEBUG: Shape validation: accepting unknown object_type")
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
                                      color: str, object_type: str, radius: Optional[float] = None) -> np.ndarray:
        """Create annotated detection visualization image.
        
        Args:
            bgr_image: Original BGR image
            detection_result: Detection result tuple or None
            color: Target color
            object_type: Target object type
            radius: Actual radius of detected object (for proper visualization)
            
        Returns:
            Annotated detection image showing what was detected
        """
        detection_image = bgr_image.copy()
        
        if detection_result:
            centroid, confidence = detection_result
            
            # Use actual object radius for visualization, with reasonable bounds
            if radius is not None:
                # Scale down radius slightly for better visibility and add minimum/maximum bounds
                vis_radius = max(15, min(int(radius * 0.8), 150))  # 80% of actual radius, bounded between 15-150px
            else:
                vis_radius = 20  # Fallback if radius not available
            
            # Draw detection circle with actual object size
            cv2.circle(detection_image, centroid, vis_radius, (0, 255, 0), 3)
            
            # Draw center point
            cv2.circle(detection_image, centroid, 5, (0, 255, 0), -1)
            
            # Add detection label
            label = f"{color} {object_type}: {confidence:.2f}"
            if radius is not None:
                label += f" (r={radius:.0f})"
            
            cv2.putText(detection_image, label, 
                       (centroid[0]+vis_radius+10, centroid[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Draw "NO DETECTION" indicator
            cv2.putText(detection_image, "NO DETECTION", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(detection_image, f"Target: {color} {object_type}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return detection_image
