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
        
        self.color_hsv_ranges = config.color_hsv_ranges
        self.confidence_threshold = config.confidence_threshold
        self.color_match_threshold = config.color_match_threshold
        self.vision_thresholds = config.vision_thresholds
    
    @handle_vision_errors("target object detection")
    def detect_target_object(self, bgr_image: np.ndarray, object_type: str, color: str, depth_image: np.ndarray = None) -> Tuple[Optional[Tuple[Tuple[int, int], float, Tuple[int, int, int, int]]], Dict[str, np.ndarray]]:
        """Detect target object using depth-first 3D object detection approach.
        
        Enhanced version that uses depth information for reliable 3D object identification,
        then validates with color and shape information.
        
        Args:
            bgr_image: Input BGR image
            object_type: Type of object ('cube' or 'rectangular_prism')
            color: Color of object ('red', 'green', 'blue', 'yellow')
            depth_image: Depth image in millimeters (float32), None for fallback to 2D detection
            
        Returns:
            Tuple of (detection_result, debug_images):
            - detection_result: ((x, y), confidence, (x, y, w, h)) tuple or None if no object detected
              where (x, y) is the centroid and (x, y, w, h) is the bounding rectangle
            - debug_images: Dictionary containing debug visualization images
        """
        if depth_image is not None:
            return self._detect_target_object_depth_first(bgr_image, object_type, color, depth_image)
        else:
            self.logger.log_warning("No depth image provided, falling back to 2D shape-first detection")
            return self._detect_target_object_shape_first(bgr_image, object_type, color)
    
    def _detect_target_object_depth_first(self, bgr_image: np.ndarray, object_type: str, color: str, depth_image: np.ndarray) -> Tuple[Optional[Tuple[Tuple[int, int], float, Tuple[int, int, int, int]]], Dict[str, np.ndarray]]:
        """Depth-first 3D object detection implementation."""
        self.logger.log_info(f"Starting depth-first detection for {color} {object_type}")
        self.logger.log_debug(f"Input shapes: bgr_image={bgr_image.shape}, depth_image={depth_image.shape}")
        
        self.logger.log_debug("Step 1: Segmenting 3D objects from depth")
        object_candidates = self._segment_3d_objects_from_depth(depth_image)
        self.logger.log_info(f"Found {len(object_candidates)} 3D object candidates")
        
        if not object_candidates:
            self.logger.log_info("No 3D objects found in depth image")
            self.logger.log_debug("About to create empty detection result")
            try:
                result = self._create_empty_detection_result(bgr_image, color, object_type)
                self.logger.log_debug("Empty detection result created successfully")
                return result
            except Exception as e:
                self.logger.log_error(f"Error creating empty detection result: {e}")
                return None, {}
        
        self.logger.log_debug("Step 2: Classifying and filtering 3D shapes")
        shape_candidates = self._classify_and_filter_3d_shapes(object_candidates, object_type, depth_image)
        self.logger.log_info(f"Found {len(shape_candidates)} valid {object_type} candidates after shape filtering")
        
        if not shape_candidates:
            self.logger.log_info(f"No valid {object_type} shapes found")
            self.logger.log_debug("About to create empty detection result for no shapes")
            try:
                result = self._create_empty_detection_result(bgr_image, color, object_type)
                self.logger.log_debug("Empty detection result for no shapes created successfully")
                return result
            except Exception as e:
                self.logger.log_error(f"Error creating empty detection result for no shapes: {e}")
                return None, {}
        
        self.logger.log_debug("Step 3: Validating 3D objects by color")
        valid_targets = self._validate_3d_objects_by_color(shape_candidates, bgr_image, color)
        self.logger.log_info(f"Found {len(valid_targets)} targets matching color {color}")
        
        if not valid_targets:
            self.logger.log_info(f"No objects matching color {color} found")
            self.logger.log_debug("About to create empty detection result for no color matches")
            try:
                result = self._create_empty_detection_result(bgr_image, color, object_type)
                self.logger.log_debug("Empty detection result for no color matches created successfully")
                return result
            except Exception as e:
                self.logger.log_error(f"Error creating empty detection result for no color matches: {e}")
                return None, {}
        
        self.logger.log_debug("Step 4: Ranking and selecting 3D target")
        detection_result = self._rank_and_select_3d_target(valid_targets)
        
        self.logger.log_debug("Step 5: Creating debug visualizations")
        self.logger.log_debug(f"About to create debug images with: object_candidates={len(object_candidates)}, shape_candidates={len(shape_candidates)}, valid_targets={len(valid_targets)}")
        
        try:
            debug_images = self._create_3d_detection_debug_images(
                bgr_image, depth_image, object_candidates, shape_candidates, 
                valid_targets, detection_result, color, object_type
            )
            self.logger.log_debug("Debug images created successfully")
        except Exception as e:
            self.logger.log_error(f"Error creating debug images: {e}")
            self.logger.log_error(f"Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.log_error(f"Traceback: {traceback.format_exc()}")
            # Return empty debug images on error
            debug_images = {
                "initial_mask": np.zeros(bgr_image.shape[:2], dtype=np.uint8),
                "final_mask": np.zeros(bgr_image.shape[:2], dtype=np.uint8),
                "detection_visualization": bgr_image.copy()
            }
        
        # Log final result
        if detection_result:
            centroid, confidence, bbox = detection_result
            self.logger.log_info(f"Depth-first detection successful: {color} {object_type} at {centroid} with confidence {confidence:.3f}")
        else:
            self.logger.log_info("Depth-first detection failed: no valid targets after ranking")
        
        self.logger.log_debug("Depth-first detection complete")
        return detection_result, debug_images
    
    def _detect_target_object_shape_first(self, bgr_image: np.ndarray, object_type: str, color: str) -> Tuple[Optional[Tuple[Tuple[int, int], float, Tuple[int, int, int, int]]], Dict[str, np.ndarray]]:
        """Original shape-first detection as fallback."""
        # Create HSV image for color filtering
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        
        # Create initial color mask for debugging purposes
        initial_mask = self._create_color_mask_unified(hsv_image, color)
        
        shape_candidates = self._detect_shapes_in_image(bgr_image, object_type, color)
        
        if not shape_candidates:
            self.logger.log_debug(f"No shape candidates found for {object_type}")
            detection_visualization = self._create_detection_visualization(bgr_image, None, color, object_type, None)
            return None, {"initial_mask": initial_mask, "final_mask": initial_mask, "detection_visualization": detection_visualization}
        
        valid_targets = self._filter_shapes_by_color(shape_candidates, color, hsv_image)
        
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
        
        detection_visualization = self._create_detection_visualization(
            bgr_image, detection_result, color, object_type, bounding_rect
        )
        
        final_mask = np.zeros_like(initial_mask)
        if valid_targets:
            for _, _, _, contour in valid_targets:
                cv2.drawContours(final_mask, [contour], 0, 255, -1)
        
        debug_images = {
            "initial_mask": initial_mask,
            "final_mask": final_mask,
            "detection_visualization": detection_visualization
        }
        
        # Debug image saving is now handled through coordinated flow in robot_arm_interface
        
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
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        kernel_size = (5, 5)
        blurred = cv2.GaussianBlur(gray, kernel_size, 0)
        
        # Edge detection with adjusted Canny thresholds
        edges = cv2.Canny(blurred, 20, 60)
        
        if color:
            hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
            color_mask = self._create_color_mask_unified(hsv_image, color)
            # Apply color mask to edges for hybrid color-edge approach
            edges = cv2.bitwise_and(edges, color_mask)
        
        # Add morphological operations to connect edge fragments
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.logger.log_debug(f"Shape detection: Found {len(contours)} contours in edge image")
        
        shape_candidates = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            self.logger.log_debug(f"Shape detection: Contour {i} area = {area:.1f} pixels")
            
            # Use bounding rectangle approach for shape validation
            aspect_ratio = w / h if h > 0 else 0
            
            if 0.7 < aspect_ratio < 1.3 and area > self.vision_thresholds["min_area"]:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroid = (cx, cy)
                    
                    shape_type = "square" if 0.9 < aspect_ratio < 1.1 else "rectangle"
                    confidence = min(1.0, area / 10000.0)
                    
                    shape_candidates.append((contour, shape_type, confidence, centroid))
                    
                    self.logger.log_info(f"Shape detection: Valid {shape_type} found at {centroid} with confidence {confidence:.2f}")
        
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
        area = cv2.contourArea(contour)
        self.logger.log_debug(f"Validating contour with area {area}")
        
        epsilon = self.vision_thresholds["polygon_epsilon"] * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        self.logger.log_debug(f"Polygon approximation: {len(approx)} vertices (need 4)")
        
        if len(approx) != 4:
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
        
        self.logger.log_debug(f"Color filtering: Starting with {len(shape_candidates)} shape candidates for color '{color}'")
        
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
                
                self.logger.log_debug(f"Color filtering: Shape {i} - {color_pixels}/{total_pixels} pixels match '{color}' ({color_match_ratio:.2f} or {color_match_ratio * 100:.1f}%)")
                
                # Require sufficient color match
                if color_match_ratio > self.color_match_threshold:
                    # Calculate overall confidence using configured weights
                    shape_weight, color_weight = self.vision_thresholds["confidence_weights"]
                    overall_confidence = (confidence * shape_weight) + (color_match_ratio * color_weight)
                    area = cv2.contourArea(contour)
                    valid_targets.append((centroid, overall_confidence, area, contour))
                    
                    self.logger.log_info(f"Color filtering: ✅ Valid target! Shape {i} - {shape_type} at {centroid} with confidence {overall_confidence:.2f}, area {area}")
                else:
                    self.logger.log_debug(f"Color filtering: ❌ Shape {i} - Color match too low: {color_match_ratio:.2f} < threshold {self.color_match_threshold}")
            else:
                self.logger.log_debug(f"  ❌ No pixels in shape mask")
        
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
            self.logger.log_info("Target validation: No valid targets to rank")
            return None
        
        self.logger.log_debug(f"Target validation: Ranking {len(valid_targets)} valid targets")
        for i, target in enumerate(valid_targets):
            centroid, confidence, area, _ = target
            self.logger.log_info(f"Target validation: Target {i} - centroid={centroid}, confidence={confidence:.2f}, area={area}")
        
        # Return the largest valid target (highest area)
        best_target = max(valid_targets, key=lambda x: x[2])  # Sort by area
        centroid, confidence, area, _ = best_target
        
        self.logger.log_info(f"Target validation: Best target selected - centroid={centroid}, confidence={confidence:.2f}, area={area}")
        
        # Apply size threshold
        min_area = self.vision_thresholds["min_area"]
        
        self.logger.log_debug(f"Target validation: Size threshold check - area {area} vs min_area {min_area}")
        
        if area > min_area:
            self.logger.log_debug(f"Target validation: ✅ Target passes size threshold (area {area} > {min_area})")
            return (centroid, confidence)
        else:
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
            for i, range_tuple in enumerate(color_ranges):
                hsv_lower, hsv_upper = range_tuple
                mask = cv2.inRange(hsv_image, np.array(hsv_lower), np.array(hsv_upper))
                masks.append(mask)
            
            combined_mask = cv2.bitwise_or(masks[0], masks[1])
            return combined_mask
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
    
    def _should_save_debug_images(self) -> bool:
        """Check if we should save debug images (during recording).
        
        Returns:
            True if debug images should be saved
        """
        # Use existing camera manager's recording state logic
        return hasattr(self, 'camera_manager') and getattr(self.camera_manager, '_was_recording_before_scan', False)
    
    
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
    
    # ========== 3D Depth-First Detection Methods ==========
    
    def _segment_3d_objects_from_depth(self, depth_image: np.ndarray) -> List[Dict[str, Any]]:
        """Segment 3D objects from depth image using plane fitting + connected components.
        
        Based on Perplexity's recommendation for small object detection on Jetson hardware.
        
        Args:
            depth_image: Depth image in millimeters (float32)
            
        Returns:
            List of object candidate dictionaries with keys: 'mask', 'centroid', 'bbox', 'depth_stats'
        """
        depth_cleaned = self._preprocess_depth_for_segmentation(depth_image)
        
        table_plane, objects_mask = self._fit_table_plane_and_extract_objects(depth_cleaned)
        
        if objects_mask is None:
            self.logger.log_warning("Failed to extract objects above table plane")
            return []
        
        object_candidates = self._extract_individual_objects(objects_mask, depth_cleaned)
        
        self.logger.log_info(f"3D segmentation: Found {len(object_candidates)} object candidates")
        return object_candidates
    
    def _preprocess_depth_for_segmentation(self, depth_image: np.ndarray) -> np.ndarray:
        """Preprocess depth image to reduce noise while preserving object edges."""
        # Remove invalid depth values (0 or very large values)
        valid_mask = (depth_image > 10) & (depth_image < 2000)  # 1cm to 2m range
        depth_cleaned = depth_image.copy()
        depth_cleaned[~valid_mask] = 0
        
        # Apply median filter to reduce noise while preserving edges
        depth_cleaned = cv2.medianBlur(depth_cleaned.astype(np.float32), 5)
        
        # Optional: bilateral filter for edge-preserving smoothing
        # Only apply if we have enough valid pixels
        if np.sum(valid_mask) > depth_image.size * 0.5:
            depth_cleaned = cv2.bilateralFilter(depth_cleaned, 5, 10, 10)
        
        return depth_cleaned
    
    def _fit_table_plane_and_extract_objects(self, depth_image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Fit table plane using least squares and extract objects above it."""
        # Get valid depth points
        valid_mask = depth_image > 0
        if np.sum(valid_mask) < 1000:  # Need minimum points for plane fitting
            self.logger.log_warning("Insufficient valid depth points for plane fitting")
            return None, None
        
        # Convert to 3D points for plane fitting
        height, width = depth_image.shape
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Get valid points
        valid_points = valid_mask.nonzero()
        x_valid = x_coords[valid_points]
        y_valid = y_coords[valid_points]
        z_valid = depth_image[valid_points]
        
        # Simple plane fitting using least squares (faster than RANSAC for table detection)
        # Assume table is roughly horizontal, so fit z = ax + by + c
        try:
            # Create design matrix
            A = np.column_stack([x_valid, y_valid, np.ones(len(x_valid))])
            
            # Solve for plane coefficients
            plane_coeffs, residuals, rank, s = np.linalg.lstsq(A, z_valid, rcond=None)
            
            # Calculate plane depths for all pixels
            plane_depths = (plane_coeffs[0] * x_coords + 
                           plane_coeffs[1] * y_coords + 
                           plane_coeffs[2])
            
            # Objects are points significantly above the table plane
            height_threshold = 5.0  # 5mm above table to account for noise
            objects_mask = (depth_image > 0) & (depth_image < plane_depths - height_threshold)
            
            self.logger.log_info(f"Table plane fitted, found {np.sum(objects_mask)} object pixels")
            return plane_coeffs, objects_mask.astype(np.uint8) * 255
            
        except np.linalg.LinAlgError:
            self.logger.log_error("Failed to fit table plane")
            return None, None
    
    def _extract_individual_objects(self, objects_mask: np.ndarray, depth_image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract individual objects using connected component analysis."""
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(objects_mask, connectivity=8)
        
        object_candidates = []
        
        # Skip background label (0)
        for label in range(1, num_labels):
            # Get component statistics
            area = stats[label, cv2.CC_STAT_AREA]
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            
            # Filter by area - updated for rotated cubes which appear larger
            # 45° rotated 1-inch cube projects as ~44mm diagonal, appearing larger than face-on
            min_area = 80   # pixels - reduced minimum for smaller projections
            max_area = 8000 # pixels - increased maximum for rotated cube projections
            
            if min_area < area < max_area:
                # Create object mask
                object_mask = (labels == label).astype(np.uint8) * 255
                
                # Calculate depth statistics for this object - handle invalid depths
                object_depths = depth_image[labels == label]
                valid_depths = object_depths[object_depths > 0]  # Filter out zero depths
                
                if len(valid_depths) > len(object_depths) * 0.3:  # Need at least 30% valid depth
                    depth_stats = {
                        'mean_depth': np.mean(valid_depths),
                        'std_depth': np.std(valid_depths),
                        'min_depth': np.min(valid_depths),
                        'max_depth': np.max(valid_depths),
                        'valid_depth_ratio': len(valid_depths) / len(object_depths)
                    }
                    
                    object_candidates.append({
                        'mask': object_mask,
                        'centroid': (int(centroids[label][0]), int(centroids[label][1])),
                        'bbox': (x, y, w, h),
                        'area': area,
                        'depth_stats': depth_stats
                    })
                    
                    self.logger.log_info(f"Object candidate: area={area}, bbox=({x},{y},{w},{h}), mean_depth={depth_stats['mean_depth']:.1f}mm, valid_depth={depth_stats['valid_depth_ratio']:.1f}")
                else:
                    self.logger.log_debug(f"Rejected object: insufficient valid depth data ({len(valid_depths)}/{len(object_depths)})")
        
        return object_candidates
    
    def _classify_and_filter_3d_shapes(self, object_candidates: List[Dict[str, Any]], object_type: str, depth_image: np.ndarray) -> List[Dict[str, Any]]:
        """Classify 3D shapes and filter by expected dimensions."""
        # Expected dimensions in millimeters (with tolerance)
        expected_dims = self._get_expected_object_dimensions(object_type)
        
        valid_candidates = []
        
        for candidate in object_candidates:
            # Estimate 3D dimensions from depth and 2D size
            dimensions = self._estimate_3d_dimensions(candidate, depth_image)
            
            # Check if dimensions match expected object type
            if self._validate_object_dimensions(dimensions, expected_dims, object_type):
                candidate['dimensions'] = dimensions
                candidate['shape_confidence'] = self._calculate_3d_shape_confidence(dimensions, expected_dims)
                valid_candidates.append(candidate)
                
                self.logger.log_info(f"Valid {object_type}: dims={dimensions}, confidence={candidate['shape_confidence']:.3f}")
            else:
                self.logger.log_debug(f"Rejected object: dims={dimensions} don't match {object_type}")
        
        return valid_candidates
    
    def _get_expected_object_dimensions(self, object_type: str) -> Dict[str, Tuple[float, float]]:
        """Get expected dimensions for object types with tolerance.
        
        Updated to handle 45° rotated cubes which appear as diamond shapes
        with projected diagonal dimensions rather than edge dimensions.
        """
        # Dimensions in millimeters: (min, max) for each measurement
        if object_type == "cube":
            # For 45° rotated 1 inch cube (~25.4mm edge):
            # - Projected space diagonal: ~44mm (25.4 * √3)
            # - Projected face diagonal: ~36mm (25.4 * √2)
            # - Bounding box captures the diamond projection
            return {
                'width': (20.0, 80.0),   # Expanded from (35.0, 50.0) - accommodate farther/closer objects
                'height': (15.0, 70.0),  # Expanded from (28.0, 45.0) - accommodate varied angles  
                'depth': (10.0, 200.0)   # Expanded from (15.0, 170.0) - accommodate depth variation
            }
        elif object_type == "rectangular_prism":
            # 2x1x1 inch prism = ~50.8x25.4x25.4mm
            # Rotation effects depend on orientation - use generous bounds
            return {
                'width': (40.0, 65.0),   # 2 inch dimension with rotation
                'height': (20.0, 45.0),  # 1 inch dimensions with rotation
                'depth': (15.0, 40.0)    # Depth variation
            }
        else:
            # Default generous bounds
            return {
                'width': (15.0, 70.0),
                'height': (15.0, 70.0),
                'depth': (10.0, 50.0)
            }
    
    def _estimate_3d_dimensions(self, candidate: Dict[str, Any], depth_image: np.ndarray) -> Dict[str, float]:
        """Estimate 3D dimensions of object candidate.
        
        Enhanced to better handle rotated objects using improved pixel-to-mm conversion
        and depth variation analysis.
        """
        bbox = candidate['bbox']
        x, y, w, h = bbox
        depth_stats = candidate['depth_stats']
        
        # Enhanced depth dimension from depth variation
        depth_range = depth_stats['max_depth'] - depth_stats['min_depth']
        mean_depth = depth_stats['mean_depth']
        
        # Improved pixel-to-mm conversion using empirical calibration
        # Based on test data: 98x80 pixel bounding box for 45° rotated 1-inch cube
        # Expected dimensions: ~44mm x 36mm (projected diagonals)
        # This gives us ~2.2 pixels/mm at ~127mm depth
        
        if mean_depth > 0:
            # Empirical conversion factor calibrated from actual test data
            # At 127mm depth: 98 pixels = 44mm → 2.23 pixels/mm
            # Scale linearly with depth for different distances
            reference_depth = 127.0  # mm (from test data)
            reference_pixels_per_mm = 2.23  # pixels/mm at reference depth
            
            # Scale conversion factor based on actual depth
            pixels_per_mm = reference_pixels_per_mm * (reference_depth / mean_depth)
            
            estimated_width = w / pixels_per_mm
            estimated_height = h / pixels_per_mm
        else:
            # Fallback for invalid depth
            estimated_width = w * 0.5  # Conservative estimate
            estimated_height = h * 0.5
        
        # Enhanced depth estimation using depth variation
        # For rotated cubes, depth variation should be significant
        estimated_depth = max(depth_range, 15.0)  # Minimum 15mm depth for rotated objects
        
        # Apply rotation correction for cubes
        # 45° rotated cube appears larger in projection
        if abs(estimated_width - estimated_height) / max(estimated_width, estimated_height) < 0.3:
            # Likely a rotated cube - adjust for diamond projection
            # Diamond projection is ~1.4x larger than face-on view
            estimated_width = estimated_width / 1.2  # Slight correction factor
            estimated_height = estimated_height / 1.2
        
        return {
            'width': estimated_width,
            'height': estimated_height,
            'depth': estimated_depth
        }
    
    def _validate_object_dimensions(self, dimensions: Dict[str, float], expected_dims: Dict[str, Tuple[float, float]], object_type: str) -> bool:
        """Validate if estimated dimensions match expected object type.
        
        Updated to handle 45° rotated cubes with diamond-shaped projections.
        Enhanced with adaptive validation based on object distance.
        """
        width, height, depth = dimensions['width'], dimensions['height'], dimensions['depth']
        
        # For cube: handle both face-on and rotated orientations
        if object_type == "cube":
            # Check if all dimensions are within expected range
            w_valid = expected_dims['width'][0] <= width <= expected_dims['width'][1]
            h_valid = expected_dims['height'][0] <= height <= expected_dims['height'][1]
            d_valid = expected_dims['depth'][0] <= depth <= expected_dims['depth'][1]
            
            self.logger.log_debug(f"Dimension validation: w={width:.1f} ({w_valid}), h={height:.1f} ({h_valid}), d={depth:.1f} ({d_valid})")
            
            # Updated aspect ratio validation for rotated cubes
            # 45° rotated cube has aspect ratio ~0.82 (diamond shape)
            aspect_ratio = min(width, height) / max(width, height)
            
            # More lenient aspect ratio validation for varied camera positions
            # Allow for both face-on (aspect ~1.0) and rotated (aspect ~0.8) cubes
            # Expanded ranges to accommodate perspective distortion
            rotated_range_valid = 0.6 <= aspect_ratio <= 1.0  # Expanded from 0.75
            face_on_range_valid = 0.8 <= aspect_ratio <= 1.2  # Expanded from 0.9-1.1
            aspect_ratio_valid = rotated_range_valid or face_on_range_valid
            
            self.logger.log_debug(f"Aspect ratio: {aspect_ratio:.3f}, rotated_valid={rotated_range_valid}, face_on_valid={face_on_range_valid}, overall_valid={aspect_ratio_valid}")
            
            final_result = w_valid and h_valid and d_valid and aspect_ratio_valid
            self.logger.log_debug(f"Final validation result: {final_result} (w={w_valid}, h={h_valid}, d={d_valid}, aspect={aspect_ratio_valid})")
            
            return final_result
        
        elif object_type == "rectangular_prism":
            # For 2x1x1 prism, one dimension should be ~2x the others
            dims = sorted([width, height, depth])
            small1, small2, large = dims
            
            # Check if two dimensions are ~1 inch and one is ~2 inches
            small_valid = (expected_dims['height'][0] <= small1 <= expected_dims['height'][1] and
                          expected_dims['depth'][0] <= small2 <= expected_dims['depth'][1])
            large_valid = expected_dims['width'][0] <= large <= expected_dims['width'][1]
            
            # More lenient aspect ratio for rotated prisms
            aspect_ratio_valid = 1.3 <= (large / small1) <= 3.0  # Allow more variation
            
            return small_valid and large_valid and aspect_ratio_valid
        
        return True  # Default: accept if we don't know the object type
    
    def _calculate_3d_shape_confidence(self, dimensions: Dict[str, float], expected_dims: Dict[str, Tuple[float, float]]) -> float:
        """Calculate confidence score based on how well dimensions match expected values."""
        width, height, depth = dimensions['width'], dimensions['height'], dimensions['depth']
        
        # Calculate how close each dimension is to the expected range center
        confidences = []
        
        for dim_name, dim_value in [('width', width), ('height', height), ('depth', depth)]:
            expected_min, expected_max = expected_dims[dim_name]
            expected_center = (expected_min + expected_max) / 2
            expected_range = expected_max - expected_min
            
            # Distance from center as fraction of range
            distance_from_center = abs(dim_value - expected_center)
            confidence = max(0.0, 1.0 - (distance_from_center / (expected_range / 2)))
            confidences.append(confidence)
        
        # Return average confidence
        return np.mean(confidences)
    
    def _validate_3d_objects_by_color(self, shape_candidates: List[Dict[str, Any]], bgr_image: np.ndarray, color: str) -> List[Dict[str, Any]]:
        """Validate 3D objects by checking color within their masks."""
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        valid_targets = []
        
        self.logger.log_debug(f"Color validation: bgr_image shape={bgr_image.shape}, hsv_image shape={hsv_image.shape}")
        
        for candidate in shape_candidates:
            object_mask = candidate['mask']
            self.logger.log_debug(f"Object mask shape: {object_mask.shape}")
            
            # Apply color filtering to the object region
            color_mask = self._create_color_mask_unified(hsv_image, color)
            self.logger.log_debug(f"Color mask shape: {color_mask.shape}")
            
            # Resize object mask to match color mask dimensions if needed
            if object_mask.shape != color_mask.shape:
                self.logger.log_debug(f"Resizing object mask from {object_mask.shape} to {color_mask.shape}")
                object_mask_resized = cv2.resize(object_mask, (color_mask.shape[1], color_mask.shape[0]))
            else:
                object_mask_resized = object_mask
            
            # Now both masks have the same dimensions
            combined_mask = cv2.bitwise_and(object_mask_resized, color_mask)
            
            # Calculate color match percentage
            total_pixels = cv2.countNonZero(object_mask_resized)
            color_pixels = cv2.countNonZero(combined_mask)
            
            if total_pixels > 0:
                color_match_ratio = color_pixels / total_pixels
                
                self.logger.log_info(f"Color validation: {color_pixels}/{total_pixels} pixels match '{color}' ({color_match_ratio:.2f})")
                
                # More lenient color threshold for 3D objects
                color_threshold = max(0.3, self.color_match_threshold * 0.7)  # 30% minimum, or 70% of configured threshold
                
                if color_match_ratio > color_threshold:
                    # Combine shape and color confidence
                    shape_confidence = candidate['shape_confidence']
                    overall_confidence = (shape_confidence * 0.6) + (color_match_ratio * 0.4)
                    
                    candidate['color_confidence'] = color_match_ratio
                    candidate['overall_confidence'] = overall_confidence
                    valid_targets.append(candidate)
                    
                    self.logger.log_info(f"Valid 3D target: shape_conf={shape_confidence:.3f}, color_conf={color_match_ratio:.3f}, overall={overall_confidence:.3f}")
        
        return valid_targets
    
    def _rank_and_select_3d_target(self, valid_targets: List[Dict[str, Any]]) -> Optional[Tuple[Tuple[int, int], float, Tuple[int, int, int, int]]]:
        """Rank valid 3D targets and select the best one."""
        if not valid_targets:
            return None
        
        # Sort by overall confidence (highest first)
        valid_targets.sort(key=lambda x: x['overall_confidence'], reverse=True)
        
        # Select the best target
        best_target = valid_targets[0]
        
        # Extract information in depth image coordinates
        depth_centroid = best_target['centroid']
        confidence = best_target['overall_confidence']
        depth_bbox = best_target['bbox']
        
        # Translate coordinates from depth space to RGB space
        # Depth: (360, 640), RGB: (720, 1280) -> 2x scaling factor
        scale_x = 2.0  # 1280 / 640
        scale_y = 2.0  # 720 / 360
        
        # Scale centroid
        rgb_centroid = (
            int(depth_centroid[0] * scale_x),
            int(depth_centroid[1] * scale_y)
        )
        
        # Scale bounding box (x, y, w, h)
        x, y, w, h = depth_bbox
        rgb_bbox = (
            int(x * scale_x),
            int(y * scale_y), 
            int(w * scale_x),
            int(h * scale_y)
        )
        
        # Debug logging for coordinate translation
        self.logger.log_debug(f"Coordinate translation:")
        self.logger.log_debug(f"  Depth centroid: {depth_centroid} -> RGB centroid: {rgb_centroid}")
        self.logger.log_debug(f"  Depth bbox: {depth_bbox} -> RGB bbox: {rgb_bbox}")
        self.logger.log_debug(f"  Scale factors: x={scale_x}, y={scale_y}")
        
        return (rgb_centroid, confidence, rgb_bbox)
    
    def _create_3d_detection_debug_images(self, bgr_image: np.ndarray, depth_image: np.ndarray, 
                                        object_candidates: List[Dict[str, Any]], shape_candidates: List[Dict[str, Any]],
                                        valid_targets: List[Dict[str, Any]], detection_result: Optional[Tuple],
                                        color: str, object_type: str) -> Dict[str, np.ndarray]:
        """Create debug visualization images for 3D detection."""
        debug_images = {}
        
        # Use BGR image dimensions for debug visualization (this is what gets displayed)
        bgr_shape = bgr_image.shape[:2]  # (height, width)
        depth_shape = depth_image.shape[:2]
        
        self.logger.log_debug(f"Debug images: bgr_shape={bgr_shape}, depth_shape={depth_shape}")
        
        # Create initial mask showing all detected 3D objects
        initial_mask = np.zeros(bgr_shape, dtype=np.uint8)
        for candidate in object_candidates:
            candidate_mask = candidate['mask']
            # Resize depth-derived mask to match BGR image dimensions
            if candidate_mask.shape != bgr_shape:
                candidate_mask_resized = cv2.resize(candidate_mask, (bgr_shape[1], bgr_shape[0]))
                self.logger.log_debug(f"Resized candidate mask from {candidate_mask.shape} to {candidate_mask_resized.shape}")
            else:
                candidate_mask_resized = candidate_mask
            
            initial_mask = cv2.bitwise_or(initial_mask, candidate_mask_resized)
        
        # Create final mask showing valid targets
        final_mask = np.zeros(bgr_shape, dtype=np.uint8)
        for target in valid_targets:
            target_mask = target['mask']
            # Resize depth-derived mask to match BGR image dimensions
            if target_mask.shape != bgr_shape:
                target_mask_resized = cv2.resize(target_mask, (bgr_shape[1], bgr_shape[0]))
                self.logger.log_debug(f"Resized target mask from {target_mask.shape} to {target_mask_resized.shape}")
            else:
                target_mask_resized = target_mask
                
            final_mask = cv2.bitwise_or(final_mask, target_mask_resized)
        
        # Create detection visualization
        detection_visualization = self._create_detection_visualization(
            bgr_image, detection_result, color, object_type, 
            detection_result[2] if detection_result else None
        )
        
        # PERPLEXITY'S CRITICAL INSIGHT: Create coordinate debugging visualizations
        # Show bounding boxes on both depth and RGB images for comparison
        depth_debug_image = self._create_depth_coordinate_debug(depth_image, valid_targets, detection_result)
        rgb_debug_image = self._create_rgb_coordinate_debug(bgr_image, valid_targets, detection_result)
        
        debug_images = {
            "initial_mask": initial_mask,
            "final_mask": final_mask,
            "detection_visualization": detection_visualization,
            "depth_coordinates": depth_debug_image,
            "rgb_coordinates": rgb_debug_image
        }
        
        return debug_images
    
    def _create_depth_coordinate_debug(self, depth_image: np.ndarray, valid_targets: List[Dict[str, Any]], 
                                     detection_result: Optional[Tuple]) -> np.ndarray:
        """Create debug image showing bounding boxes in depth coordinate space."""
        # Convert depth to 3-channel for visualization
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_debug = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
        
        # Draw all valid target bounding boxes in depth coordinates
        for i, target in enumerate(valid_targets):
            depth_bbox = target['bbox']  # This is in depth coordinates
            depth_centroid = target['centroid']  # This is in depth coordinates
            x, y, w, h = depth_bbox
            
            # Draw bounding box in depth space (RED for depth)
            cv2.rectangle(depth_debug, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(depth_debug, depth_centroid, 3, (0, 0, 255), -1)
            
            # Add label
            cv2.putText(depth_debug, f"DEPTH_{i}: {w}x{h}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Add coordinate space label
        cv2.putText(depth_debug, f"DEPTH SPACE: {depth_image.shape[1]}x{depth_image.shape[0]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return depth_debug
    
    def _create_rgb_coordinate_debug(self, bgr_image: np.ndarray, valid_targets: List[Dict[str, Any]], 
                                   detection_result: Optional[Tuple]) -> np.ndarray:
        """Create debug image showing bounding boxes in RGB coordinate space."""
        rgb_debug = bgr_image.copy()
        
        # Calculate scaling factors
        scale_x = 2.0  # 1280 / 640
        scale_y = 2.0  # 720 / 360
        
        # Draw all valid target bounding boxes translated to RGB coordinates
        for i, target in enumerate(valid_targets):
            depth_bbox = target['bbox']  # Original depth coordinates
            depth_centroid = target['centroid']  # Original depth coordinates
            
            # Translate to RGB coordinates
            x, y, w, h = depth_bbox
            rgb_x = int(x * scale_x)
            rgb_y = int(y * scale_y)
            rgb_w = int(w * scale_x)
            rgb_h = int(h * scale_y)
            rgb_centroid = (int(depth_centroid[0] * scale_x), int(depth_centroid[1] * scale_y))
            
            # Draw bounding box in RGB space (GREEN for RGB)
            cv2.rectangle(rgb_debug, (rgb_x, rgb_y), (rgb_x + rgb_w, rgb_y + rgb_h), (0, 255, 0), 3)
            cv2.circle(rgb_debug, rgb_centroid, 5, (0, 255, 0), -1)
            
            # Add label
            cv2.putText(rgb_debug, f"RGB_{i}: {rgb_w}x{rgb_h}", (rgb_x, rgb_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw final detection result if available (BLUE for final result)
        if detection_result:
            final_centroid, confidence, final_bbox = detection_result
            x, y, w, h = final_bbox
            cv2.rectangle(rgb_debug, (x, y), (x + w, y + h), (255, 0, 0), 4)
            cv2.circle(rgb_debug, final_centroid, 7, (255, 0, 0), -1)
            cv2.putText(rgb_debug, f"FINAL: {confidence:.3f}", (x, y - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Add coordinate space label
        cv2.putText(rgb_debug, f"RGB SPACE: {bgr_image.shape[1]}x{bgr_image.shape[0]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return rgb_debug
    
    def _create_empty_detection_result(self, bgr_image: np.ndarray, color: str, object_type: str) -> Tuple[None, Dict[str, np.ndarray]]:
        """Create empty detection result with debug images."""
        empty_mask = np.zeros(bgr_image.shape[:2], dtype=np.uint8)
        detection_visualization = self._create_detection_visualization(bgr_image, None, color, object_type, None)
        
        debug_images = {
            "initial_mask": empty_mask,
            "final_mask": empty_mask,
            "detection_visualization": detection_visualization
        }
        
        return None, debug_images
