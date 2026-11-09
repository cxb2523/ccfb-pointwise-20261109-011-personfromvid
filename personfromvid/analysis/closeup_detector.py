"""Close-up shot detection and distance estimation.

This module provides comprehensive closeup detection capabilities including:
- Shot type classification (extreme closeup, closeup, medium closeup, etc.)
- Distance estimation using facial landmarks and geometry
- Face size ratio analysis for shot classification
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..data.detection_results import CloseupDetection, FaceDetection, PoseDetection
from ..data.frame_data import FrameData, ImageProperties
from ..utils.exceptions import AnalysisError

logger = logging.getLogger(__name__)

# Shot classification thresholds (face area ratio relative to frame)
EXTREME_CLOSEUP_THRESHOLD = 0.25  # Face takes up >25% of frame
CLOSEUP_THRESHOLD = 0.15  # Face takes up >15% of frame
MEDIUM_CLOSEUP_THRESHOLD = 0.08  # Face takes up >8% of frame
MEDIUM_SHOT_THRESHOLD = 0.03  # Face takes up >3% of frame

# Distance estimation thresholds (inter-ocular distance in pixels)
VERY_CLOSE_IOD_THRESHOLD = 80  # >80 pixels between eyes
CLOSE_IOD_THRESHOLD = 50  # >50 pixels between eyes
MEDIUM_IOD_THRESHOLD = 25  # >25 pixels between eyes

SHOULDER_WIDTH_CLOSEUP_THRESHOLD = 0.35  # Shoulder width ratio for closeup detection

# Confidence thresholds
MIN_FACE_CONFIDENCE = 0.3
MIN_LANDMARK_CONFIDENCE = 0.5

# Edge/overlap handling thresholds
EDGE_MARGIN_RATIO = 0.02  # Face bbox within 2% of frame border counts as "near edge"
FACE_OVERLAP_IOU_THRESHOLD = 0.2  # IoU above this counts as "overlapping" faces

# Bounded retry so a single malformed face can't silently erase its detection
# entry, but also can't stall the whole selection pipeline indefinitely.
MAX_DETECTION_ATTEMPTS = 2


class CloseupDetectionError(AnalysisError):
    """Raised when closeup detection fails."""

    pass


class CloseupDetector:
    """Comprehensive closeup detection and distance estimation.

    This class provides advanced closeup detection capabilities including:
    - Multi-criteria shot classification
    - Distance estimation using facial geometry
    - Portrait suitability analysis

    Examples:
        Basic usage with FrameData:
        >>> detector = CloseupDetector()
        >>> detector.detect_closeups_in_frame(frame_data)

        Batch processing:
        >>> detector.process_frame_batch(frames_with_faces)
    """

    def __init__(
        self,
        extreme_closeup_threshold: float = EXTREME_CLOSEUP_THRESHOLD,
        closeup_threshold: float = CLOSEUP_THRESHOLD,
        medium_closeup_threshold: float = MEDIUM_CLOSEUP_THRESHOLD,
        medium_shot_threshold: float = MEDIUM_SHOT_THRESHOLD,
        edge_margin_ratio: float = EDGE_MARGIN_RATIO,
        face_overlap_iou_threshold: float = FACE_OVERLAP_IOU_THRESHOLD,
    ):
        """Initialize closeup detector with configurable thresholds.

        Args:
            extreme_closeup_threshold: Face area ratio for extreme closeup
            closeup_threshold: Face area ratio for closeup
            medium_closeup_threshold: Face area ratio for medium closeup
            medium_shot_threshold: Face area ratio for medium shot
            edge_margin_ratio: Fraction of frame width/height within which a face
                bbox is considered to be touching the frame edge
            face_overlap_iou_threshold: IoU above which two face bboxes in the same
                frame are considered overlapping
        """
        self.extreme_closeup_threshold = extreme_closeup_threshold
        self.closeup_threshold = closeup_threshold
        self.medium_closeup_threshold = medium_closeup_threshold
        self.medium_shot_threshold = medium_shot_threshold
        self.edge_margin_ratio = edge_margin_ratio
        self.face_overlap_iou_threshold = face_overlap_iou_threshold

        logger.info(
            f"Initialized CloseupDetector with thresholds: "
            f"extreme={extreme_closeup_threshold}, closeup={closeup_threshold}, "
            f"medium_closeup={medium_closeup_threshold}, medium_shot={medium_shot_threshold}, "
            f"edge_margin_ratio={edge_margin_ratio}, face_overlap_iou_threshold={face_overlap_iou_threshold}"
        )

    def detect_closeups_in_frame(self, frame: FrameData) -> None:
        """Detect closeup characteristics for all faces in a frame and update the frame in place.

        This is the primary method that operates on FrameData objects and follows the
        standardized pattern of using FrameData as the unit of work.

        Args:
            frame: FrameData object containing face detections and image properties
        """
        if not frame.face_detections:
            return

        image_properties = frame.image_properties

        # Process each face detection. The result list must always end up the
        # same length as face_detections: every face gets either a real
        # CloseupDetection or an explicit detection_failed=True placeholder, so
        # a frame can never silently lose a face's worth of classification.
        for face_idx, face_detection in enumerate(frame.face_detections):
            # Check if we have corresponding pose data for enhanced detection
            pose_detection = None
            if frame.pose_detections and len(frame.pose_detections) > face_idx:
                pose_detection = frame.pose_detections[face_idx]

            closeup_result = self._detect_closeup_for_face(
                face_detection,
                face_idx,
                frame,
                pose_detection,
                image_properties,
            )
            frame.closeup_detections.append(closeup_result)

    def _detect_closeup_for_face(
        self,
        face_detection: FaceDetection,
        face_idx: int,
        frame: FrameData,
        pose_detection: Optional[PoseDetection],
        image_properties: ImageProperties,
    ) -> CloseupDetection:
        """Detect closeup for a single face with explicit edge/overlap handling.

        Edge-of-frame and multi-face-overlap cases are resolved with explicit
        decision branches (never a silent skip). Only genuinely unexpected
        errors go through the bounded retry below; if every attempt fails, an
        explicit detection_failed placeholder is returned instead of dropping
        the face's entry.

        Args:
            face_detection: Face detection result with bbox and landmarks
            face_idx: Index of this face within frame.face_detections
            frame: Parent frame (used for id/logging and sibling-face overlap checks)
            pose_detection: Corresponding pose detection, if available
            image_properties: Image properties containing dimensions

        Returns:
            CloseupDetection: always returned, even on failure (detection_failed=True)
        """
        near_edge, edge_reason = self._check_frame_edge_proximity(
            face_detection.bbox, image_properties
        )
        overlaps, overlap_reason = self._check_face_overlap(
            face_detection, face_idx, frame.face_detections
        )

        if near_edge:
            logger.info(
                f"Frame {frame.frame_id} face {face_idx}: {edge_reason}"
            )
        if overlaps:
            logger.info(
                f"Frame {frame.frame_id} face {face_idx}: {overlap_reason}"
            )

        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_DETECTION_ATTEMPTS + 1):
            try:
                if pose_detection:
                    result = self._detect_closeup_with_pose_data(
                        face_detection, pose_detection, image_properties
                    )
                else:
                    result = self._detect_closeup_from_face(
                        face_detection, image_properties
                    )
                result.near_frame_edge = near_edge
                result.overlaps_other_face = overlaps
                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Closeup detection attempt {attempt}/{MAX_DETECTION_ATTEMPTS} "
                    f"failed for frame {frame.frame_id} face {face_idx}: {e}"
                )

        # All attempts exhausted: explicit failure marker instead of a silent
        # drop, so counts (len(closeup_detections) == len(face_detections))
        # and logs stay reconcilable.
        logger.error(
            f"Closeup detection permanently failed for frame {frame.frame_id} "
            f"face {face_idx} after {MAX_DETECTION_ATTEMPTS} attempt(s): {last_error}"
        )
        return CloseupDetection(
            is_closeup=False,
            shot_type="unknown",
            confidence=0.0,
            face_area_ratio=0.0,
            near_frame_edge=near_edge,
            overlaps_other_face=overlaps,
            detection_failed=True,
            failure_reason=str(last_error),
        )

    @staticmethod
    def _clamp_bbox_to_frame(
        bbox: Tuple[int, int, int, int],
        image_properties: ImageProperties,
    ) -> Tuple[int, int, int, int]:
        """Clamp a bbox to the visible frame region (never raises).

        Args:
            bbox: Face bounding box as (x1, y1, x2, y2), possibly out of bounds
            image_properties: Image properties containing frame dimensions

        Returns:
            Bbox clamped to [0, width] x [0, height]
        """
        x1, y1, x2, y2 = bbox
        width, height = image_properties.width, image_properties.height

        clamped_x1 = max(0, min(x1, width))
        clamped_y1 = max(0, min(y1, height))
        clamped_x2 = max(clamped_x1, min(x2, width))
        clamped_y2 = max(clamped_y1, min(y2, height))

        return (clamped_x1, clamped_y1, clamped_x2, clamped_y2)

    def _check_frame_edge_proximity(
        self,
        bbox: Tuple[int, int, int, int],
        image_properties: ImageProperties,
    ) -> Tuple[bool, str]:
        """Explicitly detect whether a face bbox touches/exceeds the frame edge.

        Args:
            bbox: Face bounding box as (x1, y1, x2, y2)
            image_properties: Image properties containing frame dimensions

        Returns:
            Tuple of (near_edge: bool, human_readable_reason: str)
        """
        x1, y1, x2, y2 = bbox
        width, height = image_properties.width, image_properties.height

        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return True, (
                f"face bbox {bbox} extends beyond frame bounds ({width}x{height}); "
                "using the visible (clamped) region for shot classification"
            )

        margin_x = width * self.edge_margin_ratio
        margin_y = height * self.edge_margin_ratio
        if (
            x1 <= margin_x
            or y1 <= margin_y
            or x2 >= width - margin_x
            or y2 >= height - margin_y
        ):
            return True, (
                f"face bbox {bbox} is within {self.edge_margin_ratio:.0%} of the "
                f"frame edge ({width}x{height})"
            )

        return False, ""

    def _check_face_overlap(
        self,
        face_detection: FaceDetection,
        face_idx: int,
        all_faces: List[FaceDetection],
    ) -> Tuple[bool, str]:
        """Explicitly detect whether this face bbox overlaps another face in the frame.

        Args:
            face_detection: The face being checked
            face_idx: Index of face_detection within all_faces
            all_faces: All face detections in the same frame

        Returns:
            Tuple of (overlaps: bool, human_readable_reason: str)
        """
        for other_idx, other in enumerate(all_faces):
            if other_idx == face_idx:
                continue

            iou = self._calculate_bbox_iou(face_detection.bbox, other.bbox)
            if iou >= self.face_overlap_iou_threshold:
                return True, (
                    f"face bbox overlaps face {other_idx} with IoU={iou:.2f} "
                    f">= threshold {self.face_overlap_iou_threshold}"
                )

        return False, ""

    @staticmethod
    def _calculate_bbox_iou(
        bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]
    ) -> float:
        """Calculate intersection-over-union between two bounding boxes."""
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        if inter_area <= 0:
            return 0.0

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area

        return inter_area / union if union > 0 else 0.0

    def _detect_closeup_from_face(
        self, face_detection: FaceDetection, image_properties: ImageProperties
    ) -> CloseupDetection:
        """Detect closeup shot characteristics from face detection using data models.

        Args:
            face_detection: Face detection result with bbox and landmarks
            image_properties: Image properties containing dimensions and metadata

        Returns:
            CloseupDetection with comprehensive analysis results

        Raises:
            CloseupDetectionError: If detection fails
        """
        try:
            frame_area = image_properties.total_pixels

            # Calculate face area ratio using the bbox clamped to the visible
            # frame region. A face bbox extending past the frame edge is a
            # normal, expected case (handled explicitly here) rather than an
            # error - it must not inflate the ratio above 1.0 and must not
            # raise, since that would silently erase this face's detection.
            visible_bbox = self._clamp_bbox_to_frame(
                face_detection.bbox, image_properties
            )
            visible_width = max(0, visible_bbox[2] - visible_bbox[0])
            visible_height = max(0, visible_bbox[3] - visible_bbox[1])
            face_area = visible_width * visible_height
            face_area_ratio = min(1.0, face_area / frame_area) if frame_area > 0 else 0.0

            # Classify shot type based on face area ratio
            shot_type = self._classify_shot_type(face_area_ratio)

            # Calculate inter-ocular distance if landmarks available
            inter_ocular_distance = None
            estimated_distance = None
            if face_detection.landmarks and len(face_detection.landmarks) >= 5:
                inter_ocular_distance = self._calculate_inter_ocular_distance(
                    face_detection.landmarks
                )
                estimated_distance = self._estimate_distance(inter_ocular_distance)

            # Determine if this is a closeup
            is_closeup = shot_type in ["extreme_closeup", "closeup", "medium_closeup"]

            # Calculate confidence based on multiple factors
            confidence = self._calculate_detection_confidence(
                face_detection,
                face_area_ratio,
                inter_ocular_distance,
            )

            return CloseupDetection(
                is_closeup=is_closeup,
                shot_type=shot_type,
                confidence=confidence,
                face_area_ratio=face_area_ratio,
                inter_ocular_distance=inter_ocular_distance,
                estimated_distance=estimated_distance,
            )

        except Exception as e:
            raise CloseupDetectionError(f"Failed to detect closeup: {str(e)}") from e

    def _detect_closeup_with_pose_data(
        self,
        face_detection: FaceDetection,
        pose_detection: PoseDetection,
        image_properties: ImageProperties,
    ) -> CloseupDetection:
        """Enhanced closeup detection using both face and pose data models.

        Args:
            face_detection: Face detection result
            pose_detection: Pose detection with keypoints
            image_properties: Image properties containing dimensions

        Returns:
            CloseupDetection with enhanced analysis including shoulder width
        """
        # Start with basic face-based detection
        result = self._detect_closeup_from_face(face_detection, image_properties)

        # Add shoulder width analysis from pose data
        shoulder_width_ratio = self._calculate_shoulder_width_ratio_from_pose(
            pose_detection, image_properties
        )
        if shoulder_width_ratio is not None:
            result.shoulder_width_ratio = shoulder_width_ratio

            # Update shot type if shoulder analysis suggests different classification
            if (
                shoulder_width_ratio >= SHOULDER_WIDTH_CLOSEUP_THRESHOLD
                and result.shot_type in ["medium_shot", "wide_shot"]
            ):
                result.shot_type = "medium_closeup"
                result.is_closeup = True
                # Update confidence with shoulder information
                result.confidence = min(1.0, result.confidence + 0.1)

        return result

    def _calculate_shoulder_width_ratio_from_pose(
        self, pose_detection: PoseDetection, image_properties: ImageProperties
    ) -> Optional[float]:
        """Calculate shoulder width ratio from pose detection data model.

        Args:
            pose_detection: Pose detection with keypoints
            image_properties: Image properties containing dimensions

        Returns:
            Shoulder width ratio or None if keypoints unavailable
        """
        pose_keypoints = pose_detection.keypoints
        left_shoulder = pose_keypoints.get("left_shoulder")
        right_shoulder = pose_keypoints.get("right_shoulder")

        if (
            left_shoulder
            and right_shoulder
            and left_shoulder[2] >= MIN_LANDMARK_CONFIDENCE
            and right_shoulder[2] >= MIN_LANDMARK_CONFIDENCE
        ):
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            return shoulder_width / image_properties.width

        return None

    def _classify_shot_type(self, face_area_ratio: float) -> str:
        """Classify shot type based on face area ratio.

        Args:
            face_area_ratio: Ratio of face area to total frame area

        Returns:
            Shot type classification string
        """
        if face_area_ratio >= self.extreme_closeup_threshold:
            return "extreme_closeup"
        elif face_area_ratio >= self.closeup_threshold:
            return "closeup"
        elif face_area_ratio >= self.medium_closeup_threshold:
            return "medium_closeup"
        elif face_area_ratio >= self.medium_shot_threshold:
            return "medium_shot"
        else:
            return "wide_shot"

    def _calculate_inter_ocular_distance(
        self, landmarks: List[Tuple[float, float]]
    ) -> float:
        """Calculate distance between eyes using facial landmarks.

        Args:
            landmarks: List of facial landmark points (typically 5 points)
                      Expected format: [left_eye, right_eye, nose, left_mouth, right_mouth]

        Returns:
            Distance between eyes in pixels
        """
        if len(landmarks) < 2:
            return 0.0

        # Assuming first two landmarks are left and right eyes
        left_eye = landmarks[0]
        right_eye = landmarks[1]

        # Calculate Euclidean distance
        distance = math.sqrt(
            (right_eye[0] - left_eye[0]) ** 2 + (right_eye[1] - left_eye[1]) ** 2
        )
        return distance

    def _estimate_distance(self, inter_ocular_distance: float) -> str:
        """Estimate relative distance based on inter-ocular distance.

        Args:
            inter_ocular_distance: Distance between eyes in pixels

        Returns:
            Distance category string
        """
        if inter_ocular_distance >= VERY_CLOSE_IOD_THRESHOLD:
            return "very_close"
        elif inter_ocular_distance >= CLOSE_IOD_THRESHOLD:
            return "close"
        elif inter_ocular_distance >= MEDIUM_IOD_THRESHOLD:
            return "medium"
        else:
            return "far"

    def _calculate_detection_confidence(
        self,
        face_detection: FaceDetection,
        face_area_ratio: float,
        inter_ocular_distance: Optional[float],
    ) -> float:
        """Calculate overall detection confidence based on multiple factors.

        Args:
            face_detection: Face detection result
            face_area_ratio: Face area ratio
            inter_ocular_distance: Inter-ocular distance

        Returns:
            Overall confidence score (0.0 to 1.0)
        """
        confidence_factors = []

        # Factor 1: Face detection confidence
        confidence_factors.append(face_detection.confidence)

        # Factor 2: Face area ratio consistency with classification
        area_confidence = min(1.0, face_area_ratio / self.medium_closeup_threshold)
        confidence_factors.append(area_confidence)

        # Factor 3: Landmark quality (if available)
        if face_detection.landmarks and inter_ocular_distance:
            landmark_confidence = min(1.0, inter_ocular_distance / CLOSE_IOD_THRESHOLD)
            confidence_factors.append(landmark_confidence)

        return max(0.3, np.mean(confidence_factors))

    def process_frame_batch(
        self,
        frames_with_faces: List["FrameData"],
        progress_callback: Optional[callable] = None,
        interruption_check: Optional[callable] = None,
    ) -> None:
        """Process a batch of frames with closeup detection.

        Args:
            frames_with_faces: List of FrameData objects with face detections
            progress_callback: Optional callback for progress updates
            interruption_check: Optional callback to check for interruption
        """
        if not frames_with_faces:
            return

        total_frames = len(frames_with_faces)
        start_time = time.time()

        logger.info(f"Starting closeup detection on {total_frames} frames")

        for i, frame_data in enumerate(frames_with_faces):
            # Check for interruption at regular intervals
            if interruption_check and i % 10 == 0:
                interruption_check()

            try:
                # Use the new standardized method
                self.detect_closeups_in_frame(frame_data)

            except Exception as e:
                frame_id = getattr(frame_data, "frame_id", f"frame_{i}")
                logger.error(f"Closeup detection failed for frame {frame_id}: {e}")
                # Continue processing other frames

            # Update progress with rate calculation
            if progress_callback:
                processed_count = i + 1
                # Calculate current processing rate
                elapsed = time.time() - start_time
                current_rate = processed_count / elapsed if elapsed > 0 else 0
                # Check if progress_callback accepts rate parameter
                try:
                    progress_callback(processed_count, rate=current_rate)
                except TypeError:
                    # Fallback to single argument for backwards compatibility
                    progress_callback(processed_count)

        logger.info(f"Closeup detection completed: {total_frames} frames processed")

    def get_detection_info(self) -> Dict[str, Any]:
        """Get information about the current detection settings.

        Returns:
            Dictionary containing thresholds and configuration
        """
        return {
            "shot_thresholds": {
                "extreme_closeup_threshold": self.extreme_closeup_threshold,
                "closeup_threshold": self.closeup_threshold,
                "medium_closeup_threshold": self.medium_closeup_threshold,
                "medium_shot_threshold": self.medium_shot_threshold,
            },
            "distance_thresholds": {
                "very_close_iod": VERY_CLOSE_IOD_THRESHOLD,
                "close_iod": CLOSE_IOD_THRESHOLD,
                "medium_iod": MEDIUM_IOD_THRESHOLD,
            },
            "detection_constants": {
                "shoulder_width_closeup_threshold": SHOULDER_WIDTH_CLOSEUP_THRESHOLD,
            },
            "edge_and_overlap_handling": {
                "edge_margin_ratio": self.edge_margin_ratio,
                "face_overlap_iou_threshold": self.face_overlap_iou_threshold,
                "max_detection_attempts": MAX_DETECTION_ATTEMPTS,
            },
        }
