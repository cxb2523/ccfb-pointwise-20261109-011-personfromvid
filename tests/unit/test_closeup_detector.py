"""Unit tests for closeup detection functionality."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np

from personfromvid.analysis.closeup_detector import (
    CLOSEUP_THRESHOLD,
    EXTREME_CLOSEUP_THRESHOLD,
    MEDIUM_CLOSEUP_THRESHOLD,
    MEDIUM_SHOT_THRESHOLD,
    CloseupDetector,
)
from personfromvid.data.detection_results import FaceDetection, PoseDetection
from personfromvid.data.frame_data import FrameData, ImageProperties, SourceInfo


class TestCloseupDetector:
    """Test suite for CloseupDetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = CloseupDetector()
        self.image_shape = (1080, 1920)  # height, width

        # Create sample FrameData components
        self.source_info = SourceInfo(
            video_timestamp=1.0,
            extraction_method="test",
            original_frame_number=1,
            video_fps=30.0
        )

        self.image_properties = ImageProperties(
            width=1920,
            height=1080,
            channels=3,
            file_size_bytes=1000000,
            format="JPG"
        )

    def create_face_detection(self, bbox, confidence=0.9, landmarks=None):
        """Helper to create FaceDetection object."""
        return FaceDetection(
            bbox=bbox,
            confidence=confidence,
            landmarks=landmarks
        )

    def create_frame_data(self, face_detections=None, pose_detections=None):
        """Helper to create FrameData object with detections."""
        return FrameData(
            frame_id="test_frame",
            file_path=Path("/tmp/test.jpg"),
            source_info=self.source_info,
            image_properties=self.image_properties,
            face_detections=face_detections or [],
            pose_detections=pose_detections or []
        )

    def test_initialization(self):
        """Test CloseupDetector initialization."""
        detector = CloseupDetector(
            extreme_closeup_threshold=0.3,
            closeup_threshold=0.2,
            medium_closeup_threshold=0.1,
            medium_shot_threshold=0.05
        )

        assert detector.extreme_closeup_threshold == 0.3
        assert detector.closeup_threshold == 0.2
        assert detector.medium_closeup_threshold == 0.1
        assert detector.medium_shot_threshold == 0.05

    def test_shot_type_classification_extreme_closeup(self):
        """Test classification of extreme closeup shots."""
        # Large face taking up 30% of frame
        face_area = int(self.image_shape[0] * self.image_shape[1] * 0.30)
        bbox_size = int(np.sqrt(face_area))
        bbox = (500, 300, 500 + bbox_size, 300 + bbox_size)

        face_detection = self.create_face_detection(bbox)
        frame_data = self.create_frame_data([face_detection])

        self.detector.detect_closeups_in_frame(frame_data)

        assert len(frame_data.closeup_detections) == 1
        result = frame_data.closeup_detections[0]
        assert result.shot_type == "extreme_closeup"
        assert result.is_closeup is True
        assert result.confidence > 0.5
        assert result.face_area_ratio >= EXTREME_CLOSEUP_THRESHOLD

    def test_shot_type_classification_closeup(self):
        """Test classification of closeup shots."""
        # Face taking up 18% of frame
        face_area = int(self.image_shape[0] * self.image_shape[1] * 0.18)
        bbox_size = int(np.sqrt(face_area))
        bbox = (500, 300, 500 + bbox_size, 300 + bbox_size)

        face_detection = self.create_face_detection(bbox)
        frame_data = self.create_frame_data([face_detection])

        self.detector.detect_closeups_in_frame(frame_data)

        result = frame_data.closeup_detections[0]
        assert result.shot_type == "closeup"
        assert result.is_closeup is True
        assert CLOSEUP_THRESHOLD <= result.face_area_ratio < EXTREME_CLOSEUP_THRESHOLD

    def test_shot_type_classification_medium_closeup(self):
        """Test classification of medium closeup shots."""
        # Face taking up 10% of frame
        face_area = int(self.image_shape[0] * self.image_shape[1] * 0.10)
        bbox_size = int(np.sqrt(face_area))
        bbox = (500, 300, 500 + bbox_size, 300 + bbox_size)

        face_detection = self.create_face_detection(bbox)
        frame_data = self.create_frame_data([face_detection])

        self.detector.detect_closeups_in_frame(frame_data)

        result = frame_data.closeup_detections[0]
        assert result.shot_type == "medium_closeup"
        assert result.is_closeup is True
        assert MEDIUM_CLOSEUP_THRESHOLD <= result.face_area_ratio < CLOSEUP_THRESHOLD

    def test_shot_type_classification_medium_shot(self):
        """Test classification of medium shots."""
        # Face taking up 5% of frame
        face_area = int(self.image_shape[0] * self.image_shape[1] * 0.05)
        bbox_size = int(np.sqrt(face_area))
        bbox = (500, 300, 500 + bbox_size, 300 + bbox_size)

        face_detection = self.create_face_detection(bbox)
        frame_data = self.create_frame_data([face_detection])

        self.detector.detect_closeups_in_frame(frame_data)

        result = frame_data.closeup_detections[0]
        assert result.shot_type == "medium_shot"
        assert result.is_closeup is False
        assert MEDIUM_SHOT_THRESHOLD <= result.face_area_ratio < MEDIUM_CLOSEUP_THRESHOLD

    def test_shot_type_classification_wide_shot(self):
        """Test classification of wide shots."""
        # Small face taking up 1% of frame
        face_area = int(self.image_shape[0] * self.image_shape[1] * 0.01)
        bbox_size = int(np.sqrt(face_area))
        bbox = (500, 300, 500 + bbox_size, 300 + bbox_size)

        face_detection = self.create_face_detection(bbox)
        frame_data = self.create_frame_data([face_detection])

        self.detector.detect_closeups_in_frame(frame_data)

        result = frame_data.closeup_detections[0]
        assert result.shot_type == "wide_shot"
        assert result.is_closeup is False
        assert result.face_area_ratio < MEDIUM_SHOT_THRESHOLD

    def test_inter_ocular_distance_calculation(self):
        """Test inter-ocular distance calculation."""
        # Create landmarks with known eye positions
        landmarks = [
            (100.0, 200.0),  # left eye
            (180.0, 200.0),  # right eye (80 pixels apart)
            (140.0, 220.0),  # nose
            (120.0, 250.0),  # left mouth
            (160.0, 250.0)   # right mouth
        ]

        bbox = (50, 150, 250, 300)
        face_detection = self.create_face_detection(bbox, landmarks=landmarks)
        frame_data = self.create_frame_data([face_detection])

        self.detector.detect_closeups_in_frame(frame_data)

        result = frame_data.closeup_detections[0]
        assert result.inter_ocular_distance == 80.0
        assert result.estimated_distance == "very_close"  # 80 > VERY_CLOSE_IOD_THRESHOLD

    def test_distance_estimation_categories(self):
        """Test distance estimation based on inter-ocular distance."""
        # Test various inter-ocular distances
        test_cases = [
            (100, "very_close"),
            (60, "close"),
            (30, "medium"),
            (10, "far")
        ]

        for iod, expected_distance in test_cases:
            distance = self.detector._estimate_distance(iod)
            assert distance == expected_distance

    def test_detect_closeup_with_pose_keypoints(self):
        """Test enhanced closeup detection with pose keypoints."""
        bbox = (500, 300, 700, 500)
        face_detection = self.create_face_detection(bbox)

        # Pose keypoints with wide shoulders indicating closeup
        pose_keypoints = {
            'left_shoulder': (400.0, 450.0, 0.9),
            'right_shoulder': (1000.0, 450.0, 0.9),  # Wide shoulders
            'nose': (600.0, 400.0, 0.8)
        }

        pose_detection = PoseDetection(
            bbox=(400, 300, 1000, 600),
            confidence=0.9,
            keypoints=pose_keypoints
        )

        frame_data = self.create_frame_data([face_detection], [pose_detection])

        self.detector.detect_closeups_in_frame(frame_data)

        result = frame_data.closeup_detections[0]
        assert result.shoulder_width_ratio is not None
        assert result.shoulder_width_ratio > 0.3  # Wide shoulders

    def test_batch_processing(self):
        """Test batch processing of frames."""
        from pathlib import Path

        from personfromvid.data.detection_results import FaceDetection
        from personfromvid.data.frame_data import FrameData, ImageProperties, SourceInfo

        # Create mock FrameData objects
        frames_with_faces = [
            FrameData(
                frame_id='frame_001',
                file_path=Path('/path/to/frame1.jpg'),
                source_info=SourceInfo(
                    video_timestamp=1.0,
                    extraction_method='test',
                    original_frame_number=30,
                    video_fps=30.0
                ),
                image_properties=ImageProperties(
                    width=self.image_shape[1],
                    height=self.image_shape[0],
                    channels=3,
                    file_size_bytes=1024000,
                    format='jpg'
                ),
                face_detections=[
                    FaceDetection(
                        bbox=(500, 300, 800, 600),
                        confidence=0.9,
                        landmarks=None
                    )
                ],
                pose_detections=[]
            ),
            FrameData(
                frame_id='frame_002',
                file_path=Path('/path/to/frame2.jpg'),
                source_info=SourceInfo(
                    video_timestamp=2.0,
                    extraction_method='test',
                    original_frame_number=60,
                    video_fps=30.0
                ),
                image_properties=ImageProperties(
                    width=self.image_shape[1],
                    height=self.image_shape[0],
                    channels=3,
                    file_size_bytes=1024000,
                    format='jpg'
                ),
                face_detections=[
                    FaceDetection(
                        bbox=(100, 100, 200, 200),
                        confidence=0.8,
                        landmarks=[
                            (120.0, 140.0), (160.0, 140.0),
                            (140.0, 160.0), (130.0, 180.0), (150.0, 180.0)
                        ]
                    )
                ],
                pose_detections=[]
            )
        ]

        # The method now modifies frames in-place and returns None
        result = self.detector.process_frame_batch(frames_with_faces)

        # Verify the method returns None (in-place modification)
        assert result is None

        # Verify frame data was updated in-place
        assert len(frames_with_faces) == 2

        # First frame should have closeup detection results
        assert len(frames_with_faces[0].closeup_detections) == 1
        closeup_result = frames_with_faces[0].closeup_detections[0]
        assert hasattr(closeup_result, 'shot_type')
        assert hasattr(closeup_result, 'is_closeup')
        assert hasattr(closeup_result, 'confidence')

        # Second frame should also have results
        assert len(frames_with_faces[1].closeup_detections) == 1

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with very small face (edge case)
        invalid_face = Mock()
        invalid_face.area = 0
        invalid_face.bbox = (0, 0, 0, 0)
        invalid_face.landmarks = None
        invalid_face.confidence = 0.9

        frame_data = self.create_frame_data([invalid_face])

        # The detector should handle this gracefully without raising an exception
        self.detector.detect_closeups_in_frame(frame_data)

        assert len(frame_data.closeup_detections) == 1
        result = frame_data.closeup_detections[0]
        assert result.shot_type == "wide_shot"
        assert result.face_area_ratio == 0.0
        assert result.is_closeup is False

    def test_confidence_calculation_factors(self):
        """Test confidence calculation considers multiple factors."""
        # High quality face detection with landmarks
        landmarks = [
            (100.0, 200.0), (180.0, 200.0), (140.0, 220.0),
            (120.0, 250.0), (160.0, 250.0)
        ]
        bbox = (500, 300, 900, 700)  # Large face for closeup
        face_detection = self.create_face_detection(bbox, confidence=0.95, landmarks=landmarks)
        frame_data = self.create_frame_data([face_detection])

        self.detector.detect_closeups_in_frame(frame_data)

        result = frame_data.closeup_detections[0]
        # Should have high confidence due to:
        # - High face detection confidence
        # - Large face area ratio
        # - Available landmarks
        assert result.confidence > 0.7

    def test_get_detection_info(self):
        """Test getting detection configuration information."""
        info = self.detector.get_detection_info()

        assert 'shot_thresholds' in info
        assert 'distance_thresholds' in info
        assert 'detection_constants' in info

        # Check shot thresholds
        shot_thresholds = info['shot_thresholds']
        assert shot_thresholds['extreme_closeup_threshold'] == EXTREME_CLOSEUP_THRESHOLD
        assert shot_thresholds['closeup_threshold'] == CLOSEUP_THRESHOLD

        # Check distance thresholds
        distance_thresholds = info['distance_thresholds']
        assert 'very_close_iod' in distance_thresholds
        assert 'close_iod' in distance_thresholds

    def test_empty_frame_batch(self):
        """Test processing empty frame batch."""
        result = self.detector.process_frame_batch([])
        assert result is None

    def test_frame_with_no_faces(self):
        """Test processing frame with no face detections."""
        frame_data = self.create_frame_data([])  # No face detections

        self.detector.detect_closeups_in_frame(frame_data)

        # Should not add any closeup detections
        assert len(frame_data.closeup_detections) == 0

    def test_multiple_faces_in_frame(self):
        """Test processing frame with multiple face detections."""
        # Create two different face detections
        face1 = self.create_face_detection((100, 100, 300, 300))  # Small face
        face2 = self.create_face_detection((400, 200, 1000, 800))  # Large face for closeup

        frame_data = self.create_frame_data([face1, face2])

        self.detector.detect_closeups_in_frame(frame_data)

        # Should have two closeup detection results
        assert len(frame_data.closeup_detections) == 2

        # First face should be wide shot or medium shot
        result1 = frame_data.closeup_detections[0]
        assert result1.shot_type in ["wide_shot", "medium_shot"]

        # Second face should be closeup (large face area)
        result2 = frame_data.closeup_detections[1]
        assert result2.is_closeup is True

    def test_batch_processing_edge_cases(self):
        """Test batch processing with edge cases."""
        # Frame with face but no pose data
        frame1 = self.create_frame_data([
            self.create_face_detection((400, 300, 600, 500))
        ])

        # Frame with both face and pose data
        pose_keypoints = {
            'left_shoulder': (300.0, 400.0, 0.9),
            'right_shoulder': (700.0, 400.0, 0.9),
            'nose': (500.0, 350.0, 0.8)
        }
        pose_detection = PoseDetection(
            bbox=(300, 300, 700, 600),
            confidence=0.9,
            keypoints=pose_keypoints
        )
        frame2 = self.create_frame_data([
            self.create_face_detection((400, 300, 600, 500))
        ], [pose_detection])

        # Frame with no faces
        frame3 = self.create_frame_data([])

        frames = [frame1, frame2, frame3]

        # Process batch
        self.detector.process_frame_batch(frames)

        # Verify results
        assert len(frame1.closeup_detections) == 1
        assert len(frame2.closeup_detections) == 1
        assert len(frame3.closeup_detections) == 0

        # Frame with pose data should have shoulder width ratio
        assert frame2.closeup_detections[0].shoulder_width_ratio is not None

        # Frame without pose data should not have shoulder width ratio
        assert frame1.closeup_detections[0].shoulder_width_ratio is None


class TestCloseupDetectorEdgeAndOverlapHandling:
    """Boundary tests for faces near the frame edge, overlapping faces, and
    detection failures - these must always produce an explicit, traceable
    result instead of silently dropping the face's entry.
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = CloseupDetector()
        self.image_properties = ImageProperties(
            width=1920,
            height=1080,
            channels=3,
            file_size_bytes=1000000,
            format="JPG",
        )
        self.source_info = SourceInfo(
            video_timestamp=1.0,
            extraction_method="test",
            original_frame_number=1,
            video_fps=30.0,
        )

    def create_frame_data(self, face_detections=None, pose_detections=None):
        return FrameData(
            frame_id="edge_test_frame",
            file_path=Path("/tmp/test.jpg"),
            source_info=self.source_info,
            image_properties=self.image_properties,
            face_detections=face_detections or [],
            pose_detections=pose_detections or [],
        )

    def test_face_bbox_extending_past_frame_edge_is_not_silently_dropped(self):
        """A face bbox extending beyond the frame boundary must still produce
        a usable, explicitly-flagged CloseupDetection instead of raising
        inside CloseupDetection.__post_init__ (face_area_ratio > 1.0) and
        being swallowed - that silent drop is exactly what makes a
        legitimately-selected frame vanish from the "closeup" pose category.
        """
        # 500x500 bbox positioned so it extends 120px past the right/bottom edges
        bbox = (1700, 900, 2200, 1400)
        face = FaceDetection(bbox=bbox, confidence=0.9, landmarks=None)
        frame = self.create_frame_data([face])

        self.detector.detect_closeups_in_frame(frame)

        # Every face must produce exactly one entry - never silently dropped.
        assert len(frame.closeup_detections) == 1
        result = frame.closeup_detections[0]

        assert result.detection_failed is False
        assert result.near_frame_edge is True
        # Ratio must be computed from the visible (clamped) region, never > 1.0
        assert 0.0 <= result.face_area_ratio <= 1.0
        visible_area = (1920 - 1700) * (1080 - 900)
        expected_ratio = visible_area / (1920 * 1080)
        assert result.face_area_ratio == expected_ratio

    def test_grossly_oversized_edge_bbox_no_longer_raises_or_vanishes(self):
        """Regression test for the root-cause bug: a bbox whose raw (unclamped)
        area exceeds the frame area used to raise ValueError inside
        CloseupDetection.__post_init__ (face_area_ratio > 1.0), which the old
        blanket except/continue swallowed - leaving closeup_detections empty
        for a face that legitimately exists. is_closeup_shot() would then
        return False and frame_selector.group_by_pose() would never place
        this frame in the "closeup" bucket, so a frame that should have been
        selected silently disappears instead.
        """
        bbox = (-200, -200, 2200, 1300)  # raw area > frame area
        face = FaceDetection(bbox=bbox, confidence=0.9, landmarks=None)
        frame = self.create_frame_data([face])

        self.detector.detect_closeups_in_frame(frame)

        assert len(frame.closeup_detections) == 1
        result = frame.closeup_detections[0]
        assert result.detection_failed is False
        assert result.near_frame_edge is True
        assert result.face_area_ratio == 1.0
        assert result.is_closeup is True
        assert frame.is_closeup_shot() is True

    def test_face_bbox_touching_edge_within_margin_is_flagged(self):
        """A bbox fully inside the frame but within the configured edge margin
        must be explicitly flagged, not treated as an ordinary interior face.
        """
        # Small face starting at x=0,y=0 - inside the default 2% margin
        bbox = (0, 0, 50, 50)
        face = FaceDetection(bbox=bbox, confidence=0.9, landmarks=None)
        frame = self.create_frame_data([face])

        self.detector.detect_closeups_in_frame(frame)

        result = frame.closeup_detections[0]
        assert result.detection_failed is False
        assert result.near_frame_edge is True

    def test_face_well_inside_frame_is_not_flagged_as_near_edge(self):
        """Sanity check: a face far from any edge should not be flagged."""
        bbox = (800, 400, 1000, 600)
        face = FaceDetection(bbox=bbox, confidence=0.9, landmarks=None)
        frame = self.create_frame_data([face])

        self.detector.detect_closeups_in_frame(frame)

        result = frame.closeup_detections[0]
        assert result.near_frame_edge is False

    def test_overlapping_faces_are_explicitly_flagged_not_skipped(self):
        """Two heavily-overlapping face bboxes (e.g. a double-detection on the
        same person, or two people standing very close) must each get an
        explicit overlaps_other_face=True result with a recorded reason,
        rather than being processed as if nothing unusual were happening or
        being skipped.
        """
        face_a = FaceDetection(bbox=(500, 300, 700, 500), confidence=0.9, landmarks=None)
        face_b = FaceDetection(bbox=(520, 310, 720, 510), confidence=0.85, landmarks=None)
        frame = self.create_frame_data([face_a, face_b])

        self.detector.detect_closeups_in_frame(frame)

        assert len(frame.closeup_detections) == 2
        assert frame.closeup_detections[0].overlaps_other_face is True
        assert frame.closeup_detections[1].overlaps_other_face is True
        assert frame.closeup_detections[0].detection_failed is False
        assert frame.closeup_detections[1].detection_failed is False

    def test_non_overlapping_faces_are_not_flagged(self):
        """Sanity check: faces far apart must not be flagged as overlapping."""
        face_a = FaceDetection(bbox=(100, 100, 200, 200), confidence=0.9, landmarks=None)
        face_b = FaceDetection(bbox=(1500, 800, 1600, 900), confidence=0.9, landmarks=None)
        frame = self.create_frame_data([face_a, face_b])

        self.detector.detect_closeups_in_frame(frame)

        assert frame.closeup_detections[0].overlaps_other_face is False
        assert frame.closeup_detections[1].overlaps_other_face is False

    def test_detection_failure_recovers_on_retry(self, monkeypatch):
        """A transient failure on the first attempt must be retried, and
        succeed if the second attempt doesn't fail - proving the retry path
        actually runs rather than failing fast on the first exception.
        """
        face = FaceDetection(bbox=(800, 400, 1000, 600), confidence=0.9, landmarks=None)
        frame = self.create_frame_data([face])

        original = CloseupDetector._detect_closeup_from_face
        call_count = {"n": 0}

        def flaky(self, face_detection, image_properties):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated transient failure")
            return original(self, face_detection, image_properties)

        monkeypatch.setattr(CloseupDetector, "_detect_closeup_from_face", flaky)

        self.detector.detect_closeups_in_frame(frame)

        assert call_count["n"] == 2  # first attempt failed, second attempt ran
        assert len(frame.closeup_detections) == 1
        result = frame.closeup_detections[0]
        assert result.detection_failed is False

    def test_detection_permanent_failure_produces_explicit_failure_marker(self, monkeypatch):
        """When every retry attempt fails, the face must still get an explicit
        detection_failed=True entry with a failure_reason - never a silent
        drop that shrinks frame.closeup_detections below len(face_detections).
        """
        face = FaceDetection(bbox=(800, 400, 1000, 600), confidence=0.9, landmarks=None)
        frame = self.create_frame_data([face])

        def always_fails(self, face_detection, image_properties):
            raise RuntimeError("simulated permanent failure")

        monkeypatch.setattr(
            CloseupDetector, "_detect_closeup_from_face", always_fails
        )

        self.detector.detect_closeups_in_frame(frame)

        # Count must match - the failure is traceable, not a silent drop.
        assert len(frame.closeup_detections) == len(frame.face_detections) == 1
        result = frame.closeup_detections[0]
        assert result.detection_failed is True
        assert result.is_closeup is False
        assert "simulated permanent failure" in result.failure_reason

    def test_permanent_failure_is_logged_at_error_level(self, monkeypatch, caplog):
        """The permanent-failure path must log at ERROR level with enough
        detail (frame id, face index) to reconcile against output counts.
        """
        import logging

        face = FaceDetection(bbox=(800, 400, 1000, 600), confidence=0.9, landmarks=None)
        frame = self.create_frame_data([face])

        def always_fails(self, face_detection, image_properties):
            raise RuntimeError("simulated permanent failure")

        monkeypatch.setattr(
            CloseupDetector, "_detect_closeup_from_face", always_fails
        )

        logger_name = "personfromvid.analysis.closeup_detector"
        with caplog.at_level(logging.ERROR, logger=logger_name):
            self.detector.detect_closeups_in_frame(frame)

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) == 1
        assert frame.frame_id in error_records[0].message
        assert "face 0" in error_records[0].message

    def test_multiple_faces_one_failing_does_not_drop_the_others(self, monkeypatch):
        """If one face's detection permanently fails, the other faces in the
        same frame must still be processed normally and keep their results -
        one bad face must not take down or hide the rest.
        """
        good_face_1 = FaceDetection(bbox=(100, 100, 300, 300), confidence=0.9, landmarks=None)
        bad_face = FaceDetection(bbox=(800, 400, 1000, 600), confidence=0.9, landmarks=None)
        good_face_2 = FaceDetection(bbox=(1200, 600, 1400, 800), confidence=0.9, landmarks=None)
        frame = self.create_frame_data([good_face_1, bad_face, good_face_2])

        original = CloseupDetector._detect_closeup_from_face

        def fail_only_for_bad_face(self, face_detection, image_properties):
            if face_detection is bad_face:
                raise RuntimeError("simulated permanent failure")
            return original(self, face_detection, image_properties)

        monkeypatch.setattr(
            CloseupDetector, "_detect_closeup_from_face", fail_only_for_bad_face
        )

        self.detector.detect_closeups_in_frame(frame)

        assert len(frame.closeup_detections) == 3
        assert frame.closeup_detections[0].detection_failed is False
        assert frame.closeup_detections[1].detection_failed is True
        assert frame.closeup_detections[2].detection_failed is False

