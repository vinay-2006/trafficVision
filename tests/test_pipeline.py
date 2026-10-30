"""
Tests for the core pipeline components. Run with: pytest tests/ -v

These use small synthetic inputs and the stock COCO yolo11n checkpoint,
so they run without needing UA-DETRAC downloaded or a fine-tuned model.
"""
import numpy as np
import pytest
import supervision as sv

from src.detector import VehicleDetector, COCO_VEHICLE_CLASS_IDS
from src.tracker import VehicleTracker
from src.perspective import PerspectiveTransformer, example_calibration
from src.analytics import SpeedEstimator, LineCounter


@pytest.fixture(scope="module")
def detector():
    return VehicleDetector(weights_path="yolo11n.pt")


def test_detector_returns_detections_object(detector):
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    detections = detector.detect(frame)
    assert isinstance(detections, sv.Detections)


def test_detector_filters_to_vehicle_classes_only(detector):
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    detections = detector.detect(frame)
    for class_id in detections.class_id:
        assert int(class_id) in COCO_VEHICLE_CLASS_IDS


def test_tracker_assigns_ids_across_frames():
    tracker = VehicleTracker(frame_rate=25)
    detections = sv.Detections(
        xyxy=np.array([[100, 100, 180, 160]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([2]),
    )
    tracked = tracker.update(detections)
    assert tracked.tracker_id is not None
    assert len(tracked.tracker_id) == 1


def test_perspective_transform_round_trip():
    transformer = example_calibration()
    points = np.array([[330, 250], [630, 250]])
    result = transformer.transform_points(points)
    assert result.shape == (2, 2)
    # top-left should map close to (0,0), top-right close to (10,0)
    assert np.isclose(result[0][0], 0, atol=1e-3)
    assert np.isclose(result[1][0], 10, atol=1e-3)


def test_perspective_calibration_json_round_trip(tmp_path):
    """A calibration saved to JSON and reloaded should transform points
    identically to the original in-memory object."""
    original = example_calibration()
    json_path = tmp_path / "calibration.json"
    original.to_json(str(json_path))

    reloaded = PerspectiveTransformer.from_json(str(json_path))

    test_points = np.array([[400, 300], [600, 400]])
    original_result = original.transform_points(test_points)
    reloaded_result = reloaded.transform_points(test_points)
    assert np.allclose(original_result, reloaded_result)


def test_speed_estimator_computes_zero_for_stationary_object():
    transformer = example_calibration()
    estimator = SpeedEstimator(transformer, fps=25)
    detections = sv.Detections(
        xyxy=np.array([[330, 250, 400, 320]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([2]),
        tracker_id=np.array([1]),
    )
    speeds_frame1 = estimator.update(detections, frame_idx=0)
    speeds_frame2 = estimator.update(detections, frame_idx=1)
    # Same position across frames -> speed should be ~0
    assert speeds_frame2.get(1, 0) == pytest.approx(0, abs=0.5)


def test_line_counter_counts_crossing():
    counter = LineCounter(start=(0, 300), end=(640, 300))
    # Detection starts above the line
    det_above = sv.Detections(
        xyxy=np.array([[100, 200, 160, 260]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([2]),
        tracker_id=np.array([1]),
    )
    counter.update(det_above)
    # Same track now below the line
    det_below = sv.Detections(
        xyxy=np.array([[100, 320, 160, 380]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([2]),
        tracker_id=np.array([1]),
    )
    in_count, out_count = counter.update(det_below)
    assert (in_count + out_count) >= 1


def _make_yolo_fixture(base_dir):
    """Helper: builds a tiny synthetic YOLO dataset for oversampling tests."""
    train_images = base_dir / "train" / "images"
    train_labels = base_dir / "train" / "labels"
    val_images = base_dir / "val" / "images"
    val_labels = base_dir / "val" / "labels"
    for d in (train_images, train_labels, val_images, val_labels):
        d.mkdir(parents=True, exist_ok=True)

    for i in range(5):
        (train_labels / f"car_{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        (train_images / f"car_{i}.jpg").touch()

    (train_labels / "rare_others.txt").write_text("3 0.5 0.5 0.1 0.1\n")
    (train_images / "rare_others.jpg").touch()

    (val_labels / "val_car.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (val_images / "val_car.jpg").touch()


def test_oversample_rare_classes_duplicates_correctly(tmp_path):
    from train.oversample_rare_classes import oversample

    _make_yolo_fixture(tmp_path)
    oversample(tmp_path, multipliers={3: 4})  # "others" -> 4x

    train_images = list((tmp_path / "train" / "images").glob("*.jpg"))
    # 5 car images + 1 original "others" + 3 duplicates = 9 total
    assert len(train_images) == 9

    val_images = list((tmp_path / "val" / "images").glob("*.jpg"))
    assert len(val_images) == 1, "val split must never be touched by oversampling"


def test_oversample_uses_max_not_product_for_multi_class_images(tmp_path):
    from train.oversample_rare_classes import oversample

    train_images = tmp_path / "train" / "images"
    train_labels = tmp_path / "train" / "labels"
    val_images = tmp_path / "val" / "images"
    val_labels = tmp_path / "val" / "labels"
    for d in (train_images, train_labels, val_images, val_labels):
        d.mkdir(parents=True, exist_ok=True)

    # One image contains BOTH bus (class 1) and others (class 3)
    (train_labels / "double_rare.txt").write_text("1 0.3 0.3 0.1 0.1\n3 0.6 0.6 0.1 0.1\n")
    (train_images / "double_rare.jpg").touch()

    oversample(tmp_path, multipliers={1: 3, 3: 8})

    # Should use max(3, 8) = 8 total appearances (7 extra), NOT 3*8=24
    all_images = list(train_images.glob("*.jpg"))
    assert len(all_images) == 8


def _make_undersample_fixture(base_dir):
    train_images = base_dir / "train" / "images"
    train_labels = base_dir / "train" / "labels"
    for d in (train_images, train_labels):
        d.mkdir(parents=True, exist_ok=True)

    for i in range(10):
        (train_labels / f"car_only_{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        (train_images / f"car_only_{i}.jpg").touch()

    for i in range(2):
        (train_labels / f"car_and_bus_{i}.txt").write_text("0 0.3 0.3 0.1 0.1\n1 0.6 0.6 0.1 0.1\n")
        (train_images / f"car_and_bus_{i}.jpg").touch()

    (train_labels / "others_only.txt").write_text("3 0.5 0.5 0.1 0.1\n")
    (train_images / "others_only.jpg").touch()


def test_undersample_car_removes_correct_fraction(tmp_path):
    from train.undersample_car import undersample

    _make_undersample_fixture(tmp_path)
    undersample(tmp_path, remove_fraction=0.5, seed=42)

    remaining = list((tmp_path / "train" / "images").glob("*.jpg"))
    removed = list((tmp_path / "train" / "removed_images").glob("*.jpg"))
    # 10 car-only, 50% removed = 5 removed, 5 remain + 2 car+bus + 1 others = 8 remaining
    assert len(removed) == 5
    assert len(remaining) == 8


def test_undersample_car_never_removes_rare_class_images(tmp_path):
    from train.undersample_car import undersample

    _make_undersample_fixture(tmp_path)
    # Remove ALL car-only images (fraction=1.0) -- the strictest possible test
    undersample(tmp_path, remove_fraction=1.0, seed=42)

    remaining_names = {p.name for p in (tmp_path / "train" / "images").glob("*.jpg")}
    # Images containing a rare class must survive even at 100% car-only removal
    assert "car_and_bus_0.jpg" in remaining_names
    assert "car_and_bus_1.jpg" in remaining_names
    assert "others_only.jpg" in remaining_names
    # All 10 pure car-only images should be gone
    assert not any(name.startswith("car_only_") for name in remaining_names)


def test_undersample_car_is_reversible_not_destructive(tmp_path):
    from train.undersample_car import undersample

    _make_undersample_fixture(tmp_path)
    undersample(tmp_path, remove_fraction=0.5, seed=42)

    # Removed files must be MOVED, not deleted -- still exist somewhere
    removed_images = list((tmp_path / "train" / "removed_images").glob("*.jpg"))
    removed_labels = list((tmp_path / "train" / "removed_labels").glob("*.txt"))
    assert len(removed_images) == 5
    assert len(removed_labels) == 5


def test_train_yolo_requires_data_unless_resuming():
    """--data is required for a fresh run, but not when --resume is set
    (the checkpoint's own saved config supplies it instead)."""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "train/train_yolo.py", "--epochs", "1"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "--data is required" in result.stderr


def test_train_yolo_rejects_nonexistent_resume_path():
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "train/train_yolo.py", "--resume", "/tmp/definitely_does_not_exist_12345.pt"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "does not exist" in result.stderr


def test_train_yolo_refuses_resume_with_missing_data_and_no_override(tmp_path):
    """If the checkpoint's saved data path doesn't exist and no --data
    override is given, must fail loudly rather than let Ultralytics
    silently fall back to its own bundled sample dataset (verified real
    behavior, not a guess -- this exact silent fallback was observed
    during manual testing before this safety check was added)."""
    import subprocess
    import sys
    import torch
    from ultralytics import YOLO

    # Build a minimal real checkpoint referencing a nonexistent data path
    ckpt_path = tmp_path / "fake_last.pt"
    model = YOLO("yolo11n.pt")
    torch.save({
        "epoch": 0,
        "optimizer": {"fake": "state"},
        "model": model.model,
        "train_args": {"data": "/nonexistent/path/data.yaml", "epochs": 3},
    }, ckpt_path)

    result = subprocess.run(
        [sys.executable, "train/train_yolo.py", "--resume", str(ckpt_path), "--device", "cpu"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "does not exist in this session" in result.stderr
    assert "silently fall back" in result.stderr


def _make_test_split_fixture(base_dir):
    xml_dir = base_dir / "xml"
    img_dir = base_dir / "images" / "MVI_TEST01"
    train_dir = base_dir / "existing_train"
    for d in (xml_dir, img_dir, train_dir / "images", train_dir / "labels"):
        d.mkdir(parents=True, exist_ok=True)

    (xml_dir / "MVI_TEST01.xml").write_text("""<?xml version="1.0" encoding="utf-8"?>
<sequence name="MVI_TEST01">
  <frame num="1">
    <target_list>
      <target id="1">
        <box left="100" top="100" width="50" height="40"/>
        <attribute vehicle_type="car"/>
      </target>
    </target_list>
  </frame>
  <frame num="2">
    <target_list>
      <target id="1">
        <box left="200" top="150" width="60" height="45"/>
        <attribute vehicle_type="car"/>
      </target>
      <target id="2">
        <box left="400" top="300" width="80" height="60"/>
        <attribute vehicle_type="bus"/>
      </target>
    </target_list>
  </frame>
  <frame num="3">
    <target_list>
      <target id="1">
        <box left="500" top="400" width="30" height="25"/>
        <attribute vehicle_type="others"/>
      </target>
    </target_list>
  </frame>
</sequence>
""")
    from PIL import Image
    for i in (1, 2, 3):
        Image.new("RGB", (960, 540), (100, 100, 100)).save(img_dir / f"img{i:05d}.jpg")

    for i in range(3):
        (train_dir / "labels" / f"existing_car_{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        (train_dir / "images" / f"existing_car_{i}.jpg").touch()

    return xml_dir, img_dir.parent, train_dir


def test_add_test_split_rare_data_skips_car_only_frames(tmp_path):
    import sys
    sys.path.insert(0, "train")
    from add_test_split_rare_data import main as _unused  # ensures importable
    import subprocess

    xml_dir, img_dir, train_dir = _make_test_split_fixture(tmp_path)
    result = subprocess.run(
        [sys.executable, "train/add_test_split_rare_data.py",
         "--xml-dir", str(xml_dir), "--img-dir", str(img_dir), "--train-dir", str(train_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "Added 2 real" in result.stdout

    remaining_images = {p.name for p in (train_dir / "images").glob("*.jpg")}
    # Car-only frame (img00001) must be excluded
    assert not any("img00001" in name for name in remaining_images)
    # car+bus (img00002) and others-only (img00003) must be included
    assert any("img00002" in name for name in remaining_images)
    assert any("img00003" in name for name in remaining_images)
    # Original 3 existing images must still be present, untouched
    assert sum(1 for name in remaining_images if name.startswith("existing_car_")) == 3


def test_compute_image_weights_orders_by_true_rarity():
    import numpy as np
    from train.weighted_sampler_trainer import compute_image_weights

    class FakeDataset:
        def __init__(self, labels):
            self.labels = labels

    # car appears in 8 images, bus in 2, others in 1 -- a true rarity gradient
    labels = (
        [{"cls": np.array([0])} for _ in range(6)]
        + [{"cls": np.array([0, 1])}]  # car + bus
        + [{"cls": np.array([1])}]  # bus only
        + [{"cls": np.array([0])}]
        + [{"cls": np.array([2])}]  # others only, rarest
    )
    dataset = FakeDataset(labels)
    weights = compute_image_weights(dataset, num_classes=3)

    car_only_w, bus_only_w, others_only_w = weights[0], weights[7], weights[9]
    assert others_only_w > bus_only_w > car_only_w

    # Multi-class image must use MAX applicable weight, not sum/product
    mixed_w = weights[6]  # the car+bus image
    assert mixed_w == bus_only_w


def test_weighted_random_sampler_matches_assigned_weights():
    """Empirical check: a weight-5 image should be drawn ~5x as often as
    a weight-1 image over many draws, not just theoretically."""
    from collections import Counter
    from torch.utils.data import WeightedRandomSampler

    weights = [1.0, 1.0, 5.0]
    sampler = WeightedRandomSampler(weights, num_samples=10000, replacement=True)
    counts = Counter(list(sampler))
    ratio = counts[2] / counts[0]
    assert 4.0 < ratio < 6.0, f"Expected ~5x sampling ratio, got {ratio}"


def test_distributed_weighted_sampler_partitions_correctly():
    """Multi-GPU sampler must: give equal-length shards per rank, be
    deterministic for the same seed+epoch+rank, and reshuffle on a new
    epoch. This is the exact mechanism relied on for --device 0,1
    training -- a silently wrong partitioning here would break
    distributed training without any visible error."""
    from train.weighted_sampler_trainer import DistributedWeightedSampler

    weights = [1.0] * 8 + [5.0] * 2

    sampler0 = DistributedWeightedSampler(weights, num_replicas=2, rank=0, seed=42)
    sampler1 = DistributedWeightedSampler(weights, num_replicas=2, rank=1, seed=42)
    sampler0.set_epoch(0)
    sampler1.set_epoch(0)
    indices0 = list(sampler0)
    indices1 = list(sampler1)

    assert len(indices0) == len(indices1) == len(sampler0)

    # Same seed+epoch+rank must reproduce identically
    sampler0b = DistributedWeightedSampler(weights, num_replicas=2, rank=0, seed=42)
    sampler0b.set_epoch(0)
    assert list(sampler0b) == indices0

    # A new epoch must reshuffle, not repeat the same draw
    sampler0.set_epoch(1)
    assert list(sampler0) != indices0


def test_weighted_trainer_runs_real_training_end_to_end(tmp_path):
    """Full integration test: actually run a real (tiny) Ultralytics
    training job using WeightedDetectionTrainer, not just unit-test the
    pieces in isolation. Confirms the custom get_dataloader() override
    genuinely works inside Ultralytics' real training loop."""
    import shutil
    from ultralytics import YOLO
    from train.weighted_sampler_trainer import WeightedDetectionTrainer

    out_dir = str(tmp_path / "weighted_trainer_pytest")
    shutil.rmtree(out_dir, ignore_errors=True)
    try:
        model = YOLO("yolo11n.pt")
        model.train(
            trainer=WeightedDetectionTrainer,
            data="coco8.yaml", epochs=1, imgsz=64, batch=2, device="cpu",
            project=out_dir, name="run", verbose=False,
        )
        from pathlib import Path
        assert (Path(out_dir) / "run" / "weights" / "last.pt").exists()
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_train_yolo_cli_sampler_weighted_runs_end_to_end(tmp_path):
    """Same as above, but through the actual train_yolo.py CLI --sampler
    weighted flag -- catches wiring bugs (wrong import path, broken
    argument plumbing) that a module-level test alone wouldn't."""
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    out_dir = str(tmp_path / "cli_weighted_pytest")
    shutil.rmtree(out_dir, ignore_errors=True)
    try:
        result = subprocess.run(
            [sys.executable, "train/train_yolo.py",
             "--data", "coco8.yaml", "--model", "yolo11n.pt",
             "--imgsz", "64", "--batch", "2", "--device", "cpu",
             "--epochs", "1", "--project", out_dir, "--name", "run",
             "--sampler", "weighted"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Using WeightedDetectionTrainer" in result.stdout
        assert (Path(out_dir) / "run" / "weights" / "last.pt").exists()
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


# ============================================================================
# AUDIT REMEDIATION: CALIBRATION INTEGRITY & SPEED ESTIMATION TESTS
# ============================================================================

def test_perspective_transformer_geometry_validation():
    """Verify geometry validator rejects degenerate, non-finite, and negative dimensions."""
    # 1. Collinear points (zero area)
    collinear = np.array([[100, 100], [200, 200], [300, 300], [400, 400]], dtype=np.float32)
    valid_target = np.array([[0, 0], [10, 0], [10, 30], [0, 30]], dtype=np.float32)
    valid, msg = PerspectiveTransformer.validate_geometry(collinear, valid_target)
    assert not valid
    assert "area too small or degenerate" in msg

    # 2. Non-finite values (NaN)
    nan_pts = valid_target.copy()
    nan_pts[0, 0] = np.nan
    valid_nan, msg_nan = PerspectiveTransformer.validate_geometry(valid_target, nan_pts)
    assert not valid_nan
    assert "non-finite" in msg_nan

    # 3. Target dimensions too small or negative
    zero_target = np.zeros((4, 2), dtype=np.float32)
    valid_zero, msg_zero = PerspectiveTransformer.validate_geometry(valid_target, zero_target)
    assert not valid_zero


def test_perspective_transformer_provenance_and_status_gates():
    """Verify sample/illustrative files cannot claim genuine ground truth."""
    # Sample file dictionary
    sample_dict = {
        "source_points": [[480, 280], [800, 280], [1100, 720], [180, 720]],
        "target_points_meters": [[0, 0], [10, 0], [10, 50], [0, 50]],
        "is_calibrated": True,  # Attacker or misconfigured JSON claiming true
        "provenance": "illustrative_sample",
        "status": "illustrative",
    }
    valid, msg, info = PerspectiveTransformer.validate_calibration_dict(sample_dict)
    assert valid
    # Must be forced to False and illustrative because provenance is sample
    assert info["is_calibrated"] is False
    assert info["status"] == "illustrative"

    # Genuinely measured user calibration
    genuine_dict = {
        "source_points": [[480, 280], [800, 280], [1100, 720], [180, 720]],
        "target_points_meters": [[0, 0], [10, 0], [10, 50], [0, 50]],
        "is_calibrated": True,
        "provenance": "field_survey_measured",
        "status": "calibrated",
    }
    valid_g, msg_g, info_g = PerspectiveTransformer.validate_calibration_dict(genuine_dict)
    assert valid_g
    assert info_g["is_calibrated"] is True
    assert info_g["status"] == "calibrated"


def test_speed_estimator_known_displacement_and_frame_skipping():
    """Verify speed is mathematically exact (displacement / dt) across frame skipping."""
    # Construct a synthetic 1 px = 0.1 meter metric transformer
    src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    dst = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    transformer = PerspectiveTransformer(src, dst, is_calibrated=True, status="calibrated", provenance="synthetic_test")

    fps = 25.0
    estimator = SpeedEstimator(transformer, fps=fps, max_speed_kmh=200.0)

    # Frame 0: Vehicle 1 at (50, 0) -> real-world (5, 0)
    det0 = sv.Detections(
        xyxy=np.array([[40, -10, 60, 0]], dtype=np.float32),  # bottom center is (50, 0)
        tracker_id=np.array([1]),
        class_id=np.array([2]),
    )
    estimator.update(det0, frame_idx=0)

    # Frame 25 (dt = 1.0 sec): Vehicle 1 at (50, 100) -> real-world (5, 10)
    # Displacement = 10 meters in 1.0 second = 10 m/s = 36.0 km/h
    det25 = sv.Detections(
        xyxy=np.array([[40, 90, 60, 100]], dtype=np.float32),  # bottom center is (50, 100)
        tracker_id=np.array([1]),
        class_id=np.array([2]),
    )
    speeds_continuous = estimator.update(det25, frame_idx=25)
    assert pytest.approx(speeds_continuous[1], 0.1) == 36.0

    # Test Frame Skipping: Frame 0 -> Frame 20 (dt = 20 / 25 = 0.8s, within 1.5s gap threshold)
    # Displacement = 80 pixels = 8.0 meters -> 8.0 / 0.8 = 10 m/s = 36.0 km/h
    estimator_skip = SpeedEstimator(transformer, fps=fps, max_speed_kmh=200.0)
    estimator_skip.update(det0, frame_idx=0)
    det20 = sv.Detections(
        xyxy=np.array([[40, 70, 60, 80]], dtype=np.float32),  # bottom center is (50, 80) -> 8.0m
        tracker_id=np.array([1]),
        class_id=np.array([2]),
    )
    speeds_skipped = estimator_skip.update(det20, frame_idx=20)
    assert pytest.approx(speeds_skipped[1], 0.1) == 36.0


def test_speed_estimator_outlier_detection_and_exclusion():
    """Verify speeds > max_speed_kmh are excluded from statistics but kept in raw diagnostics."""
    src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    dst = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    transformer = PerspectiveTransformer(src, dst, is_calibrated=True, status="calibrated")
    estimator = SpeedEstimator(transformer, fps=25.0, max_speed_kmh=180.0)

    # Frame 0
    det0 = sv.Detections(
        xyxy=np.array([[40, 0, 60, 10]], dtype=np.float32),
        tracker_id=np.array([10]),
        class_id=np.array([2]),
    )
    estimator.update(det0, frame_idx=0)

    # Frame 1 (1/25 sec = 0.04s): Jump by 1000px = 100 meters in 0.04s => 2500 m/s = 9000 km/h!
    det1 = sv.Detections(
        xyxy=np.array([[40, 1000, 60, 1010]], dtype=np.float32),
        tracker_id=np.array([10]),
        class_id=np.array([2]),
    )
    speeds = estimator.update(det1, frame_idx=1)

    # Outlier must be excluded from reliable speeds
    assert 10 not in speeds
    assert estimator.outlier_count == 1
    assert 10 not in estimator.max_speeds

    # Raw diagnostic measurement must record the raw outlier for inspection
    raw = estimator.get_raw_diagnostics(10)
    assert len(raw) == 1
    assert raw[0]["is_outlier"] is True
    assert raw[0]["raw_kmh"] > 1000.0


def test_speed_estimator_track_discontinuity_gap_reset():
    """Verify track disappearing for > 1.5s resets history to prevent false velocity jump."""
    src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    dst = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    transformer = PerspectiveTransformer(src, dst, is_calibrated=True, status="calibrated")
    # max_gap_seconds = 1.5s -> 37 frames at 25 fps
    estimator = SpeedEstimator(transformer, fps=25.0, max_gap_seconds=1.5)

    # Frame 10
    det_first = sv.Detections(
        xyxy=np.array([[40, 0, 60, 10]], dtype=np.float32),
        tracker_id=np.array([5]),
        class_id=np.array([2]),
    )
    estimator.update(det_first, frame_idx=10)

    # Frame 70 (60 frames later = 2.4 sec > 1.5 sec threshold)
    det_reappear = sv.Detections(
        xyxy=np.array([[40, 80, 60, 90]], dtype=np.float32),
        tracker_id=np.array([5]),
        class_id=np.array([2]),
    )
    speeds = estimator.update(det_reappear, frame_idx=70)

    # Because gap exceeded threshold, track history reset: no speed calculation on first frame of reappearance
    assert 5 not in speeds
