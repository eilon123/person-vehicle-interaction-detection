from person_vehicle.tracking import stitch_occluded_tracklets


def _row(frame, person_id, x, vehicle_id="v1"):
    return {"frame_index": frame, "timestamp_s": frame * .1, "scene": 0, "objects": [
        {"id": person_id, "kind": "person", "bbox": [x, 10, x + 20, 70],
         "confidence": .8, "observed": True},
        {"id": vehicle_id, "kind": "car", "bbox": [0, 0, 150, 90],
         "confidence": .8, "observed": True},
    ]}


def test_stitches_short_same_vehicle_occlusion_with_motion_continuity():
    rows = [_row(0, "p1", 10), _row(1, "p1", 20),
            _row(4, "p2", 50), _row(5, "p2", 60)]
    config = {"occlusion_tracklet_stitch_enabled": True,
              "occlusion_tracklet_max_gap_s": .5,
              "occlusion_tracklet_max_predicted_distance": 1.0,
              "occlusion_tracklet_min_direction_cosine": 0.0}
    stitched = stitch_occluded_tracklets(rows, config)
    assert {obj["id"] for row in stitched for obj in row["objects"] if obj["kind"] == "person"} == {"p1"}


def test_does_not_stitch_tracklets_anchored_to_different_vehicles():
    rows = [_row(0, "p1", 10, "v1"), _row(1, "p1", 20, "v1"),
            _row(4, "p2", 50, "v2"), _row(5, "p2", 60, "v2")]
    config = {"occlusion_tracklet_stitch_enabled": True,
              "occlusion_tracklet_max_gap_s": .5,
              "occlusion_tracklet_max_predicted_distance": 1.0,
              "occlusion_tracklet_min_direction_cosine": 0.0}
    stitched = stitch_occluded_tracklets(rows, config)
    assert {obj["id"] for row in stitched for obj in row["objects"] if obj["kind"] == "person"} == {"p1", "p2"}
