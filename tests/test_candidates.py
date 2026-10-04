from person_vehicle.candidates import propose


def test_short_encounter_proposed_but_far_passby_ignored():
    rows = [{"scene": 0, "timestamp_s": 2, "objects": [
        {"id": "p1", "kind": "person", "bbox": [9, 0, 11, 4]},
        {"id": "p2", "kind": "person", "bbox": [100, 0, 110, 4]},
        {"id": "v1", "kind": "car", "bbox": [0, 0, 10, 5]}]}]
    config = dict(near_margin=0.2, candidate_gap_s=1, context_s=1, window_s=8, window_overlap_s=2)
    candidates = propose(rows, 5, config)
    assert len(candidates) == 1
    assert candidates[0]["person_id"] == "p1"
    assert candidates[0]["start_s"] == 1


def test_scene_ids_do_not_join_encounters():
    objects = [{"id": "p", "kind": "person", "bbox": [0, 0, 2, 2]},
               {"id": "v", "kind": "car", "bbox": [0, 0, 5, 5]}]
    rows = [{"scene": s, "timestamp_s": s + 1, "objects": objects} for s in [0, 1]]
    config = dict(near_margin=0.2, candidate_gap_s=1, context_s=1, window_s=8, window_overlap_s=2)
    assert len(propose(rows, 5, config)) == 2


def test_head_scaled_proximity_accounts_for_perspective():
    config = dict(near_margin=0.01, candidate_gap_s=1, context_s=1, window_s=8,
                  window_overlap_s=2, head_scaled_candidate_distance=True,
                  candidate_max_head_widths=2.0)
    rows = [{"scene": 0, "timestamp_s": 1, "objects": [
        {"id": "v", "kind": "car", "bbox": [20, 0, 40, 20]},
        {"id": "near", "kind": "person", "bbox": [0, 0, 10, 20],
         "head_bbox": [2, 0, 8, 6]},
        {"id": "far", "kind": "person", "bbox": [50, 0, 60, 20],
         "head_bbox": [53, 0, 55, 2]}]}]
    candidates = propose(rows, 3, config)
    assert [candidate["person_id"] for candidate in candidates] == ["near"]


def test_overlapping_vehicle_choice_uses_center_distance_not_id():
    config = dict(near_margin=.3, candidate_gap_s=1, context_s=.1, window_s=8,
                  window_overlap_s=2, exclusive_person_vehicle=True,
                  vehicle_assignment_center_weight=.2)
    rows = [{"scene": 0, "timestamp_s": 1, "objects": [
        {"id": "p", "kind": "person", "bbox": [95, 45, 115, 85]},
        {"id": "v_far", "kind": "car", "bbox": [0, 0, 105, 50], "confidence": .9},
        {"id": "v_near", "kind": "car", "bbox": [10, 35, 100, 100], "confidence": .8},
    ]}]
    assert propose(rows, 2, config)[0]["vehicle_id"] == "v_near"


def test_vehicle_assignment_does_not_switch_during_short_detection_gap():
    config = dict(near_margin=.2, candidate_gap_s=2, context_s=.1, window_s=8,
                  window_overlap_s=2, exclusive_person_vehicle=True,
                  vehicle_assignment_hysteresis=True,
                  vehicle_assignment_max_gap_s=3,
                  vehicle_assignment_hold_when_missing=True)
    rows = [
        {"scene": 0, "timestamp_s": 0, "objects": [
            {"id": "p", "kind": "person", "bbox": [90, 20, 110, 70]},
            {"id": "correct", "kind": "car", "bbox": [0, 0, 100, 80]}]},
        {"scene": 0, "timestamp_s": 1, "objects": [
            {"id": "p", "kind": "person", "bbox": [90, 20, 110, 70]},
            {"id": "wrong", "kind": "car", "bbox": [95, 0, 195, 80]}]},
        {"scene": 0, "timestamp_s": 2, "objects": [
            {"id": "p", "kind": "person", "bbox": [90, 20, 110, 70]},
            {"id": "correct", "kind": "car", "bbox": [0, 0, 100, 80]},
            {"id": "wrong", "kind": "car", "bbox": [95, 0, 195, 80]}]},
    ]
    candidates = propose(rows, 3, config)
    assert {item["vehicle_id"] for item in candidates} == {"correct"}


def test_short_upright_track_touching_vehicle_bypasses_duration_filter():
    config = dict(near_margin=.2, candidate_gap_s=1, context_s=.1, window_s=8,
                  window_overlap_s=2, person_track_min_detections=5,
                  person_track_min_duration_s=.75, short_track_vehicle_exception=True,
                  short_track_vehicle_min_detections=2,
                  short_track_vehicle_min_aspect_ratio=.85)
    rows = [
        {"scene": 0, "timestamp_s": 0.0, "objects": [
            {"id": "p", "kind": "person", "bbox": [8, 0, 10, 5]},
            {"id": "v", "kind": "car", "bbox": [10, 0, 20, 6]}]},
        {"scene": 0, "timestamp_s": 0.1, "objects": [
            {"id": "p", "kind": "person", "bbox": [9, 0, 11, 5]},
            {"id": "v", "kind": "car", "bbox": [10, 0, 20, 6]}]},
    ]
    assert propose(rows, 1, config)[0]["person_id"] == "p"


def test_short_wide_detection_inside_vehicle_remains_filtered():
    config = dict(near_margin=.2, candidate_gap_s=1, context_s=.1, window_s=8,
                  window_overlap_s=2, person_track_min_detections=5,
                  person_track_min_duration_s=.75, short_track_vehicle_exception=True,
                  short_track_vehicle_min_detections=2,
                  short_track_vehicle_min_aspect_ratio=.85)
    rows = [
        {"scene": 0, "timestamp_s": 0.0, "objects": [
            {"id": "noise", "kind": "person", "bbox": [11, 2, 17, 5]},
            {"id": "v", "kind": "car", "bbox": [10, 0, 20, 6]}]},
        {"scene": 0, "timestamp_s": 0.1, "objects": [
            {"id": "noise", "kind": "person", "bbox": [11, 2, 17, 5]},
            {"id": "v", "kind": "car", "bbox": [10, 0, 20, 6]}]},
    ]
    assert propose(rows, 1, config) == []


def test_horizontal_body_axis_prefers_vehicle_at_person_depth():
    config = dict(near_margin=.6, candidate_gap_s=1, context_s=.1, window_s=8,
                  window_overlap_s=2, exclusive_person_vehicle=True,
                  vehicle_assignment_center_weight=0,
                  horizontal_body_vehicle_assignment=True,
                  horizontal_body_min_aspect_ratio=1.5,
                  horizontal_body_min_confidence=.2,
                  horizontal_body_anchor_fraction=.6)
    rows = [{"scene": 0, "timestamp_s": 1, "objects": [
        {"id": "p", "kind": "person", "bbox": [250, 170, 290, 270], "confidence": .8},
        {"id": "lower", "kind": "car", "bbox": [0, 165, 240, 335]},
        {"id": "upper", "kind": "car", "bbox": [44, 75, 260, 183]},
    ]}]
    assert propose(rows, 2, config)[0]["vehicle_id"] == "lower"


def test_horizontal_body_rule_does_not_change_single_vehicle_path():
    config = dict(near_margin=.6, candidate_gap_s=1, context_s=.1, window_s=8,
                  window_overlap_s=2, exclusive_person_vehicle=True,
                  horizontal_body_vehicle_assignment=True)
    rows = [{"scene": 0, "timestamp_s": 1, "objects": [
        {"id": "p", "kind": "person", "bbox": [250, 170, 290, 270], "confidence": .8},
        {"id": "only", "kind": "car", "bbox": [44, 75, 260, 183]},
    ]}]
    assert propose(rows, 2, config)[0]["vehicle_id"] == "only"
