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
