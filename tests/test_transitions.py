from person_vehicle.transitions import infer_person_vehicle_transitions, fuse_transition_events


def row(index, person=None, vehicle=True):
    objects = []
    if vehicle:
        objects.append({"id": "v1", "kind": "car", "bbox": [40, 40, 100, 100]})
    if person:
        objects.append({"id": "p1", "kind": "person", "bbox": person})
    return {"frame_index": index, "timestamp_s": index / 10, "scene": 0, "objects": objects}


def settings():
    return {"geometry_transitions": True, "transition_min_observations": 4,
            "transition_distance_change": .18, "transition_size_ratio": 1.25,
            "transition_vehicle_persistence_s": .35, "transition_event_context_s": 1.5}


def test_person_appearing_inside_then_growing_is_exit_without_requiring_distance_change():
    rows = [row(i) for i in range(5)] + [
        row(5, [65, 55, 72, 70]), row(6, [70, 55, 80, 75]),
        row(7, [62, 48, 82, 84]), row(8, [55, 42, 90, 96])]
    events = infer_person_vehicle_transitions(rows, [{"person_id": "p1", "vehicle_id": "v1", "scene": 0}],
                                             {"duration_s": 1}, settings())
    assert len(events) == 1 and events[0]["type"] == "exit"


def test_person_approaching_shrinking_and_disappearing_inside_is_enter():
    rows = [row(0, [5, 45, 35, 95]), row(1, [20, 48, 45, 90]),
            row(2, [38, 50, 56, 85]), row(3, [55, 55, 68, 78])] + [row(i) for i in range(4, 9)]
    events = infer_person_vehicle_transitions(rows, [{"person_id": "p1", "vehicle_id": "v1", "scene": 0}],
                                             {"duration_s": 1}, settings())
    assert len(events) == 1 and events[0]["type"] == "enter"


def test_disappearance_without_motion_or_size_change_is_not_interaction():
    rows = [row(i, [60, 50, 75, 80]) for i in range(4)] + [row(i) for i in range(4, 9)]
    assert infer_person_vehicle_transitions(rows, [{"person_id": "p1", "vehicle_id": "v1", "scene": 0}],
                                            {"duration_s": 1}, settings()) == []


def test_geometry_corrects_vlm_type_and_boundaries_instead_of_duplicating():
    base = [{"event_id": "e1", "type": "other_interaction",
             "persons": [{"person_id": "p1", "description": "person"}],
             "vehicle": {"vehicle_id": "v1", "description": "car"},
             "spans": [{"start_s": 1, "end_s": 4}], "evidence_frames": [10],
             "truncated_start": False, "truncated_end": False, "group_id": None}]
    transition = [{**base[0], "type": "exit", "spans": [{"start_s": 2, "end_s": 3}],
                   "evidence_frames": [20, 30], "group_id": "geometry_transition"}]
    fused = fuse_transition_events(base, transition, {"geometry_fusion_mode": "correct_then_add"})
    assert len(fused) == 1
    assert fused[0]["type"] == "exit"
    assert fused[0]["spans"] == [{"start_s": 2, "end_s": 3}]
    assert fused[0]["group_id"] == "vlm_geometry_corrected"
