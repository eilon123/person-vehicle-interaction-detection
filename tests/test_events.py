import copy

import pytest

from person_vehicle.events import ClipOutput, active_events, collapse_physical_episodes, merge_events
from person_vehicle.evaluate import evaluate, match_events, temporal_iou
from person_vehicle.verify import (build_ballots, correct_enter_exit_direction,
                                   parse_and_aggregate_votes, parse_decision, select_sample_rows)


def event(start=1, end=3, person="p1", action="enter"):
    return {"event_id": "e1", "type": action, "persons": [{"person_id": person, "description": "dark jacket"}],
            "vehicle": {"vehicle_id": "v1", "description": "white car"},
            "spans": [{"start_s": start, "end_s": end}], "evidence_frames": [10],
            "truncated_start": False, "truncated_end": False, "group_id": None}


def test_disjoint_spans_iou():
    spans = [{"start_s": 0, "end_s": 1}, {"start_s": 3, "end_s": 4}]
    assert temporal_iou(spans, [{"start_s": 1, "end_s": 3}]) == 0
    assert temporal_iou(spans, [{"start_s": 0, "end_s": 4}]) == 0.5


def test_duplicate_and_wrong_pair():
    assert len(match_events([event(), event()], [event()])) == 1
    assert match_events([event(person="p2")], [event()]) == []
    assert len(match_events([event(person="p2")], [event()], mapping={"p2": "p1"})) == 1
    assert match_events([event(action="exit")], [event()], typed=True) == []


def test_half_open_boundaries_and_gap():
    e = event()
    e["spans"].append({"start_s": 5, "end_s": 6})
    assert active_events([e], 1) == [e]
    assert active_events([e], 3) == []
    assert active_events([e], 4) == []
    assert active_events([e], 5) == [e]


def test_merge_preserves_separate_episodes():
    merged = merge_events([event(), event(2, 4), event(5, 6)])
    assert len(merged) == 2
    assert merged[0]["spans"] == [{"start_s": 1, "end_s": 4}]


def test_temporal_event_nms_merges_overlap_but_never_gap():
    rows = [{"objects": [
        {"id": "p1", "kind": "person", "bbox": [0, 0, 10, 20]},
        {"id": "v1", "kind": "car", "bbox": [8, 0, 30, 20]}]}]
    config = {"collapse_physical_episodes": True, "event_nms_temporal_iou": .5,
              "event_nms_temporal_containment": .8}
    overlapping = collapse_physical_episodes(
        [event(1, 4, action="enter"), event(1.2, 3.8, action="exit")], rows, config)
    assert len(overlapping) == 1
    assert overlapping[0]["type"] == "other_interaction"
    separate = collapse_physical_episodes(
        [event(0, 1, action="exit"), event(3, 4, action="enter")], rows, config)
    assert len(separate) == 2


def test_schema_rejects_out_of_range():
    clip = dict(clip_id="x", source_sha256="abc", status="ok", duration_s=5, frame_count=100, interactions=[event()])
    ClipOutput.model_validate(clip)
    clip["interactions"][0]["spans"][0]["end_s"] = 7
    with pytest.raises(ValueError):
        ClipOutput.model_validate(clip)


def test_verifier_rejects_invented_evidence():
    text = '{"decision":"interaction","reason":"entry","events":[{"type":"enter","start_frame":0,"end_frame":10,"person_description":"coat","vehicle_description":"car","evidence_frames":[8]}]}'
    with pytest.raises(ValueError):
        parse_decision(text, [0, 5, 10])
    assert parse_decision('{"decision":"uncertain","reason":"occluded","events":[]}', [0])["decision"] == "uncertain"


def test_transition_dense_sampling_keeps_context_and_visibility_change():
    rows = [{"frame_index": i, "objects": ([{"id": "v"}] + ([{"id": "p"}] if 4 <= i < 7 else []))}
            for i in range(10)]
    selected = select_sample_rows(rows, {"person_id": "p", "vehicle_id": "v"}, 6, "transition_dense")
    indices = [row["frame_index"] for row in selected]
    assert indices[0] == 0 and indices[-1] == 9
    assert 4 in indices and 7 in indices


def test_temporal_ballot_majority_requires_consecutive_support():
    ballots = build_ballots(list(range(16)), count=5, width=6)
    votes = []
    for ballot in ballots:
        positive = ballot["ballot_id"] in (2, 3, 4)
        votes.append({"ballot_id": ballot["ballot_id"], "decision": "interaction" if positive else "no_interaction",
                      "type": "enter" if positive else None,
                      "evidence_frames": ballot["frame_indices"][:2] if positive else [], "reason": "test"})
    text = __import__("json").dumps({"person_description": "dark coat", "vehicle_description": "white car", "votes": votes})
    decision, raw = parse_and_aggregate_votes(text, ballots, .6, 2)
    assert decision["decision"] == "interaction"
    assert decision["events"][0]["type"] == "enter"
    assert decision["events"][0]["start_frame"] == ballots[1]["frame_indices"][0]
    assert decision["events"][0]["end_frame"] == ballots[3]["frame_indices"][-1]
    assert len(raw["votes"]) == 5


def test_temporal_ballot_parser_accepts_singleton_array():
    ballots = build_ballots(list(range(16)), count=5, width=6)
    votes = [{"ballot_id": ballot["ballot_id"], "decision": "interaction", "type": "exit",
              "evidence_frames": ballot["frame_indices"][:1], "reason": "test"}
             for ballot in ballots]
    payload = [{"person_description": "person", "vehicle_description": "car", "votes": votes}]
    decision, _ = parse_and_aggregate_votes(__import__("json").dumps(payload), ballots, .6, 2)
    assert decision["decision"] == "interaction"
    assert decision["events"][0]["type"] == "exit"


def test_eval_counts_and_undefined_rates():
    reference = {"clip": {"interactions": [event()], "duration_s": 60}}
    predictions = {"clip": {"status": "ok", "interactions": [event(), copy.deepcopy(event())]}}
    result = evaluate(predictions, reference)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)
    assert result["precision"] == 0.5
    assert result["false_alarms_per_minute"] == 1
    with pytest.raises(ValueError):
        evaluate({}, reference)


def test_direction_correction_only_flips_existing_enter_exit():
    rows = []
    for index in range(6):
        inside = index < 2
        person_box = [40, 30, 60, 80] if inside else [115 + index * 5, 30, 135 + index * 5, 80]
        rows.append({"objects": [
            {"id": "p1", "kind": "person", "bbox": person_box},
            {"id": "v1", "kind": "car", "bbox": [20, 20, 120, 100]},
        ]})
    config = {"direction_correction_enabled": True, "direction_context_frames": 0,
              "direction_edge_observations": 2, "direction_min_gap_change": .05}
    candidate = {"person_id": "p1", "vehicle_id": "v1"}
    wrong = {"type": "enter", "start_frame": 0, "end_frame": 5}
    corrected, note = correct_enter_exit_direction(wrong, rows, candidate, config)
    assert corrected["type"] == "exit" and "corrected" in note
    other, note = correct_enter_exit_direction({**wrong, "type": "other_interaction"}, rows, candidate, config)
    assert other["type"] == "other_interaction" and note is None


def test_track_rule_flips_enter_when_person_walks_along_vehicle_to_edge():
    from person_vehicle.events import apply_track_consistency_rules
    rows = []
    for index, x in enumerate([75, 65, 50, 35, 15, -5]):
        rows.append({"timestamp_s": float(index), "objects": [
            {"id": "p1", "kind": "person", "bbox": [x, 20, x + 20, 80]},
            {"id": "v1", "kind": "car", "bbox": [0, 10, 100, 90]},
        ]})
    candidate = event()
    candidate["type"] = "enter"
    candidate["persons"] = [{"person_id": "p1", "description": "person"}]
    candidate["vehicle"] = {"vehicle_id": "v1", "description": "car"}
    candidate["spans"] = [{"start_s": 0, "end_s": 5}]
    config = {"track_consistency_rules": True,
              "enter_exit_min_gap_change": .2,
              "enter_exit_min_longitudinal_change": .45,
              "enter_exit_vehicle_edge_fraction": .18}
    corrected = apply_track_consistency_rules([candidate], rows, config)
    assert corrected[0]["type"] == "exit"


def test_disappearance_at_persistent_vehicle_flips_other_to_enter():
    from person_vehicle.events import apply_track_consistency_rules
    rows = []
    for index, x in enumerate([0, 35, 70]):
        rows.append({"timestamp_s": index * .2, "objects": [
            {"id": "p1", "kind": "person", "bbox": [x, 20, x + 20, 80]},
            {"id": "v1", "kind": "car", "bbox": [90, 10, 190, 90]},
        ]})
    for index in range(3, 7):
        rows.append({"timestamp_s": index * .2, "objects": [
            {"id": "v1", "kind": "car", "bbox": [90, 10, 190, 90]},
        ]})
    candidate = event(action="other_interaction")
    candidate["persons"] = [{"person_id": "p1", "description": "person"}]
    candidate["vehicle"] = {"vehicle_id": "v1", "description": "car"}
    candidate["spans"] = [{"start_s": 0, "end_s": .8}]
    config = {"track_consistency_rules": True, "disappearance_enter_cue": True,
              "disappearance_enter_min_approach": .08,
              "disappearance_enter_max_final_gap": .1}
    corrected = apply_track_consistency_rules([candidate], rows, config)
    assert corrected[0]["type"] == "enter"


def test_event_cannot_start_before_first_tracked_person_frame():
    from person_vehicle.events import clip_events_to_person_visibility
    rows = [
        {"frame_index": 0, "timestamp_s": 0.0, "objects": []},
        {"frame_index": 1, "timestamp_s": 1.0, "objects": [
            {"id": "p1", "kind": "person", "bbox": [0, 0, 2, 4], "observed": False}]},
        {"frame_index": 2, "timestamp_s": 2.0, "objects": [
            {"id": "p1", "kind": "person", "bbox": [0, 0, 2, 4], "observed": True}]},
        {"frame_index": 3, "timestamp_s": 3.0, "objects": [
            {"id": "p1", "kind": "person", "bbox": [0, 0, 2, 4], "observed": True}]},
    ]
    candidate = event()
    candidate["persons"] = [{"person_id": "p1", "description": "person"}]
    candidate["spans"] = [{"start_s": 0.0, "end_s": 3.5}]
    candidate["evidence_frames"] = [0, 2]
    candidate["truncated_start"] = True
    clipped = clip_events_to_person_visibility([candidate], rows)
    assert clipped[0]["spans"] == [{"start_s": 1.0, "end_s": 3.5}]
    assert clipped[0]["evidence_frames"] == [2]
    assert clipped[0]["truncated_start"] is True


def test_door_operation_is_exposed_as_load_unload():
    from person_vehicle.events import normalize_door_operations
    door = event(action="door_operation")
    enter = event(action="enter")
    normalized = normalize_door_operations([door, enter])
    assert [item["type"] for item in normalized] == ["load_unload", "enter"]
