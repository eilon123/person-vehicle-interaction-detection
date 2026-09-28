import copy

import pytest

from person_vehicle.events import ClipOutput, active_events, merge_events
from person_vehicle.evaluate import evaluate, match_events, temporal_iou
from person_vehicle.verify import parse_decision


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


def test_eval_counts_and_undefined_rates():
    reference = {"clip": {"interactions": [event()], "duration_s": 60}}
    predictions = {"clip": {"status": "ok", "interactions": [event(), copy.deepcopy(event())]}}
    result = evaluate(predictions, reference)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)
    assert result["precision"] == 0.5
    assert result["false_alarms_per_minute"] == 1
    with pytest.raises(ValueError):
        evaluate({}, reference)
