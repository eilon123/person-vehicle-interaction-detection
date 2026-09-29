import pytest

from person_vehicle.annotate import empty_clip, make_event


def test_manual_event_matches_evaluator_shape():
    event = make_event(1, 1.0, 2.0, 30, 60, "enter", "person_a", "dark coat",
                       "vehicle_a", "silver car")
    assert event["event_id"] == "ref001"
    assert event["spans"] == [{"start_s": 1.0, "end_s": 2.0}]
    assert empty_clip(3.5) == {"duration_s": 3.5, "interactions": []}


def test_manual_event_rejects_zero_length_interval():
    with pytest.raises(ValueError, match="end"):
        make_event(1, 1.0, 1.0, 30, 30, "enter", "p", "person", "v", "car")
