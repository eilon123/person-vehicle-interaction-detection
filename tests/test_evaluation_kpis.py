from person_vehicle.evaluate import evaluate, proposal_recall, passerby_false_positive_rate, rates, temporal_occupancy


def test_candidate_requires_pair_and_coverage():
    reference = {"clip": {"interactions": [{"persons": [{"person_id": "p"}],
        "vehicle": {"vehicle_id": "v"}, "spans": [{"start_s": 1, "end_s": 3}]}]}}
    candidates = {"clip": [{"person_id": "wrong", "vehicle_id": "v", "start_s": 1, "end_s": 3}]}
    assert proposal_recall(candidates, reference)["recall"] == 0
    candidates["clip"][0]["person_id"] = "p"
    candidates["clip"][0]["end_s"] = 2
    assert proposal_recall(candidates, reference)["recall"] == 1


def test_false_positive_encounter_counted_once():
    event = {"persons": [{"person_id": "p"}], "vehicle": {"vehicle_id": "v"},
             "spans": [{"start_s": 1, "end_s": 2}]}
    predicted = {"clip": {"interactions": [event, event]}}
    negative = {"clip": [{"person_id": "p", "vehicle_id": "v", "spans": [{"start_s": 0, "end_s": 3}]}]}
    result = passerby_false_positive_rate(predicted, negative)
    assert result["false_positive_encounters"] == 1
    assert result["rate"] == 1


def test_empty_rates_are_not_perfect():
    assert rates(0, 0, 0) == {"tp": 0, "fp": 0, "fn": 0, "precision": None, "recall": None, "f1": None}


def test_binary_timeline_can_ignore_participant_ids():
    prediction = {"clip": {"status": "ok", "interactions": [{"persons": [{"person_id": "p01"}],
        "vehicle": {"vehicle_id": "v01"}, "spans": [{"start_s": 1, "end_s": 2}]}]}}
    reference = {"clip": {"duration_s": 3, "interactions": [{"persons": [{"person_id": "manual_person"}],
        "vehicle": {"vehicle_id": "manual_vehicle"}, "spans": [{"start_s": 1, "end_s": 2}]}]}}
    result = evaluate(prediction, reference, ignore_participants=True)
    assert result["f1"] == 1
    assert result["participant_matching"] == "ignored"


def test_threshold_free_temporal_occupancy_unions_parallel_events():
    prediction = {"clip": {"interactions": [
        {"spans": [{"start_s": 0, "end_s": 4}]}, {"spans": [{"start_s": 2, "end_s": 5}]}]}}
    reference = {"clip": {"interactions": [{"spans": [{"start_s": 3, "end_s": 6}]}]}}
    result = temporal_occupancy(prediction, reference)
    assert result["predicted_s"] == 5
    assert result["reference_s"] == 3
    assert result["intersection_s"] == 2
    assert result["temporal_iou"] == 1 / 3
