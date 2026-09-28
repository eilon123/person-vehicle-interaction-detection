from person_vehicle.evaluate import proposal_recall, passerby_false_positive_rate, rates


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
