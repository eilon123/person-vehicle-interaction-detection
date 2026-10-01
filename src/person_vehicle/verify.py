import json
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import Field

from .events import StrictModel
from .io import fingerprint, read_json, write_json
from .video import frames


PROMPT = """You are examining chronological video frames for a person-vehicle interaction.
Only evaluate person {person_id} (green box) with vehicle {vehicle_id} (blue box).
Frame indices and timestamps are printed on each image. A nearby person, walking past,
or disappearing behind a car is NOT sufficient evidence of interaction.
Require visible directed action: entering, exiting, opening/closing a door,
loading/unloading, or working on/touching the vehicle. If evidence is ambiguous,
answer uncertain. Do not infer action from proximity alone. Use no demographic claims.
Return a JSON object with keys decision, reason, events.
Choose exactly one decision: interaction, no_interaction, uncertain.
reason must explain the specific visible action, not just say 'visible evidence'.
events is a list of objects with keys: type, start_frame, end_frame,
person_description, vehicle_description, evidence_frames.
Choose exactly one type for each event: enter, exit, door_operation, load_unload,
other_interaction. Frame fields are integers and evidence_frames is a list of integers.
Describe the person's visible clothing (not just what they are doing) and the
vehicle's visible color/type. Do not copy these instructions into your answer.
Enter means the body moves INTO the cabin; exit means OUT. Bending into a window
while staying outside is other_interaction. load_unload requires a visible OBJECT
transfer; a person getting into a car is enter, not load_unload.
Use only supplied frame indices. start_frame and end_frame delimit visible action,
end_frame is inclusive. events must be empty unless decision is interaction.
Multiple distinct actions may be returned. Include associated door action in an
entry/exit rather than duplicating it. Explain what supports action direction.
First compare the person's state in the earliest, middle, and latest frames:
outside the vehicle, crossing a visible doorway/cabin boundary, inside/absent with
direct transition evidence, or unknown. Enter requires an outside-to-inside body
transition; exit requires the reverse. A door operation requires visible door
movement caused by the target person. Reaching toward a door, standing beside it,
or a static open door is not enough. If the body transition is visible, prefer
enter/exit and include its associated door action in that event.
Use the full visible action range, from the first directed movement to completion.
Do not create a one-frame event unless the action is cut by a clip boundary.
Before returning interaction, identify at least two chronological observations
that support the action. Otherwise return uncertain or no_interaction.
"""

LOAD_UNLOAD_ACCESS_RULE = ("For load_unload, look for a directed loading or unloading action at a "
                           "vehicle access opening such as a door, trunk, or cargo compartment. The "
                           "opening may already be open or may open during the action. Evidence may "
                           "be an object moving into or out of the vehicle, hands or arms working "
                           "inside, or a person leaning part of their body into the opening and then "
                           "withdrawing while remaining outside. The object and hands need not both "
                           "be visible. Include an associated opening in the load_unload event rather "
                           "than duplicating it as door_operation. Door movement alone, proximity, "
                           "or standing beside an open vehicle is insufficient. Full-body movement "
                           "into or out of the cabin is enter or exit, not load_unload. If the "
                           "directed access action is not visible, answer uncertain.")

UNTRACKED_EXIT_PROMPT = """Examine the chronological frames around vehicle {vehicle_id} (blue box).
The person detector did not provide a person track at the beginning of this clip.
Look for a person who is initially inside or obscured by this vehicle and then
visibly emerges from its cabin and moves outward. If that body transition is
visible, classify it as exit even if there is no green person box. Give the
first and last supplied frame indices supporting the transition and at least
two supplied evidence frame indices. Describe the visible person and vehicle.
Do not infer an exit from proximity, a passerby, a person appearing behind the
vehicle, or camera motion alone. If the cabin-to-outside transition cannot be
seen, return uncertain or no_interaction.
Return a JSON object with decision, reason, events. decision must be one of
interaction, no_interaction, uncertain. events must be empty unless decision
is interaction; each event has type exit, start_frame, end_frame,
person_description, vehicle_description, evidence_frames. Frame fields are
integers and end_frame is inclusive. Allowed frame indices in chronological
order: {indices}.
"""

VOTING_PROMPT = """You are examining chronological video frames for person {person_id} (green box)
and vehicle {vehicle_id} (blue box). Evaluate each listed temporal ballot independently.
A ballot supports interaction only when it shows directed action: entering, exiting, opening or
closing a door, loading or unloading an object, or touching/working on the vehicle. Proximity,
walking past, or disappearance behind a vehicle is not sufficient. Enter is outside-to-cabin;
exit is cabin-to-outside. Return one JSON object with person_description, vehicle_description,
and votes. votes must contain exactly one object per ballot with keys ballot_id, decision, type,
evidence_frames, reason. decision is interaction, no_interaction, or uncertain. type is one of
enter, exit, door_operation, load_unload, other_interaction when decision is interaction, and
null otherwise. Use only frame indices belonging to that ballot. Do not let one ballot's answer
determine another ballot. Ballots:
{ballots}
"""


class VerifiedEvent(StrictModel):
    type: Literal["enter", "exit", "door_operation", "load_unload", "other_interaction"]
    start_frame: int
    end_frame: int
    person_description: str = Field(min_length=1)
    vehicle_description: str = Field(min_length=1)
    evidence_frames: list[int] = Field(min_length=1)


class Decision(StrictModel):
    decision: Literal["interaction", "no_interaction", "uncertain"]
    reason: str
    events: list[VerifiedEvent]


class BallotVote(StrictModel):
    ballot_id: int
    decision: Literal["interaction", "no_interaction", "uncertain"]
    type: Literal["enter", "exit", "door_operation", "load_unload", "other_interaction"] | None
    evidence_frames: list[int]
    reason: str


class VotingDecision(StrictModel):
    person_description: str = Field(min_length=1)
    vehicle_description: str = Field(min_length=1)
    votes: list[BallotVote]


def build_ballots(indices, count=5, width=6):
    """Create overlapping chronological ballots over sampled frame indices."""
    if not indices:
        return []
    width = min(width, len(indices))
    starts = np.linspace(0, len(indices) - width, min(count, len(indices) - width + 1), dtype=int)
    return [{"ballot_id": number, "frame_indices": indices[start:start + width]}
            for number, start in enumerate(sorted(set(starts)), 1)]


def parse_and_aggregate_votes(text, ballots, minimum_fraction=.6, minimum_consecutive=2):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    payload = json.loads(text)
    # Qwen sometimes wraps the requested object in a singleton JSON array.
    # Accept that harmless formatting variation while keeping schema validation strict.
    if isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    result = VotingDecision.model_validate(payload).model_dump()
    expected = {ballot["ballot_id"]: set(ballot["frame_indices"]) for ballot in ballots}
    if {vote["ballot_id"] for vote in result["votes"]} != set(expected):
        raise ValueError("Voting response does not contain exactly the requested ballots")
    for vote in result["votes"]:
        if not set(vote["evidence_frames"]) <= expected[vote["ballot_id"]]:
            raise ValueError("Vote evidence is outside its ballot")
        if (vote["decision"] == "interaction") != (vote["type"] is not None):
            raise ValueError("Vote action type disagrees with decision")
    positive = [vote for vote in result["votes"] if vote["decision"] == "interaction"]
    fraction = len(positive) / len(ballots)
    positive_ids = {vote["ballot_id"] for vote in positive}
    longest = current = 0
    for ballot_id in sorted(expected):
        current = current + 1 if ballot_id in positive_ids else 0
        longest = max(longest, current)
    if fraction < minimum_fraction or longest < minimum_consecutive:
        return {"decision": "no_interaction", "reason":
                f"Voting rejected: {len(positive)}/{len(ballots)} positive ballots; longest run {longest}",
                "events": []}, result
    counts = {}
    for vote in positive:
        counts[vote["type"]] = counts.get(vote["type"], 0) + 1
    action = max(sorted(counts), key=lambda label: counts[label])
    winning = [vote for vote in positive if vote["type"] == action]
    evidence = sorted({frame for vote in winning for frame in vote["evidence_frames"]})
    if not evidence:
        return {"decision": "uncertain", "reason": "Winning votes supplied no evidence frames", "events": []}, result
    winning_ids = {vote["ballot_id"] for vote in winning}
    positive_window_frames = sorted({frame for ballot in ballots if ballot["ballot_id"] in winning_ids
                                     for frame in ballot["frame_indices"]})
    event = {"type": action, "start_frame": positive_window_frames[0],
             "end_frame": positive_window_frames[-1],
             "person_description": result["person_description"],
             "vehicle_description": result["vehicle_description"], "evidence_frames": evidence}
    return {"decision": "interaction", "reason":
            f"Voting accepted: {len(positive)}/{len(ballots)} positive ballots; {counts[action]} support {action}",
            "events": [event]}, result


def parse_decision(text, allowed):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = Decision.model_validate_json(text)
    if (result.decision == "interaction") != bool(result.events):
        raise ValueError("Decision and events disagree")
    for event in result.events:
        if event.start_frame > event.end_frame:
            raise ValueError("Reversed event bounds")
        if not {event.start_frame, event.end_frame, *event.evidence_frames} <= set(allowed):
            raise ValueError("Verifier invented a frame index")
        if not all(event.start_frame <= frame <= event.end_frame for frame in event.evidence_frames):
            raise ValueError("Evidence outside event")
    return result.model_dump()


class QwenVerifier:
    def __init__(self, config):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
        self.config = config
        self.processor = AutoProcessor.from_pretrained(config["vlm_model"],
            revision=config["vlm_revision"], cache_dir=".cache/huggingface", max_pixels=config["max_pixels"], local_files_only=True)
        quantized = config.get("load_in_4bit", False) and torch.cuda.is_available()
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True) if quantized else None
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config["vlm_model"],
            revision=config["vlm_revision"], cache_dir=".cache/huggingface",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=config["vlm_device"], attn_implementation="sdpa", local_files_only=True,
            quantization_config=quantization,
            max_memory={0: "6GiB", "cpu": "20GiB"} if torch.cuda.is_available() else None)
        self.model.eval()

    def generate(self, images, prompt, timestamps=None):
        import torch
        from transformers.video_utils import VideoMetadata
        messages = [{"role": "user", "content": [{"type": "video"}] +
                     [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sampled_fps = (len(images) - 1) / (timestamps[-1] - timestamps[0]) if timestamps and len(images) > 1 else 2.0
        metadata = VideoMetadata(total_num_frames=len(images), fps=sampled_fps, frames_indices=list(range(len(images))))
        inputs = self.processor(text=[text], videos=[np.stack([np.asarray(image) for image in images])],
            video_metadata=[metadata], do_sample_frames=False,
            size={"shortest_edge": 128 * 28 * 28, "longest_edge": self.config["max_pixels"]},
            padding=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.config["max_new_tokens"])
        return self.processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]


def select_sample_rows(eligible, candidate, count, strategy="uniform"):
    if not eligible:
        return []
    count = min(count, len(eligible))
    if strategy != "transition_dense" or count < 6:
        positions = np.linspace(0, len(eligible) - 1, count, dtype=int)
        return [eligible[i] for i in sorted(set(positions))]
    selected = set(np.linspace(0, len(eligible) - 1, max(4, count // 2), dtype=int).tolist())
    target_ids = {candidate["person_id"], candidate["vehicle_id"]}
    visibility = [frozenset(obj["id"] for obj in row["objects"] if obj["id"] in target_ids)
                  for row in eligible]
    transitions = {0, len(eligible) - 1}
    transitions.update(i for i in range(1, len(eligible)) if visibility[i] != visibility[i - 1])
    radius = 0
    while len(selected) < count and radius < len(eligible):
        for center in sorted(transitions):
            for position in (center - radius, center + radius):
                if 0 <= position < len(eligible):
                    selected.add(position)
                    if len(selected) == count:
                        break
            if len(selected) == count:
                break
        radius += 1
    return [eligible[i] for i in sorted(selected)]


def sample_window(path, rows, candidate, count, strategy="uniform", related_person_ids=None):
    eligible = [row for row in rows if candidate["start_s"] <= row["timestamp_s"] < candidate["end_s"]
                and row["scene"] == candidate["scene"]]
    if not eligible:
        return [], []
    sampled_rows = select_sample_rows(eligible, candidate, count, strategy)
    selected = {row["frame_index"]: row for row in sampled_rows}
    related_person_ids = set(related_person_ids or [])
    focus_ids = {candidate["person_id"], candidate["vehicle_id"], *related_person_ids}
    pair_boxes = [obj["bbox"] for row in eligible for obj in row["objects"] if obj["id"] in focus_ids]
    # One fixed crop across time avoids artificial camera movement and keeps a
    # disappearing person in view, with a margin for nearby scene context.
    crop = None
    if pair_boxes:
        crop = [min(b[0] for b in pair_boxes), min(b[1] for b in pair_boxes),
                max(b[2] for b in pair_boxes), max(b[3] for b in pair_boxes)]
    images, frame_ids = [], []
    for index, timestamp, frame in frames(path):
        if index > max(selected):
            break
        if index not in selected:
            continue
        image = frame.to_image()
        draw = ImageDraw.Draw(image)
        for obj in selected[index]["objects"]:
            if obj["id"] in focus_ids:
                color = ("lime" if obj["id"] == candidate["person_id"] else
                         "orange" if obj["id"] in related_person_ids else "cyan")
                draw.rectangle(obj["bbox"], outline=color, width=3)
                draw.text((obj["bbox"][0], max(20, obj["bbox"][1] - 15)), obj["id"], fill=color)
        if crop:
            margin = max(crop[2] - crop[0], crop[3] - crop[1]) * (0.3 if candidate["person_id"].startswith("untracked_") else 0.12)
            bounds = (max(0, int(crop[0] - margin)), max(0, int(crop[1] - margin)),
                      min(image.width, int(crop[2] + margin)), min(image.height, int(crop[3] + margin)))
            focused = image.crop(bounds)
            focused.thumbnail((640, 480))
            canvas = Image.new("RGB", (max(focused.width, 280), focused.height + 24), "black")
            canvas.paste(focused, (0, 24))
            ImageDraw.Draw(canvas).text((5, 5), f"frame {index} | {timestamp:.3f}s", fill="white")
            image = canvas
        images.append(image)
        frame_ids.append(index)
    return images, frame_ids


def exit_motion_cue(rows, candidate, indices):
    """Describe a tracked person moving from the vehicle box to outside it."""
    observations = []
    for index in indices:
        row = rows[index]
        objects = {obj["id"]: obj for obj in row["objects"]}
        person = objects.get(candidate["person_id"])
        vehicle = objects.get(candidate["vehicle_id"])
        if not person or not vehicle:
            continue
        px1, py1, px2, py2 = person["bbox"]
        vx1, vy1, vx2, vy2 = vehicle["bbox"]
        cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
        inside_box = vx1 <= cx <= vx2 and vy1 <= cy <= vy2
        dx = max(vx1 - px2, px1 - vx2, 0)
        dy = max(vy1 - py2, py1 - vy2, 0)
        gap = (dx * dx + dy * dy) ** 0.5 / max(vx2 - vx1, vy2 - vy1, 1e-6)
        observations.append((index, inside_box, gap))
    for first in observations:
        if not first[1]:
            continue
        later = [item for item in observations if item[0] > first[0] and not item[1]]
        if len(later) >= 2 and later[-1][2] > first[2] + 0.1:
            return (f"Track motion suggests a possible exit: the person center is within the vehicle "
                    f"box at frame {first[0]}, then outside it by frame {later[0][0]}, with "
                    f"greater separation by frame {later[-1][0]}. Inspect those frames for actual "
                    "cabin-to-outside body movement. If visible, prefer exit over other_interaction "
                    "or door_operation. Box overlap alone does not prove the person was in the cabin.")
    return "No clear tracked inside-to-outside motion cue; decide from the visual frames."


def correct_enter_exit_direction(event, rows, candidate, config):
    """Correct only an existing enter/exit label from a strong temporal track cue.

    This function never creates an interaction and never changes another action
    type into enter/exit. Ambiguous trajectories retain the VLM label.
    """
    if event.get("type") not in {"enter", "exit"} or not config.get("direction_correction_enabled", False):
        return event, None
    start, end = event["start_frame"], event["end_frame"]
    context = int(config.get("direction_context_frames", 4))
    observations = []
    for index in range(max(0, start - context), min(len(rows), end + context + 1)):
        objects = {obj["id"]: obj for obj in rows[index]["objects"]}
        person, vehicle = objects.get(candidate["person_id"]), objects.get(candidate["vehicle_id"])
        if not person or not vehicle:
            continue
        px1, py1, px2, py2 = person["bbox"]
        vx1, vy1, vx2, vy2 = vehicle["bbox"]
        center = ((px1 + px2) / 2, (py1 + py2) / 2)
        inside = vx1 <= center[0] <= vx2 and vy1 <= center[1] <= vy2
        dx, dy = max(vx1 - px2, px1 - vx2, 0), max(vy1 - py2, py1 - vy2, 0)
        gap = (dx * dx + dy * dy) ** .5 / max(vx2 - vx1, vy2 - vy1, 1e-6)
        observations.append((index, inside, gap))
    edge_count = int(config.get("direction_edge_observations", 2))
    if len(observations) < edge_count * 2:
        return event, None
    early, late = observations[:edge_count], observations[-edge_count:]
    early_inside = all(item[1] for item in early)
    late_inside = all(item[1] for item in late)
    early_gap = sum(item[2] for item in early) / edge_count
    late_gap = sum(item[2] for item in late) / edge_count
    minimum_change = float(config.get("direction_min_gap_change", .08))
    inferred = None
    if early_inside and not late_inside and late_gap >= early_gap + minimum_change:
        inferred = "exit"
    elif not early_inside and late_inside and early_gap >= late_gap + minimum_change:
        inferred = "enter"
    if inferred is None or inferred == event["type"]:
        return event, None
    corrected = {**event, "type": inferred}
    return corrected, (f"Direction corrected from {event['type']} to {inferred}: tracked body center "
                       f"changed from {'inside' if early_inside else 'outside'} to "
                       f"{'inside' if late_inside else 'outside'} the vehicle, with normalized gap "
                       f"{early_gap:.2f} -> {late_gap:.2f}.")


def disappearance_return_cue(rows, candidate, config):
    """Describe a person-track gap followed by return near the same vehicle."""
    eligible = [row for row in rows if row["scene"] == candidate["scene"] and
                candidate["start_s"] <= row["timestamp_s"] < candidate["end_s"]]
    states = []
    for row in eligible:
        objects = {obj["id"]: obj for obj in row["objects"]}
        person = objects.get(candidate["person_id"])
        vehicle = objects.get(candidate["vehicle_id"])
        # Predicted boxes bridge short detector misses. For this cue, only an
        # observed person detection counts as visible.
        visible = bool(person and person.get("observed", True))
        near_vehicle = False
        if visible and vehicle:
            px1, py1, px2, py2 = person["bbox"]
            vx1, vy1, vx2, vy2 = vehicle["bbox"]
            dx = max(vx1 - px2, px1 - vx2, 0)
            dy = max(vy1 - py2, py1 - vy2, 0)
            near_vehicle = (dx * dx + dy * dy) ** .5 <= config.get("near_margin", .2) * max(
                vx2 - vx1, vy2 - vy1, 1e-6)
        alternatives = []
        if vehicle:
            vx1, vy1, vx2, vy2 = vehicle["bbox"]
            scale = max(vx2 - vx1, vy2 - vy1, 1e-6)
            for obj in objects.values():
                if obj["kind"] != "person" or obj["id"] == candidate["person_id"] or not obj.get("observed", True):
                    continue
                ox1, oy1, ox2, oy2 = obj["bbox"]
                dx = max(vx1 - ox2, ox1 - vx2, 0)
                dy = max(vy1 - oy2, oy1 - vy2, 0)
                if (dx * dx + dy * dy) ** .5 <= config.get("near_margin", .2) * scale:
                    alternatives.append(obj)
        states.append((row["frame_index"], row["timestamp_s"], visible, near_vehicle,
                       person["bbox"] if visible else None, alternatives))
    minimum = config.get("load_unload_gap_min_s", .3)
    maximum = config.get("load_unload_gap_max_s", 3.0)
    for start in range(1, len(states) - 1):
        if states[start][2] or not states[start - 1][2] or not states[start - 1][3]:
            continue
        end = start
        while end < len(states) and not states[end][2]:
            end += 1
        if end >= len(states):
            continue
        gap_s = states[end][1] - states[start - 1][1]
        related_id = candidate["person_id"] if states[end][3] else None
        return_index = end
        alternative_points = [(position, obj) for position in range(start, end + 1)
                              for obj in states[position][5]]
        if alternative_points:
            # A tracker ID switch is accepted only when the new person's center
            # is close to the last target position, normalized by vehicle size.
            before_box = states[start - 1][4]
            bx = (before_box[0] + before_box[2]) / 2
            by = (before_box[1] + before_box[3]) / 2
            position, closest = min(alternative_points, key=lambda item: (
                ((item[1]["bbox"][0] + item[1]["bbox"][2]) / 2 - bx) ** 2 +
                ((item[1]["bbox"][1] + item[1]["bbox"][3]) / 2 - by) ** 2))
            row = next(row for row in eligible if row["frame_index"] == states[position][0])
            vehicle = next(obj for obj in row["objects"] if obj["id"] == candidate["vehicle_id"])
            vx1, vy1, vx2, vy2 = vehicle["bbox"]
            scale = max(vx2 - vx1, vy2 - vy1, 1e-6)
            cx = (closest["bbox"][0] + closest["bbox"][2]) / 2
            cy = (closest["bbox"][1] + closest["bbox"][3]) / 2
            if ((cx - bx) ** 2 + (cy - by) ** 2) ** .5 <= config.get("load_unload_id_switch_distance", .35) * scale:
                related_id = closest["id"]
                return_index = position
                gap_s = states[return_index][1] - states[start - 1][1]
        if minimum <= gap_s <= maximum and related_id:
            identity_note = ("the same track ID" if related_id == candidate["person_id"] else
                             f"nearby person track {related_id}, a possible spatial ID continuation")
            return ("Temporal track cue: the target person is visibly near the target vehicle at "
                    f"frame {states[start - 1][0]}, is not observed for about {gap_s:.2f}s, and is "
                    f"followed by {identity_note} near the same vehicle at frame {states[return_index][0]}. "
                    "An orange box marks a possible continuation with a different tracker ID. Inspect the "
                    "chronological images around this gap. If they show the person leaning/reaching "
                    "into a door, trunk, or cargo opening and then withdrawing while remaining "
                    "outside, you may classify load_unload even when the handled object is occluded. "
                    "Do not classify load_unload from the track gap alone: ordinary occlusion, an ID "
                    "switch, walking behind the vehicle, or a full-body cabin transition are not "
                    "load_unload. A visible full-body transition is enter or exit."), {related_id}
        start = end
    return (("No qualifying near-vehicle disappearance-and-return pattern was found for this target "
             "track; do not infer load_unload from track continuity."), set())


def verify_candidates(path, rows, candidates, metadata, config, output, verifier_factory=QwenVerifier, resume=True):
    events, reviews = [], []
    verifier = None
    cache = Path(output) / "verification" / Path(path).stem
    prefilter = {}
    prefilter_dir = config.get("vote_prefilter_review_dir")
    if config.get("voting_enabled") and prefilter_dir:
        prefilter_path = Path(prefilter_dir) / f"{Path(path).stem}.json"
        if prefilter_path.exists():
            prefilter = {row["candidate_id"]: row for row in read_json(prefilter_path)}
    reusable_votes = {}
    reuse_dir = config.get("vote_reuse_verification_dir")
    if config.get("voting_enabled") and reuse_dir:
        source_cache = Path(reuse_dir) / Path(path).stem
        for cached_file in source_cache.glob("*.json"):
            cached_row = read_json(cached_file)
            reusable_votes[cached_row["candidate"]["candidate_id"]] = cached_row
    for candidate in candidates:
        print(f"{Path(path).stem}: verify {candidate['candidate_id']}/{len(candidates)}", flush=True)
        prior = prefilter.get(candidate["candidate_id"])
        if prior and prior["decision"] == "no_interaction":
            reviews.append({**candidate, "decision": "no_interaction",
                            "reason": "Voting prefilter retained the Experiment 2 no-interaction decision",
                            "events": [], "voting": None, "postprocess_rejected_events": []})
            continue
        load_cue, related_person_ids = ("", set())
        if config.get("load_unload_disappearance_cue", False):
            load_cue, related_person_ids = disappearance_return_cue(rows, candidate, config)
        images, indices = sample_window(path, rows, candidate, config["sample_frames"],
                                        config.get("sampling_strategy", "uniform"), related_person_ids)
        ballots = build_ballots(indices, config.get("vote_ballots", 5), config.get("vote_ballot_frames", 6))
        voting = config.get("voting_enabled", False)
        untracked_exit = candidate["person_id"].startswith("untracked_")
        if untracked_exit:
            prompt = UNTRACKED_EXIT_PROMPT.format(**candidate, indices=indices)
        else:
            prompt = (VOTING_PROMPT.format(**candidate, ballots=json.dumps(ballots)) if voting else
                      PROMPT.format(**candidate) + "\nAllowed frame indices in chronological order: " + str(indices))
            if config.get("load_unload_access_rule", False):
                if voting:
                    prompt = prompt.replace("loading or unloading an object",
                                            "handling inside a vehicle compartment that opens or is open")
                    prompt += "\n" + LOAD_UNLOAD_ACCESS_RULE
                else:
                    prompt = prompt.replace(
                        "load_unload requires a visible OBJECT\ntransfer; a person getting into a car is enter, not load_unload.",
                        LOAD_UNLOAD_ACCESS_RULE)
            if config.get("exit_motion_cue", False):
                prompt += "\n" + exit_motion_cue(rows, candidate, indices)
            if config.get("load_unload_disappearance_cue", False):
                prompt += "\n" + load_cue
        signature = fingerprint({"candidate": candidate, "source": metadata["source_sha256"],
                                 "config": config, "prompt": prompt, "indices": indices,
                                 "tracks": [rows[i] for i in indices], "sampling_version": 5})
        file = cache / f"{signature}.json"
        voting_record = None
        if resume and file.exists():
            cached = read_json(file)
            decision = cached["decision"]
            voting_record = cached.get("voting")
            # Recover cached model output produced before singleton-array responses
            # were accepted. This avoids repeating expensive VLM inference.
            if voting and cached.get("raw"):
                try:
                    decision, voting_record = parse_and_aggregate_votes(
                        cached["raw"], ballots, config.get("vote_min_fraction", .6),
                        config.get("vote_min_consecutive", 2))
                    cached["decision"] = decision
                    cached["voting"] = voting_record
                    write_json(file, cached)
                except (ValueError, json.JSONDecodeError):
                    pass
        elif voting and candidate["candidate_id"] in reusable_votes:
            cached = reusable_votes[candidate["candidate_id"]]
            decision, voting_record = parse_and_aggregate_votes(
                cached["raw"], ballots, config.get("vote_min_fraction", .6),
                config.get("vote_min_consecutive", 2))
            write_json(file, {"candidate": candidate, "frame_ids": indices, "ballots": ballots,
                              "raw": cached["raw"], "decision": decision, "voting": voting_record})
        elif not untracked_exit and not any({candidate["person_id"], candidate["vehicle_id"]} <= {obj["id"] for obj in rows[i]["objects"]} for i in indices):
            decision = {"decision": "uncertain", "reason": "No sampled frame shows both target track IDs", "events": []}
        elif config["verifier"] == "review":
            decision = {"decision": "uncertain", "reason": "Review-only mode: no action verifier run", "events": []}
        else:
            if verifier is None:
                verifier = verifier_factory(config)
            text = verifier.generate(images, prompt, [rows[i]["timestamp_s"] for i in indices])
            try:
                if voting:
                    decision, voting_record = parse_and_aggregate_votes(
                        text, ballots, config.get("vote_min_fraction", .6),
                        config.get("vote_min_consecutive", 2))
                else:
                    decision = parse_decision(text, indices)
            except (ValueError, json.JSONDecodeError) as error:
                # Invalid model output is exposed, never silently treated as a negative.
                decision = {"decision": "uncertain", "reason": f"Invalid verifier response: {error}", "events": []}
            write_json(file, {"candidate": candidate, "frame_ids": indices, "ballots": ballots if voting else None,
                              "raw": text, "decision": decision, "voting": voting_record})
        rejected = []
        if untracked_exit and any(event["type"] != "exit" for event in decision["events"]):
            decision = {"decision": "uncertain", "reason": "Initial vehicle probe supports only exit events", "events": []}
        direction_notes = []
        corrected_events = []
        for original_event in decision["events"]:
            event, direction_note = correct_enter_exit_direction(original_event, rows, candidate, config)
            corrected_events.append(event)
            if direction_note:
                direction_notes.append(direction_note)
            start, end = event["start_frame"], event["end_frame"]
            boundary_context = max(0, config.get("vote_boundary_context_s", 0))
            if voting and boundary_context:
                start_time = rows[start]["timestamp_s"] - boundary_context
                end_time = rows[end]["timestamp_s"] + boundary_context
                start = next((i for i, row in enumerate(rows) if row["timestamp_s"] >= start_time), 0)
                end = max(i for i, row in enumerate(rows) if row["timestamp_s"] <= end_time)
            if config.get("require_multi_frame_event", False) and start == end and start not in (0, len(rows) - 1):
                rejected.append({**event, "postprocess_reason": "single-frame event without clip-boundary truncation"})
                continue
            events.append({"event_id": "pending", "type": event["type"],
                "persons": [{"person_id": candidate["person_id"], "description": event["person_description"]}],
                "vehicle": {"vehicle_id": candidate["vehicle_id"], "description": event["vehicle_description"]},
                "spans": [{"start_s": rows[start]["timestamp_s"],
                           "end_s": rows[end + 1]["timestamp_s"] if end + 1 < len(rows) else metadata["duration_s"]}],
                "truncated_start": start == 0, "truncated_end": end == len(rows) - 1,
                "evidence_frames": event["evidence_frames"], "group_id": None})
        review_decision = dict(decision)
        review_decision["events"] = corrected_events
        if direction_notes:
            review_decision["reason"] = decision["reason"] + " " + " ".join(direction_notes)
        reviews.append({**candidate, **review_decision, "voting": voting_record,
                        "postprocess_rejected_events": rejected})
    return events, reviews
