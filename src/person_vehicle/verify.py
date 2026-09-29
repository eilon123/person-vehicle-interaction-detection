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
    window_frames = sorted({frame for ballot in ballots for frame in ballot["frame_indices"]})
    event = {"type": action, "start_frame": window_frames[0], "end_frame": window_frames[-1],
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


def sample_window(path, rows, candidate, count, strategy="uniform"):
    eligible = [row for row in rows if candidate["start_s"] <= row["timestamp_s"] < candidate["end_s"]
                and row["scene"] == candidate["scene"]]
    if not eligible:
        return [], []
    sampled_rows = select_sample_rows(eligible, candidate, count, strategy)
    selected = {row["frame_index"]: row for row in sampled_rows}
    pair_boxes = [obj["bbox"] for row in eligible for obj in row["objects"]
                  if obj["id"] in (candidate["person_id"], candidate["vehicle_id"])]
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
            if obj["id"] in (candidate["person_id"], candidate["vehicle_id"]):
                color = "lime" if obj["id"] == candidate["person_id"] else "cyan"
                draw.rectangle(obj["bbox"], outline=color, width=3)
                draw.text((obj["bbox"][0], max(20, obj["bbox"][1] - 15)), obj["id"], fill=color)
        if crop:
            margin = max(crop[2] - crop[0], crop[3] - crop[1]) * 0.12
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
    for candidate in candidates:
        print(f"{Path(path).stem}: verify {candidate['candidate_id']}/{len(candidates)}", flush=True)
        prior = prefilter.get(candidate["candidate_id"])
        if prior and prior["decision"] == "no_interaction":
            reviews.append({**candidate, "decision": "no_interaction",
                            "reason": "Voting prefilter retained the Experiment 2 no-interaction decision",
                            "events": [], "voting": None, "postprocess_rejected_events": []})
            continue
        images, indices = sample_window(path, rows, candidate, config["sample_frames"],
                                        config.get("sampling_strategy", "uniform"))
        ballots = build_ballots(indices, config.get("vote_ballots", 5), config.get("vote_ballot_frames", 6))
        voting = config.get("voting_enabled", False)
        prompt = (VOTING_PROMPT.format(**candidate, ballots=json.dumps(ballots)) if voting else
                  PROMPT.format(**candidate) + "\nAllowed frame indices in chronological order: " + str(indices))
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
        elif not any({candidate["person_id"], candidate["vehicle_id"]} <= {obj["id"] for obj in rows[i]["objects"]} for i in indices):
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
        for event in decision["events"]:
            start, end = event["start_frame"], event["end_frame"]
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
        reviews.append({**candidate, **decision, "voting": voting_record,
                        "postprocess_rejected_events": rejected})
    return events, reviews
