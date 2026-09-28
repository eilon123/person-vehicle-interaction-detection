from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Span(StrictModel):
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_s <= self.start_s:
            raise ValueError("Span must have positive duration")
        return self


class Person(StrictModel):
    person_id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class Vehicle(StrictModel):
    vehicle_id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class Event(StrictModel):
    event_id: str
    type: Literal["enter", "exit", "door_operation", "load_unload", "other_interaction"]
    persons: list[Person] = Field(min_length=1, max_length=1)
    vehicle: Vehicle
    spans: list[Span] = Field(min_length=1)
    truncated_start: bool = False
    truncated_end: bool = False
    evidence_frames: list[int] = Field(min_length=1)
    group_id: str | None = None

    @model_validator(mode="after")
    def ordered(self):
        for left, right in zip(self.spans, self.spans[1:]):
            if right.start_s < left.end_s:
                raise ValueError("Spans must be sorted and disjoint")
        if any(i < 0 for i in self.evidence_frames):
            raise ValueError("Frame indices cannot be negative")
        return self


class ClipOutput(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    clip_id: str
    source_sha256: str
    status: Literal["ok", "error"]
    duration_s: float = Field(ge=0)
    frame_count: int = Field(ge=0)
    interactions: list[Event]
    error: str | None = None

    @model_validator(mode="after")
    def consistent(self):
        if self.status == "error" and (not self.error or self.interactions):
            raise ValueError("Error outputs need a reason and no events")
        if self.status == "ok" and (self.error or self.frame_count == 0):
            raise ValueError("Successful outputs need decoded frames and no error")
        if len({e.event_id for e in self.interactions}) != len(self.interactions):
            raise ValueError("Duplicate event IDs")
        for event in self.interactions:
            if event.spans[-1].end_s > self.duration_s + 1e-6:
                raise ValueError("Event exceeds clip duration")
            if max(event.evidence_frames) >= self.frame_count:
                raise ValueError("Evidence frame outside clip")
        return self


def active_events(events, timestamp):
    return [e for e in events if any(s["start_s"] <= timestamp < s["end_s"] for s in e["spans"])]


def merge_events(events):
    """Merge overlapping proposals for the same pair and action, never across gaps."""
    result = []
    for event in sorted(events, key=lambda e: e["spans"][0]["start_s"]):
        match = next((old for old in result if old["type"] == event["type"]
                      and old["persons"][0]["person_id"] == event["persons"][0]["person_id"]
                      and old["vehicle"]["vehicle_id"] == event["vehicle"]["vehicle_id"]
                      and any(a["start_s"] < b["end_s"] and b["start_s"] < a["end_s"]
                              for a in old["spans"] for b in event["spans"])), None)
        if match is None:
            result.append(event)
            continue
        spans = sorted(match["spans"] + event["spans"], key=lambda s: s["start_s"])
        merged = []
        for span in spans:
            if merged and span["start_s"] <= merged[-1]["end_s"]:
                merged[-1]["end_s"] = max(merged[-1]["end_s"], span["end_s"])
            else:
                merged.append(dict(span))
        match["spans"] = merged
        match["evidence_frames"] = sorted(set(match["evidence_frames"] + event["evidence_frames"]))
        match["truncated_start"] |= event["truncated_start"]
        match["truncated_end"] |= event["truncated_end"]
    for index, event in enumerate(result, 1):
        event["event_id"] = f"e{index:03d}"
    return result
