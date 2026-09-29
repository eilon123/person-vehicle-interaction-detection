"""Small local GUI for creating temporal person--vehicle reference labels."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from .io import write_json
from .video import probe


EVENT_TYPES = ("enter", "exit", "door_operation", "load_unload", "other_interaction")


def empty_clip(duration_s: float) -> dict:
    return {"duration_s": duration_s, "interactions": []}


def make_event(number: int, start_s: float, end_s: float, start_frame: int, end_frame: int,
               event_type: str, person_id: str, person_description: str,
               vehicle_id: str, vehicle_description: str) -> dict:
    """Build one event in the evaluator's reference format."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unsupported event type: {event_type}")
    if end_s <= start_s:
        raise ValueError("The end must be later than the start")
    if not all((person_id.strip(), person_description.strip(), vehicle_id.strip(), vehicle_description.strip())):
        raise ValueError("Person and vehicle IDs and descriptions are required")
    return {
        "event_id": f"ref{number:03d}",
        "type": event_type,
        "persons": [{"person_id": person_id.strip(), "description": person_description.strip()}],
        "vehicle": {"vehicle_id": vehicle_id.strip(), "description": vehicle_description.strip()},
        "spans": [{"start_s": round(start_s, 6), "end_s": round(end_s, 6)}],
        "truncated_start": False,
        "truncated_end": False,
        "evidence_frames": sorted({int(start_frame), int(end_frame)}),
        "group_id": None,
    }


class AnnotationApp:
    def __init__(self, input_path: str | Path, output_dir: str | Path):
        import tkinter as tk
        from tkinter import messagebox, ttk
        from PIL import Image, ImageTk

        self.tk, self.messagebox, self.ttk = tk, messagebox, ttk
        self.Image, self.ImageTk = Image, ImageTk
        source = Path(input_path)
        self.paths = [source] if source.is_file() else sorted(source.glob("*.mp4"))
        if not self.paths:
            raise ValueError(f"No MP4 files at {source}")
        self.output_dir = Path(output_dir)
        self.reference_path = self.output_dir / "events.json"
        self.uncertain_path = self.output_dir / "uncertain.json"
        self.reference = self._load(self.reference_path)
        self.uncertain = self._load(self.uncertain_path)
        self.clip_index = 0
        self.cap = None
        self.metadata = None
        self.current_frame = 0
        self.start = None
        self.playing = False

        self.root = tk.Tk()
        self.root.title("Person-vehicle temporal annotator")
        self.root.minsize(920, 720)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._build()
        self.open_clip(0)

    @staticmethod
    def _load(path: Path) -> dict:
        if not path.exists():
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{path} must contain a JSON object")
        return value

    def _build(self):
        tk, ttk = self.tk, self.ttk
        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill="x")
        ttk.Button(controls, text="◀ Previous video", command=lambda: self.open_clip(self.clip_index - 1)).pack(side="left")
        self.video_label = ttk.Label(controls, text="")
        self.video_label.pack(side="left", padx=12)
        ttk.Button(controls, text="Next video ▶", command=lambda: self.open_clip(self.clip_index + 1)).pack(side="left")
        ttk.Button(controls, text="Save now", command=self.save).pack(side="right")

        self.image_label = ttk.Label(self.root, anchor="center")
        self.image_label.pack(fill="both", expand=True, padx=8)
        self.position_label = ttk.Label(self.root, anchor="center")
        self.position_label.pack(fill="x")
        self.slider = tk.Scale(self.root, from_=0, to=1, orient="horizontal", showvalue=False,
                               command=self.seek, length=850)
        self.slider.pack(fill="x", padx=12)

        playback = ttk.Frame(self.root, padding=(8, 2))
        playback.pack(fill="x")
        ttk.Button(playback, text="⏮ -1 s", command=lambda: self.jump(-1)).pack(side="left")
        self.play_button = ttk.Button(playback, text="Play", command=self.toggle_play)
        self.play_button.pack(side="left", padx=6)
        ttk.Button(playback, text="+1 s ⏭", command=lambda: self.jump(1)).pack(side="left")
        ttk.Button(playback, text="Set start", command=self.set_start).pack(side="left", padx=25)
        self.range_label = ttk.Label(playback, text="No start selected")
        self.range_label.pack(side="left")

        fields = ttk.LabelFrame(self.root, text="Event details", padding=8)
        fields.pack(fill="x", padx=8, pady=6)
        self.type_var = tk.StringVar(value="other_interaction")
        self.person_id = tk.StringVar(value="ref_person_1")
        self.person_description = tk.StringVar(value="Visible person involved in interaction")
        self.vehicle_id = tk.StringVar(value="ref_vehicle_1")
        self.vehicle_description = tk.StringVar(value="Visible vehicle involved in interaction")
        for column, (label, variable, values) in enumerate((
            ("Action", self.type_var, EVENT_TYPES), ("Person ID", self.person_id, None),
            ("Person description", self.person_description, None), ("Vehicle ID", self.vehicle_id, None),
            ("Vehicle description", self.vehicle_description, None),
        )):
            ttk.Label(fields, text=label).grid(row=0, column=column, sticky="w")
            if values:
                widget = ttk.Combobox(fields, textvariable=variable, values=values, state="readonly", width=18)
            else:
                widget = ttk.Entry(fields, textvariable=variable, width=26)
            widget.grid(row=1, column=column, padx=(0, 7), sticky="ew")
            fields.columnconfigure(column, weight=1)

        actions = ttk.Frame(self.root, padding=8)
        actions.pack(fill="x")
        ttk.Button(actions, text="Add confirmed interval", command=lambda: self.add_interval(False)).pack(side="left")
        ttk.Button(actions, text="Add uncertain interval", command=lambda: self.add_interval(True)).pack(side="left", padx=8)
        ttk.Button(actions, text="Delete selected", command=self.delete_selected).pack(side="right")

        self.items = tk.Listbox(self.root, height=8)
        self.items.pack(fill="both", padx=8, pady=(0, 8))

    def open_clip(self, index: int):
        if not 0 <= index < len(self.paths):
            return
        self.save()
        self.playing = False
        if self.cap:
            self.cap.release()
        self.clip_index = index
        path = self.paths[index]
        self.metadata = probe(path)
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open {path}")
        self.current_frame = 0
        self.start = None
        self.slider.configure(to=max(0, self.metadata["frame_count"] - 1))
        self.slider.set(0)
        self.video_label.configure(text=f"{index + 1}/{len(self.paths)}: {path.name}")
        self.show_frame(0)
        self.refresh_items()

    @property
    def clip_id(self) -> str:
        return self.paths[self.clip_index].stem

    @property
    def fps(self) -> float:
        return float(self.metadata.get("nominal_fps") or self.cap.get(cv2.CAP_PROP_FPS) or 30.0)

    def frame_time(self, frame: int) -> float:
        return min(frame / self.fps, self.metadata["duration_s"])

    def show_frame(self, frame: int):
        frame = max(0, min(int(frame), self.metadata["frame_count"] - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = self.cap.read()
        if not ok:
            return
        self.current_frame = frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        scale = min(880 / width, 480 / height, 1)
        if scale < 1:
            image = cv2.resize(image, (round(width * scale), round(height * scale)))
        photo = self.ImageTk.PhotoImage(self.Image.fromarray(image))
        self.image_label.configure(image=photo)
        self.image_label.image = photo
        self.slider.set(frame)
        self.position_label.configure(text=f"Frame {frame} / {self.metadata['frame_count'] - 1}    |    {self.frame_time(frame):.3f} s")

    def seek(self, value):
        if not self.playing:
            self.show_frame(float(value))

    def jump(self, seconds: float):
        self.show_frame(self.current_frame + round(seconds * self.fps))

    def toggle_play(self):
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self.play_step()

    def play_step(self):
        if not self.playing:
            return
        if self.current_frame >= self.metadata["frame_count"] - 1:
            self.toggle_play()
            return
        self.show_frame(self.current_frame + 1)
        self.root.after(max(1, round(1000 / self.fps)), self.play_step)

    def set_start(self):
        self.start = (self.current_frame, self.frame_time(self.current_frame))
        self.range_label.configure(text=f"Start: {self.start[1]:.3f} s; move to the end, then add interval")

    def add_interval(self, uncertain: bool):
        if self.start is None:
            self.messagebox.showerror("Set a start", "Move to the first interaction frame and click 'Set start'.")
            return
        start_frame, start_s = self.start
        end_frame, end_s = self.current_frame, self.frame_time(self.current_frame)
        if end_s <= start_s:
            self.messagebox.showerror("Invalid interval", "Move to a frame after the start.")
            return
        if uncertain:
            clip = self.uncertain.setdefault(self.clip_id, {"duration_s": self.metadata["duration_s"], "uncertain_spans": []})
            clip["uncertain_spans"].append({"start_s": round(start_s, 6), "end_s": round(end_s, 6), "note": "Needs review"})
        else:
            clip = self.reference.setdefault(self.clip_id, empty_clip(self.metadata["duration_s"]))
            try:
                event = make_event(len(clip["interactions"]) + 1, start_s, end_s, start_frame, end_frame,
                                   self.type_var.get(), self.person_id.get(), self.person_description.get(),
                                   self.vehicle_id.get(), self.vehicle_description.get())
            except ValueError as error:
                self.messagebox.showerror("Cannot add interval", str(error))
                return
            clip["interactions"].append(event)
        self.start = None
        self.range_label.configure(text="No start selected")
        self.save()
        self.refresh_items()

    def current_entries(self):
        confirmed = [("confirmed", e) for e in self.reference.get(self.clip_id, {}).get("interactions", [])]
        uncertain = [("uncertain", s) for s in self.uncertain.get(self.clip_id, {}).get("uncertain_spans", [])]
        return confirmed + uncertain

    def refresh_items(self):
        self.items.delete(0, self.tk.END)
        for kind, value in self.current_entries():
            span = value["spans"][0] if kind == "confirmed" else value
            label = value.get("type", "review")
            self.items.insert(self.tk.END, f"{kind.upper():9} {span['start_s']:.3f}–{span['end_s']:.3f} s  {label}")

    def delete_selected(self):
        selection = self.items.curselection()
        if not selection:
            return
        confirmed_count = len(self.reference.get(self.clip_id, {}).get("interactions", []))
        index = selection[0]
        if index < confirmed_count:
            self.reference[self.clip_id]["interactions"].pop(index)
        else:
            self.uncertain[self.clip_id]["uncertain_spans"].pop(index - confirmed_count)
        self.save()
        self.refresh_items()

    def save(self):
        write_json(self.reference_path, self.reference)
        write_json(self.uncertain_path, self.uncertain)

    def close(self):
        self.save()
        if self.cap:
            self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def launch(input_path: str | Path, output_dir: str | Path):
    """Open the local annotation window."""
    AnnotationApp(input_path, output_dir).run()
