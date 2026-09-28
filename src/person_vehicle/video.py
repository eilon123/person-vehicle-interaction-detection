from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from .io import sha256, write_json


def frames(path):
    """Yield presentation-order frames with timestamps relative to first frame."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = None
        previous = -1.0
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError(f"Missing frame timestamp in {path}")
            absolute = float(frame.pts * frame.time_base)
            if origin is None:
                origin = absolute
            timestamp = absolute - origin
            if timestamp <= previous:
                raise ValueError("Non-increasing presentation timestamps")
            previous = timestamp
            yield index, timestamp, frame


def probe(path):
    times = []
    last_duration = 0
    width = height = 0
    for _, timestamp, frame in frames(path):
        times.append(timestamp)
        width, height = frame.width, frame.height
        last_duration = float(frame.duration * frame.time_base) if frame.duration else 0
    if not times:
        raise ValueError(f"No decodable frames in {path}")
    if not last_duration:
        last_duration = float(np.median(np.diff(times))) if len(times) > 1 else 1 / 25
    return {"clip_id": Path(path).stem, "source_sha256": sha256(path),
            "frame_count": len(times), "width": width, "height": height,
            "duration_s": times[-1] + last_duration,
            "timestamps": times, "last_frame_duration_s": last_duration}


def audit(path, output):
    metadata = probe(path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / f"{Path(path).stem}.json", metadata)
    selected = set(np.linspace(0, metadata["frame_count"] - 1, 20, dtype=int).tolist())
    sheet = Image.new("RGB", (5 * 320, 4 * 208), "#171a21")
    draw = ImageDraw.Draw(sheet)
    count = 0
    for index, timestamp, frame in frames(path):
        if index in selected:
            image = frame.to_image()
            image.thumbnail((320, 180))
            x, y = (count % 5) * 320, (count // 5) * 208
            sheet.paste(image, (x, y))
            draw.text((x + 5, y + 183), f"{timestamp:.2f}s | frame {index}", fill="white")
            count += 1
    sheet.save(output / f"{Path(path).stem}.jpg")
    return {k: v for k, v in metadata.items() if k != "timestamps"}
