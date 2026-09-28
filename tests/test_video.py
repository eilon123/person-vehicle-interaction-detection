from fractions import Fraction

import av
import numpy as np

from person_vehicle.render import render
from person_vehicle.video import probe


def test_variable_frame_timing_and_empty_overlay(tmp_path):
    source = tmp_path / "sample.mp4"
    stamps = [0, 40, 100, 140, 220]
    with av.open(str(source), "w") as output:
        stream = output.add_stream("libx264", rate=25)
        stream.width, stream.height = 320, 240
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, 1000)
        stream.options = {"bf": "0"}
        for stamp in stamps:
            frame = av.VideoFrame.from_ndarray(np.full((240, 320, 3), 100, np.uint8), format="bgr24")
            frame.pts, frame.time_base = stamp, Fraction(1, 1000)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    info = probe(source)
    rows = [{"frame_index": i, "timestamp_s": t, "objects": [], "scene": 0} for i, t in enumerate(info["timestamps"])]
    clip = {"schema_version": "1.0", "clip_id": "sample", "source_sha256": info["source_sha256"],
            "status": "ok", "duration_s": info["duration_s"], "frame_count": len(rows), "interactions": []}
    qa = render(source, rows, clip, tmp_path / "annotated.mp4")
    assert qa["frame_count"] == 5
    assert qa["max_timestamp_error_s"] < 0.001
    assert qa["overlay_agreement"] == 1
