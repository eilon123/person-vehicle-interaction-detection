"""Export dense frame sheets for manual reference annotation (not model output).

Example: python tools/review_frames.py --input clip.mp4 --start 3 --end 7 --fps 5
Optional normalized crop: --crop 0.2 0.1 0.8 0.9
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from person_vehicle.video import frames


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--start", type=float, default=0)
parser.add_argument("--end", type=float, default=float("inf"))
parser.add_argument("--fps", type=float, default=4)
parser.add_argument("--crop", type=float, nargs=4)
parser.add_argument("--output", default="outputs/manual_review")
args = parser.parse_args()
if args.fps <= 0:
    raise ValueError("fps must be positive")
destination = Path(args.output)
destination.mkdir(parents=True, exist_ok=True)
sheet = None
count = page = 0
next_time = args.start
for index, timestamp, frame in frames(args.input):
    if timestamp > args.end:
        break
    if timestamp + 1e-6 < next_time:
        continue
    next_time = timestamp + 1 / args.fps
    if count % 16 == 0:
        if sheet is not None:
            sheet.save(destination / f"{Path(args.input).stem}_{args.start:g}_{page:02d}.jpg")
        page += 1
        sheet = Image.new("RGB", (1600, 1000), "#161a20")
        draw = ImageDraw.Draw(sheet)
    image = frame.to_image()
    if args.crop:
        x1, y1, x2, y2 = args.crop
        image = image.crop((int(x1 * image.width), int(y1 * image.height), int(x2 * image.width), int(y2 * image.height)))
    image.thumbnail((400, 225))
    x, y = (count % 4) * 400, ((count % 16) // 4) * 250
    sheet.paste(image, (x, y))
    draw.text((x + 4, y + 229), f"{timestamp:.3f}s | frame {index}", fill="white")
    count += 1
if sheet:
    sheet.save(destination / f"{Path(args.input).stem}_{args.start:g}_{page:02d}.jpg")
print(f"Exported {count} frames across {page} sheets")
