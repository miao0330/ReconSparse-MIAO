#!/usr/bin/env python3
"""Create one six-camera + annotation BEV overview per ReconDreamer-RL scene.

Images come from the original nuScenes keyframes. No Gaussian models are loaded.
Dependencies: numpy, Pillow. Example:
    python tools/visualize_recondreamer_dataset.py --scene-ids 000 019 027 077
    python tools/visualize_recondreamer_dataset.py --workers 4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_DATASET = Path("/htfs/filesystem_pfs/waytous/datasets/miao/ReconDreamer-RL")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs/recondreamer_dataset_overview"
CAMERAS = (
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
)
COLORS = {
    "car": "#60a5fa", "large vehicle": "#fbbf24", "pedestrian": "#fb7185",
    "two-wheeler": "#c084fc", "other": "#cbd5e1",
}


@dataclass(frozen=True)
class Scene:
    scene_id: str
    official_name: str
    description: str
    location: str
    frame: int
    sample_token: str
    timestamp_us: int
    relative_time_s: float
    has_checkpoint: bool
    keyframes: tuple[int, ...]


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict]:
    """Stream a JSON array without loading the 1.3 GB sample_data table into RAM."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def refill() -> None:
            nonlocal buffer, position, eof
            more = handle.read(chunk_size)
            buffer = buffer[position:] + more
            position = 0
            eof = not more

        refill()
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                break
            if eof:
                raise ValueError(f"Empty JSON array file: {path}")
            refill()
        if buffer[position] != "[":
            raise ValueError(f"Expected a JSON array in {path}")
        position += 1
        first = True
        while True:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                if eof:
                    raise ValueError(f"Unterminated JSON array: {path}")
                refill()
            if buffer[position] == "]":
                position += 1
                if buffer[position:].strip() or handle.read().strip():
                    raise ValueError(f"Unexpected content after JSON array: {path}")
                return
            if not first:
                if buffer[position] != ",":
                    raise ValueError(f"Expected a comma between JSON records: {path}")
                position += 1
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                try:
                    item, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(f"Invalid or incomplete JSON record in {path}") from None
                    refill()
                    continue
                if not isinstance(item, dict):
                    raise ValueError(f"Expected JSON objects inside {path}")
                position = end
                first = False
                yield item
                break


def select_scenes(assets: Path, fraction: float, scene_ids: list[str] | None,
                  require_checkpoint: bool, limit: int | None) -> list[Scene]:
    meta = assets / "nuscenes/v1.0-trainval"
    samples = {row["token"]: row for row in read_json(meta / "sample.json")}
    scenes = {row["token"]: row for row in read_json(meta / "scene.json")}
    logs = {row["token"]: row for row in read_json(meta / "log.json")}
    requested = {f"{int(value):03d}" for value in scene_ids} if scene_ids else None
    selected = []
    for directory in sorted((assets / "nus/data").iterdir()):
        if not directory.is_dir() or not directory.name.isdecimal():
            continue
        if requested is not None and directory.name not in requested:
            continue
        has_checkpoint = (directory / "3DGS_without_prior/checkpoint_final.pth").is_file()
        if require_checkpoint and not has_checkpoint:
            continue
        mapping = read_json(directory / "token_frame_map.json")
        entries = sorted((int(frame), token) for token, frame in mapping.items())
        index = int((len(entries) - 1) * fraction + 0.5)
        # Some exports have shorter pose sequences; choose the closest existing keyframe.
        candidates = sorted(range(len(entries)), key=lambda i: (abs(i - index), i))
        chosen = next((i for i in candidates if
                       (directory / "ego_pose" / f"{entries[i][0]:03d}.txt").is_file()), None)
        if chosen is None:
            raise ValueError(f"No keyframe pose found for scene {directory.name}")
        frame, token = entries[chosen]
        sample = samples[token]
        scene = scenes[sample["scene_token"]]
        first_timestamp = samples[scene["first_sample_token"]]["timestamp"]
        selected.append(Scene(
            directory.name, scene["name"], scene.get("description", ""),
            logs[scene["log_token"]]["location"], frame, token, int(sample["timestamp"]),
            (sample["timestamp"] - first_timestamp) / 1e6, has_checkpoint,
            tuple(entry[0] for entry in entries),
        ))
    if requested is not None:
        missing = requested - {s.scene_id for s in selected}
        if missing:
            raise ValueError(f"Requested scene IDs unavailable under these filters: {sorted(missing)}")
    if not selected:
        raise ValueError("No scene directories matched the requested selection")
    return selected[:limit] if limit is not None else selected


def camera_index(assets: Path, scenes: list[Scene], output: Path) -> dict:
    meta = assets / "nuscenes/v1.0-trainval"
    source = meta / "sample_data.json"
    requested = {s.sample_token for s in scenes}
    signature = hashlib.sha256(json.dumps({
        "path": str(source.resolve()), "size": source.stat().st_size,
        "mtime_ns": source.stat().st_mtime_ns, "tokens": sorted(requested),
    }, sort_keys=True).encode()).hexdigest()[:16]
    cache = output / f"camera_index_{signature}.json"
    if cache.is_file():
        print(f"Using camera metadata cache: {cache.name}", flush=True)
        return read_json(cache)
    sensors = {row["token"]: row for row in read_json(meta / "sensor.json")}
    channels = {
        row["token"]: sensors[row["sensor_token"]]["channel"]
        for row in read_json(meta / "calibrated_sensor.json")
    }
    index: dict[str, dict] = {token: {} for token in requested}
    expected = len(requested) * len(CAMERAS)
    found = 0
    started = time.monotonic()
    print(f"Indexing original camera images for {len(requested)} scenes...", flush=True)
    for number, row in enumerate(iter_json_array(source), 1):
        token = row.get("sample_token")
        if row.get("is_key_frame") and token in requested:
            channel = channels.get(row.get("calibrated_sensor_token"))
            if channel in CAMERAS and channel not in index[token]:
                index[token][channel] = {key: row[key] for key in
                                         ("filename", "timestamp", "width", "height")}
                found += 1
        if number % 200000 == 0:
            print(f"  scanned {number:,} records; cameras {found}/{expected}; "
                  f"{time.monotonic() - started:.0f}s", flush=True)
        if found == expected:
            break
    cache.write_text(json.dumps(index), encoding="utf-8")
    print(f"Camera index ready: {found}/{expected}", flush=True)
    return index


@lru_cache(maxsize=16)
def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    suffix = "-Bold" if bold else ""
    candidates = [f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf",
                  f"/usr/share/fonts/dejavu/DejaVuSans{suffix}.ttf"]
    for name in candidates:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def wrapped_text(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int],
                 width: int, size: int = 20, fill: str = "#cbd5e1") -> int:
    x, y = xy
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and draw.textbbox((0, 0), candidate, font=font(size))[2] > width:
            draw.text((x, y), line, font=font(size), fill=fill)
            y += size + 9
            line = word
        else:
            line = candidate
    if line:
        draw.text((x, y), line, font=font(size), fill=fill)
        y += size + 9
    return y


def object_group(category: str) -> str:
    if category.startswith("human."):
        return "pedestrian"
    if "bicycle" in category or "motorcycle" in category:
        return "two-wheeler"
    if any(kind in category for kind in ("truck", "bus", "trailer", "construction")):
        return "large vehicle"
    return "car" if category == "vehicle.car" else "other"


def load_pose(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"Expected a finite 4x4 pose: {path}")
    return matrix


def draw_bev(directory: Path, scene: Scene, extent: float) -> tuple[Image.Image, dict]:
    size = 500
    image = Image.new("RGB", (size, size), "#111d2d")
    draw = ImageDraw.Draw(image)
    ego_pose = load_pose(directory / "ego_pose" / f"{scene.frame:03d}.txt")
    world_to_ego = np.linalg.inv(ego_pose)
    scale = (size - 1) / (2 * extent)

    def pixels(points: np.ndarray) -> list[tuple[float, float]]:
        # Ego x is forward (up on the page); ego y is left (left on the page).
        return [(size / 2 - float(p[1]) * scale, size / 2 - float(p[0]) * scale)
                for p in points]

    for meters in range(math.ceil(-extent / 10) * 10, math.floor(extent / 10) * 10 + 1, 10):
        coordinate = size / 2 - meters * scale
        draw.line((coordinate, 0, coordinate, size), fill="#25354a")
        draw.line((0, coordinate, size, coordinate), fill="#25354a")
    draw.text((9, 8), f"Annotation BEV  |  +/-{extent:g}m  |  grid 10m",
              font=font(15), fill="#e2e8f0")

    path = []
    for frame in scene.keyframes:
        pose_file = directory / "ego_pose" / f"{frame:03d}.txt"
        if pose_file.is_file():
            pose = load_pose(pose_file)
            path.append((frame, world_to_ego @ pose[:, 3]))
    for color, points in [
        ("#64748b", [p for frame, p in path if frame <= scene.frame]),
        ("#22d3ee", [p for frame, p in path if frame >= scene.frame]),
    ]:
        if len(points) > 1:
            draw.line(pixels(np.asarray(points)), fill=color, width=3)

    frame_objects = read_json(directory / "instances/frame_instances.json")
    instances = read_json(directory / "instances/instances_info.json")
    present = frame_objects.get(str(scene.frame), [])
    categories: Counter = Counter()
    shown = 0
    for object_id in present:
        info = instances[str(object_id)]
        annotations = info["frame_annotations"]
        index = annotations["frame_idx"].index(scene.frame)
        transform = world_to_ego @ np.asarray(annotations["obj_to_world"][index])
        length, width = np.asarray(annotations["box_size"][index], dtype=float)[:2]
        category = object_group(info["class_name"])
        categories[category] += 1
        if max(abs(transform[0, 3]), abs(transform[1, 3])) > extent + max(length, width):
            continue
        corners = np.array([[length / 2, width / 2, 0, 1], [length / 2, -width / 2, 0, 1],
                            [-length / 2, -width / 2, 0, 1], [-length / 2, width / 2, 0, 1]])
        polygon = pixels((transform @ corners.T).T)
        draw.polygon(polygon, outline=COLORS[category])
        center = pixels(transform[:, 3][None])[0]
        nose = pixels((transform @ np.array([length / 2, 0, 0, 1]))[None])[0]
        draw.line((*center, *nose), fill=COLORS[category], width=2)
        if size * 0.04 < center[0] < size * 0.94 and size * 0.06 < center[1] < size * 0.94:
            draw.text((center[0] + 3, center[1] + 2), str(object_id), font=font(12), fill=COLORS[category])
        shown += 1
    cx, cy = size / 2, size / 2
    draw.polygon([(cx, cy - 12), (cx - 8, cy + 9), (cx + 8, cy + 9)], fill="#4ade80")
    draw.text((cx + 11, cy - 8), "EGO", font=font(13, True), fill="#4ade80")
    draw.text((size / 2 - 28, 31), "Forward", font=font(13), fill="#cbd5e1")
    return image, {"annotated_objects": len(present), "bev_objects": shown,
                   "object_categories": dict(categories), "ego_world_pose": ego_pose.tolist()}


def render_scene(assets: Path, output: Path, scene: Scene, cameras: dict, extent: float) -> dict:
    directory = assets / "nus/data" / scene.scene_id
    canvas = Image.new("RGB", (1536, 1300), "#0b1220")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 15), f"{scene.scene_id}  |  {scene.official_name}", font=font(31, True), fill="white")
    badge = "GS CHECKPOINT AVAILABLE" if scene.has_checkpoint else "GS CHECKPOINT MISSING"
    draw.text((1035, 24), badge, font=font(18, True),
              fill="#4ade80" if scene.has_checkpoint else "#fbbf24")
    draw.text((20, 61), f"Original nuScenes cameras  |  frame {scene.frame:03d}  |  "
              f"t = {scene.relative_time_s:.2f}s  |  {scene.location}", font=font(19), fill="#cbd5e1")
    missing = []
    camera_sources = {}
    for i, channel in enumerate(CAMERAS):
        x, y = (i % 3) * 512, 104 + (i // 3) * 316
        draw.rectangle((x, y, x + 511, y + 27), fill="#1e293b")
        draw.text((x + 11, y + 3), channel, font=font(17, True), fill="white")
        info = cameras.get(channel)
        try:
            if info is None:
                raise FileNotFoundError("No keyframe camera metadata")
            source = assets / "nuscenes" / info["filename"]
            with Image.open(source) as image:
                panel = ImageOps.contain(image.convert("RGB"), (512, 288),
                                         method=Image.Resampling.LANCZOS)
            canvas.paste(panel, (x + (512 - panel.width) // 2, y + 28 + (288 - panel.height) // 2))
            camera_sources[channel] = dict(info)
        except (OSError, ValueError) as error:
            missing.append(f"{channel}: {error}")
            draw.text((x + 20, y + 140), "CAMERA IMAGE UNAVAILABLE", font=font(19), fill="#fb7185")

    bev, stats = draw_bev(directory, scene, extent)
    canvas.paste(bev, (16, 752))
    x, y, width = 550, 754, 958
    draw.text((x, y), "Scene description", font=font(24, True), fill="white")
    y = wrapped_text(draw, scene.description, (x, y + 42), width, 22) + 20
    draw.text((x, y), "Annotation legend", font=font(22, True), fill="white")
    y += 38
    for category, color in COLORS.items():
        draw.rectangle((x, y + 4, x + 16, y + 20), fill=color)
        draw.text((x + 30, y), f"{category}: {stats['object_categories'].get(category, 0)}",
                  font=font(19), fill="#cbd5e1")
        y += 30
    y += 12
    for text in [
        "Green: ego position and forward direction",
        "Gray / cyan: recorded past / future ego path",
        f"Objects inside BEV view: {stats['bev_objects']} / {stats['annotated_objects']}",
        "BEV: ego forward points up; ego left points left.",
        "Image source: original nuScenes camera keyframes.",
    ]:
        y = wrapped_text(draw, text, (x, y), width, 17)
    if missing:
        wrapped_text(draw, f"Missing camera images: {len(missing)} (see manifest.json)",
                     (x, y + 8), width, 18, "#fb7185")
    draw.text((18, 1270), "Six-camera snapshot + annotation BEV. Object tracks are dataset annotations; "
              "the recorded ego path is a reference.", font=font(15), fill="#94a3b8")
    filename = f"{scene.scene_id}_{scene.official_name}.jpg"
    canvas.save(output / filename, quality=90, subsampling=0)
    return {**asdict(scene), "keyframes": list(scene.keyframes), "image": filename,
            "status": "partial" if missing else "ok", "missing_cameras": missing,
            "camera_sources": camera_sources, **stats}


def write_gallery(output: Path, records: list[dict], metadata: dict) -> None:
    cards = []
    for row in records:
        if "image" not in row:
            continue
        label = f"{row['scene_id']} / {row['official_name']}"
        search = html.escape(f"{label} {row['description']} {row['location']}", quote=True)
        image = html.escape(row["image"], quote=True)
        badge = "GS available" if row["has_checkpoint"] else "GS missing"
        cards.append(f'<article data-search="{search}" data-gs="{int(row["has_checkpoint"])}">'
                     f'<a href="{image}" target="_blank"><img src="{image}" loading="lazy" '
                     f'width="1536" height="1300" alt="{html.escape(label)}"></a>'
                     f'<strong>{html.escape(label)}</strong><span>{badge} · frame {row["frame"]} '
                     f'· {row["relative_time_s"]:.2f}s</span><p>{html.escape(row["description"])}</p></article>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ReconDreamer-RL 场景总览</title><style>
body{margin:24px;background:#0b1220;color:#e2e8f0;font:16px system-ui,sans-serif}
header{position:sticky;top:0;background:#0b1220;padding:12px 0;z-index:1}
h1{font-size:26px;margin:4px 0 12px}input,select{padding:10px;margin:4px;background:#1e293b;color:white;border:1px solid #475569;border-radius:6px}
input{width:min(500px,75vw)}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:18px}
article{background:#172235;border-radius:8px;overflow:hidden}article img{width:100%;height:auto}strong,span,article p{display:block;margin:10px 14px}span{color:#94a3b8}article[hidden]{display:none}
@media(max-width:480px){body{margin:10px}main{grid-template-columns:1fr}}
</style><header><h1>ReconDreamer-RL：每场景一张总览</h1>
<p>原始 nuScenes 六相机关键帧 + 标注鸟瞰图。点击图片放大。图像来源为原始相机数据；重建模型状态另行标注。</p>
<input id="search" placeholder="搜索场景编号、描述、地点，例如 019 / pedestrian / construction">
<select id="gs"><option value="all">全部场景</option><option value="1">有重建模型</option><option value="0">缺少重建模型</option></select>
<span id="count"></span></header><main>''' + "\n".join(cards) + '''</main><script>
const search=document.querySelector('#search'),gs=document.querySelector('#gs'),cards=[...document.querySelectorAll('article')];
function filter(){let count=0;const query=search.value.toLowerCase().trim();for(const card of cards){card.hidden=!card.dataset.search.toLowerCase().includes(query)||(gs.value!=='all'&&card.dataset.gs!==gs.value);if(!card.hidden)count++;}document.querySelector('#count').textContent=`显示 ${count} / ${cards.length} 个场景`;}
search.addEventListener('input',filter);gs.addEventListener('change',filter);filter();
</script></html>'''
    (output / "index.html").write_text(page, encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({**metadata, "scenes": records},
                                                   ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["scene_id", "official_name", "location", "frame", "relative_time_s",
                  "has_checkpoint", "status", "image", "description", "error"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scene-ids", nargs="+", help="Recon scene IDs, e.g. 000 019 027 077")
    parser.add_argument("--frame-fraction", type=float, default=0.5,
                        help="Representative keyframe position: 0=first, 0.5=middle, 1=last")
    parser.add_argument("--bev-range", type=float, default=40.0, help="BEV half-width in meters")
    parser.add_argument("--require-checkpoint", action="store_true", help="Only include scenes with GS model files")
    parser.add_argument("--limit", type=int, help="Render only the first N selected scenes")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing overview run")
    args = parser.parse_args(argv)
    if not 0 <= args.frame_fraction <= 1 or not math.isfinite(args.bev_range) or args.bev_range <= 0:
        parser.error("--frame-fraction must be in [0,1] and --bev-range must be positive and finite")
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers and --limit must be positive")
    assets = args.dataset_root.resolve() / "assets"
    output = args.output_dir.resolve()
    if not (assets / "nus/data").is_dir():
        parser.error(f"Dataset scene directory not found: {assets / 'nus/data'}")
    if output == assets or assets in output.parents or output == args.dataset_root.resolve():
        parser.error("Choose an output directory outside the dataset assets")
    if output.exists() and any(output.glob("*.jpg")) and not args.overwrite:
        parser.error("Output already contains scene images; use a new directory or --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    scenes = select_scenes(assets, args.frame_fraction, args.scene_ids, args.require_checkpoint, args.limit)
    print(f"Selected {len(scenes)} scenes; {sum(s.has_checkpoint for s in scenes)} have GS checkpoints", flush=True)
    cameras = camera_index(assets, scenes, output)
    records = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(render_scene, assets, output, s, cameras[s.sample_token],
                                   args.bev_range): s for s in scenes}
        for future in as_completed(futures):
            scene = futures[future]
            try:
                records.append(future.result())
            except Exception as error:
                records.append({**asdict(scene), "status": "error", "error": str(error)})
                print(f"ERROR scene {scene.scene_id}: {error}", flush=True)
            if len(records) % 20 == 0 or len(records) == len(scenes):
                print(f"Rendered {len(records)}/{len(scenes)} scenes in {time.monotonic()-started:.0f}s", flush=True)
    records.sort(key=lambda row: row["scene_id"])
    statuses = dict(Counter(row["status"] for row in records))
    write_gallery(output, records, {
        "dataset_root": str(args.dataset_root.resolve()), "image_source": "original_nuscenes_keyframes",
        "frame_fraction": args.frame_fraction, "bev_range_m": args.bev_range,
        "selected_scenes": len(scenes), "status_counts": statuses,
    })
    print(f"Finished: {statuses}\nGallery: {output / 'index.html'}", flush=True)
    return 0 if statuses.get("ok", 0) == len(scenes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
