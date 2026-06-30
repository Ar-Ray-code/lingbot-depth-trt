#!/usr/bin/env python3
"""Live RealSense demo for raw depth vs TensorRT LingBot-Depth refined depth."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
import torch


def depth_to_color(depth: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    clean = np.where(valid, depth, 0.0)
    if vmin is None:
        vmin = float(np.percentile(clean[valid], 1)) if valid.any() else 0.0
    if vmax is None:
        vmax = float(np.percentile(clean[valid], 99)) if valid.any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = np.clip((clean - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color[~valid] = (0, 0, 0)
    return color


def put_label(image: np.ndarray, label: str) -> None:
    cv2.rectangle(image, (8, 8), (8 + 12 * len(label), 36), (0, 0, 0), -1)
    cv2.putText(image, label, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def put_top_right_text(image: np.ndarray, text: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    padding = 7
    margin = 8
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x2 = image.shape[1] - margin
    y1 = margin
    x1 = x2 - text_w - padding * 2
    y2 = y1 + text_h + baseline + padding * 2
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (x1 + padding, y2 - baseline - padding),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def make_comparison(raw_depth: np.ndarray, refined_depth: np.ndarray, inference_ms: float) -> np.ndarray:
    valid = np.isfinite(raw_depth) & (raw_depth > 0) & np.isfinite(refined_depth) & (refined_depth > 0)
    if valid.any():
        vals = np.concatenate([raw_depth[valid], refined_depth[valid]])
        vmin = float(np.percentile(vals, 1))
        vmax = float(np.percentile(vals, 99))
    else:
        vmin, vmax = 0.0, 1.0

    raw_vis = depth_to_color(raw_depth, vmin=vmin, vmax=vmax)
    refined_vis = depth_to_color(refined_depth, vmin=vmin, vmax=vmax)
    put_label(raw_vis, "raw depth")
    put_label(refined_vis, "refined depth")
    comparison = np.concatenate([raw_vis, refined_vis], axis=1)
    put_top_right_text(comparison, f"{inference_ms:.1f} ms")
    return comparison


def device_info(device: rs.device) -> dict[str, str]:
    fields = {
        "name": rs.camera_info.name,
        "serial": rs.camera_info.serial_number,
        "firmware": rs.camera_info.firmware_version,
        "physical_port": rs.camera_info.physical_port,
        "product_id": rs.camera_info.product_id,
        "product_line": rs.camera_info.product_line,
        "usb_type": rs.camera_info.usb_type_descriptor,
    }
    out: dict[str, str] = {}
    for key, info in fields.items():
        try:
            out[key] = device.get_info(info)
        except Exception:
            pass
    return out


def list_realsense_devices() -> list[dict[str, str]]:
    ctx = rs.context()
    return [device_info(device) for device in ctx.query_devices()]


def print_devices(devices: list[dict[str, str]]) -> None:
    if not devices:
        print("No RealSense devices found.")
        return
    for i, info in enumerate(devices):
        print(f"[{i}] {info.get('name', 'unknown')}")
        for key in ["serial", "physical_port", "product_id", "product_line", "usb_type", "firmware"]:
            if key in info:
                print(f"    {key}: {info[key]}")


def resolve_realsense_serial(args: argparse.Namespace) -> tuple[str, dict[str, str]]:
    devices = list_realsense_devices()
    if args.list_devices:
        print_devices(devices)
        raise SystemExit(0)
    if not devices:
        raise RuntimeError("No RealSense devices found")

    target = args.serial or args.usb_id or args.realsense
    if target in (None, "", "auto"):
        if len(devices) > 1:
            print_devices(devices)
            raise RuntimeError("Multiple RealSense devices found; specify --realsense, --serial, or --usb-id")
        info = devices[0]
        return info["serial"], info

    target_lower = str(target).lower()
    matches = []
    for info in devices:
        searchable = " ".join(str(v) for v in info.values()).lower()
        if target_lower in searchable:
            matches.append(info)

    if len(matches) == 1:
        return matches[0]["serial"], matches[0]
    print_devices(devices)
    if not matches:
        raise RuntimeError(f"No RealSense device matched target: {target}")
    raise RuntimeError(f"Multiple RealSense devices matched target: {target}")


def trt_dtype_to_torch(dtype: Any) -> torch.dtype:
    import tensorrt as trt

    if dtype == trt.DataType.FLOAT:
        return torch.float32
    if dtype == trt.DataType.HALF:
        return torch.float16
    if dtype == trt.DataType.INT32:
        return torch.int32
    if dtype == trt.DataType.INT8:
        return torch.int8
    raise TypeError(f"Unsupported TensorRT dtype: {dtype}")


class TensorRTRefiner:
    def __init__(self, args: argparse.Namespace):
        import tensorrt as trt

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT backend requires CUDA")
        model_path = Path(args.model)
        if model_path.suffix != ".engine":
            raise ValueError(f"--model must point to a TensorRT .engine file, got: {model_path}")
        self.args = args
        self.device = torch.device("cuda")
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(model_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {model_path}")
        self.context = self.engine.create_execution_context()

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        input_names = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        output_names = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.image_name = "image" if "image" in input_names else input_names[0]
        self.depth_name = "depth" if "depth" in input_names else input_names[1]
        self.output_name = "depth_refined" if "depth_refined" in output_names else output_names[0]
        self.image_shape = tuple(self.engine.get_tensor_shape(self.image_name))
        self.depth_shape = tuple(self.engine.get_tensor_shape(self.depth_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self.image_dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(self.image_name))
        self.depth_dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(self.depth_name))
        self.output_dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(self.output_name))
        self.stream = torch.cuda.Stream()
        print(f"Loaded TensorRT engine: {args.model}")
        print(f"  {self.image_name}: {self.image_shape} {self.image_dtype}")
        print(f"  {self.depth_name}: {self.depth_shape} {self.depth_dtype}")
        print(f"  {self.output_name}: {self.output_shape} {self.output_dtype}")

    def refine(self, color_bgr: np.ndarray, depth_m: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = depth_m.shape
        expected_h, expected_w = self.depth_shape[-2:]
        if (h, w) != (expected_h, expected_w):
            raise RuntimeError(f"Engine expects {expected_w}x{expected_h}, got {w}x{h}")

        image_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image_rgb / 255.0, dtype=self.image_dtype, device=self.device).permute(2, 0, 1).unsqueeze(0).contiguous()
        depth = torch.tensor(depth_m, dtype=self.depth_dtype, device=self.device).unsqueeze(0).contiguous()
        output = torch.empty(self.output_shape, dtype=self.output_dtype, device=self.device).contiguous()

        self.context.set_tensor_address(self.image_name, int(image.data_ptr()))
        self.context.set_tensor_address(self.depth_name, int(depth.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(output.data_ptr()))

        start = time.perf_counter()
        with torch.cuda.stream(self.stream):
            ok = self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        if not ok:
            raise RuntimeError("TensorRT execution failed")
        infer_s = time.perf_counter() - start
        return output.squeeze(0).detach().float().cpu().numpy(), infer_s


def open_video_writer(args: argparse.Namespace, frame_size: tuple[int, int]) -> cv2.VideoWriter | None:
    if not args.output_video:
        return None
    output_video = Path(args.output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*args.video_fourcc)
    writer = cv2.VideoWriter(str(output_video), fourcc, args.video_fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_video}")
    return writer


def write_run_metadata(args: argparse.Namespace, output_dir: Path, camera_info: dict[str, str]) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "camera": camera_info,
        "model": args.model,
        "backend": "tensorrt",
        "stream": {"width": args.width, "height": args.height, "fps": args.fps},
        "display": args.show_display,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_demo(args: argparse.Namespace) -> int:
    serial, selected_info = resolve_realsense_serial(args)
    print(f"Selected RealSense: {selected_info.get('name', 'unknown')} serial={serial}")
    if selected_info.get("physical_port"):
        print(f"  physical_port={selected_info['physical_port']}")
    if selected_info.get("product_id"):
        print(f"  product_id={selected_info['product_id']}")

    output_dir: Path | None = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif not args.show_display:
        output_dir = Path("live_demo_output") / datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_run_metadata(args, output_dir, selected_info)
        print(f"Writing latest frames to: {output_dir}")

    refiner = TensorRTRefiner(args)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    video_writer: cv2.VideoWriter | None = None
    last_report = time.perf_counter()
    frames = 0
    saved = 0
    try:
        for _ in range(args.warmup):
            pipeline.wait_for_frames(args.timeout_ms)

        while True:
            frameset = pipeline.wait_for_frames(args.timeout_ms)
            aligned = align.process(frameset)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())
            depth_units = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
            depth_m = depth_units.astype(np.float32) * depth_scale

            refined, infer_s = refiner.refine(color_bgr, depth_m)
            comparison = make_comparison(depth_m, refined, infer_s * 1000.0)
            frames += 1

            if video_writer is None:
                h, w = comparison.shape[:2]
                video_writer = open_video_writer(args, (w, h))
            if video_writer is not None:
                video_writer.write(comparison)

            if output_dir:
                cv2.imwrite(str(output_dir / "latest_comparison.png"), comparison)
                if args.save_every > 0 and frames % args.save_every == 0:
                    saved += 1
                    cv2.imwrite(str(output_dir / f"comparison_{saved:06d}.png"), comparison)

            if args.show_display:
                cv2.imshow("LingBot-Depth RealSense Live", comparison)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            now = time.perf_counter()
            if args.print_every > 0 and (frames % args.print_every == 0 or now - last_report >= args.print_seconds):
                valid_raw = float(((depth_m > 0) & np.isfinite(depth_m)).mean())
                valid_refined = float(((refined > 0) & np.isfinite(refined)).mean())
                print(
                    f"frame={frames} infer={infer_s * 1000:.1f}ms "
                    f"raw_valid={valid_raw:.3f} refined_valid={valid_refined:.3f}"
                )
                last_report = now

            if args.max_frames > 0 and frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        if video_writer is not None:
            video_writer.release()
        pipeline.stop()
        if args.show_display:
            cv2.destroyAllWindows()

    print(f"Done. frames={frames}")
    if output_dir:
        print(f"Latest comparison: {output_dir / 'latest_comparison.png'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to TensorRT .engine")
    parser.add_argument("--realsense", default="auto", help="Device selector: serial, name, USB port, product id, or auto")
    parser.add_argument("--serial", default=None, help="Exact RealSense serial number")
    parser.add_argument("--usb-id", default=None, help="Substring matched against physical port or product id")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--show-display", action="store_true", help="Show OpenCV GUI window")
    parser.add_argument("--output-dir", default=None, help="Directory for latest_comparison.png and periodic snapshots")
    parser.add_argument("--save-every", type=int, default=30, help="Save numbered PNG every N frames; 0 disables numbered snapshots")
    parser.add_argument("--output-video", default=None, help="Optional video output path")
    parser.add_argument("--video-fourcc", default="mp4v")
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 means run until interrupted")
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--print-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
