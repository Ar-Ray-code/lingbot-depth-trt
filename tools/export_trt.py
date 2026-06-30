#!/usr/bin/env python3
"""ONNX/TensorRT conversion experiments for LingBot-Depth.

This is intentionally a development script.  It exports a fixed-shape,
fixed-token LingBot-Depth depth-regression graph that avoids the default
variable depth-token masking path.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import tensorrt as trt
import torch
import torch.nn.functional as F
from torch import nn

from mdm.model.modules_rgbd_encoder import DINOv2_RGBD_Encoder
from mdm.model.v2 import MDMModel
from mdm.utils.geo import normalized_view_plane_uv


def _patched_encoder_forward(
    self: DINOv2_RGBD_Encoder,
    image: torch.Tensor,
    depth: torch.Tensor,
    token_rows: int,
    token_cols: int,
    return_class_token: bool = False,
    remap_depth_in: str = "linear",
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None, None, None]:
    image_14 = F.interpolate(
        image,
        (token_rows * 14, token_cols * 14),
        mode="bilinear",
        align_corners=False,
        antialias=not self.onnx_compatible_mode,
    )
    image_14 = (image_14 - self.image_mean) / self.image_std

    depth_14 = F.interpolate(depth, (token_rows * 14, token_cols * 14), mode="nearest")
    depth_14[torch.isinf(depth_14)] = 0.0
    depth_14[torch.isnan(depth_14)] = 0.0
    dmask_14 = (depth_14 > 0.01).detach()
    depth_14 = depth_14 * dmask_14.to(dtype=depth_14.dtype)

    if remap_depth_in == "log":
        depth_14 = torch.log(depth_14)
        depth_14[~dmask_14] = 0.0
        depth_14 = torch.nan_to_num(depth_14, nan=0.0, posinf=0.0, neginf=0.0)
    elif remap_depth_in != "linear":
        raise NotImplementedError

    features = self.backbone.get_intermediate_layers_mae(
        x_img=image_14,
        x_depth=depth_14,
        n=self.intermediate_layers,
        return_class_token=True,
        **kwargs,
    )
    assert self.img_mask_ratio == 0, "img_mask_ratio is not supported in this encoder"

    if isinstance(features[0][0], list):
        num_valid_tokens = token_rows * token_cols
        features = tuple(
            (
                torch.cat([feat[:, :num_valid_tokens].contiguous() for feat in feats], dim=0),
                torch.cat(cls_tokens, dim=0),
            )
            for feats, cls_tokens in features
        )

    x = torch.stack(
        [
            proj(feat.permute(0, 2, 1)[:, :, : token_rows * token_cols].unflatten(2, (token_rows, token_cols)).contiguous())
            for proj, (feat, _clstoken) in zip(self.output_projections, features)
        ],
        dim=1,
    ).sum(dim=1)
    cls_token = features[-1][1]

    if return_class_token:
        return x, cls_token, None, None
    return x, None, None


def _patched_model_forward(
    self: MDMModel,
    image: torch.Tensor,
    num_tokens: int,
    depth: torch.Tensor | None = None,
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    batch_size, _, img_h, img_w = image.shape
    device, dtype = image.device, image.dtype

    assert depth is not None
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)

    aspect_ratio = img_w / img_h
    base_h, base_w = (num_tokens / aspect_ratio) ** 0.5, (num_tokens * aspect_ratio) ** 0.5
    if isinstance(base_h, torch.Tensor):
        base_h, base_w = int(base_h.round().item()), int(base_w.round().item())
    else:
        base_h, base_w = round(base_h), round(base_w)

    features, cls_token, _, _ = self.encoder(
        image,
        depth,
        base_h,
        base_w,
        return_class_token=True,
        remap_depth_in=self.remap_depth_in,
        **kwargs,
    )

    features = features + cls_token[..., None, None]
    features = [features, None, None, None, None]

    for level in range(5):
        uv = normalized_view_plane_uv(
            width=base_w * 2**level,
            height=base_h * 2**level,
            aspect_ratio=aspect_ratio,
            dtype=dtype,
            device=device,
        )
        uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
        if features[level] is None:
            features[level] = uv
        else:
            features[level] = torch.cat([features[level], uv], dim=1)

    features = [feature.to(dtype=dtype) for feature in features]
    features = self.neck(features)

    depth_reg, normal, mask = (
        getattr(self, head)(features)[-1] if hasattr(self, head) else None
        for head in ["depth_head", "normal_head", "mask_head"]
    )
    depth_reg, normal, mask = (
        F.interpolate(v, (img_h, img_w), mode="bilinear", align_corners=False, antialias=False) if v is not None else None
        for v in [depth_reg, normal, mask]
    )

    if depth_reg is not None:
        if self.remap_depth_out == "exp":
            depth_reg = depth_reg.exp().squeeze(1)
        elif self.remap_depth_out == "linear":
            depth_reg = depth_reg.squeeze(1)
        else:
            raise ValueError(f"Invalid remap_depth_out: {self.remap_depth_out}")
    if normal is not None:
        normal = normal.permute(0, 2, 3, 1)
        normal = F.normalize(normal, dim=-1)
    if mask is not None:
        mask_prob = mask.squeeze(1).sigmoid()
    else:
        mask_prob = None

    return_dict = {
        "depth_reg": depth_reg,
        "normal": normal,
        "mask": mask_prob,
    }
    return {k: v for k, v in return_dict.items() if v is not None}


def apply_lingbot_export_patches() -> None:
    """Keep FP16 ONNX export type-stable without modifying upstream files."""
    DINOv2_RGBD_Encoder.forward = _patched_encoder_forward
    MDMModel.forward = _patched_model_forward


class ExportableLingBotDepth(nn.Module):
    def __init__(self, model: MDMModel, num_tokens: int):
        super().__init__()
        self.model = model
        self.num_tokens = int(num_tokens)

    def forward(self, image: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        output = self.model.forward(
            image,
            num_tokens=self.num_tokens,
            depth=depth,
            enable_depth_mask=False,
        )
        return output["depth_reg"]


def load_capture(capture_dir: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    rgb_path = capture_dir / "rgb.png"
    depth_path = capture_dir / "raw_depth.png"
    metadata_path = capture_dir / "metadata.json"
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read RGB image: {rgb_path}")
    if depth_mm is None:
        raise FileNotFoundError(f"Failed to read depth image: {depth_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0).contiguous()
    depth = torch.tensor(depth_mm.astype(np.float32) / 1000.0, dtype=torch.float32, device=device).unsqueeze(0).contiguous()

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({"height": int(rgb.shape[0]), "width": int(rgb.shape[1])})
    return image, depth, metadata


def make_dummy_inputs(width: int, height: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    image_np = np.zeros((height, width, 3), dtype=np.float32)
    image_np[..., 0] = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    image_np[..., 1] = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    image_np[..., 2] = 0.5
    depth_np = np.full((height, width), 1.0, dtype=np.float32)

    image = torch.tensor(image_np, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0).contiguous()
    depth = torch.tensor(depth_np, dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    metadata = {
        "source": "dummy",
        "height": height,
        "width": width,
        "note": "Synthetic input used only for fixed-shape export and smoke validation.",
    }
    return image, depth, metadata


def summarize_depth(depth: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(depth) & (depth > 0)
    out: dict[str, Any] = {
        "shape": list(depth.shape),
        "valid_fraction": float(valid.mean()),
        "valid_pixels": int(valid.sum()),
    }
    if valid.any():
        vals = depth[valid].astype(np.float64)
        out.update(
            {
                "min_m": float(vals.min()),
                "median_m": float(np.median(vals)),
                "mean_m": float(vals.mean()),
                "p95_m": float(np.percentile(vals, 95)),
                "max_m": float(vals.max()),
            }
        )
    return out


def load_export_model(args: argparse.Namespace, device: torch.device) -> ExportableLingBotDepth:
    apply_lingbot_export_patches()
    model = MDMModel.from_pretrained(args.model).to(device).eval()
    model.encoder.onnx_compatible_mode = True
    model.enable_pytorch_native_sdpa()
    if args.precision == "fp16":
        model = model.half()
    return ExportableLingBotDepth(model, args.num_tokens).to(device).eval()


def export_onnx(
    wrapper: ExportableLingBotDepth,
    image: torch.Tensor,
    depth: torch.Tensor,
    onnx_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (image, depth),
            str(onnx_path),
            input_names=["image", "depth"],
            output_names=["depth_refined"],
            opset_version=args.opset,
            do_constant_folding=True,
            external_data=args.external_data,
            dynamo=args.dynamo,
        )
    elapsed = time.perf_counter() - start
    model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.checker.check_model(model)
    return {
        "path": str(onnx_path),
        "bytes": onnx_path.stat().st_size,
        "export_s": elapsed,
        "ir_version": model.ir_version,
        "opset": [(op.domain, op.version) for op in model.opset_import],
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
    }


def run_torch(wrapper: ExportableLingBotDepth, image: torch.Tensor, depth: torch.Tensor) -> tuple[np.ndarray, float]:
    device = image.device
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output = wrapper(image, depth)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return output.detach().float().cpu().numpy(), elapsed


def run_onnxruntime(onnx_path: Path, image: torch.Tensor, depth: torch.Tensor, args: argparse.Namespace) -> dict[str, Any]:
    providers: list[Any]
    if args.ort_provider == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif args.ort_provider == "tensorrt":
        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(args.work_dir / "ort_trt_cache"),
                    "trt_fp16_enable": False,
                },
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]

    start = time.perf_counter()
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    load_s = time.perf_counter() - start
    inputs = {
        "image": image.detach().cpu().numpy(),
        "depth": depth.detach().cpu().numpy(),
    }
    start = time.perf_counter()
    output = session.run(["depth_refined"], inputs)[0]
    infer_s = time.perf_counter() - start
    return {
        "providers": session.get_providers(),
        "load_s": load_s,
        "infer_s": infer_s,
        "output": output,
    }


def compare_outputs(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    diff = b.astype(np.float64) - a.astype(np.float64)
    abs_diff = np.abs(diff)
    return {
        "max_abs": float(abs_diff.max()),
        "mean_abs": float(abs_diff.mean()),
        "median_abs": float(np.median(abs_diff)),
        "p95_abs": float(np.percentile(abs_diff, 95)),
        "allclose_rtol1e-3_atol1e-3": bool(np.allclose(a, b, rtol=1e-3, atol=1e-3)),
    }


def build_tensorrt_engine(onnx_path: Path, engine_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.INFO if args.verbose_trt else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)

    start = time.perf_counter()
    with builder.create_network(network_flags) as network, trt.OnnxParser(network, logger) as parser:
        if not parser.parse(onnx_path.read_bytes()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            return {"ok": False, "stage": "parse", "errors": errors}

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb * (1 << 30))
        config.set_memory_pool_limit(trt.MemoryPoolType.TACTIC_DRAM, args.tactic_gb * (1 << 30))
        if hasattr(trt.BuilderFlag, "TF32") and args.tf32:
            config.set_flag(trt.BuilderFlag.TF32)

        serialized_engine = builder.build_serialized_network(network, config)
        elapsed = time.perf_counter() - start
        if serialized_engine is None:
            return {
                "ok": False,
                "stage": "build",
                "build_s": elapsed,
                "inputs": [
                    {"name": network.get_input(i).name, "shape": list(network.get_input(i).shape)}
                    for i in range(network.num_inputs)
                ],
                "outputs": [
                    {"name": network.get_output(i).name, "shape": list(network.get_output(i).shape)}
                    for i in range(network.num_outputs)
                ],
                "layers": network.num_layers,
            }
        engine_path.write_bytes(serialized_engine)
        return {
            "ok": True,
            "path": str(engine_path),
            "bytes": engine_path.stat().st_size,
            "build_s": elapsed,
            "inputs": [
                {"name": network.get_input(i).name, "shape": list(network.get_input(i).shape)}
                for i in range(network.num_inputs)
            ],
            "outputs": [
                {"name": network.get_output(i).name, "shape": list(network.get_output(i).shape)}
                for i in range(network.num_outputs)
            ],
            "layers": network.num_layers,
        }


def run_tensorrt_engine(
    engine_path: Path,
    image: torch.Tensor,
    depth: torch.Tensor,
    torch_output: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        return {"ok": False, "stage": "deserialize"}

    context = engine.create_execution_context()
    output_shape = tuple(engine.get_tensor_shape("depth_refined"))
    output_dtype = trt_dtype_to_torch(engine.get_tensor_dtype("depth_refined"))
    output = torch.empty(output_shape, dtype=output_dtype, device=image.device).contiguous()

    context.set_tensor_address("image", int(image.data_ptr()))
    context.set_tensor_address("depth", int(depth.data_ptr()))
    context.set_tensor_address("depth_refined", int(output.data_ptr()))

    if image.device.type == "cuda":
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(args.trt_warmup):
                if not context.execute_async_v3(stream.cuda_stream):
                    return {"ok": False, "stage": "warmup_execute"}
        stream.synchronize()

        start = time.perf_counter()
        with torch.cuda.stream(stream):
            for _ in range(args.trt_runs):
                if not context.execute_async_v3(stream.cuda_stream):
                    return {"ok": False, "stage": "execute"}
        stream.synchronize()
        infer_s = (time.perf_counter() - start) / args.trt_runs
    else:
        start = time.perf_counter()
        for _ in range(args.trt_runs):
            if not context.execute_v2([]):
                return {"ok": False, "stage": "execute_v2"}
        infer_s = (time.perf_counter() - start) / args.trt_runs

    trt_output = output.detach().cpu().numpy()
    return {
        "ok": True,
        "engine": str(engine_path),
        "runs": args.trt_runs,
        "warmup": args.trt_warmup,
        "infer_s_avg": infer_s,
        "depth": summarize_depth(trt_output.squeeze(0)),
        "vs_torch": compare_outputs(torch_output, trt_output),
    }


def trt_dtype_to_torch(dtype: trt.DataType) -> torch.dtype:
    if dtype == trt.DataType.FLOAT:
        return torch.float32
    if dtype == trt.DataType.HALF:
        return torch.float16
    if dtype == trt.DataType.INT32:
        return torch.int32
    if dtype == trt.DataType.INT8:
        return torch.int8
    if hasattr(trt.DataType, "BF16") and dtype == trt.DataType.BF16:
        return torch.bfloat16
    raise TypeError(f"Unsupported TensorRT dtype: {dtype}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="lingbot-depth-pretrain-vitl-14-v0.5.pt")
    parser.add_argument("--capture", type=Path, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--work-dir", type=Path, default=Path("trt_work/lingbot_depth_nt300"))
    parser.add_argument("--num-tokens", type=int, default=300)
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--dynamo", action="store_true")
    parser.add_argument("--no-external-data", dest="external_data", action="store_false")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-ort", action="store_true")
    parser.add_argument("--skip-trt-build", action="store_true")
    parser.add_argument("--skip-trt-run", action="store_true")
    parser.add_argument("--ort-provider", choices=["cuda", "cpu", "tensorrt"], default="cuda")
    parser.add_argument("--workspace-gb", type=int, default=6)
    parser.add_argument("--tactic-gb", type=int, default=4)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--verbose-trt", action="store_true")
    parser.add_argument("--trt-warmup", type=int, default=3)
    parser.add_argument("--trt-runs", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.work_dir / f"lingbot_depth_nt{args.num_tokens}.onnx"
    engine_path = args.work_dir / f"lingbot_depth_nt{args.num_tokens}.engine"
    report_path = args.work_dir / "conversion_report.json"

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.capture is None:
        image, depth, capture_meta = make_dummy_inputs(args.width, args.height, device)
        capture_label = "dummy"
    else:
        image, depth, capture_meta = load_capture(args.capture, device)
        capture_label = str(args.capture)
    if args.precision == "fp16":
        image = image.half()
        depth = depth.half()
    wrapper = load_export_model(args, device)

    torch_output, torch_s = run_torch(wrapper, image, depth)
    report: dict[str, Any] = {
        "capture": capture_label,
        "capture_meta": capture_meta,
        "model": str(args.model),
        "device": str(device),
        "precision": args.precision,
        "num_tokens": args.num_tokens,
        "torch": {
            "version": torch.__version__,
            "infer_s": torch_s,
            "depth": summarize_depth(torch_output.squeeze(0)),
        },
        "conversion_notes": [
            "Export path uses model.forward(depth_reg) with enable_depth_mask=False.",
            "Default LingBot-Depth variable depth-token masking is not included in this first TensorRT graph.",
            "Point-cloud projection and mask application remain Python-side postprocessing for now.",
        ],
    }

    if not args.skip_export:
        report["onnx"] = export_onnx(wrapper, image, depth, onnx_path, args)
    else:
        onnx.checker.check_model(onnx.load(str(onnx_path), load_external_data=True))
        report["onnx"] = {"path": str(onnx_path), "skipped_export": True}

    if not args.skip_ort:
        ort_result = run_onnxruntime(onnx_path, image, depth, args)
        report["onnxruntime"] = {
            "providers": ort_result["providers"],
            "load_s": ort_result["load_s"],
            "infer_s": ort_result["infer_s"],
            "depth": summarize_depth(ort_result["output"].squeeze(0)),
            "vs_torch": compare_outputs(torch_output, ort_result["output"]),
        }

    if not args.skip_trt_build:
        report["tensorrt_build"] = build_tensorrt_engine(onnx_path, engine_path, args)

    if not args.skip_trt_run and engine_path.exists():
        report["tensorrt_run"] = run_tensorrt_engine(engine_path, image, depth, torch_output, args)

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    trt_info = report.get("tensorrt_build")
    if isinstance(trt_info, dict) and trt_info.get("ok") is False:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
