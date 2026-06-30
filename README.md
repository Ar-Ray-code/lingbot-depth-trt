# lingbot-depth-trt

TensorRT conversion tools and a RealSense live demo for [LingBot-Depth](https://github.com/Robbyant/lingbot-depth).

LingBot-Depth is not vendored in this repository. Please refer to and cite the upstream project.

![](media/test.png)

## Setup

- Ubuntu 26.04 (NVIDIA driver version : 595.71.05)
- RTX 3060 (12GB RAM)
- uv 0.11.22 (x86_64-unknown-linux-gnu)
- RealSense D405

```bash
cd python
uv venv --python 3.12
uv sync
```

If you want to use a local LingBot-Depth checkout instead of the GitHub dependency:

```bash
cd python
uv pip install -e ../../lingbot-depth
```

## Download Model

Download the pretrained model from Hugging Face:

```bash
cd python
mkdir -p ../output/models
uv run huggingface-cli download \
  robbyant/lingbot-depth-pretrain-vitl-14-v0.5 \
  model.pt \
  --local-dir ../output/models
```

The model page is:

```text
https://huggingface.co/robbyant/lingbot-depth-pretrain-vitl-14-v0.5/blob/main/model.pt
```

## Convert to TensorRT

Convert the model to a fixed-shape TensorRT engine. The default shape is 640 x 480.

```bash
cd python
uv run python ../tools/export_trt.py \
  --model ../output/models/model.pt \
  --precision fp16 \
  --num-tokens 1200 \
  --work-dir ../output/trt_nt1200_fp16
```

The generated TensorRT engine is written to:

```text
output/trt_nt1200_fp16/lingbot_depth_nt1200.engine
```

To build another fixed input size, pass `--width` and `--height`.

You may pass `--capture /path/to/capture` for smoke validation with real RGB-D data. The capture directory should contain `rgb.png` and `raw_depth.png`. It is optional and is not required for conversion.

## RealSense Live Demo

List RealSense devices:

```bash
cd python
uv run python scripts/live_demo.py \
  --model ../output/trt_nt1200_fp16/lingbot_depth_nt1200.engine \
  --list-devices
```

Run with an OpenCV display window:

```bash
cd python
uv run python scripts/live_demo.py \
  --model ../output/trt_nt1200_fp16/lingbot_depth_nt1200.engine \
  --realsense auto \
  --show-display
```


## Citation

```
@article{lingbot-depth2026,
  title={Masked Depth Modeling for Spatial Perception},
  author={Tan, Bin and Sun, Changjiang and Qin, Xiage and Adai, Hanat and Fu, Zelin and Zhou, Tianxiang and Zhang, Han and Xu, Yinghao and Zhu, Xing and Shen, Yujun and Xue, Nan},
  journal={arXiv preprint arXiv:2601.17895},
  year={2026}
}
```

```
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={Transactions on Machine Learning Research},
  year={2024}
}
```
