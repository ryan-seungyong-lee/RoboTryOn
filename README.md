<div align="center">
<img src="assets/logo.png" width="120" alt="RoboTryOn">
<h2>Synthesizing Dexterous Robot Demonstrations from Human Videos via Virtual Try-On</h2>

<a href="https://arxiv.org/abs/2611.05163"><img src='https://img.shields.io/badge/arXiv-2611.05163-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'></a>
<a href="https://ryan-seungyong-lee.github.io/RoboTryOn"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://huggingface.co/RyanL22/RoboTryOn"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Checkpoints-yellow" alt="Checkpoints"></a>
<a href="https://huggingface.co/spaces/RyanL22/RoboTryOn"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Demo-blue" alt="Demo"></a>

<a href="https://ryan-seungyong-lee.github.io/">Seungyong Lee</a><sup>1</sup>&emsp;<a href="https://suhyeok-jang.github.io/">Suhyeok Jang</a><sup>1,2</sup>&emsp;<a href="https://eurominyoung186.github.io/">Minyoung Kim</a><sup>3</sup>&emsp;<a href="https://alinlab.kaist.ac.kr/shin.html">Jinwoo Shin</a><sup>1,2</sup><br>
<sup>1</sup>KAIST&emsp;<sup>2</sup>RLWRLD&emsp;<sup>3</sup>Korea University
</div>

<!-- <p align="center"><img src="assets/teaser.jpg" width="90%"></p> -->

<p align="center">
<b>RoboTryOn</b> treats human-to-robot translation as virtual try-on: try-on keeps the person and
swaps the garment, RoboTryOn keeps the demonstrated motion and swaps the human for the target embodiment.
An image model translates keyframes of a human video into the target embodiment, a video model
synthesizes the motion between them under the guidance of the hand skeleton, and an inverse
dynamics model labels the actions.
</p>

## News
- **[2026-11]** Paper released on [arXiv](https://arxiv.org/abs/2611.05163).
- **[2026-11]** Code and [checkpoints](https://huggingface.co/RyanL22/RoboTryOn) released.

## Overview

| Stage | Model | Code |
|---|---|---|
| Preprocessing | HaWoR hands, SAM3 masks, keyframe selection, hand skeleton | `preprocessing/` |
| Keyframe translation f<sub>θ</sub> | FLUX.2 [klein] 9B Base (full fine-tune or LoRA) | `src/robotryon/image`, `scripts/{train_image,infer_keyframes}.py` |
| Video synthesis g<sub>φ</sub> | LTX-2.5 in-context LoRA | `src/robotryon/video`, `scripts/{train_video,infer_video}.py` |
| Action labeling h<sub>ψ</sub> | Depth Anything V2-L + flow-matching action head | `src/robotryon/idm`, `scripts/{train_idm,label_actions}.py` |

```
configs/          all hyperparameters (YAML); every script takes --config file.yaml key.sub=value ...
src/robotryon/    models, datasets, samplers (common, skeleton, masks, embodiment, image, video, idm)
preprocessing/    human / embodiment episode preparation
scripts/          training, inference, end-to-end pipeline
```

## Installation

The three models were developed with different library versions, so we recommend one environment
per stage (Python 3.10, CUDA 12.4, PyTorch 2.5).

```bash
git clone https://github.com/ryan-seungyong-lee/RoboTryOn && cd RoboTryOn
pip install -e ".[gen,preprocess]"   # image / video models and preprocessing
pip install -e ".[idm]"              # IDM (separate environment recommended)
```

- Image model: `diffusers>=0.39` (FLUX.2 [klein] support). Video model: `diffusers>=0.40` (LTX-2.5).
- Preprocessing: `mujoco==3.11.0` (rendering needs `MUJOCO_GL=egl` or `osmesa`).
  [HaWoR](https://github.com/ThunderVVV/HaWoR) (with the MANO model) and
  [SAM 3](https://github.com/facebookresearch/sam3) are installed separately.

## Inference

### Checkpoints

| Model | Checkpoint |
|---|---|
| Image model (LoRA) | [`image/`](https://huggingface.co/RyanL22/RoboTryOn/tree/main/image) |
| Video model (LoRA) | [`video/`](https://huggingface.co/RyanL22/RoboTryOn/tree/main/video) |
| IDM | [`idm/`](https://huggingface.co/RyanL22/RoboTryOn/tree/main/idm) |

```bash
hf download RyanL22/RoboTryOn --local-dir ckpt
```

Base weights: [FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B),
[LTX-2.5-Diffusers](https://huggingface.co/Lightricks/LTX-2.5-Diffusers),
[Depth-Anything-V2-Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf).

### End-to-end

Given a human video recorded from the embodiment's head-camera position, a description of the
target embodiment (`configs/embodiments/gr1.yaml`) and a reference image per view (the
embodiment's pixels on a white background; `scripts/make_reference.py` cuts one out of an
episode of that embodiment):

```bash
python scripts/run_pipeline.py --config configs/pipeline.yaml \
    embodiment=path/to/my_embodiment.yaml idm_embodiment=my_embodiment \
    input.videos.left=human_left.mp4 input.videos.right=human_right.mp4 \
    input.caption="put the ball in the box" input.object_prompts='[ball]' \
    reference.left=ref_left.png reference.right=ref_right.png \
    checkpoints.image=ckpt/image checkpoints.video=ckpt/video checkpoints.idm=ckpt/idm \
    hawor_root=third_party/HaWoR output_dir=runs/demo
```

Each stage runs as its own process (`python.<stage>` selects the interpreter), and `stages=[...]`
runs a subset. Output:

```
runs/demo/episode/     preprocessed human episode
runs/demo/keyframes/   keyframe_<view>_<frame>.png    translated keyframes
runs/demo/video/       video_<view>.mp4               synthesized video (20 fps, 1024x576 per view)
runs/demo/actions/     actions.npy, states.npy        joint actions and states (T x D)
```

### Stage by stage

```bash
# 1. human video -> episode directory (hands, keyframes, retargeted skeleton, masks)
python -m preprocessing.prepare_human_clip --embodiment my_embodiment.yaml \
    --video left human_left.mp4 --video right human_right.mp4 --caption "..." \
    --hawor_root third_party/HaWoR --object_prompt ball --out runs/demo/episode

# 2. keyframe translation
python scripts/infer_keyframes.py --config configs/image/infer.yaml \
    checkpoint=ckpt/image episodes='[runs/demo/episode]' \
    reference.left=ref_left.png reference.right=ref_right.png output_dir=runs/demo/keyframes

# 3. video synthesis
python scripts/infer_video.py --config configs/video/infer.yaml model.lora=ckpt/video \
    input.episode=runs/demo/episode input.keyframes_dir=runs/demo/keyframes/episode \
    output.dir=runs/demo/video

# 4. action labeling: IDM joint actions, then the wrists re-solved from the reconstructed human wrists
python scripts/label_actions.py --config configs/idm/label.yaml \
    label.checkpoint=ckpt/idm label.embodiment=my_embodiment \
    label.videos='[runs/demo/video/video_left.mp4,runs/demo/video/video_right.mp4]' \
    label.output_dir=runs/demo/actions
python scripts/correct_wrists.py --config configs/idm/label.yaml label.output_dir=runs/demo/actions \
    wrist_correction.hand3d=runs/demo/episode/hands.npz wrist_correction.embodiment=my_embodiment.yaml
```

Useful options:

| Option | Values | Description |
|---|---|---|
| `mask_human` (video) | `true` \| `false` | skeleton flow for the video model on black (`true`) or drawn on the human video (`false`); the latter can help with a moving camera or heavy occlusion. The image model always blacks out the human arm |
| `ref_skeleton` (image) | `false` \| `true` | reuse the previous keyframe's latent as the reference, or decode it and draw the current skeleton on it |
| `views` (image) | `auto` \| `stereo` \| `mono` | generate both views jointly or one at a time |
| `two_stage.mode` (video) | `same_res` \| `upsample` | both stages at 1024x576, or 512x288 followed by the latent upsampler |
| `input.stereo` (video) | `true` \| `false` | generate both views in one sequence |

## Training

The three models are trained on teleoperated episodes of the embodiments only.

**Data format.** One directory per episode (`preprocessing/prepare_robot_episode.py` builds it
from a LeRobot v2.1 dataset and an embodiment YAML):

```
<root>/<episode_id>/
  meta.json               {"fps", "views", "num_frames", "caption", "embodiment"}
  video_<view>.mp4        RGB
  keypoints_<view>.npz    kp (T, 2, 17, 2) float32 pixels, vis (T, 2, 17) bool
  mask_<view>.mp4         arm + hand mask of the embodiment (>127)
  object_mask_<view>.mp4  manipulated object (optional)
```

The IDM reads LeRobot v2.1 datasets directly; embodiments are declared in `configs/idm/da2l_flowmatch.yaml`.

**Image model**

```bash
# full fine-tune
accelerate launch --num_processes 4 scripts/train_image.py --config configs/image/klein9b_fft.yaml \
    data.roots='[data/embodiment_a,data/embodiment_b]' output_dir=runs/image_fft
# LoRA
accelerate launch --num_processes 4 scripts/train_image.py --config configs/image/klein9b_lora.yaml \
    data.roots='[data/embodiment_a,data/embodiment_b]' output_dir=runs/image_lora
```

**Video model**

```bash
accelerate launch --num_processes 4 scripts/train_video.py --config configs/video/ltx25_iclora_stage1.yaml data.roots='[...]'
accelerate launch --num_processes 4 scripts/train_video.py --config configs/video/ltx25_iclora_stage2.yaml data.roots='[...]' \
    lora.init=outputs/video/stage1/checkpoint-004000/pytorch_lora_weights.safetensors
```

**IDM**

```bash
torchrun --standalone --nproc_per_node=4 scripts/train_idm.py --config configs/idm/da2l_flowmatch.yaml \
    data.roots.<embodiment>=path/to/lerobot_dataset ...
```

## Adding an embodiment

Copy `configs/embodiments/gr1.yaml` and fill in the MJCF/URDF path, the joint layout, the
links of the 17 hand keypoints (wrist; thumb base, two joints, tip; knuckle, middle joint and
tip of the other four fingers) and the camera model. The same file drives the embodiment skeleton and
mask for training data, the retargeted human skeleton at inference, and the IDM's wrist
correction. Embodiment models are not distributed with this repository.

## Citation

```bibtex
@article{lee2026robotryon,
  title={Synthesizing Dexterous Robot Demonstrations from Human Videos via Virtual Try-On},
  author={Lee, Seungyong and Jang, Suhyeok and Kim, Minyoung and Shin, Jinwoo},
  journal={arXiv preprint arXiv:2611.05163},
  year={2026}
}
```

## Acknowledgments

This work was supported by RLWRLD Inc. The IDM is adapted from
[GR00T-Dreams](https://github.com/NVIDIA/GR00T-Dreams) (Apache-2.0), and the image-model trainer
from the FLUX.2 training examples of [diffusers](https://github.com/huggingface/diffusers)
(Apache-2.0). We build on [FLUX.2 [klein]](https://bfl.ai), [LTX-2](https://github.com/Lightricks/LTX-2),
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2),
[HaWoR](https://github.com/ThunderVVV/HaWoR) and [SAM 3](https://github.com/facebookresearch/sam3).
