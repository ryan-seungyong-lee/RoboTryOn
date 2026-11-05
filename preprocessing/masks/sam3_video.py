"""Text- or box-prompted SAM3 video segmentation (propagated over the whole clip).

SAM3 (https://github.com/facebookresearch/sam3) is installed separately; its checkpoint is
fetched from the Hugging Face hub on first use. A video session holds one prompt at a time,
so every prompt is detected on its anchor frame, propagated forward and backward in its own
reset session, and all returned objects are ORed into one mask:

* ``union=False``: the prompt list is a fallback chain of wordings for one thing, and the
  first prompt that detects anything is used;
* ``union=True``: the list names several things (an arm in a sleeve, a ball and a cup) and
  every prompt that detects something is added.
"""

from __future__ import annotations

import cv2
import numpy as np


class Sam3Video:
    def __init__(self):
        import torch
        from sam3.model_builder import build_sam3_video_predictor
        self.predictor = build_sam3_video_predictor(gpus_to_use=[torch.cuda.current_device()])
        self.model = self.predictor.model
        self._defaults = {k: getattr(self.model, k)
                          for k in ("score_threshold_detection", "new_det_thresh")}

    def configure(self, score_threshold: float | None = None,
                  new_det_thresh: float | None = None) -> None:
        """Detection score threshold for new prompts and for objects found while tracking."""
        for k, v in (("score_threshold_detection", score_threshold),
                     ("new_det_thresh", new_det_thresh)):
            setattr(self.model, k, self._defaults[k] if v is None else float(v))

    def _req(self, **kw):
        return self.predictor.handle_request(kw)

    def _detect(self, sid, frame, text=None, box=None) -> int:
        self._req(type="reset_session", session_id=sid)
        kw = dict(type="add_prompt", session_id=sid, frame_index=int(frame))
        if text is not None:
            kw["text"] = text
        if box is not None:
            kw["bounding_boxes"] = [list(box)]
            kw["bounding_box_labels"] = [1]
        return len(self._req(**kw)["outputs"]["out_obj_ids"])

    def _propagate(self, sid, frame, out: np.ndarray) -> None:
        T, H, W = out.shape
        for r in self.predictor.handle_stream_request(
                dict(type="propagate_in_video", session_id=sid, start_frame_index=int(frame))):
            t = r["frame_index"]
            if t >= T:
                continue
            for m in r["outputs"]["out_binary_masks"]:
                m = np.asarray(m.cpu().numpy() if hasattr(m, "cpu") else m).squeeze()
                if m.ndim != 2:
                    continue
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                out[t] |= m.astype(bool)

    def segment(self, video_path: str, num_frames: int, size, prompts=(), boxes=(),
                anchor_frames=(0,), union: bool = False) -> tuple[np.ndarray, list]:
        """Segment a video.

        Args:
            prompts: text prompts. boxes: ``(frame, (x, y, w, h))`` boxes, relative [0, 1]
                coordinates, each tracked as one object.
            anchor_frames: frames on which text prompts are tried, in order.
        Returns:
            bool ``(T, H, W)`` mask and a log of ``(prompt, frame, n_objects)`` hits.
        """
        W, H = size
        out = np.zeros((num_frames, H, W), bool)
        sid = self._req(type="start_session", resource_path=str(video_path))["session_id"]
        hits = []
        try:
            for text in prompts:
                for f in anchor_frames:
                    n = self._detect(sid, f, text=text)
                    if n:
                        hits.append((text, int(f), n))
                        self._propagate(sid, f, out)
                        break
                if hits and not union:
                    break
            for f, box in boxes:
                if self._detect(sid, f, box=box):
                    hits.append(("box", int(f), 1))
                    self._propagate(sid, f, out)
        finally:
            self._req(type="close_session", session_id=sid)
        return out, hits

    def shutdown(self):
        self.predictor.shutdown()


def anchor_indices(anchors, num_frames: int) -> list[int]:
    """Anchor frames; floats in [0, 1) are fractions of the clip."""
    out = []
    for a in anchors:
        f = round(a * (num_frames - 1)) if isinstance(a, float) and 0 <= a < 1 else int(a)
        f = min(max(f, 0), num_frames - 1)
        if f not in out:
            out.append(f)
    return out


def segment_to_video(video: str, out: str, cfg, prompts=None, boxes=(),
                     sam: Sam3Video | None = None) -> np.ndarray:
    """Segment ``video`` with one config section and write the mask mp4 to ``out``.

    ``cfg`` holds ``prompts, union, score_threshold, new_det_thresh, anchor_frames``;
    ``prompts`` overrides ``cfg.prompts``. Pass ``sam`` to reuse a loaded predictor.
    """
    from preprocessing.episode_io import video_info, write_video
    fps, W, H, n = video_info(video)
    sam = sam or Sam3Video()
    sam.configure(cfg.get("score_threshold"), cfg.get("new_det_thresh"))
    mask, hits = sam.segment(
        video, n, (W, H), prompts=list(prompts or cfg.get("prompts", [])), boxes=boxes,
        anchor_frames=anchor_indices(cfg.get("anchor_frames", [0]), n),
        union=bool(cfg.get("union", False)))
    write_video(out, mask, fps, gray=True)
    print(f"{out}: {mask.shape} coverage={mask.mean():.4f} hits={hits}")
    return mask
