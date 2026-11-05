"""Video model g_phi: an in-context LoRA on LTX-2.5 that synthesizes the robot video between
translated keyframes under the guidance of the hand-skeleton flow (paper Sec. 3.3).

Modules:
    conditioning  training sequence layout, keyframe sampling, flow-matching loss
    dataset       random training clips with on-the-fly skeleton rendering and augmentation
    rope4d        fourth RoPE axis distinguishing the two stereo views
    pipeline      diffusers pipeline subclass and the two-stage generation of one clip
    chunks        splitting long videos at keyframes and concatenating the clips
    text          caption embeddings (cached once; the text encoder is never co-resident)
    training      gradient checkpointing and LoRA weight I/O
    io            mp4 / png helpers

Heavy dependencies (torch, diffusers) are imported by the submodules that need them.
"""
