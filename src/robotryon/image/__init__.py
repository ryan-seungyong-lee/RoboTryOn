"""Keyframe translation f_theta (paper Sec. 3.2): FLUX.2 [klein] 9B Base as a reference-guided
inpainting model.

Modules:
  fill_channels  widened patch embedding, channel-concatenated [noisy | conditioned frame | mask]
  stereo_ids     RoPE ids with the camera view on the 4th axis
  episodes       data layout access and model-resolution geometry
  dataset        training samples (reference window, robot-only exemplar, masks, skeleton)
  text           prompts, caption dropout and the text-embedding cache
  training       flow-matching loss
  model          FFT / LoRA build, save and load
  sampler        true-CFG Euler sampler returning latent tokens
  ar_keyframes   autoregressive translation with latent reuse
"""
