#!/usr/bin/env python
# Adapted from GR00T-Dreams (Apache-2.0): scripts/idm_training.py, gr00t/experiment/{runner_idm,trainer}.py.
"""Train the shared multi-embodiment IDM (paper App. B.2.6).

    torchrun --standalone --nproc_per_node=4 scripts/train_idm.py \
        --config configs/idm/da2l_flowmatch.yaml data.roots.openarm_rh56f1=/data/openarm ...

Each embodiment's LeRobot v2.1 dataset is normalised with its own statistics; embodiments are
mixed by ``weight`` (1/3 each). Checkpoints (``checkpoint-<step>/``) and the final model hold
``model.safetensors``, ``idm_config.json`` and ``metadata.json`` (per-embodiment specs and
action statistics, needed for labelling).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import Sampler
from transformers import Trainer, TrainingArguments
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from robotryon.common.config import parse_args, save_config
from robotryon.idm.data import LeRobotIDMDataset, MixtureDataset, collate, embodiment_specs
from robotryon.idm.model import IDM


class EpochSampler(Sampler):
    """Seeded shuffle that re-draws the mixture table every epoch."""

    def __init__(self, ds: MixtureDataset, seed: int):
        self.ds, self.seed, self.epoch = ds, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.ds.set_epoch(epoch)

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.ds), generator=g).tolist())

    def __len__(self):
        return len(self.ds)


class IDMTrainer(Trainer):
    def __init__(self, *a, metadata: dict, **k):
        self.metadata = metadata
        super().__init__(*a, **k)

    def _get_train_sampler(self, *_):
        return EpochSampler(self.train_dataset, self.args.seed)

    def compute_loss(self, model, inputs, return_outputs=False, **_):
        loss = model(inputs)
        return (loss, {"loss": loss}) if return_outputs else loss

    def create_optimizer(self):
        """AdamW, no weight decay on biases and LayerNorm weights."""
        if self.optimizer is None:
            decay = {n for n in get_parameter_names(self.model, ALL_LAYERNORM_LAYERS) if "bias" not in n}
            params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
            groups = [{"params": [p for n, p in params if n in decay], "weight_decay": self.args.weight_decay},
                      {"params": [p for n, p in params if n not in decay], "weight_decay": 0.0}]
            cls, kw = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = cls(groups, **kw)
        return self.optimizer

    def save_model(self, output_dir=None, _internal_call=False):
        if self.args.should_save:
            self.accelerator.unwrap_model(self.model).save_pretrained(
                output_dir or self.args.output_dir, self.metadata)


def main():
    cfg = parse_args(__doc__)
    tc, dc = cfg.train, cfg.data
    specs = embodiment_specs(dc)
    names = dc.use or list(specs)
    datasets = [LeRobotIDMDataset(specs[n], cfg.model, dc, train=True) for n in names]
    train_ds = MixtureDataset(datasets, [specs[n]["weight"] for n in names], seed=tc.seed)
    metadata = {n: d.metadata() for n, d in zip(names, datasets)}
    print({n: len(d) for n, d in zip(names, datasets)}, flush=True)

    model = IDM(cfg.model, pretrained_vision=True)
    out = Path(tc.output_dir)
    if int(os.environ.get("RANK", 0)) == 0:
        out.mkdir(parents=True, exist_ok=True)
        save_config(cfg, out / "config.yaml")

    args = TrainingArguments(
        output_dir=str(out), per_device_train_batch_size=tc.batch_size,
        gradient_accumulation_steps=tc.grad_accum_steps, max_steps=tc.max_steps,
        learning_rate=tc.learning_rate, lr_scheduler_type=tc.lr_scheduler, warmup_ratio=tc.warmup_ratio,
        weight_decay=tc.weight_decay, adam_beta1=tc.adam_betas[0], adam_beta2=tc.adam_betas[1],
        adam_epsilon=1e-8, max_grad_norm=tc.max_grad_norm, optim="adamw_torch",
        bf16=tc.bf16, tf32=True, seed=tc.seed, logging_steps=10, save_strategy="steps",
        save_steps=tc.save_steps, save_total_limit=8, report_to=tc.report_to,
        dataloader_num_workers=tc.num_workers, dataloader_persistent_workers=tc.num_workers > 0,
        dataloader_pin_memory=False, remove_unused_columns=False, ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100, num_train_epochs=300)
    trainer = IDMTrainer(model=model, args=args, train_dataset=train_ds, data_collator=collate,
                         metadata=metadata)
    if tc.resume:
        # Trainer's rng_state.pth holds numpy objects that torch>=2.6 refuses by default.
        _load = torch.load
        torch.load = lambda *a, **k: _load(*a, **{"weights_only": False, **k})
    trainer.train(resume_from_checkpoint=True if tc.resume else None)
    trainer.save_model(str(out))


if __name__ == "__main__":
    main()
