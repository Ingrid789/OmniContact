"""Export the inference-only actor and verify parity with the training policy."""

import argparse
import os
from pathlib import Path
import tempfile

import torch
from torch import nn

from .model import Actor


class DeploymentActor(nn.Module):
    def __init__(self, actor, flow_only):
        super().__init__()
        self.actor = actor
        self.flow_only = flow_only

    def forward(self, reference, history):
        if self.flow_only:
            reference = torch.cat((reference[..., :49], torch.zeros_like(reference[..., 49:])), -1)
        return self.actor(reference, history).clamp(-10., 10.)


def export_actor(checkpoint, output):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    actor = Actor(first_width=cfg["encoder_width"], decoder_width=cfg["decoder_width"])
    actor.load_state_dict(saved["models"][1])
    wrapper = DeploymentActor(actor.eval(), cfg.get("goal_mode", "full") == "flow").eval()
    del saved
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator().manual_seed(37)
    def example(batch):
        return (torch.randn(batch, 11, 113, generator=rng), torch.randn(batch, 10, 141, generator=rng))
    with torch.inference_mode(), tempfile.TemporaryDirectory(dir=output.parent) as folder:
        module = torch.jit.freeze(torch.jit.trace(wrapper, example(1)))
        path = Path(folder) / "actor.pt"
        module.save(str(path))
        loaded = torch.jit.load(str(path)).eval()
        for batch in (1, 4):
            inputs = example(batch)
            torch.testing.assert_close(loaded(*inputs), wrapper(*inputs), rtol=2e-5, atol=2e-6)
        os.replace(path, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Trusted training checkpoint")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export_actor(args.checkpoint, args.output)
