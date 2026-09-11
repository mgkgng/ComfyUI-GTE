"""Seed Range Noise — batch noise where item i is drawn from seed + i.

Why this exists: sampling a batch of N draws ONE noise tensor from a single
seeded generator and slices it N ways (comfy/sample.py: prepare_noise). Batch
item i therefore has no seed of its own — its identity is the pair
(seed, batch_index), and that pair only reproduces inside a batch of the same
size. A candidate you liked is then awkward to write down, to re-run on its
own, or to hand to someone else.

This reseeds per item instead: item i gets a full, independent generator at
seed + i. Its noise is then bit-identical to what a plain batch_size=1 run at
seed + i would draw, so a candidate's identity collapses back to a single
integer that means the same thing in any stock workflow, with no batch context
to carry around.

What that does NOT promise is a pixel-identical image when you later re-run a
candidate alone. The noise going in is exact, but a UNet forward pass at batch
5 and at batch 1 pick different GPU kernels and reduction orders, and three
CFG-guided steps amplify the difference. Measured here: re-running candidate 2
alone landed 2.2/255 mean pixel difference from its batched original, against
38-51 for every other candidate in the batch — unmistakably the same image, not
the same bytes. Sampling IS deterministic at a fixed batch size (repeat runs
diff by 0), so exact reproduction means re-running at the same batch size, or
continuing from the saved latent instead of regenerating.

Feed the NOISE output to SamplerCustomAdvanced. Plain KSampler builds its own
noise internally and has no NOISE input, so it cannot use this.
"""

from __future__ import annotations

import torch

from ...base import GteNode
from ...registry import register


def seeds_for(latent: dict, seed: int) -> list[int]:
	"""The seed each batch item is drawn from, in batch order.

	A latent that has been through Latent From Batch carries "batch_index" —
	the positions it was sliced out of. Honouring it keeps a picked candidate
	on the same seed after the pick as it was before, which is what makes
	"select one, then continue it" reproduce rather than drift.
	"""
	inds = latent.get("batch_index")
	if inds is None:
		return [seed + i for i in range(latent["samples"].shape[0])]
	return [seed + int(i) for i in inds]


class Noise_SeedRange:
	"""The NOISE object handed to SamplerCustomAdvanced.

	ComfyUI's NOISE contract is one method — generate_noise(latent) -> tensor
	(see comfy_extras/nodes_custom_sampler.py: Noise_RandomNoise).
	"""

	def __init__(self, seed: int):
		self.seed = seed

	def generate_noise(self, input_latent: dict):
		samples = input_latent["samples"]
		if getattr(samples, "is_nested", False):
			raise RuntimeError(
				"Seed Range Noise does not support nested latents. "
				"Use the core RandomNoise node for this model."
			)

		# One item at a time, each from its own generator — this is the whole
		# point, and it is also what makes each item match a solo run. A private
		# generator rather than torch.manual_seed() keeps the global RNG intact;
		# the values are identical either way, since they depend only on the seed.
		single = (1,) + tuple(samples.shape[1:])
		noises = [
			torch.randn(
				single,
				dtype=torch.float32,
				layout=samples.layout,
				generator=torch.Generator(device="cpu").manual_seed(s),
				device="cpu",
			)
			for s in seeds_for(input_latent, self.seed)
		]
		# Stays on CPU like core's prepare_noise — the sampler moves it.
		return torch.cat(noises, dim=0).to(dtype=samples.dtype)


@register
class SeedRangeNoise(GteNode):
	DISPLAY_NAME = "Seed Range Noise (gte)"
	CATEGORY = "gte/sampling"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"seed": ("INT", {
					"default": 0,
					"min": 0,
					# Leave headroom so seed + batch cannot overflow the field.
					"max": 0xffffffffffffffff - 0xffff,
					"control_after_generate": True,
				}),
			},
		}

	RETURN_TYPES = ("NOISE",)
	RETURN_NAMES = ("noise",)
	FUNCTION = "execute"

	def execute(self, seed):
		return (Noise_SeedRange(seed),)
