"""Candidate Select — pick one candidate out of a generated batch.

Step 1 renders N candidates from consecutive seeds and you choose one. Core's
Latent From Batch can slice the batch, but it only hands back a latent: it does
not tell you WHICH seed you just chose, and a seed is the whole identity of a
candidate — the thing you write down, re-run months later, or send to someone
else. This node returns the latent, its preview, and that seed together.

The seed arithmetic mirrors Seed Range Noise: candidate i was drawn from
`origin_seed + i`, so selecting index i reports `origin_seed + i`. Wire the same
seed that fed the noise node and the number that comes out is directly re-usable
in any stock workflow.

Both of a sampler's latent outputs can be sliced at once: pass `denoised` too
and the same index comes back on both. That is what lets a chosen candidate be
branched again — Noise Rotate needs the pair, and one ◀ ▶ choice should not have
to be repeated on a second node that could drift out of step with this one.

`batch_index` is stamped on the emitted latent, exactly as Latent From Batch
does. Nothing downstream needs it while a continuation runs on DisableNoise, but
the moment noise IS regenerated from this latent, that stamp is what keeps the
candidate on its own seed instead of silently sliding to another.

The widget has two views. The GRID shows every candidate at once, for the
comparison the choice actually is; clicking one selects it and opens the SINGLE
view, which shows that candidate large — a thumbnail is enough to tell two
compositions apart, never enough to judge one. The single view carries ◀ ▶ so
the batch can be walked at full size, and those buttons move the selection with
the view: a cursor that pointed somewhere other than `index` would be this
project's founding bug in miniature, a picture that has moved while the number
under it has not.

Index is 0-based on the wire and 1-based in the readout, which keeps
`seed = origin_seed + index` free of an off-by-one. Out-of-range
clamps rather than raising, because a cursor that refuses to move is worse than
one that stops at the end, and the readout always shows where you actually are.
"""

from __future__ import annotations

import os
import random

import numpy as np
from PIL import Image

from ...base import GteNode, first
from ...registry import register


# Two sizes per candidate. The grid draws 56px tiles, so a tile that arrived at
# full resolution would cost megabytes per run to render a stamp; the single view
# is a real look at the picture and 320px would be a blurry insult. Publishing
# both keeps the grid cheap AND the inspection honest, because the browser only
# fetches a full image for the ONE candidate actually being viewed.
_THUMB = 320
_FULL = 1024


def _publish(images):
	"""Write the batch to ComfyUI's temp dir so the widget can show it.

	The same route PreviewImage uses (`/view?...&type=temp`), rather than this
	pack's RAM store: these are small stills, the store exists for multi-hundred
	-megabyte clips, and temp is already swept by ComfyUI.

	Each candidate is published twice — a `filename` thumbnail for the grid and a
	`full` copy for the single view — plus the full copy's pixel size, so the
	viewer can shape its box to the real aspect ratio instead of letterboxing a
	portrait candidate into a square.

	folder_paths is imported here rather than at module scope: it only exists
	with ComfyUI's root on the path, and the test suite imports this module
	without it.
	"""
	import folder_paths  # noqa: PLC0415 — runtime-only, see above

	out = folder_paths.get_temp_directory()
	os.makedirs(out, exist_ok=True)
	tag = "".join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))
	published = []
	for i in range(images.shape[0]):
		arr = np.clip(images[i].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
		im = Image.fromarray(arr)

		full = im.copy()
		full.thumbnail((_FULL, _FULL), Image.LANCZOS)
		full_name = f"gte_cand_{tag}_{i:03d}_full.png"
		full.save(os.path.join(out, full_name), compress_level=4)

		# thumbnail() mutates in place, so this must come after the full copy.
		im.thumbnail((_THUMB, _THUMB), Image.LANCZOS)
		name = f"gte_cand_{tag}_{i:03d}.png"
		im.save(os.path.join(out, name), compress_level=4)

		published.append({"filename": name, "subfolder": "", "type": "temp",
						  "full": full_name, "width": full.width, "height": full.height})
	return published


@register
class CandidateSelect(GteNode):
	DISPLAY_NAME = "Candidate Select (gte)"
	CATEGORY = "gte/sampling"
	# Run even when the downstream is cached, so the readout still reports the
	# batch size and the chosen seed after a re-queue.
	OUTPUT_NODE = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"latents": ("LATENT", {"tooltip":
					"The candidate batch — the sampler output from Step 1."}),
				"index": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1,
					"tooltip": "Which candidate to keep. Use the ◀ ▶ buttons; "
							   "the readout counts from 1 the way the previews do."}),
			},
			"optional": {
				"denoised": ("LATENT", {"tooltip":
					"The matching `denoised_output` batch. Sliced at the same index, "
					"so the pair stays together for a further Noise Rotate."}),
				"images": ("IMAGE", {"tooltip":
					"The decoded previews, so the chosen one can be routed on its own."}),
				"origin_seed": ("INT", {"default": 0, "min": 0,
					"max": 0xffffffffffffffff, "tooltip":
					"The seed that fed Seed Range Noise. Candidate i came from "
					"origin_seed + i, so this is what makes `seed` meaningful."}),
			},
		}

	RETURN_TYPES = ("LATENT", "LATENT", "IMAGE", "INT", "INT")
	RETURN_NAMES = ("latent", "denoised", "image", "seed", "index")
	OUTPUT_TOOLTIPS = (
		"The chosen candidate, carrying its batch_index.",
		"Its denoised_output, when `denoised` was supplied — wire both into Noise Rotate.",
		"Its preview, when `images` was supplied.",
		"origin_seed + index — the candidate's portable identity.",
		"The index actually used, after clamping.",
	)
	FUNCTION = "execute"

	def execute(self, latents, index=0, denoised=None, images=None, origin_seed=0):
		samples = latents["samples"]
		n = samples.shape[0]
		if n == 0:
			raise RuntimeError("Candidate Select: the batch is empty — nothing to choose.")

		i = max(0, min(int(first(index, 0)), n - 1))
		seed = int(first(origin_seed, 0)) + i

		out = latents.copy()
		out["samples"] = samples[i:i + 1].clone()
		if "noise_mask" in latents:
			mask = latents["noise_mask"]
			out["noise_mask"] = (mask.clone() if mask.shape[0] == 1
								 else mask[i:i + 1].clone())
		# Keep the candidate addressable: a later noise regeneration reads this.
		existing = latents.get("batch_index")
		out["batch_index"] = [existing[i]] if existing else [i]

		picked_denoised = None
		if denoised is not None:
			dn = denoised["samples"]
			if dn.shape[0] != n:
				raise RuntimeError(
					f"Candidate Select: `denoised` holds {dn.shape[0]} latents but "
					f"`latents` holds {n}. They must be the two outputs of the SAME "
					f"sampler run, or the index would point at different candidates."
				)
			picked_denoised = denoised.copy()
			picked_denoised["samples"] = dn[i:i + 1].clone()

		image = None
		if images is not None:
			image = images[i:i + 1] if images.shape[0] > i else images[:1]

		# The grid needs every candidate, not just the chosen one — that is the
		# point of picking by eye instead of by number.
		try:
			thumbs = _publish(images) if images is not None else []
		except Exception as exc:  # noqa: BLE001
			# The grid is a convenience; losing it must never cost you the pick.
			print(f"[gte] Candidate Select: preview grid unavailable ({exc!r})")
			thumbs = []
		base = int(first(origin_seed, 0))

		print(f"[gte] Candidate Select: {i + 1}/{n}  seed {seed}")
		return {"ui": {"gte_candidate": [{"index": i, "count": n, "seed": seed,
										 "origin_seed": base, "thumbs": thumbs}]},
				"result": (out, picked_denoised, image, seed, i)}
