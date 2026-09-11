"""Stamp Step / Resume Step — let a partial latent say where it is.

A latent that has been sampled part-way is meaningless without the sigma it sits
at. Everything else in this pack takes that seriously; the graph did not. A
staged workflow carries the position on a SEPARATE wire — the stage's checkpoint
widget — while the latent itself says nothing. Two facts, two places, free to
disagree.

They disagree the moment a stage is skipped. Bypass Step C and ComfyUI passes
the latent through correctly (the bypass rule matches inputs to outputs by slot
position), so the FINAL stage receives Step B's latent — but its `start_step` is
still wired to Checkpoint C. The sampler is told the latent holds 34% noise when
it holds 69%, under-denoises, and returns a soft, still-noisy image with no
error anywhere. Nothing is broken enough to complain.

A stamp fixes that by making the latent self-describing.

WHAT A STAMP IS
---------------
A ComfyUI LATENT is a plain dict: `{"samples": tensor}` plus whatever else rides
along. Core already does this — `batch_index` from Latent From Batch,
`noise_mask` from the inpaint nodes — and every node that passes a latent on
rebuilds it with `latent.copy()`, so extra keys survive untouched. Verified for
the whole chain this pack uses: SamplerCustomAdvanced (nodes_custom_sampler.py,
`out = latent.copy()`), Candidate Select, and Noise Rotate.

So stamping is just:

    latent["gte_step"] = 11

The number travels WITH the tensor it describes, the way EXIF travels with a
photo, instead of alongside it on a wire that can be rerouted, bypassed or left
pointing at the wrong stage. It cannot get out of sync, because there is only
one copy of the fact.

WHY THIS BEATS A SWITCH
-----------------------
The obvious fix is a switch that picks the right checkpoint number depending on
which stages are enabled. That works, but it means the graph encodes the
enabled-stage set twice — once in the bypass state, once in the switch wiring —
and those two can drift apart.

A stamp needs no such knowledge. Bypass a stage and its Stamp Step goes with it,
so the latent arrives still carrying whichever stage last actually ran, and the
next stage reads the truth. Correct for every combination of enabled stages,
including ones nobody planned for: skipping the middle stage becomes a valid
two-branch tree rather than a silent mis-render.

USE
---
    end of a stage:    sampler output -> Stamp Step (step = this stage's end)
    start of a stage:  incoming latent -> Resume Step -> Sigma Segment.start_step

`end_step` still comes from the stage's own control. Only `start_step` is read
from the latent, because only the latent knows where it actually is.
"""

from __future__ import annotations

from ...base import GteNode, first
from ...registry import register

# The dict key the stamp lives under. Namespaced so it cannot collide with a key
# core might add later, and so it is obvious in a debugger which pack wrote it.
STAMP_KEY = "gte_step"


SIGMA_KEY = "gte_sigma"


def stamped_sigma(latent: dict):
	"""The noise level a latent says it is at, or None if nothing stamped one."""
	value = latent.get(SIGMA_KEY)
	return None if value is None else float(value)


def stamp(latent: dict, step: int, sigma=None) -> dict:
	"""Return a copy of `latent` carrying `step`, or raise if that goes backwards.

	A copy rather than a mutation: the input latent may be feeding other nodes,
	and a stage stamping its own number onto a tensor someone else is still
	reading is the kind of action-at-a-distance this pack tries not to have.
	"""
	step = int(step)
	if step < 0:
		raise ValueError(f"Stamp Step: step must be >= 0, got {step}.")
	if sigma is not None:
		sigma = float(sigma)
		if sigma < 0.0:
			raise ValueError(f"Stamp Step: sigma must be >= 0, got {sigma}.")

	previous = latent.get(STAMP_KEY)
	if previous is not None and step < int(previous):
		# Sampling only ever moves forward through the schedule, so a stage
		# claiming to end BEFORE the latent already is means the checkpoints are
		# out of order. Catch it here, where the numbers are still on screen.
		raise ValueError(
			f"Stamp Step: this latent is already at step {previous}, but the stage "
			f"claims to end at {step}. A stage cannot finish earlier than it began "
			f"— check that the checkpoint controls increase along the chain."
		)

	out = latent.copy()
	out[STAMP_KEY] = step
	if sigma is not None:
		# The step says WHERE on the schedule; the sigma says HOW MUCH noise is
		# actually left there. They are not interchangeable: the same step number
		# means wildly different noise levels under different schedulers (karras
		# reaches 2% noise by step 5 of 8 where a linear flow schedule still has
		# 37%). Anything reconstructing a state needs the sigma, not the index.
		out[SIGMA_KEY] = sigma
	return out


def stamped_step(latent: dict):
	"""The step a latent says it is at, or None if nothing has stamped it."""
	value = latent.get(STAMP_KEY)
	return None if value is None else int(value)


@register
class StampStep(GteNode):
	DISPLAY_NAME = "Stamp Step (gte)"
	CATEGORY = "gte/sampling"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"latent": ("LATENT", {"tooltip":
					"The latent this stage is handing on — normally the sampler's "
					"`output`."}),
				"step": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip":
					"The absolute step this stage ended on: the same number as its "
					"Sigma Segment `end_step`. Wire the stage's checkpoint control "
					"to both and they cannot drift apart."}),
			},
			"optional": {
				# Optional so every existing graph keeps working unchanged; a
				# latent with no sigma stamp simply cannot be varied by the
				# model-aware path, and Noise Rotate says so rather than guessing.
				"sigma": ("FLOAT", {"forceInput": True, "tooltip":
					"The noise level this stage ended at — wire Sigma Segment's "
					"`end_sigma`. The latent then carries it onward, so the next "
					"stage's Noise Rotate knows how much noise it is looking at "
					"without anything else being wired. A step number alone is not "
					"enough: the same step means different noise on different "
					"schedulers."}),
			},
		}

	RETURN_TYPES = ("LATENT",)
	RETURN_NAMES = ("latent",)
	OUTPUT_TOOLTIPS = (
		"The same latent, now carrying its position for the next stage to read.",
	)
	FUNCTION = "execute"

	def execute(self, latent, step, sigma=None):
		sig = first(sigma, None)
		out = stamp(latent, int(first(step, 0)), None if sig is None else float(sig))
		where = f"step {out[STAMP_KEY]}"
		if SIGMA_KEY in out:
			where += f", sigma {out[SIGMA_KEY]:.4f}"
		else:
			where += " (no sigma — wire Sigma Segment `end_sigma` if this latent "
			where += "will be branched)"
		print(f"[gte] Stamp Step: latent is at {where}")
		return (out,)


@register
class ResumeStep(GteNode):
	DISPLAY_NAME = "Resume Step (gte)"
	CATEGORY = "gte/sampling"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"latent": ("LATENT", {"tooltip":
					"The latent this stage is resuming from. Its stamp says where "
					"in the schedule it actually sits."}),
			},
			"optional": {
				"fallback": ("INT", {"default": 0, "min": 0, "max": 10000,
					"tooltip":
					"Used when the latent carries no stamp. 0 is right for a fresh "
					"Empty Latent — it has not been sampled, so it is at the start "
					"of the schedule. Set it if you are resuming a latent loaded "
					"from disk, which will have lost its stamp."}),
			},
		}

	RETURN_TYPES = ("INT",)
	RETURN_NAMES = ("start_step",)
	OUTPUT_TOOLTIPS = (
		"Where this latent is — wire it to Sigma Segment's `start_step`.",
	)
	FUNCTION = "execute"

	def execute(self, latent, fallback=0):
		step = stamped_step(latent)
		if step is None:
			step = int(first(fallback, 0))
			# Not an error: an Empty Latent genuinely is at step 0. Said out loud
			# anyway, because the other way to get here is a latent that lost its
			# stamp, and that is worth noticing before the render looks wrong.
			print(f"[gte] Resume Step: latent carries no stamp, using fallback {step}")
		else:
			print(f"[gte] Resume Step: latent resumes at step {step}")
		return (step,)
