"""Schedule Info — how much noise is left at every step, before you spend a run.

A step number is not a noise level, and the gap between the two is where staged
diffusion goes wrong. The SCHEDULER decides how the denoising is distributed
across the steps, and two schedulers with identical step counts put wildly
different amounts of noise at the same index. Measured on an 8-step
rectified-flow schedule:

    step        karras   kl_optimal   simple      exponential
       1         54.1%        79.8%    87.5%            41.2%
       5          2.2%        23.0%    37.5%             1.2%
       7          0.2%         0.2%    12.5%             0.2%

Branching at "step 5" therefore means something completely different depending
on a dropdown three groups away. Under karras there is 2% of the image left to
vary and the branch cannot work; under simple there is 37%. Nothing on the
canvas says so, and the failure is silent: the descendants come back looking
like damaged copies of the parent rather than alternatives.

So this node reads the schedule and says it out loud. Wire the same SIGMAS the
stages use, look at the column, and put the checkpoints where the noise is.

`marks` highlights your checkpoints in the table, so the question "is there
anything left to branch on at step 5" is answered by reading one line.
"""

from __future__ import annotations

from ...base import GteNode, first
from ...registry import register


def describe(sigmas, marks=()) -> str:
	"""The schedule as a table: noise remaining, and how much each step removes."""
	vals = [float(s) for s in sigmas]
	if len(vals) < 2:
		raise RuntimeError(
			f"Schedule Info: need at least two sigmas to describe a schedule, "
			f"got {len(vals)}."
		)
	top = max(vals) or 1.0
	marks = {int(m) for m in marks}

	rows = [f"{'step':>5}{'sigma':>10}{'noise left':>12}{'this step':>11}  "]
	for i, s in enumerate(vals):
		left = 100.0 * s / top
		# What this step actually removes — the flat rows at the end are why a
		# late checkpoint has nothing to offer.
		drop = (100.0 * (vals[i - 1] - s) / top) if i else 0.0
		tag = "  <-- checkpoint" if i in marks else ""
		rows.append(f"{i:>5}{s:>10.4f}{left:>11.1f}%{drop:>10.1f}%{tag}")
	return "\n".join(rows)


@register
class ScheduleInfo(GteNode):
	DISPLAY_NAME = "Schedule Info (gte)"
	CATEGORY = "gte/sampling"
	# Runs even when nothing downstream needs it: this is a readout, and a
	# readout that only appears when something else happens to want it is no use.
	OUTPUT_NODE = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"sigmas": ("SIGMAS", {"tooltip":
					"The FULL schedule — the scheduler's output, before any Sigma "
					"Segment slicing. Percentages are relative to its first value."}),
			},
			"optional": {
				"marks": ("STRING", {"default": "", "tooltip":
					"Checkpoint steps to highlight, comma separated (e.g. \"1,5,7\"). "
					"Just for reading; it changes nothing."}),
			},
		}

	RETURN_TYPES = ("STRING",)
	RETURN_NAMES = ("table",)
	OUTPUT_TOOLTIPS = (
		"The table as text — wire it into Show Text to keep it on the canvas.",
	)
	FUNCTION = "execute"

	def execute(self, sigmas, marks=""):
		raw = str(first(marks, "") or "")
		want = []
		for part in raw.replace(";", ",").split(","):
			part = part.strip()
			if part:
				try:
					want.append(int(float(part)))
				except ValueError:
					# A typo in a decoration must not cost you the run.
					print(f"[gte] Schedule Info: ignoring unreadable mark {part!r}")
		table = describe(sigmas, want)
		print("[gte] Schedule Info:\n" + table)
		return {"ui": {"text": [table]}, "result": (table,)}
