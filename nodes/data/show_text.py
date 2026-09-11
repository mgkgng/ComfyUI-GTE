"""Show Text · Lines — display several texts, one per line, in the node.

Wire up to six strings (or a LIST from a per-item node) and it prints them
stacked in the node body, numbered, so you can read a batch of values at a
glance instead of opening six separate previews.

INPUT_IS_LIST, so a list-producing upstream node shows every element as its own
line rather than firing the node once per item.
"""

from __future__ import annotations

from ...base import GteNode
from ...registry import register


def _flatten(value):
	"""One socket's value -> a flat list of strings ([] when nothing is wired)."""
	if value is None:
		return []
	items = value if isinstance(value, list) else [value]
	out = []
	for v in items:
		if v is None:
			continue
		if isinstance(v, list):
			out.extend(str(x) for x in v if x is not None)
		else:
			out.append(str(v))
	return out


@register
class ShowTextLines(GteNode):
	DISPLAY_NAME = "Show Text · Lines (gte)"
	CATEGORY = "gte/data"
	OUTPUT_NODE = True
	INPUT_IS_LIST = True

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"optional": {
				f"text_{i}": ("STRING", {"forceInput": True, "tooltip":
					f"Line {i}. A LIST here becomes one line per element."})
				for i in range(1, 7)
			},
		}

	RETURN_TYPES = ("STRING",)
	RETURN_NAMES = ("text",)
	OUTPUT_TOOLTIPS = ("Every line joined with newlines, to chain onward.",)
	FUNCTION = "execute"

	@classmethod
	def IS_CHANGED(cls, **kwargs):
		return float("nan")      # always re-display, even when the inputs repeat

	def execute(self, **kwargs):
		lines = []
		for i in range(1, 7):
			lines.extend(_flatten(kwargs.get(f"text_{i}")))
		body = "\n".join(f"{n}. {t}" for n, t in enumerate(lines, 1)) if lines else "(nothing wired)"
		print(f"[gte] Show Text · Lines:\n{body}")
		return {"ui": {"text": [body]}, "result": ("\n".join(lines),)}
