"""Base class for gte nodes.

ComfyUI does not require a base class — it duck-types on INPUT_TYPES,
RETURN_TYPES and FUNCTION. This base just supplies sane defaults so each
node file stays small, and gives one place to add shared helpers later
(logging, error wrapping, common type constants, etc.).
"""

from __future__ import annotations


def first(v, default=None):
	"""Unwrap the first element of an INPUT_IS_LIST argument.

	With INPUT_IS_LIST every input arrives wrapped in a list, including scalar
	widgets — so a widget that reads 8 arrives as [8]. Nodes that consume the
	whole batch still want the scalar. Non-list values pass straight through, so
	this is safe to call whether or not the node sets INPUT_IS_LIST.
	"""
	if isinstance(v, list):
		return v[0] if v else default
	return v


# What a linked BOOLEAN can plausibly arrive as. Widget clicks give a real bool;
# anything wired in can be a string, and Python's own bool() is dangerously
# wrong about those — bool("false") and bool("0") are both True.
_TRUE = {"true", "1", "yes", "on", "y", "t"}
_FALSE = {"false", "0", "no", "off", "n", "f", ""}


def as_bool(v, default=False, *, where="value"):
	"""A BOOLEAN input as a real bool, whether it came from the widget or a link.

	ComfyUI validates LITERAL widget values only — the check sits in the else
	branch of the is-link test in execution.py — so a value arriving over a wire
	reaches the node exactly as the upstream node emitted it. For an INT that
	means an unclamped number; for a BOOLEAN it means possibly the STRING
	"false", which `bool()` happily reports as True. A toggle that silently
	turns itself on is the worst kind of bug this pack can ship, so parse rather
	than cast, and refuse anything genuinely ambiguous instead of guessing.
	"""
	v = first(v, default)
	if v is None:
		return bool(default)
	if isinstance(v, bool):          # before int: bool IS an int in Python
		return v
	if isinstance(v, (int, float)):
		return v != 0
	if isinstance(v, str):
		s = v.strip().lower()
		if s in _TRUE:
			return True
		if s in _FALSE:
			return False
		raise RuntimeError(
			f"{where}: expected a true/false value, got {v!r}. Wire a boolean, "
			f"or one of {sorted(_TRUE)} / {sorted(_FALSE)}."
		)
	raise RuntimeError(
		f"{where}: expected a true/false value, got {type(v).__name__} ({v!r})."
	)


class GteNode:
	# --- registry metadata (read by @register) -------------------------
	# Override per node. NODE_ID defaults to the class name when None.
	NODE_ID: str | None = None
	DISPLAY_NAME: str | None = None

	# --- ComfyUI contract defaults -------------------------------------
	# Right-click menu path. Convention: "gte/<group>".
	CATEGORY = "gte"
	# Name of the instance method ComfyUI calls to run the node.
	FUNCTION = "execute"

	# Sensible empty defaults so an unfinished node still loads as a no-op
	# instead of crashing the pack at import time. Real nodes override both.
	RETURN_TYPES: tuple = ()

	@classmethod
	def INPUT_TYPES(cls):  # noqa: N802 — ComfyUI requires this exact name
		return {"required": {}}
