"""Auto-discovery registry for gte custom nodes.

Why this exists: ComfyUI loads nodes from two module-level dicts,
NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS. Maintaining those by
hand is error-prone, and node IDs are GLOBAL across every installed pack — a
duplicate silently shadows another author's node.

Instead: each node class is tagged with @register. The package __init__
calls discover() which imports every module under nodes/, running each
decorator. IDs are namespaced with ID_PREFIX to avoid global collisions,
and duplicates raise loudly instead of shadowing.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType
from typing import Type

# Global namespace prefix for every node ID this pack publishes.
# Keeps our IDs from colliding with core nodes or other custom packs.
ID_PREFIX = "GTE_"

NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}


def register(cls: Type) -> Type:
	"""Class decorator: add a node to the global ComfyUI mappings.

	Node ID resolution order:
	  1. explicit cls.NODE_ID
	  2. class name
	The ID_PREFIX is prepended unless already present.

	Display name resolution order:
	  1. explicit cls.DISPLAY_NAME
	  2. class name
	"""
	raw_id = getattr(cls, "NODE_ID", None) or cls.__name__
	full_id = raw_id if raw_id.startswith(ID_PREFIX) else ID_PREFIX + raw_id

	if full_id in NODE_CLASS_MAPPINGS:
		existing = NODE_CLASS_MAPPINGS[full_id]
		raise ValueError(
			f"Duplicate node id {full_id!r}: "
			f"{cls.__module__}.{cls.__qualname__} clashes with "
			f"{existing.__module__}.{existing.__qualname__}"
		)

	NODE_CLASS_MAPPINGS[full_id] = cls
	NODE_DISPLAY_NAME_MAPPINGS[full_id] = getattr(cls, "DISPLAY_NAME", None) or cls.__name__
	cls._GTE_FULL_ID = full_id
	return cls


def discover(package: ModuleType) -> None:
	"""Recursively import every submodule of `package` so @register fires.

	Import errors in a single node module are caught and reported but do not
	abort loading the rest of the pack — one broken node should not take the
	whole UI down.
	"""
	for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
		try:
			importlib.import_module(info.name)
		except Exception as exc:  # noqa: BLE001 — isolate per-module failures
			print(f"[gte] failed to load {info.name}: {exc!r}")
