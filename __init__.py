"""gte — Generative Trajectory Explorer nodes for ComfyUI.

ComfyUI imports this package (the folder dropped into custom_nodes/) and reads
three names: NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS and WEB_DIRECTORY.
Everything else is auto-discovered — see registry.py.
"""

from . import nodes, registry

# Walk nodes/ and run every @register decorator.
registry.discover(nodes)

# Names ComfyUI looks for at the top level of a custom_node package.
NODE_CLASS_MAPPINGS = registry.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = registry.NODE_DISPLAY_NAME_MAPPINGS

# Serve frontend extensions from ./web (JS that runs in the ComfyUI browser).
WEB_DIRECTORY = "./web"

__all__ = [
	"NODE_CLASS_MAPPINGS",
	"NODE_DISPLAY_NAME_MAPPINGS",
	"WEB_DIRECTORY",
]

print(f"[gte] loaded {len(NODE_CLASS_MAPPINGS)} node(s)")
