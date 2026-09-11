// Render "Show Text · Lines (gte)"'s lines inside the node.
//
// The Python node returns {"ui": {"text": ["1. a\n2. b"]}}; onExecuted drops
// that into a scrollable read-only panel mounted on the node, so several values
// are readable at a glance on the graph.

import { app } from "../../scripts/app.js";

function panel(node) {
	if (node._gte_lines) return node._gte_lines;
	const el = document.createElement("div");
	el.style.cssText = [
		"width:100%", "height:100%", "box-sizing:border-box", "padding:6px 8px",
		"font:12px/1.5 ui-monospace,monospace", "color:#e6e6e6",
		"background:#181818", "border-radius:4px", "overflow:auto",
		"white-space:pre-wrap", "word-break:break-word",
	].join(";");
	el.textContent = "run to display";
	node.addDOMWidget("gte_lines", "gte_lines", el, {
		serialize: false, hideOnZoom: false, getMinHeight: () => 120,
	});
	node._gte_lines = el;
	return el;
}

app.registerExtension({
	name: "gte.showTextLines",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== "GTE_ShowTextLines") return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated?.apply(this, arguments);
			panel(this);
			this.setSize([Math.max(this.size[0], 300), Math.max(this.size[1], 200)]);
			return r;
		};

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			const t = message?.text;
			if (t && t.length) {
				panel(this).textContent = Array.isArray(t) ? t.join("\n") : String(t);
				this.setDirtyCanvas?.(true, true);
			}
		};
	},
});
