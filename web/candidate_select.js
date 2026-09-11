// Click-to-pick grid + single-candidate viewer for "Candidate Select (gte)".
//
// Choosing a candidate is a visual judgement, so the node shows the candidates
// rather than a number. It has two views:
//
//   GRID    every candidate at once — the comparison the choice actually is.
//           Click one and it is selected AND opened in the single view.
//   SINGLE  that candidate large, the way Preview Image shows an image, with
//           ◀ ▶ to walk the batch without shrinking back down, and Back to
//           return to the grid.
//
// A thumbnail is enough to tell two compositions apart and never enough to
// judge one, which is why clicking has to do more than move a number.
//
// The ◀ ▶ buttons move the SELECTION, not a separate viewing cursor. Two
// cursors would let the picture on screen disagree with the `index` that the
// graph will actually run — the same class of mistake as a latent whose sigma
// stayed behind. Selecting is still an ordinary graph edit either way, so
// nothing runs until you re-queue.
//
// The grid is whatever the last run produced. When the cursor sits on a
// candidate the run did not cover (index typed by hand, or a re-queue pending),
// the header says so instead of implying the picture below is current.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "GTE_CandidateSelect";

/** The widget object by name, or undefined. */
function getWidget(node, name) {
	return node.widgets?.find((w) => w.name === name);
}

// Tiles are a fixed small size rather than a fraction of the node. With `1fr`
// columns a wide node grew the thumbnails until the grid outgrew the widget's
// reserved height and spilled past the node border. A candidate only has to be
// recognisable, so cap it and let the row wrap instead.
const TILE = 56;
const GAP = 4;
const HEAD_H = 24;
const BTN_H = 26;
// Ceiling on the single view's image box. Without one, a tall candidate on a
// wide node would grow the node taller than the screen the moment it is opened.
const VIEW_MAX_H = 420;
const VIEW_MIN_H = 120;
// Litegraph gives the DOM widget a content-sized box, so `width:100%` inside it
// resolves against nothing and fixed-width grid tracks push the widget WIDER
// than the node instead of wrapping. Size it from node.size[0] instead, which is
// the only authoritative width, and never measure our own (possibly overflowing)
// element to decide the layout.
const PAD = 26;

// How many tiles fit across the node as it is RIGHT NOW. Fixing the column
// count instead (repeat(5, ...)) forced everything onto one row and ran off the
// edge as soon as the batch outgrew the node's width.
//   n columns occupy n*TILE + (n-1)*GAP, so n <= (width + GAP) / (TILE + GAP)
function innerWidth(node) {
	return Math.max(TILE, Math.round(node.size?.[0] ?? 300) - PAD);
}

function colsFor(node, n) {
	//   n columns occupy n*TILE + (n-1)*GAP,  so  n <= (width + GAP) / (TILE + GAP)
	const fit = Math.floor((innerWidth(node) + GAP) / (TILE + GAP));
	return Math.max(1, Math.min(n, fit));
}

function cur(node) { return Math.round(getWidget(node, "index")?.value ?? 0); }

/** The view mode lives in node.properties so it is SERIALISED with the graph:
 *  reopening a workflow puts the node back in the view you left it in. The DOM
 *  widget itself is serialize:false, so it cannot remember anything. */
function mode(node) {
	return node.properties?.gte_view === "single" ? "single" : "grid";
}

/** Fetch every full-size image once, so Prev/Next is a cache hit rather than a
 *  round trip. Only on entering the single view — doing it on every execute
 *  would pull the full batch for people who never leave the grid. The array is
 *  retained so the Image objects are not collected before they land. */
function warmFullImages(node) {
	const t = node._tcs;
	if (!t || !t.state.thumbs.length) return;
	t.warm = t.state.thumbs.map((th) => {
		const im = new Image();
		im.decoding = "async";
		im.src = urlFor(t, th, true);
		return im;
	});
}

function setMode(node, m) {
	node.properties = node.properties || {};
	node.properties.gte_view = m;
	if (m === "single") warmFullImages(node);
	paint(node);
	node.setDirtyCanvas?.(true, true);
}

/** Move the selection, clamped to the batch. Clamping rather than wrapping
 *  matches the node's own out-of-range behaviour: a cursor that stops at the
 *  end is honest, one that silently jumps to the other end is not. */
function setIndex(node, v) {
	const w = getWidget(node, "index");
	if (!w) return;
	const count = node._tcs?.state.count || 0;
	const hi = count ? count - 1 : 99999;
	w.value = Math.max(0, Math.min(Math.round(v), hi));
	// The callback is wrapped in setup() to repaint, so this must not paint
	// again: a second pass would reassign the image src and make it blink.
	w.callback?.(w.value, app.canvas, node);
	node.setDirtyCanvas?.(true, true);
}

function urlFor(t, th, full) {
	const name = (full && th.full) || th.filename;
	return api.apiURL(`/view?filename=${encodeURIComponent(name)}`
		+ `&type=${th.type}&subfolder=${encodeURIComponent(th.subfolder || "")}`
		+ `&t=${t.state.stamp}`);
}

function headline(node) {
	const t = node._tcs;
	const idx = cur(node);
	const { count, origin } = t.state;
	if (!count) return "queue once to see the candidates";
	const shown = Math.min(idx, count - 1);
	return `${shown + 1} / ${count}`
		+ (origin !== null ? `   ·   seed ${origin + shown}` : "")
		+ (idx >= count ? "   ·   out of range" : "");
}

/** The height this widget needs for the view it is currently in. Drives
 *  getMinHeight, so litegraph reserves the right box instead of clipping the
 *  last grid row or leaving a gap under a small image. */
function neededHeight(node) {
	const t = node._tcs;
	if (!t) return HEAD_H + TILE + 10;
	if (mode(node) === "single") {
		return HEAD_H + 6 + viewHeight(node) + 4 + BTN_H + 4;
	}
	return HEAD_H + 6 + Math.max(1, t.rows || 1) * (TILE + 4);
}

/** How tall the single view's image box should be: the candidate's own aspect
 *  ratio at the node's current width, capped. Sizing to the REAL ratio is why
 *  the backend sends width/height — a portrait candidate letterboxed into a
 *  square box would waste half the room you opened the view to get. */
function viewHeight(node) {
	const t = node._tcs;
	const th = t.state.thumbs[Math.min(cur(node), t.state.thumbs.length - 1)];
	const w = innerWidth(node);
	if (!th || !th.width || !th.height) return Math.min(VIEW_MAX_H, Math.max(VIEW_MIN_H, w));
	const h = Math.round(w * (th.height / th.width));
	return Math.min(VIEW_MAX_H, Math.max(VIEW_MIN_H, h));
}

function paintGrid(node) {
	const t = node._tcs;
	const idx = cur(node);
	const thumbs = t.state.thumbs;

	t.grid.replaceChildren();
	if (!thumbs.length) return;
	const cols = colsFor(node, thumbs.length);
	// Fixed-width columns: the row fills up, then wraps to the next line.
	t.grid.style.gridTemplateColumns = `repeat(${cols}, ${TILE}px)`;
	thumbs.forEach((th, i) => {
		const cell = document.createElement("div");
		const on = i === idx;
		cell.style.cssText = "position:relative;border-radius:4px;overflow:hidden;cursor:pointer;"
			+ `min-width:0;height:${TILE}px;box-sizing:border-box;`
			+ `border:2px solid ${on ? "#7fd1ff" : "#2a2f3a"};`
			+ (on ? "box-shadow:0 0 0 2px rgba(127,209,255,.35);" : "opacity:.72;");
		const img = document.createElement("img");
		img.src = urlFor(t, th, false);
		img.style.cssText = "width:100%;height:100%;display:block;object-fit:cover;";
		const tag = document.createElement("div");
		tag.textContent = i + 1;
		tag.style.cssText = "position:absolute;left:3px;top:2px;font:600 10px sans-serif;"
			+ "color:#fff;text-shadow:0 1px 3px #000;";
		cell.append(img, tag);
		cell.onmouseenter = () => { if (i !== idx) cell.style.opacity = "1"; };
		cell.onmouseleave = () => { if (i !== idx) cell.style.opacity = ".72"; };
		// One click both picks the candidate and opens it: the two are the same
		// intention. setMode paints, so the index is set without painting twice.
		cell.onclick = (e) => {
			e.stopPropagation();
			const w = getWidget(node, "index");
			// Set the value WITHOUT its repaint (setMode paints), because this
			// handler is running inside the very grid a repaint would replace.
			if (w) { w.value = i; }
			setMode(node, "single");
			w?.callback?.(i, app.canvas, node);
		};
		t.grid.append(cell);
	});
}

function paintSingle(node) {
	const t = node._tcs;
	const thumbs = t.state.thumbs;
	const i = Math.min(cur(node), thumbs.length - 1);
	const th = thumbs[i];

	t.view.style.height = viewHeight(node) + "px";
	// object-fit:contain, not cover: the grid may crop a tile to keep the row
	// tidy, but the view exists to show the picture and must not hide any of it.
	t.img.style.display = t.thumbImg.style.display = th ? "block" : "none";
	if (th) {
		// The THUMBNAIL is already in the browser cache — the grid fetched it —
		// so it paints in the same frame as the click. Show it upscaled
		// underneath and let the full image fade in on top when it arrives.
		// Navigation then feels immediate even when the full fetch is not,
		// which over a remote proxy is most of the time.
		const small = urlFor(t, th, false);
		if (t.thumbImg.getAttribute("src") !== small) t.thumbImg.setAttribute("src", small);

		const want = urlFor(t, th, true);
		// Only reassign on a real change — writing the same src restarts the
		// fetch and makes the image blink on every ◀ ▶ repaint.
		if (t.img.getAttribute("src") !== want) {
			t.img.style.opacity = "0";
			t.img.onload = () => { t.img.style.opacity = "1"; };
			t.img.setAttribute("src", want);
		} else if (t.img.complete) {
			t.img.style.opacity = "1";
		}
	}
	const count = t.state.count || 0;
	t.prev.disabled = i <= 0;
	t.next.disabled = count === 0 || i >= count - 1;
	for (const b of [t.prev, t.next]) {
		b.style.opacity = b.disabled ? ".4" : "1";
		b.style.cursor = b.disabled ? "default" : "pointer";
	}
}

function paint(node) {
	const t = node._tcs;
	if (!t) return;
	const single = mode(node) === "single" && t.state.thumbs.length > 0;

	t.head.textContent = headline(node);

	// Pin the box to the node's width so nothing inside can widen it.
	const inner = innerWidth(node) + "px";
	t.wrap.style.width = inner;
	t.wrap.style.maxWidth = inner;
	t.grid.style.width = inner;
	t.grid.style.maxWidth = inner;
	t.view.style.width = inner;
	t.view.style.maxWidth = inner;

	t.grid.style.display = single ? "none" : "grid";
	t.view.style.display = single ? "flex" : "none";
	t.row.style.display = single ? "flex" : "none";

	// Row count decides the node height, and Back must not cause a one-frame
	// collapse — but that number is arithmetic, so compute it without touching
	// the DOM and rebuild the grid ONLY when it is visible and actually stale.
	// Rebuilding it on every Prev/Next re-created six <img> elements behind a
	// display:none, which is most of what made navigation feel slow.
	const n = t.state.thumbs.length;
	t.rows = n ? Math.ceil(n / colsFor(node, n)) : 0;
	const key = `${t.state.stamp}|${colsFor(node, n)}|${cur(node)}`;
	if (!single && key !== t.gridKey) {
		t.gridKey = key;
		paintGrid(node);
	}
	if (single) paintSingle(node);

	// Switching view, or a different candidate count, needs a different height.
	// Let litegraph re-measure (it consults getMinHeight) and grow the node if
	// the content no longer fits. Growing is the normal case; shrinking would
	// fight a manual resize, so it happens only when LEAVING the single view,
	// where the tall box we reserved is demonstrably no longer needed and
	// keeping it would strand a large empty rectangle under a two-row grid.
	//
	// A manual height set in grid mode is remembered across the trip rather than
	// lost to that shrink: you get your node back, not the minimum one.
	const want = neededHeight(node);
	if (want !== t.lastWant || (single ? "single" : "grid") !== t.lastMode) {
		const leaving = t.lastMode === "single" && !single;
		if (t.lastMode === "grid" && single) t.gridH = node.size[1];
		t.lastWant = want;
		t.lastMode = single ? "single" : "grid";
		const size = node.computeSize?.();
		if (size) {
			if (leaving) node.setSize([node.size[0], Math.max(size[1], t.gridH || 0)]);
			else if (node.size[1] < size[1]) node.setSize([node.size[0], size[1]]);
		}
		node.setDirtyCanvas?.(true, true);
	}
}

function setup(node) {
	if (node._tcs) return;
	const wrap = document.createElement("div");
	// overflow:hidden is the backstop — whatever the grid does, it cannot paint
	// outside the box litegraph reserved for this widget.
	wrap.style.cssText = "display:flex;flex-direction:column;gap:4px;padding:2px;"
		+ "box-sizing:border-box;width:100%;overflow:hidden;";

	const head = document.createElement("div");
	head.style.cssText = "font:11px monospace;color:#cfe3ff;background:#1b1f27;"
		+ `border:1px solid #333a46;border-radius:4px;padding:3px 6px;text-align:center;`
		+ `height:${HEAD_H}px;box-sizing:border-box;flex:0 0 auto;`
		+ "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";

	const grid = document.createElement("div");
	// justify-content:start keeps a short row left-aligned instead of stretching.
	grid.style.cssText = "display:grid;gap:4px;justify-content:start;"
		+ "align-content:start;overflow-y:auto;overflow-x:hidden;";

	const view = document.createElement("div");
	view.style.cssText = "display:none;position:relative;align-items:center;"
		+ "justify-content:center;background:#11141a;border:1px solid #2a2f3a;"
		+ "border-radius:4px;box-sizing:border-box;overflow:hidden;flex:0 0 auto;";
	const layer = "position:absolute;inset:0;width:100%;height:100%;display:block;"
		+ "object-fit:contain;";
	// Underneath: the cached thumbnail, upscaled. Soft, but instant.
	const thumbImg = document.createElement("img");
	thumbImg.style.cssText = layer + "filter:blur(1px);";
	// On top: the real thing, revealed once decoded.
	const img = document.createElement("img");
	img.style.cssText = layer + "opacity:0;transition:opacity .12s linear;";
	view.append(thumbImg, img);

	const row = document.createElement("div");
	row.style.cssText = `display:none;gap:5px;height:${BTN_H}px;flex:0 0 auto;`;
	const mk = (text, title, onClick, grow) => {
		const b = document.createElement("button");
		b.textContent = text;
		b.title = title;
		b.style.cssText = `flex:${grow || 1};padding:0;border:1px solid #3a4150;`
			+ "border-radius:4px;background:#2a2f3a;color:#cfe3ff;cursor:pointer;"
			+ "font:600 12px sans-serif;";
		b.onclick = (e) => { e.stopPropagation(); if (!b.disabled) onClick(); };
		return b;
	};
	const back = mk("⊞ Back", "Back to the candidate grid", () => setMode(node, "grid"), 1.1);
	const prev = mk("◀ Prev", "Select the previous candidate", () => setIndex(node, cur(node) - 1));
	const next = mk("Next ▶", "Select the next candidate", () => setIndex(node, cur(node) + 1));
	row.append(back, prev, next);

	wrap.append(head, grid, view, row);
	node._tcs = { wrap, head, grid, view, img, thumbImg, row, back, prev, next,
				  rows: 0, lastWant: -1, lastMode: null, gridH: 0, gridKey: "", warm: [],
				  state: { count: 0, origin: null, thumbs: [], stamp: 0 } };
	node.addDOMWidget("candidate_ui", "gte_candidate_ui", wrap, {
		serialize: false, hideOnZoom: false,
		// Ask for exactly the height the current view needs, so the node neither
		// clips the last row nor reserves empty space for candidates it lacks.
		getMinHeight: () => neededHeight(node),
	});

	const w = getWidget(node, "index");
	if (w) {
		const prevCb = w.callback;
		w.callback = function (...a) { const r = prevCb?.apply(this, a); paint(node); return r; };
	}
	requestAnimationFrame(() => paint(node));
}

app.registerExtension({
	name: "gte.candidateSelect",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== NODE_TYPE) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated?.apply(this, arguments);
			setup(this);
			// Wide enough for a full row of tiles, tall enough for one row of them.
			this.setSize([Math.max(this.size[0], 300), Math.max(this.size[1], 180)]);
			return r;
		};

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			const m = message?.gte_candidate;
			if (m && m.length && this._tcs) {
				this._tcs.state = {
					count: m[0].count || 0,
					origin: m[0].origin_seed ?? null,
					thumbs: m[0].thumbs || [],
					// Filenames are reused across runs; bust the browser cache.
					stamp: Date.now(),
				};
				if (mode(this) === "single") warmFullImages(this);
				// The view mode is left alone on purpose. This node re-runs on
				// EVERY queue (OUTPUT_NODE), including queues that were about
				// the downstream stages, so snapping back to the grid each time
				// would fight whoever is inspecting a candidate.
				paint(this);
			}
		};

		// Dragging the node's edge changes how many tiles fit, and rescales the
		// single view's image box — so rebuild then. Watching our own element
		// instead would just observe the width we set.
		const onResize = nodeType.prototype.onResize;
		nodeType.prototype.onResize = function (...a) {
			const r = onResize?.apply(this, a);
			if (this._tcs) paint(this);
			return r;
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			const r = onConfigure?.apply(this, arguments);
			if (this._tcs) requestAnimationFrame(() => paint(this));
			return r;
		};
	},
});
