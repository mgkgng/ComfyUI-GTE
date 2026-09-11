# Generative Trajectory Explorer

A ComfyUI node pack for **stopping a generation part-way, looking at it, and
branching from there.**

A normal generation runs from pure noise to a finished image in one go — you pay
full price before you know whether the seed was any good. These nodes let you
stop at intermediate checkpoints, inspect the candidates, keep the directions
worth continuing, and create controlled variations from them. You only pay full
price for the branches you keep.

```
STEP A   DISCOVER    0 -> A     many seeds, cheap. Pick a direction.
STEP B   BRANCH      A -> B     vary the one you picked. Pick again.
STEP C   BRANCH      B -> C     vary again, more finely. Pick again.
FINAL                C -> end   finish the one you chose.
```

The unit of work is **a partial latent plus the noise level it sits at**. A
latent without its sigma is meaningless, and that idea is why every node here is
built the way it is.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mgkgng/ComfyUI-GTE
```

Restart ComfyUI. No extra Python dependencies — everything runs on the torch,
numpy and Pillow that ComfyUI already ships.

### The example workflow needs three other packs

The **nodes** above have no dependencies beyond ComfyUI itself. The example
*workflow* is a different matter — it is built with helpers from:

| pack | used for |
|---|---|
| [rgthree-comfy](https://github.com/rgthree/rgthree-comfy) | `Any Switch`, `Power Lora Loader`, `KSampler Config`, the group bypasser and the mute/bypass relays |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | `SetNode` / `GetNode` — the virtual wires that carry the shared controls into every stage |
| [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use) | `easy int`, `easy seed`, `easy ifElse` (the lazy guider switch) |

Install those if you want the shipped graph. If you are only after the nodes,
you need none of them.

The loaders ship **empty** — pick your own checkpoint, or UNET + CLIP + VAE.
Nothing in these nodes is tied to a model family. Set the CLIP `type` to match
whatever you load.

Then load `workflows/GTE-example.json` and read `WORKFLOW_GUIDE.md`, or the
notes on the canvas itself.

## The nodes

| node | what it does |
|---|---|
| **Seed Range Noise** | a batch where item *i* uses `seed + i` as a full independent reseed, so each candidate has a real, reproducible identity |
| **Sigma Segment** | slices an **absolute** range out of one immutable schedule: `sigmas[start : end+1]`. Adjacent segments share their boundary sigma, which is what makes continuation seamless |
| **Stamp Step / Resume Step** | writes the position onto the latent itself, so the next stage reads where the latent *actually* is rather than where the wiring assumes |
| **Candidate Select** | a cursor over a batch: click a tile, see it large, keep one. Reports that candidate's own seed |
| **Noise Rotate** | controlled descendants of a partial latent — `theta` sets how far from the parent, `spread` how widely the siblings fan out |
| **Schedule Info** | how much noise is left at every step, before you spend a run on it |
| **Show Text · Lines** | puts that table on the canvas |

## Two ideas worth knowing

**Rotate the noise, never add to it.** `x + d·eps` inflates the noise level by
`sqrt(1+d^2)` — +41% at d=1 — while the scheduler still believes the old sigma.
The sampler then under-denoises, and variation strength and quality loss end up
tangled in one dial with no error anywhere. Rotating preserves `||x - x0||`
exactly, so `theta` moves the direction and never the amount.

**Ask the model, never name it.** `x - x0` is the noise only for additive models
(SD1.5/SDXL). Rectified flow (Flux, SD3, krea) mixes noise in instead, so the
same subtraction yields noise *and* signal. There is no list of model families
anywhere in this pack: Noise Rotate probes the model's own noise scaling twice
to recover its noise coordinate, which works for every parameterisation ComfyUI
ships and for ones that do not exist yet.

## Constraints, measured

- **euler only.** Splitting a trajectory works only for samplers whose step
  depends on `(x, sigma)` alone. `dpmpp_2m` carries derivative history and
  degrades; `dpmpp_2m_sde` injects its own noise and breaks outright. euler,
  heun and ddim are safe.
- **Branch where there is noise left to branch on.** Read the Schedule Info
  table rather than trusting step numbers — the same step means very different
  things under different schedulers.
- **Keep the variation seeds different from the origin seed.** If they collide,
  the "new" noise is the tensor the candidate was born from, and both the angle
  and the noise level quietly stop meaning what they say.

## Tests

```bash
ComfyUI/venv/bin/python tests/test_gte.py     # standalone, no pytest needed
pytest tests/                                  # if you have it
```

Every test corresponds to something that actually broke.

## A note on node IDs

Node IDs are prefixed `GTE_` (`GTE_NoiseRotate`, `GTE_SigmaSegment`, …) and the
shipped workflow refers to them by those IDs. If you have graphs saved against
an earlier build of these nodes under a different prefix, they will not resolve
— open them, and re-add the affected nodes.

## License

MIT
