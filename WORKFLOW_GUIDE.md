# Generative Trajectory Explorer — how to use it

A normal generation runs from pure noise to a finished image in one go. You pay
full price before you know whether the seed was any good.

This workflow stops part-way, lets you LOOK, and then continues only the ones you
picked — branching into variations at each stop. You pay full price only for the
branches you keep.

```
STEP A   DISCOVER    0 -> A     many seeds, cheap. Pick a direction.
STEP B   BRANCH      A -> B     vary the one you picked. Pick again.
STEP C   BRANCH      B -> C     vary again, more finely. Pick again.
FINAL                C -> end   finish the one you chose.
```

The unit of work is **a partial latent plus the noise level it sits at**. A latent
without its sigma is meaningless — that idea is the reason everything here is
built the way it is.

---

## The one thing to understand first

Every control is either **WHERE you stop**, **HOW FAR you move**, or **WHICH WAY
you move**. They are three different questions and no control answers two of them.

| question | control | |
|---|---|---|
| where do I stop? | Checkpoint A / B / C | positions on the schedule |
| how far do I move? | **theta** | the only magnitude dial |
| which way do I move? | **variation seed** | picks a direction, no magnitude |
| how do the siblings fan out? | **spread** | shares out what theta allocated |

---

## GLOBAL CONTROLS

### Total Steps
How long the full schedule is. Everything else is measured against it. 20 is a
good working value; raise it for quality, lower it for speed.

### Checkpoint A, B, C
The absolute steps where the workflow stops. They must increase:
`0 < A < B < C < Total`. There is ONE schedule and each stage takes a slice of
it, so changing Total Steps moves everything consistently — nothing to rewire.

**Pick these by noise level, not by step number.** The same step number means
wildly different things under different schedulers. Read the *Noise left at each
step* table on the canvas and put your checkpoints where there is still something
left to work with. At 20 steps on `kl_optimal`:

```
step  5  ->  65% noise left     A: the image is still a rough direction
step 10  ->  39% noise left     B: composition committing, rendering open
step 15  ->  17% noise left     C: layout fixed, only interpretation left
```

Branch too late and there is nothing left to vary. Branch too early and you are
not varying your choice, you are re-rolling it — at high noise the model has
barely decided anything yet, so rotating it produces a different picture rather
than a variation of yours.

### Width / Height
Round to a multiple of **16** on Flux2, 8 on SD1.5. Anything else gets silently
rounded down at decode time (910 comes back as 896).

### Sampler
**euler. Do not change this.** Splitting a trajectory only works for samplers
whose next step depends on `(x, sigma)` alone. `dpmpp_2m` carries derivative
history and degrades; `dpmpp_2m_sde` injects its own noise and breaks outright.
euler, heun and ddim are safe; euler is what is tested.

### Scheduler — why it matters more here than usual

In a normal generation you never see the inside of the trajectory, so the
scheduler is a mild quality preference. Here the inside IS the interface, and the
scheduler is the map from **step number -> noise level**. Your checkpoints are
step numbers, so the scheduler decides what they mean:

    step 10 is 39% noise on kl_optimal, and 88% on simple.
    Same widget, same graph, completely different image.

It also decides **how many steps land in the usable band** — enough noise left to
branch on, enough structure to judge by eye. Flux2, 20 steps:

| scheduler | usable steps | where A/B/C go | noise there |
|---|---|---|---|
| `kl_optimal` | 15 of 20 | 5 / 10 / 15 | 65% / 39% / 17% |
| `simple` | 8 of 20 | 16 / 18 / 19 | 65% / 46% / 28% |
| `karras` | 7 of 20 | 2 / 4 / 7 | 61% / 36% / 15% |

**Any scheduler works.** Nothing in the workflow assumes a schedule shape —
Sigma Segment slices whatever it is handed, continuation is exact because euler
is memoryless, and Noise Rotate uses the real sigma, never the step number. So a
front-loaded scheduler is not broken; it just squeezes your choices into fewer
steps and wastes the rest of the compute.

`kl_optimal` is the default here because it spreads the noise most evenly on a
flow model, which buys the most usable checkpoints per step paid for. If you
change it, read the table and move the checkpoints — that is the whole fix.

**Check it yourself — the table at the bottom of the graph.** *Noise left at each
step* recomputes for whichever scheduler is selected. Per step it shows the
sigma, the noise still left, and how much that step removes:

```
 step     sigma  noise left  this step
    5    0.6536       65.4%       6.1%  <-- checkpoint
   10    0.3907       39.1%       4.8%  <-- checkpoint
   15    0.1675       16.7%       4.3%  <-- checkpoint
```

The `this step` column is the one to watch. Flat all the way down means the work
is spread evenly and every step is a usable place to stop. A column that runs
22%, 14%, 6%, 1%, 0%, 0% is telling you the back half of the schedule is doing
nothing — move the checkpoints forward, or pick a scheduler that spreads better.

Set `marks` on that node to your checkpoint numbers and they get flagged in the
table, so "is step 10 a sensible place to stop" is one line to read.

Two things to know: `simple` puts the checkpoints so late that the FINAL stage
gets one step to go from ~28% noise to zero, which can cost quality. And
`linear_quadratic` is built for video models (Mochi/LTX) — on an image model it
has no usable steps at all.

### Use negative prompt?
- **ON for SD1.5/SDXL.** CFG does real work there.
- **OFF for Flux / krea2.** Guidance is distilled into the model; CFG on top is
  wrong. Turning it off also skips the text encode entirely, so it is faster.

### Flux guidance (flux2 only)
Flux2's built-in guidance scale, default 3.5. **Does nothing on krea2, SD1.5 or
SDXL** — those models never read it. Harmless to leave wired.

---

## SEEDS — three of them, and they must stay separate

- **Origin seed (Step A)** — which family of candidates Step A discovers.
- **Variation seed (Step B)** — which direction Step B's branches go.
- **Variation seed (Step C)** — which direction Step C's branches go.

They are separate because the whole workflow is "I found a good one, now re-roll
the branches". One shared seed would make that impossible: re-rolling Step C
would also re-roll Step B, changing C's parent and destroying the branch you were
happy with.

**Keep a variation seed different from the origin seed.** If they match, the new
noise is literally the same tensor the candidate was born from — you ask for 45
degrees and get 16, and the noise level inflates by up to 38%, which the next
stage then under-denoises into a soft render. Nothing errors.

**Seed distance means nothing.** Seed 501 is exactly as unrelated to 500 as
9e18 is. There is no "nearby seed". Seeds pick a point; theta sets how far away
that point is.

---

## PER-STAGE CONTROLS

### How many samples?
How many images this stage makes. Step A makes candidates from consecutive seeds
(`origin`, `origin+1`, `origin+2`, …). Steps B and C make descendants of the one
you picked.

Cost is linear, so this is your main speed dial. Many cheap candidates early,
fewer expensive ones late, is the whole point.

### How much variation? — **theta**
`0 – 90` in practice (the widget allows -360…360).

**theta is the distance from the parent**, in degrees. It is the only magnitude
control in the workflow.

```
 0      an EXACT reproduction of the parent's continuation. The control.
10-45   the useful working range
90      a fresh direction at the same noise level
```

theta is literally the angle: `cos_sim(parent, child) = cos(theta)`, accurate to a
fraction of a degree.

Past 90 the children keep moving away from the parent while collapsing back
together, so variety stops improving. At 180 the variation seed stops mattering
entirely and every descendant is the same image.

**Match theta to depth.** At a shallow checkpoint (lots of noise left) the model
has not committed to a composition, so a large theta re-rolls the picture instead
of varying it. Deeper down, the layout is fixed and a large theta varies the
rendering while the composition holds. So: modest theta early, bolder theta late.

### How spread apart? — **spread**
`0 – 90`, default `90`.

theta says how far each child is from the **parent**. spread says how far the
children are from **each other**.

```
spread 90   children fan out maximally     (what you want most of the time)
spread 45   a tighter family
spread  0   every child identical — you pay N renders for one image
```

Without spread these two were locked together: asking to move further forced the
family apart with it. "Go somewhere new, but keep this family tight" is what
spread makes expressible.

Sibling separation for a given theta:

```
spread ->    0     10     20     30     45     60     75     90
theta 25   0.0    5.9   11.7   17.2   24.4   30.0   33.6   34.8
theta 45   0.0   10.0   19.7   29.0   41.4   51.3   57.8   60.0
theta 60   0.0   12.2   24.2   35.7   51.3   64.1   72.5   75.5
                               (degrees between siblings)
```

**Why theta = 0 makes spread do nothing.** Picture the parent as the north pole.
theta is the latitude line every child stands on; spread is how far apart they
are spaced along it. At theta = 0 that circle has shrunk to the pole — six people
spaced maximally around a zero-radius circle all stand on the same spot.

It is the triangle inequality, not a quirk: two points each at angle *p* from a
common point can be at most *2p* apart. It is also required — theta = 0 being an
exact reproduction is the baseline that lets you say "this branch is identical to
not branching".

So the ceiling on how different the siblings can be is set by **theta**. spread
only shares out the room theta has opened.

### Keep the parent direction?
When ON, the first descendant is the parent's own continuation, bit-exact, and
the rest are variations around it.

Useful because a single theta cannot give you both: every child sits the same
distance from the parent, so the faithful continuation can't be one of them. This
lets exactly one child sit at distance zero instead.

### The candidate cursor (◀ Prev / Next ▶)
Click through the results, or click a tile to see it large. The buttons only move
the `index` widget — choosing is an ordinary graph edit, so **re-queue** when you
have picked. The readout says `(re-queue)` when your cursor has moved off the
candidate that actually ran, so a stale selection is never mistaken for a live one.

---

## Skipping stages

Each stage can be bypassed. The latent carries its own position — a stamp written
by the stage that produced it — so the next stage always reads where the latent
*actually* is, not where the wiring assumes it is. Skip the middle stage and you
get a valid shorter tree rather than a silent mis-render.

Turning Step A off cascades to B and C, because a branch with no parent is
meaningless. With everything off, an empty latent falls through to the FINAL
render, which becomes an ordinary one-shot generation.

---

## Reading the run

The console prints what actually happened:

```
[gte] Resume Step: latent resumes at step 5      <- where this stage began
[gte] Stamp Step: latent is at step 10           <- where it ended
[gte] Candidate Select: 2/20  seed 701           <- which one you kept
[gte] Noise Rotate: 15 descendant(s) at theta=25deg from seed 500
```

If the numbers disagree with your checkpoint controls, trust the console — it is
reporting the latent, and the latent is the thing being sampled.

---

## Quick recipes

**"Show me lots of options, cheaply."**
Total 20, A 5. Samples 20+. You are only paying 5 steps per candidate.

**"I like this one, show me nearby versions."**
theta 15–25, spread 90, keep_parent ON.

**"I like the layout but not the treatment."**
Branch at C (low noise). theta 30–45, spread 90.

**"Move somewhere new but keep the family tight."**
theta 45, spread 20–30.

**"Nothing is changing."**
theta too low, or you branched too late and there is no noise left to vary.
Check the noise table.

**"It changed completely / went off-prompt."**
theta too high for that depth. Lower it, or branch deeper.

**"The results look soft or over-worked."**
The noise level and the step number disagree. Usually the wrong scheduler
(`karras` on a flow model), or a variation seed that collides with the origin seed.

---

## Canvas note blocks

Short versions to paste into Note nodes, one per region.

**GLOBAL CONTROLS**
> One schedule, sliced. Checkpoints are absolute steps and must increase.
> Pick them from the noise table, not by step number. Sampler must stay euler —
> splitting a trajectory only works for a memoryless sampler.

**SEEDS**
> Three independent seeds so you can re-roll one stage without disturbing the
> others. Keep the variation seeds different from the origin seed. Seed distance
> carries no meaning — seeds pick a direction, theta sets the distance.

**STEP A — DISCOVER**
> Cheap candidates from consecutive seeds. No theta here: there is no parent to
> move away from yet. Click a tile, then re-queue.

**STEP B / STEP C — BRANCH**
> theta = how far from the parent (0 = exact copy, 10–45 useful, 90 = fresh
> direction at the same noise level).
> spread = how widely the children fan out around that distance (90 = max,
> 0 = all identical).
> theta sets the size of the circle; spread sets where on it each child stands.
> At theta 0 the circle is a point, so spread does nothing — that is the exact-
> reproduction control working.
> keep_parent = make child 1 the faithful continuation.

**FINAL**
> Runs the chosen branch to the end. The only stage that decodes the real output
> rather than the model's current guess.

**PLUMBING / SIGMA VISUALIZATION**
> Never bypass. These feed every stage; a bypassed Get node has no inputs, so it
> breaks the chain instead of passing through.
