"""Noise Rotate — controlled descendants of a partial latent.

You stopped a generation part-way, looked at it, and want to see several ways it
could still go. A partial latent splits into what the model has already decided
and what is still undecided:

    x  =  x0_pred  +  residual

`x0_pred` is the model's current guess at the finished image — the part you
picked. The residual is the noise it has not resolved yet, and that is where the
remaining freedom lives. So: keep x0_pred, and turn the residual toward a
different answer.

    eps     = x - x0_pred
    new     = randn(variation_seed) * eps.std()
    eps_var = cos(theta) * eps + sin(theta) * new
    variant = x0_pred + eps_var

Rotating rather than adding is the whole point. `x + d*eps'` inflates the noise
level (by sqrt(1+d^2)), so the continuation would run a schedule that expects
less noise than it gets and quietly under-denoise — variation strength and
quality loss confounded in one dial. Because cos^2 + sin^2 = 1, this leaves
||x - x0_pred|| exactly unchanged: theta moves the direction, never the amount.
Scaling `new` to the residual's OWN magnitude rather than assuming a unit
variance is what keeps that true at any checkpoint, and means the node needs
neither sigma nor a latent-space conversion.

theta is an angle, and it is exactly the cosine similarity between the old
residual and the new one: cos_sim(eps, eps_var) = cos(theta). That identity holds
because `new` is drawn independently of `eps`, and two independent gaussians in
D dimensions are near-orthogonal (cos ~ 1/sqrt(D), about 0.6% at 4x113x64). So
the number on the widget is the angle actually turned, to within a fraction of a
degree — and it gets more exact at higher resolution, not less.

    theta = 0    exact continuation (the control)
    10-45        the useful working range
    90           fresh direction at the same noise level, x0_pred still kept

theta steers TWO things at once, and they are not the same number. The angle
from the parent is theta. The angle between two SIBLINGS is wider — at the
default spread of 90:

    angle(variant_i, variant_j) = acos(cos^2 theta)

    theta:           0    10     30     45     60     90    120    150    180
    from parent:   0.0  10.1   30.2   45.3   60.2   90.0  119.8  149.8  180.0
    between sibs:  0.0  14.3   41.8   60.4   75.9   90.4   76.0   41.8    0.0

Siblings are always further from each other than from the parent, and their
spread PEAKS at 90 and then closes again. That is the real reason 90 is the
working maximum: past it the descendants keep marching away from the parent
while collapsing back together, and at 180 the term sin(theta)*new vanishes
entirely — `variation_seed` stops mattering, every descendant is the same
image, and that image is x0_pred minus the residual, the exact negative of the
continuation. At 360 you are back at the original. Values beyond 90 are allowed
because they are interesting to look at, not because they produce more variety.

`spread` separates those two distances. Without it, how different the children
are from EACH OTHER is locked to how far they are from the parent, and there is
no way to ask for "move somewhere new, but keep this family tight". With it,
every child still sits exactly theta from the parent while spread decides how
widely they fan out around a shared family direction:

    spread ->       0      10      20      30      45      60      75      90
    theta  25      0.0     5.9    11.7    17.2    24.4    30.0    33.6    34.8
    theta  45      0.0    10.0    19.7    29.0    41.4    51.3    57.8    60.0
    theta  60      0.0    12.2    24.2    35.7    51.3    64.1    72.5    75.5
                                    (sibling angle in degrees)

spread = 90 is exactly what this node did before it existed, which is why it is
the default. spread = 0 collapses the family to one image repeated `count`
times. Above 90 is a mirror (100 behaves as 80), so the range stops there —
unlike theta, where past 90 is genuinely different.

Note the ceiling is set by theta: spread can only distribute the room theta has
opened. At theta 25 the children can never be more than 34.8 degrees apart.

Negative angles are a free extra axis: cos is even and sin is odd, so -theta has
the identical strength as +theta but is a different descendant.

The node prints what each branch had to work with. Two very different failures
look identical on the canvas -- descendants that come out the same -- and only
the numbers separate them: either the residual barely matters any more (the
model has committed, nothing left to turn) or the latents WERE separated and
the model collapsed them on the way to an image. Distilled/turbo models do the
second: they are trained to jump to their answer and to ignore the noise they
are handed, which is exactly the property this node needs.

SD1.5 reference, "normal" scheduler, 12 total steps, measured on this pack:

    branch at step:      2      4      6      9     11
    residual share:  97.7%  89.3%  71.5%  21.6%   0.1%

These are an EPS-PREDICTION model and do NOT transfer to rectified flow. SD1.5
adds noise with an unbounded sigma so the share runs to ~100%; krea2/Flux mix it
in, so the share cannot exceed roughly 50% however noisy the state is. Read the
share against other checkpoints of the SAME model, never another model's table.

## Which vector gets rotated depends on the model

`x - x0` is the noise ONLY for additive models (x = x0 + sigma*eps). A mixture
model -- rectified flow: Flux, SD3, krea -- builds its state as
x = (1-sigma)*x0 + sigma*eps, so that same subtraction yields sigma*(eps - x0),
noise AND signal together. Rotating it drags the signal off the trajectory:
measured at theta 90, the x0 component came back at 1.00 where 0.125 was valid,
and the implied noise at 3.5x what the schedule expects. The sampler then
resumes on a state that is not on its path and cannot clean it, so the branch
renders as a slightly-perturbed version of where it started rather than
developing. No error anywhere -- the same silent shape as the stamp bug.

Wire `model` + `current_sigma` and the node stops guessing. It asks the model
how IT builds a noisy state, inverts that, rotates the recovered noise, and
lets the model rebuild -- all in sampler space, because Flux's latent format
shifts as well as scales (0.1159) and a rotation performed in the shifted space
would not preserve the offset.

    process_latent_in                -> sampler space
    S = f(sigma, 0, x0)              -> the state with no noise in it
    G = f(sigma, 1, x0) - S          -> the gain on unit noise (NOT sigma:
                                        CONST applies an extra noise_scale)
    eps = (x - S) / G                -> the model's own noise coordinate
    rotate eps                       -> unchanged, this part was always right
    f(sigma, eps')                   -> a valid state at the same sigma
    process_latent_out

There is no list of model names in this file and there must never be one. The
contract is a capability, not an identity: if the model's noise mapping is
affine and non-singular, the node can recover a noise coordinate, vary it and
rebuild; otherwise it refuses and says which guarantee failed. Affinity is
verified with a random probe rather than assumed -- a map can be exactly linear
at 0, 1 and 2 while curved everywhere else, and would otherwise pass with the
right gain.

`model` and `current_sigma` are a PAIR. Half-wired raises, because the fallback
is correct for additive models and wrong for flow, and choosing it silently is
the whole bug.

Measured on SD1.5: branch LATE. At sigma 4.86 (96% noise) x0_pred is barely
committed, so rotating re-rolls the composition rather than varying it. Deeper in
(sigma ~1.8 or below) the layout holds while the interpretation changes, which is
what "show me alternatives" actually means.

`keep_parent` answers a different question from either dial: "this state is
already good — continue it, but show me alternatives too". Every descendant sits
at the SAME theta from the parent, so a family cannot contain both the faithful
continuation and variations of it; two children each theta from the parent can
differ by at most 2*theta, which is why theta=0 makes them all identical no
matter what spread says. That is a fact about distances, not about this code.
`keep_parent` sidesteps it by letting ONE child sit at a different distance —
zero — instead of trying to make zero-distance children differ. Descendant 1 is
then the parent's own continuation, bit-exact, sharing a batch with its
alternatives so they can be judged against each other rather than from memory.

It REPLACES the first descendant instead of prepending one, so descendants
2..count keep the seeds and indices they had with the toggle off.

Variant i uses `variation_seed + i`, and the shared family direction is derived
from the same integer — so one number still addresses the whole set. Note the
pair (variation_seed, index) is what identifies a descendant, not a seed on its
own: because the family direction is shared, a solo run at `variation_seed + i`
builds a DIFFERENT family. Child i is stable under changing `count`, which is
what makes the index a durable address. Reproducing a descendant needs the
parent latent and theta regardless, so a standalone seed never sufficed here.
"""

from __future__ import annotations

import math

import torch

from ...base import GteNode, as_bool, first
from ...registry import register
from .step_stamp import stamped_sigma


# An odd 64-bit constant (splitmix64's) used to derive the family seed from the
# same integer the children come from. One number still addresses the whole
# family, and the derived value cannot land on a child's own seed.
_FAMILY_MIX = 0x9E3779B97F4A7C15
_U64 = 0xFFFFFFFFFFFFFFFF


def _family_seed(base: int) -> int:
	"""A seed for the shared direction, derived from — but never equal to — a child's."""
	fam = ((base + 1) * _FAMILY_MIX) & _U64
	if base <= fam <= base + 4096:
		# Cannot happen in practice, but a seed collision is exactly the bug this
		# node exists to avoid, so refuse to rely on luck.
		fam ^= 0xA5A5A5A5A5A5A5A5
	return fam


# How far the noise map may stray from affine, and how small its gain may get
# before the inverse is meaningless. Both are checked, because neither implies
# the other: a map that ignores its noise argument entirely is PERFECTLY affine
# and completely useless (IMG_TO_IMG_FLOW is exactly that).
_AFFINE_TOL = 1e-4
_GAIN_MIN = 1e-4
_PROBE_SEED = 0x5EED


def _probe_noise_map(model_sampling, sigma, x0):
	"""Recover the affine map noise -> state that THIS model uses, by asking it.

	The node must not know what a "flow model" is. ComfyUI already owns the
	parameterisation, and every scheme it ships builds a noisy state affinely in
	the noise:

	    EPS    x = x0            + sigma * noise
	    CONST  x = (1-sigma)*x0  + sigma * noise_scale * noise

	so two evaluations recover the whole map without naming a single model:

	    S = f(sigma, 0, x0)          the state with no noise in it
	    G = f(sigma, 1, x0) - S      the gain applied to unit noise

	Note G is NOT sigma. CONST multiplies by an extra `noise_scale` that the
	flow sampler exposes, and assuming a gain of sigma there is wrong by exactly
	that factor -- silently, as an under-denoise, which is the failure this
	whole path exists to prevent.

	Affinity is then VERIFIED rather than assumed, with a deterministic random
	probe. Checking only 0/1/2 is foolable: a map can be exactly linear at three
	chosen points and curved everywhere else, and such a map passes with the
	right gain while being wrong for real noise.

	Raises when the map cannot be inverted, naming which guarantee failed.
	"""
	z = torch.zeros_like(x0)
	S = model_sampling.noise_scaling(sigma, z, x0, False)
	G = model_sampling.noise_scaling(sigma, torch.ones_like(x0), x0, False) - S

	gen = torch.Generator().manual_seed(_PROBE_SEED)
	r = torch.randn(x0.shape, dtype=torch.float32, generator=gen).to(x0.device, x0.dtype)
	err = float((model_sampling.noise_scaling(sigma, r, x0, False) - (S + G * r)).abs().max())
	if err > _AFFINE_TOL:
		raise RuntimeError(
			f"Noise Rotate: this model's noise mapping is not affine in the noise "
			f"(probe error {err:.3g}). There is no noise coordinate to recover, so "
			f"a rotation cannot be defined. Leave `model`/`current_sigma` unwired to "
			f"use the additive path, or branch with a different model."
		)
	gmin = float(G.abs().min())
	if gmin < _GAIN_MIN:
		raise RuntimeError(
			f"Noise Rotate: this state has no recoverable noise component "
			f"(gain {gmin:.3g} at sigma {float(sigma):.4g}). Either the schedule has "
			f"reached sigma 0 -- nothing is unresolved, so there is nothing to vary -- "
			f"or this model's sampling ignores the noise it is given."
		)
	return S, G


def _project_out(x, a):
	"""`x` with its component along `a` removed: one step of Gram-Schmidt.

	Two independent gaussians in D dimensions are ALMOST perpendicular already
	(cos ~ 1/sqrt(D), 0.7% at 4x113x64), which is why a single rotation was
	honest to about 0.2 degrees without this. `spread` composes two rotations
	and is used at small angles, where that same 0.2 degrees is a tenth of the
	value on the dial — so here it is made exact instead of nearly-exact.
	"""
	xf, af = x.flatten().double(), a.flatten().double()
	return x - float(xf @ af / (af @ af)) * a


def rotate_family(x, x0, theta_deg: float, spread_deg: float, count: int, base_seed: int,
				  keep_parent: bool = False):
	"""`count` descendants, each exactly `theta_deg` from the parent.

	    v      one shared family direction, perpendicular to the residual
	    q_i    each child's own direction, perpendicular to both
	    w_i =  cos(spread)*v + sin(spread)*q_i
	    d_i =  cos(theta)*eps + sin(theta)*w_i

	Because every w_i is perpendicular to eps and unit-scaled to it, ||d_i|| ==
	||eps|| exactly and d_i . eps == cos(theta) for every child — so `spread`
	moves the children around each other WITHOUT moving them nearer or further
	from the parent. That separation is the whole point: theta says how far the
	family travels, spread says how far apart its members are.

	`keep_parent` REPLACES child 0 with the parent itself, rather than inserting
	it. Inserting would shift every other child one place along, and an index is
	an address here — Candidate Select reports `seed + index`, and a descendant
	is identified by (variation_seed, index). Replacing leaves children 1..n-1
	byte-identical to what the same seed produces with the flag off, so turning
	the toggle on does not silently rename anything you already chose.
	"""
	eps = x - x0
	fam = _rotate_vectors(eps, theta_deg, spread_deg, count, base_seed,
						  skip_first=keep_parent)
	return [x.clone() if v is None else x0 + v for v in fam]


def _rotate_vectors(eps, theta_deg: float, spread_deg: float, count: int,
					base_seed: int, skip_first: bool = False):
	"""`count` rotations of ONE vector. `None` marks a child left untouched.

	Separated from the reconstruction because WHICH vector gets rotated depends
	on the model: for an additive model it is the residual x - x0, for anything
	else it is the noise coordinate recovered from the model's own noise map.
	The rotation itself is identical either way -- that is the part that was
	always right.
	"""
	scale = eps.std()
	t, p = math.radians(theta_deg), math.radians(spread_deg)

	def draw(seed):
		return torch.randn(eps.shape, dtype=eps.dtype,
						   generator=torch.Generator().manual_seed(int(seed) & _U64))

	v = _project_out(draw(_family_seed(base_seed)), eps)
	v = v / v.std() * scale

	out = []
	for i in range(count):
		if skip_first and i == 0:
			# None, not a reconstruction: the caller returns the ORIGINAL tensor
			# for this child, so the parent survives bit-for-bit and never pays
			# the round trip through the model's noise map.
			out.append(None)
			continue
		q = _project_out(_project_out(draw(base_seed + i), eps), v)
		q = q / q.std() * scale
		w = math.cos(p) * v + math.sin(p) * q
		out.append(math.cos(t) * eps + math.sin(t) * w)
	return out


def _diagnose(x, x0, variants, theta_deg, spread_deg, skip_first, rotated=None):
	"""Print what the branch had to work with, and what it actually produced.

	Two different failures look identical from the canvas — descendants that
	come out the same. Either the node was given nothing to vary (the model has
	already committed, so the residual barely matters), or it varied the latents
	properly and the MODEL collapsed them back on the way to an image. The
	numbers below separate those, which no amount of staring at the previews
	will.

	`residual share` is scale-free, so it compares checkpoints OF THE SAME MODEL.
	It does NOT compare across model families, and reading it as though it did is
	a trap. An eps-prediction model ADDS noise with an unbounded sigma
	(x = x0 + sigma*eps), so the share runs to ~100%. A rectified-flow model
	MIXES it (x = (1-t)*x0 + t*eps), so the residual can never dominate x0 and
	the share tops out near 50% even at 100% noise. 45% is starved on the first
	and nearly maximal on the second.
	"""
	eps = x - x0
	s_x0, s_eps = float(x0.std()), float(eps.std())
	share = 100.0 * s_eps ** 2 / (s_eps ** 2 + s_x0 ** 2) if (s_eps or s_x0) else 0.0
	ratio = s_eps / s_x0 if s_x0 else float("inf")
	print(f"[gte]   parent: std(x0)={s_x0:.4f} std(residual)={s_eps:.4f} "
		  f"-> residual/x0 = {ratio:.2f}, {share:.1f}% of the state")

	# What the dials asked for: two children each theta from the parent, fanned
	# out by spread, sit acos(cos^2 t + sin^2 t cos^2 s) apart.
	t, p = math.radians(theta_deg), math.radians(spread_deg)
	want = math.degrees(math.acos(max(-1.0, min(1.0,
		math.cos(t) ** 2 + math.sin(t) ** 2 * math.cos(p) ** 2))))

	# Measure in the coordinate the rotation actually happened in. Under the
	# parameterisation-aware path that is the model's noise coordinate, not
	# `variant - x0` -- the reconstruction is affine but not an isometry of
	# node space, so angles measured there would not be the theta asked for and
	# the readout would accuse the node of a fault it does not have.
	src = rotated if rotated is not None else variants
	pool = [v for v in (src[1:] if skip_first else src) if v is not None][:6]
	got = None
	if len(pool) > 1:
		ds = [(v if rotated is not None else v - x0).flatten().double() for v in pool]
		cos = []
		for i in range(len(ds)):
			for j in range(i + 1, len(ds)):
				n = ds[i].norm() * ds[j].norm()
				if float(n) > 0:
					cos.append(max(-1.0, min(1.0, float(ds[i] @ ds[j] / n))))
		if cos:
			got = math.degrees(math.acos(sum(cos) / len(cos)))

	if got is None:
		print(f"[gte]   siblings: n/a (need 2+ varying descendants), asked for {want:.1f}deg")

	else:
		print(f"[gte]   siblings: {got:.1f}deg apart in latent space (asked for {want:.1f}deg)")
		if abs(got - want) < 2.0 and got > 10.0:
			print("[gte]   -> the latents ARE separated. If the images still look "
				  "alike, the MODEL is collapsing them, not this node.")
	# No fixed percentage: the share's CEILING is model-dependent (see above), so
	# an absolute threshold cried wolf on flow models, where 45% is near maximal.
	# The ratio is the safer alarm -- a residual under half the prediction's
	# magnitude is late in any parameterisation.
	if ratio < 0.5:
		print(f"[gte]   -> the residual is only {ratio:.2f}x the prediction: late in "
			  f"the schedule, little left to turn. Compare against an EARLIER "
			  f"checkpoint of this same model before concluding anything.")



def rotate_in_noise_space(x, x0, model, sigma, theta_deg: float, spread_deg: float,
						  count: int, base_seed: int, keep_parent: bool = False):
	"""Rotate the model's OWN noise coordinate, and let the model rebuild the state.

	    process_latent_in            -> sampler space (Flux shifts as well as scales)
	    S, G = probe(noise_scaling)  -> the model's affine noise map
	    eps  = (x - S) / G           -> its noise coordinate
	    rotate eps
	    noise_scaling(sigma, eps')   -> a VALID state at the same sigma
	    process_latent_out           -> back to the space the graph speaks

	The additive path rotates `x - x0`, which for an additive model IS the noise.
	For a mixture model x_t = (1-t)*x0 + t*eps that same subtraction gives
	t*(eps - x0) -- noise AND signal -- so rotating it drags the signal off the
	manifold: measured at theta 90 the x0 component came back at 1.00 where 0.125
	is valid, and the implied noise at 3.5x. The sampler then resumes on a state
	that is not on its trajectory and cannot clean it, which reads as "the
	branch never developed".

	Working in the model's own coordinate makes that impossible by construction,
	for every parameterisation ComfyUI has and any future one that stays affine.
	"""
	pin = model.get_model_object("process_latent_in")
	pout = model.get_model_object("process_latent_out")
	ms = model.get_model_object("model_sampling")

	xs, x0s = pin(x), pin(x0)
	sig = torch.as_tensor(sigma, dtype=xs.dtype, device=xs.device)
	S, G = _probe_noise_map(ms, sig, x0s)

	eps = (xs - S) / G
	fam = _rotate_vectors(eps, theta_deg, spread_deg, count, base_seed,
						  skip_first=keep_parent)
	out = []
	for v in fam:
		if v is None:
			out.append(x.clone())          # untouched, never round-tripped
		else:
			out.append(pout(ms.noise_scaling(sig, v, x0s, False)))
	return out, eps, fam


@register
class NoiseRotate(GteNode):
	DISPLAY_NAME = "Noise Rotate (gte)"
	CATEGORY = "gte/sampling"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"latent": ("LATENT", {"tooltip":
					"`output` from the sampler that stopped at the checkpoint — "
					"the state still carrying its noise."}),
				"denoised": ("LATENT", {"tooltip":
					"`denoised_output` from that same sampler: what the model "
					"currently believes the image is becoming. This is preserved."}),
				# The full circle is allowed, not just the useful quadrant: 90 is
				# where sibling spread peaks, but seeing what lies past it is
				# worth more than a guard rail. See the module docstring.
				"theta": ("FLOAT", {"default": 25.0, "min": -360.0, "max": 360.0,
					"step": 0.5, "tooltip":
					"Angle to turn the unresolved noise — literally the cosine "
					"similarity to the original residual, cos_sim = cos(theta). "
					"0 = exact continuation, 10-45 useful, 90 = uncorrelated "
					"(and the widest spread between descendants). Past 90 the "
					"descendants move further from the parent but back TOWARD "
					"each other; at 180 the variation seed stops mattering and "
					"they collapse to one image, the negative of the "
					"continuation. -theta mirrors +theta at equal strength."}),
				"count": ("INT", {"default": 4, "min": 1, "max": 64, "tooltip":
					"How many descendants. They come out as a batch — pick one "
					"with Candidate Select."}),
				"variation_seed": ("INT", {"default": 0, "min": 0,
					"max": 0xffffffffffffffff - 0xffff, "control_after_generate": True,
					"tooltip": "Descendant i is drawn from variation_seed + i. The "
							   "shared family direction is derived from the same "
							   "number, so one integer still addresses the whole set."}),
				# Declared LAST on purpose: widgets_values is positional, so adding
				# a widget in the middle would silently re-map every saved graph.
				"spread": ("FLOAT", {"default": 90.0, "min": 0.0, "max": 90.0,
					"step": 0.5, "tooltip":
					"How far apart the descendants are FROM EACH OTHER, at the "
					"same distance from the parent. 90 = independent (the "
					"default, and what this node always did); 0 = every "
					"descendant identical, so `count` costs you N renders of one "
					"image. The sibling angle it produces is capped by theta: at "
					"theta 25 they can never be more than 34.8 degrees apart."}),
				# Also last, and for the same positional reason as `spread`.
				"keep_parent": ("BOOLEAN", {"default": False,
					"label_on": "Yes", "label_off": "No", "tooltip":
					"Make the FIRST descendant the parent's own continuation, "
					"unchanged, and vary the rest. Use it when the current state "
					"is already good: you keep it in the same batch as its "
					"alternatives, so Candidate Select compares them side by "
					"side instead of you re-running the branch to get it back. "
					"It replaces descendant 1 rather than adding one, so the "
					"others keep the seeds and indices they already had."}),
			},
			"optional": {
				# A PAIR. Together they make the node parameterisation-aware; the
				# model carries its own noise map and latent scaling, the sigma
				# says where this latent actually is. Neither is guessed: one
				# without the other is an error, because silently falling back to
				# the additive path is exactly the failure this pair prevents.
				"model": ("MODEL", {"tooltip":
					"The model this latent came from. Wire it (with current_sigma) "
					"for anything that is not a plain additive/eps model — Flux, "
					"SD3, krea, any rectified-flow checkpoint. The node asks the "
					"model how it builds a noisy state rather than assuming."}),
				# forceInput so it exists only when wired: a widget would always
				# hold a value and the pair rule could never see it as absent.
				"current_sigma": ("FLOAT", {"forceInput": True, "tooltip":
					"The sigma this latent is AT — not where the next stage ends. "
					"Wire Sigma Segment's `start_sigma` from the segment that is "
					"about to run, which is the same boundary this latent stopped on."}),
			},
		}

	RETURN_TYPES = ("LATENT",)
	RETURN_NAMES = ("latents",)
	OUTPUT_TOOLTIPS = (
		"The descendants, as a batch, at the same noise level as the input.",
	)
	FUNCTION = "execute"

	def execute(self, latent, denoised, theta, count, variation_seed, spread=90.0,
				keep_parent=False, model=None, current_sigma=None):
		x, x0 = latent["samples"], denoised["samples"]
		if x.shape != x0.shape:
			raise RuntimeError(
				f"Noise Rotate: `latent` is {tuple(x.shape)} but `denoised` is "
				f"{tuple(x0.shape)}. Both must come from the SAME sampler — wire "
				f"its `output` and `denoised_output`."
			)
		if x.shape[0] != 1:
			raise RuntimeError(
				f"Noise Rotate: expected one latent, got a batch of {x.shape[0]}. "
				f"Choose a candidate first (Candidate Select)."
			)
		if float(x.sub(x0).std()) == 0.0:
			# Nothing undecided left: at sigma 0 there is no residual to turn, and
			# every "variant" would be the same finished image.
			raise RuntimeError(
				"Noise Rotate: `latent` and `denoised` are identical — this "
				"checkpoint has no unresolved noise left to vary. Branch from an "
				"earlier point in the schedule."
			)

		theta = float(first(theta, 0.0))
		spread = float(first(spread, 90.0))
		base = int(first(variation_seed, 0))
		n = int(first(count, 1))
		keep = as_bool(keep_parent, False, where="Noise Rotate: keep_parent")
		if n > 1 and spread < 1.0:
			print(f"[gte] Noise Rotate: spread={spread:g} — {n} descendants will be "
				  f"near-identical; you are paying {n} renders for one image.")
		if keep and n == 1:
			# Not an error — the graph still runs — but the node has been asked
			# for one descendant and told to make it the parent, so it is doing
			# nothing at all and the user almost certainly meant count > 1.
			print("[gte] Noise Rotate: keep_parent with count=1 — the only "
				  "descendant IS the parent, so nothing varies. Raise count.")
		model = first(model, None)
		sigma = first(current_sigma, None)
		stamped = stamped_sigma(latent)
		if sigma is None and stamped is not None:
			# The latent knows where it is. A stamped sigma cannot be the wrong
			# segment's, which a hand-wired one can: it was written by the stage
			# that actually produced this tensor.
			sigma = float(stamped)
			from_stamp = True
		else:
			from_stamp = False

		if sigma is not None and model is None:
			raise RuntimeError(
				"Noise Rotate: a sigma is available but `model` is not wired. The "
				"sigma alone is not enough — the model is what says how a noisy "
				"state is built, and without it the node cannot invert anything. "
				"Wire MODEL (the same one feeding the guider)."
			)
		if model is not None and sigma is None:
			raise RuntimeError(
				"Noise Rotate: `model` is wired but this latent carries no sigma. "
				"Either wire Sigma Segment's `end_sigma` into the Stamp Step that "
				"produced this latent — then it travels automatically — or wire "
				"`current_sigma` here directly with the sigma this latent is AT."
			)

		# theta 0 is the control the whole design rests on, so it returns the
		# ORIGINAL tensor rather than a reconstruction of it. Every path below
		# would only add float error to something already exact.
		if math.isclose(theta % 360.0, 0.0, abs_tol=1e-9):
			variants, rotated, coord = [x.clone() for _ in range(n)], None, None
		elif model is not None:
			variants, coord, rotated = rotate_in_noise_space(
				x, x0, model, sigma, theta, spread, n, base, keep)
			# Say WHICH sigma was used, not just that a sigma was used. Wiring
			# the wrong segment's start_sigma is silent and catastrophic: at
			# sigma 1.0 the signal term (1-sigma)*x0 vanishes and the rebuilt
			# state contains no image at all, only noise.
			src = "from the latent's stamp" if from_stamp else "wired by hand"
			print(f"[gte]   model-aware path: sigma={float(sigma):.4f} ({src})")
			if float(sigma) >= 0.999:
				print("[gte]   !! sigma is ~1.0: the reconstruction keeps NO signal. "
					  "You have almost certainly wired the previous segment's "
					  "start_sigma instead of this one's.")
		else:
			print("[gte]   using the ADDITIVE path (x = x0 + sigma*eps): correct for "
				  "SD/SDXL, WRONG for rectified flow. Wire `model` + `current_sigma` "
				  "if this is Flux/SD3/krea.")
			variants = rotate_family(x, x0, theta, spread, n, base, keep)
			coord, rotated = None, None

		out = latent.copy()
		out["samples"] = torch.cat(variants, dim=0)
		# The descendants are their own lineage now; the parent's batch position
		# would mislead anything that regenerates noise from them.
		out.pop("batch_index", None)
		kept = " (descendant 1 is the parent)" if keep else ""
		print(f"[gte] Noise Rotate: {len(variants)} descendant(s) at "
			  f"theta={theta:g}deg spread={spread:g}deg from seed {base}{kept}")
		try:
			_diagnose(x, x0, variants, theta, spread, keep, rotated)
		except Exception as exc:  # noqa: BLE001
			# Diagnostics must never cost you the branch.
			print(f"[gte]   (diagnostics unavailable: {exc!r})")
		return (out,)
