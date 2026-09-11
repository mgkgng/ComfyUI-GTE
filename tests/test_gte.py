"""Regression tests for the Generative Trajectory Explorer nodes.

Every test here corresponds to something that actually broke: a latent that
carried the wrong step, a rotation that inflated the noise level, a boundary
sigma that stopped being shared between adjacent segments.

Run either way (pytest is optional):
	python tests/test_gte.py
	pytest tests/

Needs torch, so use ComfyUI's interpreter:
	ComfyUI/venv/bin/python tests/test_gte.py
"""

from __future__ import annotations

import math
import os
import sys

import torch

# Import the pack as `gte` regardless of where the tests are run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import gte  # noqa: E402,F401  (registers every node)
from gte.nodes.data.show_text import ShowTextLines  # noqa: E402
from gte.nodes.sampling.candidate_select import CandidateSelect  # noqa: E402
from gte.nodes.sampling.noise_rotate import NoiseRotate, _probe_noise_map  # noqa: E402
from gte.nodes.sampling.schedule_info import ScheduleInfo, describe  # noqa: E402
from gte.nodes.sampling.seed_range_noise import Noise_SeedRange, seeds_for  # noqa: E402
from gte.nodes.sampling.sigma_segment import SigmaSegment, slice_sigmas  # noqa: E402
from gte.nodes.sampling.step_stamp import (  # noqa: E402
	StampStep, ResumeStep, stamped_step,
)


def _unwrap(r):
	"""Nodes return either a tuple or a {'ui','result'} envelope."""
	return r["result"] if isinstance(r, dict) else r

def test_show_text_lines_numbers_and_flattens():
	from gte.nodes.data.show_text import ShowTextLines
	n = ShowTextLines()
	# INPUT_IS_LIST wraps every socket; a list socket becomes one line per element
	r = n.execute(text_1=["hello"], text_2=[["a", "b"]], text_3=[None])
	assert r["result"][0] == "hello\na\nb"
	assert r["ui"]["text"][0] == "1. hello\n2. a\n3. b"
	# nothing wired says so rather than showing an empty box
	assert "nothing wired" in ShowTextLines().execute()["ui"]["text"][0]

def _core_prepare_noise():
	"""comfy.sample.prepare_noise — the behaviour Seed Range Noise must match.

	Compared against core rather than a copy of its formula on purpose: the
	point is that our items stay interchangeable with core's, so if core ever
	changes how it draws noise, these tests should fail and tell us.

	ComfyUI's root is not on the test path (only custom_nodes/ is, so `import
	gte` works), so it is added here and taken straight back out. Leaving it
	makes `folder_paths` importable, and the node that probes for it — Candidate
	Select, for its thumbnail grid — then takes its ComfyUI-present branch for
	every test that follows, quietly changing what the rest of the suite
	exercises.
	"""
	root = os.path.dirname(os.path.dirname(os.path.dirname(
		os.path.dirname(os.path.abspath(__file__)))))
	added = root not in sys.path
	if added:
		sys.path.insert(0, root)
	try:
		import comfy.sample  # noqa: PLC0415 — deliberately local, see above
		return comfy.sample.prepare_noise
	finally:
		if added:
			sys.path.remove(root)
		sys.modules.pop("folder_paths", None)

def test_seed_range_noise_item_matches_a_solo_run():
	"""The reason the node exists: item i's NOISE must equal a solo run at seed+i.

	Noise only — the rendered image still shifts slightly if you later re-run a
	candidate at a different batch size, because the UNet picks different GPU
	kernels per batch size. See the node docstring. If this drifts, a
	candidate's seed stops being a portable identity and we are back to
	carrying (seed, batch_index) pairs around.
	"""
	prepare_noise = _core_prepare_noise()
	seed = 740016770880953
	batch = Noise_SeedRange(seed).generate_noise({"samples": torch.zeros(5, 4, 64, 64)})
	assert batch.shape == (5, 4, 64, 64)
	for i in range(5):
		solo = prepare_noise(torch.zeros(1, 4, 64, 64), seed + i)
		assert torch.equal(batch[i:i + 1], solo), f"item {i} != solo run at seed+{i}"

def test_seed_range_noise_honours_batch_index():
	# After Latent From Batch picks item 3, it must still be drawn from seed+3 —
	# otherwise "select a candidate, then continue it" silently changes the image.
	prepare_noise = _core_prepare_noise()
	seed = 740016770880953
	picked = Noise_SeedRange(seed).generate_noise(
		{"samples": torch.zeros(1, 4, 64, 64), "batch_index": [3]}
	)
	assert torch.equal(picked, prepare_noise(torch.zeros(1, 4, 64, 64), seed + 3))

def test_seed_range_noise_seed_mapping():
	assert seeds_for({"samples": torch.zeros(3, 4, 8, 8)}, 100) == [100, 101, 102]
	# A picked latent carries the positions it came from, not 0..n.
	assert seeds_for({"samples": torch.zeros(2, 4, 8, 8), "batch_index": [3, 7]}, 100) == [103, 107]

def test_seed_range_noise_refuses_nested_latents():
	class _Nested(torch.Tensor):
		is_nested = True

	samples = torch.zeros(1, 4, 8, 8).as_subclass(_Nested)
	try:
		Noise_SeedRange(0).generate_noise({"samples": samples})
		raise AssertionError("a nested latent must be refused, not silently mishandled")
	except RuntimeError as exc:
		assert "nested" in str(exc)

def test_sigma_segment_keeps_the_shared_boundary():
	"""Adjacent stages must overlap on one sigma, or the hand-off is not seamless.

	0->4 and 4->9 both contain sigmas[4]: the second stage has to be told the
	noise level it resumes at, not merely the sigmas that remain.
	"""
	sig = list(range(21))                      # stand-in for a 20-step schedule
	a = slice_sigmas(sig, 0, 4)
	b = slice_sigmas(sig, 4, 9)
	assert a == [0, 1, 2, 3, 4]
	assert b == [4, 5, 6, 7, 8, 9]
	assert a[-1] == b[0]
	# a segment runs (end - start) steps, holding one more sigma than that
	assert len(a) - 1 == 4 and len(b) - 1 == 5

def test_sigma_segment_indices_stay_absolute():
	# The whole point: asking for 9 means step 9 of the ORIGINAL schedule, with no
	# regard for how the earlier stages were cut. Chained SplitSigmas would need 5.
	sig = list(range(21))
	assert slice_sigmas(sig, 9, 14)[0] == 9

def test_sigma_segment_refuses_bad_ranges():
	sig = list(range(21))                      # 20 steps, indices 0..20
	for start, end in [(4, 4), (9, 4), (0, 21), (-1, 5)]:
		try:
			slice_sigmas(sig, start, end)
			raise AssertionError(f"({start},{end}) must be refused, not clamped")
		except ValueError:
			pass
	# The two invalid orderings are different mistakes and must not share a
	# message — "one sigma performs no sampling" does not explain a swapped pair.
	try:
		slice_sigmas(sig, 9, 4)
	except ValueError as exc:
		assert "before" in str(exc) and "swapped" in str(exc)
	try:
		slice_sigmas(sig, 4, 4)
	except ValueError as exc:
		assert "one sigma" in str(exc)
	# the exact end of the schedule is still valid
	assert len(slice_sigmas(sig, 14, 20)) == 7

def test_sigma_segment_node_reports_both_ends():
	"""start_sigma puts the stage's noise level on the canvas; end_sigma is what
	the OUTPUT latent will be sitting at, and is what Stamp Step carries onward.
	A step number alone cannot substitute: the same index is a different noise
	level under a different scheduler."""
	seg, start_sigma, end_sigma = SigmaSegment().execute(
		[14.61, 10.74, 8.08, 6.20, 4.85, 3.86], 2, 4)
	assert seg == [8.08, 6.20, 4.85]
	assert abs(start_sigma - 8.08) < 1e-6
	assert abs(end_sigma - 4.85) < 1e-6

def test_schedule_info_shows_where_the_noise_actually_is():
	"""The reading that makes a bad checkpoint obvious before a render.

	A step number is not a noise level. Under karras an 8-step flow schedule is
	97.8% denoised by step 5; under a linear one it is 62.5%. Branching at
	"step 5" therefore means two completely different things, decided by a
	dropdown in another group, and nothing on the canvas said so.
	"""
	from gte.nodes.sampling.schedule_info import describe
	karras = [1.0, 0.5408, 0.2757, 0.1308, 0.0568, 0.0220, 0.0073, 0.0020, 0.0]
	table = describe(karras, marks=[1, 5, 7])
	lines = table.splitlines()
	assert "checkpoint" in lines[2] and "checkpoint" in lines[6]
	assert "54.1%" in lines[2], lines[2]      # step 1 still has noise
	assert "2.2%" in lines[6], lines[6]       # step 5 has almost none
	# percentages are relative to the schedule's own top, so any model reads the same
	assert "100.0%" in lines[1]

def test_schedule_info_survives_a_typo_in_the_marks():
	# The marks are decoration; a typo there must not cost a queued run.
	from gte.nodes.sampling.schedule_info import ScheduleInfo
	out = ScheduleInfo().execute([1.0, 0.5, 0.0], marks="1, oops, 2")
	assert "checkpoint" in out["result"][0]

def test_candidate_select_reports_the_candidates_own_seed():
	"""The seed must match Seed Range Noise's mapping, or the identity is a lie."""
	lat = {"samples": torch.arange(5 * 4 * 8 * 8, dtype=torch.float32).reshape(5, 4, 8, 8)}
	out, _dn, img, seed, idx = CandidateSelect().execute(lat, index=2, origin_seed=100)["result"]
	assert seed == 102 and idx == 2
	assert torch.equal(out["samples"][0], lat["samples"][2])

def test_candidate_select_stamps_batch_index():
	# Without this, regenerating noise from the picked latent silently lands on a
	# different candidate's seed.
	lat = {"samples": torch.zeros(4, 4, 8, 8)}
	out = CandidateSelect().execute(lat, index=3)["result"][0]
	assert out["batch_index"] == [3]
	# and an already-sliced latent keeps its ORIGINAL position, not the new one
	nested = CandidateSelect().execute(
		{"samples": torch.zeros(2, 4, 8, 8), "batch_index": [7, 9]}, index=1)["result"][0]
	assert nested["batch_index"] == [9]

def test_candidate_select_slices_the_denoised_pair_too():
	"""Branching again needs BOTH of a sampler's outputs for the SAME candidate.

	Selecting them on two separate nodes would let the two indices drift apart,
	and the mismatch would be invisible — you would rotate one candidate's noise
	around another's prediction.
	"""
	lat = {"samples": torch.arange(3 * 4 * 8 * 8, dtype=torch.float32).reshape(3, 4, 8, 8)}
	den = {"samples": torch.ones(3, 4, 8, 8) * torch.arange(3).view(3, 1, 1, 1)}
	out, dn, _, _, _ = CandidateSelect().execute(lat, index=2, denoised=den)["result"]
	assert torch.equal(out["samples"][0], lat["samples"][2])
	assert float(dn["samples"].mean()) == 2.0

	# a mismatched pair cannot have come from one run, and must not be guessed at
	try:
		CandidateSelect().execute(lat, index=0, denoised={"samples": torch.zeros(5, 4, 8, 8)})
		raise AssertionError("a mismatched denoised batch must be refused")
	except RuntimeError as exc:
		assert "SAME sampler" in str(exc)

def test_candidate_select_clamps_and_reports_where_it_landed():
	# A cursor should stop at the end, not raise — but it must say where it is.
	lat = {"samples": torch.zeros(3, 4, 8, 8)}
	res = CandidateSelect().execute(lat, index=99, origin_seed=50)
	out, _dn, _im, seed, idx = res["result"]
	assert idx == 2 and seed == 52
	ui = res["ui"]["gte_candidate"][0]
	assert (ui["index"], ui["count"], ui["seed"]) == (2, 3, 52)

def test_candidate_select_survives_a_missing_preview_grid():
	# The thumbnails need ComfyUI's folder_paths, which the suite runs without.
	# Selection is the node's job; the grid is decoration and must not break it.
	lat = {"samples": torch.zeros(2, 4, 8, 8)}
	imgs = torch.zeros(2, 8, 8, 3)
	res = CandidateSelect().execute(lat, index=1, images=imgs, origin_seed=10)
	assert res["result"][3] == 11
	assert res["ui"]["gte_candidate"][0]["thumbs"] == []

def test_candidate_select_picks_the_matching_image():
	lat = {"samples": torch.zeros(3, 4, 8, 8)}
	imgs = torch.stack([torch.full((8, 8, 3), float(i)) for i in range(3)])
	img = CandidateSelect().execute(lat, index=1, images=imgs)["result"][2]
	assert img.shape[0] == 1 and float(img.mean()) == 1.0
	# images are optional: without them the slot is simply empty
	assert CandidateSelect().execute(lat, index=1)["result"][2] is None

def test_candidate_select_refuses_an_empty_batch():
	try:
		CandidateSelect().execute({"samples": torch.zeros(0, 4, 8, 8)})
		raise AssertionError("an empty batch must be refused")
	except RuntimeError as exc:
		assert "empty" in str(exc)

def test_stamp_step_round_trips():
	"""The whole contract: what Stamp writes, Resume reads back."""
	lat = {"samples": torch.zeros(1, 4, 8, 8)}
	stamped = StampStep().execute(lat, 11)[0]
	assert ResumeStep().execute(stamped)[0] == 11

def test_stamp_survives_the_nodes_it_has_to_travel_through():
	"""The stamp is only useful if every hop preserves it.

	SamplerCustomAdvanced, Candidate Select and Noise Rotate all rebuild the
	latent with `latent.copy()`, so extra keys ride along. Pin that for the two
	nodes in this pack — if either ever stops copying, the stamp goes silently
	missing and stages resume at the wrong sigma.
	"""
	g = torch.Generator().manual_seed(3)
	batch = {"samples": torch.randn(4, 4, 8, 8, generator=g), "gte_step": 9}
	denoised = {"samples": torch.randn(4, 4, 8, 8, generator=g), "gte_step": 9}

	picked, picked_dn = CandidateSelect().execute(batch, index=2, denoised=denoised)["result"][:2]
	assert stamped_step(picked) == 9, "Candidate Select dropped the stamp"
	assert stamped_step(picked_dn) == 9, "Candidate Select dropped it on `denoised`"

	kids = NoiseRotate().execute(picked, picked_dn, 25.0, 3, 500)[0]
	assert stamped_step(kids) == 9, "Noise Rotate dropped the stamp"

def test_resume_step_falls_back_for_an_unstamped_latent():
	"""An Empty Latent has never been sampled, so step 0 is the truth."""
	assert ResumeStep().execute({"samples": torch.zeros(1, 4, 8, 8)})[0] == 0
	assert ResumeStep().execute({"samples": torch.zeros(1, 4, 8, 8)}, fallback=7)[0] == 7

def test_stamp_step_refuses_to_go_backwards():
	"""Checkpoints out of order is the mistake this catches.

	Sampling only moves forward, so a stage ending before the latent already is
	means the controls are misordered — worth an error while the numbers are
	still on screen, rather than a plausible image from the wrong sigma.
	"""
	at14 = {"samples": torch.zeros(1, 4, 8, 8), "gte_step": 14}
	try:
		StampStep().execute(at14, 9)
		assert False, "stamping backwards should raise"
	except ValueError as exc:
		assert "already at step 14" in str(exc)
	# equal is fine: re-stamping the same position is a no-op, not a mistake
	assert ResumeStep().execute(StampStep().execute(at14, 14)[0])[0] == 14

def test_stamp_step_does_not_mutate_its_input():
	"""The incoming latent may still be feeding other nodes."""
	lat = {"samples": torch.zeros(1, 4, 8, 8)}
	StampStep().execute(lat, 4)
	assert "gte_step" not in lat

def test_stamp_step_keeps_the_other_latent_keys():
	"""batch_index in particular — losing it would move a candidate's seed."""
	lat = {"samples": torch.zeros(1, 4, 8, 8), "batch_index": [3],
		   "noise_mask": torch.ones(1, 1, 8, 8)}
	out = StampStep().execute(lat, 4)[0]
	assert out["batch_index"] == [3]
	assert "noise_mask" in out

def _parent(sigma=4.8557, seed=0, size=16):
	"""A stand-in checkpoint: x = x0 + sigma*eps, the shape a stopped sampler emits.

	`size` matters for the angle tests. Sibling separation is exact only in
	expectation: the per-child directions are independent draws, so their mutual
	angles carry a 1/sqrt(D) sampling error — about 3% at 16x16 but 0.8% at
	64x64, and 0.6% at a real 4x113x64 latent.
	"""
	g = torch.Generator().manual_seed(seed)
	x0 = torch.randn(1, 4, size, size, generator=g) * 0.5      # a hedged prediction
	eps = torch.randn(1, 4, size, size, generator=g)
	return {"samples": x0 + sigma * eps}, {"samples": x0}

def test_noise_rotate_preserves_the_noise_level():
	"""The invariant the whole method rests on: theta turns, it does not inflate.

	If ||x - x0|| grows with theta, the continuation runs a schedule expecting
	less noise than it gets and under-denoises — variation strength and quality
	loss confounded in one dial.
	"""
	lat, den = _parent()
	before = (lat["samples"] - den["samples"]).std().item()
	for theta in (0, 10, 30, 45, 90):
		out = NoiseRotate().execute(lat, den, theta, 3, 500)[0]["samples"]
		for i in range(out.shape[0]):
			after = (out[i:i+1] - den["samples"]).std().item()
			assert abs(after / before - 1) < 0.06, f"theta={theta} moved the magnitude"

def test_noise_rotate_theta_zero_is_the_exact_parent():
	lat, den = _parent()
	out = NoiseRotate().execute(lat, den, 0.0, 2, 7)[0]["samples"]
	for i in range(out.shape[0]):
		assert torch.allclose(out[i:i+1], lat["samples"], atol=1e-6)

def test_noise_rotate_divergence_grows_with_theta():
	lat, den = _parent()
	d = []
	for theta in (0, 10, 30, 60, 90):
		v = NoiseRotate().execute(lat, den, theta, 1, 11)[0]["samples"]
		d.append((v - lat["samples"]).abs().mean().item())
	assert d == sorted(d), f"divergence must be monotonic in theta, got {d}"
	assert d[0] == 0.0

def test_noise_rotate_theta_is_the_cosine_similarity():
	"""The widget number is the angle actually turned — that is the whole claim.

	Holds because `new` is drawn independently of `eps`, so the two are
	near-orthogonal and cos_sim(eps, eps_var) collapses to cos(theta).
	"""
	lat, den = _parent()
	eps = (lat["samples"] - den["samples"]).flatten()
	for theta in (10, 30, 45, 90, 135):
		v = NoiseRotate().execute(lat, den, float(theta), 1, 4)[0]["samples"]
		ev = (v - den["samples"]).flatten()
		cos = float(ev @ eps / (ev.norm() * eps.norm()))
		assert abs(cos - math.cos(math.radians(theta))) < 0.05, \
			f"theta={theta} turned {math.degrees(math.acos(max(-1, min(1, cos)))):.1f} deg"

def test_noise_rotate_past_ninety_collapses_the_siblings():
	"""Beyond 90 the descendants march away from the parent but back together.

	Sibling angle is acos(cos^2 theta), so it peaks at 90 and closes again — the
	reason 90 is the working maximum even though the widget now allows the whole
	circle.
	"""
	lat, den = _parent()

	def spread(theta):
		out = NoiseRotate().execute(lat, den, float(theta), 2, 21)[0]["samples"]
		a = (out[0:1] - den["samples"]).flatten()
		b = (out[1:2] - den["samples"]).flatten()
		return float(a @ b / (a.norm() * b.norm()))

	# cos of the sibling angle: 1 = identical, 0 = maximally spread.
	assert spread(90) < 0.15, "siblings should be near-orthogonal at 90"
	assert spread(135) > spread(90), "past 90 siblings close back up"
	assert spread(179) > spread(135)

def test_noise_rotate_at_one_eighty_ignores_the_variation_seed():
	"""sin(180) = 0, so `new` drops out: every descendant is the same negative.

	Worth pinning — a user who winds theta past 180 expecting more variety gets
	N copies of one image, and that surprise should be a documented property
	rather than a bug report.
	"""
	lat, den = _parent()
	a = NoiseRotate().execute(lat, den, 180.0, 3, 1)[0]["samples"]
	b = NoiseRotate().execute(lat, den, 180.0, 3, 999999)[0]["samples"]
	assert torch.allclose(a, b, atol=1e-5), "the variation seed still mattered at 180"
	for i in range(1, a.shape[0]):
		assert torch.allclose(a[0:1], a[i:i+1], atol=1e-5), "descendants differ at 180"
	# and that one image is the exact negative of the residual
	expected = den["samples"] - (lat["samples"] - den["samples"])
	assert torch.allclose(a[0:1], expected, atol=1e-5)

def test_noise_rotate_negative_theta_mirrors_positive():
	"""-theta has the same strength as +theta but is a different descendant."""
	lat, den = _parent()
	eps = (lat["samples"] - den["samples"]).flatten()
	pos = NoiseRotate().execute(lat, den, 30.0, 1, 55)[0]["samples"]
	neg = NoiseRotate().execute(lat, den, -30.0, 1, 55)[0]["samples"]

	def cos_to_parent(v):
		ev = (v - den["samples"]).flatten()
		return float(ev @ eps / (ev.norm() * eps.norm()))

	assert abs(cos_to_parent(pos) - cos_to_parent(neg)) < 0.02, "equal strength"
	assert not torch.allclose(pos, neg, atol=1e-3), "but a different descendant"

def _angle(a, b):
	a, b = a.flatten(), b.flatten()
	return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b / (a.norm() * b.norm()))))))

def test_spread_moves_siblings_without_moving_them_from_the_parent():
	"""The whole point of `spread`: it separates two distances that were locked.

	Every descendant must stay exactly theta from the parent whatever spread is,
	while the angle BETWEEN them opens up. If distance-from-parent drifts with
	spread, the two dials are still entangled and the control is a lie.
	"""
	lat, den = _parent()
	eps = lat["samples"] - den["samples"]
	for spread in (0.0, 20.0, 45.0, 90.0):
		out = NoiseRotate().execute(lat, den, 45.0, 3, 900, spread=spread)[0]["samples"]
		for i in range(out.shape[0]):
			from_parent = _angle(out[i:i+1] - den["samples"], eps)
			assert abs(from_parent - 45.0) < 0.2, \
				f"spread={spread} moved a child to {from_parent:.2f} deg from the parent"

def test_spread_controls_the_sibling_angle():
	"""Sibling separation must grow monotonically with spread, matching the model.

	    angle(i, j) = acos(cos^2 theta + sin^2 theta * cos^2 spread)
	"""
	lat, den = _parent(size=64)
	seen = []
	for spread in (0.0, 20.0, 45.0, 90.0):
		out = NoiseRotate().execute(lat, den, 45.0, 2, 900, spread=spread)[0]["samples"]
		got = _angle(out[0:1] - den["samples"], out[1:2] - den["samples"])
		t, p = math.radians(45.0), math.radians(spread)
		want = math.degrees(math.acos(max(-1.0, min(1.0,
			math.cos(t) ** 2 + math.sin(t) ** 2 * math.cos(p) ** 2))))
		assert abs(got - want) < 1.0, f"spread={spread}: got {got:.2f}, expected {want:.2f}"
		seen.append(got)
	assert seen == sorted(seen), f"sibling angle must grow with spread, got {seen}"

def test_spread_zero_collapses_the_family():
	"""spread=0 leaves only the shared direction, so every descendant is the same.

	Worth pinning as a documented property rather than a surprise: it is the one
	setting where `count` costs N renders of a single image.
	"""
	out = NoiseRotate().execute(*_parent(), 40.0, 4, 77, spread=0.0)[0]["samples"]
	for i in range(1, out.shape[0]):
		assert torch.allclose(out[0:1], out[i:i+1], atol=1e-5)

def test_spread_default_is_the_old_independent_behaviour():
	"""Default spread must reproduce acos(cos^2 theta) — the pre-spread node.

	Anyone who never touches the new widget must get what they always got.
	"""
	lat, den = _parent(size=64)
	out = NoiseRotate().execute(lat, den, 60.0, 2, 4242)[0]["samples"]
	got = _angle(out[0:1] - den["samples"], out[1:2] - den["samples"])
	want = math.degrees(math.acos(math.cos(math.radians(60.0)) ** 2))
	assert abs(got - want) < 1.0, f"got {got:.2f}, expected {want:.2f}"

def test_noise_rotate_descendants_are_addressed_by_seed_and_index():
	"""A descendant is (variation_seed, index), and index is stable under `count`.

	Before `spread` existed each child was a solo re-run at `variation_seed + i`.
	A SHARED family direction makes that impossible by construction — it is
	derived from the base seed, so a solo run at seed+1 would build a different
	family. That promise is gone on purpose.

	What replaces it is the invariant that actually gets used: child i is the
	same latent whether you asked for 2 descendants or 64, so the pair
	(seed, index) addresses it and Candidate Select's cursor stays meaningful.
	Reproducing a descendant needs the parent latent and theta anyway — the seed
	alone never sufficed at a branch stage.
	"""
	lat, den = _parent()
	small = NoiseRotate().execute(lat, den, 30.0, 2, 900)[0]["samples"]
	large = NoiseRotate().execute(lat, den, 30.0, 6, 900)[0]["samples"]
	for i in range(small.shape[0]):
		assert torch.allclose(small[i:i+1], large[i:i+1], atol=1e-6), \
			f"descendant {i} moved when count changed"
	assert not torch.allclose(large[0:1], large[1:2]), "descendants must differ"

def test_noise_rotate_keep_parent_returns_the_parent_bit_exact():
	"""Descendant 1 must be the parent itself, not a reconstruction of it.

	The point of the toggle is to keep a state you already judged good, so
	"very close" is not good enough: if it were rebuilt as x0 + eps it would
	carry float round-trip error and stop being the thing you chose.
	"""
	lat, den = _parent()
	out = NoiseRotate().execute(lat, den, 30.0, 4, 900, keep_parent=True)[0]["samples"]
	assert out.shape[0] == 4, "keep_parent replaces a descendant, it does not add one"
	assert float((out[0:1] - lat["samples"]).abs().max()) == 0.0
	# ...and the rest must still actually vary, or the toggle has eaten the batch.
	assert not torch.allclose(out[1:2], out[2:3])

def test_noise_rotate_keep_parent_leaves_the_other_addresses_alone():
	"""Turning the toggle on must not silently rename descendants 2..count.

	Prepending the parent would shift every other child one index along, and an
	index IS an address here — Candidate Select reports `seed + index`. So this
	pins that only descendant 1 changes.
	"""
	lat, den = _parent()
	off = NoiseRotate().execute(lat, den, 30.0, 5, 900)[0]["samples"]
	on = NoiseRotate().execute(lat, den, 30.0, 5, 900, keep_parent=True)[0]["samples"]
	for i in range(1, off.shape[0]):
		assert torch.allclose(off[i:i+1], on[i:i+1], atol=0), \
			f"descendant {i} moved when keep_parent was turned on"
	assert not torch.allclose(off[0:1], on[0:1]), "descendant 1 should have changed"

def test_noise_rotate_keep_parent_puts_two_distances_in_one_batch():
	"""The whole reason the toggle exists: a family at ONE theta cannot contain
	both the faithful continuation and variations of it, because two children
	each theta from the parent differ by at most 2*theta. keep_parent gets there
	by letting one child sit at a different distance instead."""
	lat, den = _parent(size=64)
	x, x0 = lat["samples"], den["samples"]
	out = NoiseRotate().execute(lat, den, 25.0, 4, 900, keep_parent=True)[0]["samples"]

	def angle(a, b):
		a, b = a.flatten().double(), b.flatten().double()
		# Clamp: summing 16k float32 products lands cos at 1 +/- 1e-14, and acos
		# is undefined a hair outside [-1, 1].
		return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b / (a.norm() * b.norm()))))))

	dists = [angle(out[i:i+1] - x0, x - x0) for i in range(4)]
	# Not `== 0.0`: that same accumulation puts the parent at ~6e-06 degrees of
	# itself. Bit-exactness is the other test's job; this one is about distance.
	assert dists[0] < 1e-3, f"descendant 1 is the parent, so ~0 degrees, got {dists[0]}"
	for d in dists[1:]:
		assert abs(d - 25.0) < 1.0, f"the varying descendants stay at theta, got {d}"

def test_noise_rotate_keep_parent_reads_a_linked_string_correctly():
	"""A wired BOOLEAN can arrive as a string, and bool("false") is True.

	ComfyUI only validates LITERAL widget values, so anything reaching this
	input over a link is whatever the upstream node emitted. Casting would turn
	the toggle ON when the graph said off — silently, with no error and a batch
	that quietly lost a descendant.
	"""
	lat, den = _parent()
	x = lat["samples"]
	for falsey in (False, "false", "False", " no ", "0", 0):
		out = NoiseRotate().execute(lat, den, 30.0, 3, 900, keep_parent=falsey)[0]["samples"]
		assert not torch.allclose(out[0:1], x), f"{falsey!r} should mean OFF"
	for truthy in (True, "true", "TRUE", " yes ", "1", 1):
		out = NoiseRotate().execute(lat, den, 30.0, 3, 900, keep_parent=truthy)[0]["samples"]
		assert float((out[0:1] - x).abs().max()) == 0.0, f"{truthy!r} should mean ON"

def test_noise_rotate_keep_parent_refuses_a_value_it_cannot_read():
	# Guessing is what got us here; an unreadable toggle must say so.
	lat, den = _parent()
	try:
		NoiseRotate().execute(lat, den, 30.0, 2, 900, keep_parent="maybe")
	except RuntimeError as exc:
		assert "keep_parent" in str(exc) and "maybe" in str(exc)
	else:
		raise AssertionError("an unparseable boolean must raise")

class _FakeSampling:
	"""The two shapes ComfyUI actually ships, as plain maths.

	Local rather than imported so the suite stays standalone. The node is not
	tested against these CLASSES anyway -- its contract is "any affine,
	invertible noise map" -- and the real comfy.model_sampling implementations
	were probed separately and behave identically.
	"""
	def __init__(self, kind, noise_scale=1.0):
		self.kind, self.noise_scale = kind, noise_scale

	def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
		if self.kind == "eps":                       # x = x0 + sigma*eps
			return latent_image + sigma * noise
		if self.kind == "const":                     # x = (1-s)*x0 + s*gain*eps
			return sigma * (self.noise_scale * noise) + (1.0 - sigma) * latent_image
		if self.kind == "deaf":                      # ignores the noise entirely
			return latent_image
		if self.kind == "bent":                      # linear at 0,1,2 -- curved elsewhere
			return latent_image + sigma * (noise + 0.15 * noise * (noise - 1) * (noise - 2))
		raise AssertionError(self.kind)

class _FakeModel:
	"""A MODEL stand-in: a noise map plus a latent format with a real shift,
	because Flux/krea have one (0.1159) and SD1.5 does not."""
	def __init__(self, sampling, scale=0.3611, shift=0.1159):
		self.sampling, self.scale, self.shift = sampling, scale, shift

	def get_model_object(self, name):
		if name == "model_sampling":
			return self.sampling
		if name == "process_latent_in":
			return lambda l: (l - self.shift) * self.scale
		if name == "process_latent_out":
			return lambda l: (l / self.scale) + self.shift
		raise KeyError(name)

def _flow_state(sigma=0.875, size=16, seed=0):
	"""A latent as a MIXTURE model would hand it over, in node space."""
	g = torch.Generator().manual_seed(seed)
	x0 = torch.randn(1, 4, size, size, generator=g) * 2.348
	eps = torch.randn(1, 4, size, size, generator=g)
	x = sigma * eps + (1.0 - sigma) * x0                 # sampler space
	m = _FakeModel(_FakeSampling("const"))
	to_node = lambda l: l / m.scale + m.shift
	return {"samples": to_node(x)}, {"samples": to_node(x0)}, x0, m, sigma

def _validity(variant_node, x0_sampler, m, sigma):
	"""(x0 coefficient, implied noise std) of a state, in SAMPLER space.
	A valid mixture state has coefficient 1-sigma and unit noise."""
	v = (variant_node - m.shift) * m.scale
	a, b = v.flatten().double(), x0_sampler.flatten().double()
	coef = float(a @ b / (b @ b))
	implied = float(((v - (1.0 - sigma) * x0_sampler) / sigma).std())
	return coef, implied

def test_noise_rotate_flow_reconstruction_stays_on_the_manifold():
	"""The bug this path exists for, pinned.

	Rotating `x - x0` on a MIXTURE model rotates noise AND signal together, so
	the reconstruction lands off the trajectory: the x0 component comes back at
	full strength instead of faded to 1-sigma, and the implied noise at several
	times what the schedule expects. The sampler then resumes on a state that is
	not on its path and cannot clean it -- which reads as a branch that never
	develops rather than as an error.
	"""
	lat, den, x0s, m, sigma = _flow_state()
	valid_coef = 1.0 - sigma

	broken = NoiseRotate().execute(lat, den, 90.0, 2, 900)[0]["samples"]
	c, i = _validity(broken[0:1], x0s, m, sigma)
	assert abs(c - valid_coef) > 0.5, "the additive path should be visibly invalid here"
	assert i > 2.0, "and should imply far too much noise"

	fixed = NoiseRotate().execute(lat, den, 90.0, 2, 900,
								  model=m, current_sigma=sigma)[0]["samples"]
	for k in range(2):
		c, i = _validity(fixed[k:k + 1], x0s, m, sigma)
		assert abs(c - valid_coef) < 0.02, f"x0 coefficient {c:.3f}, valid {valid_coef:.3f}"
		assert abs(i - 1.0) < 0.05, f"implied noise std {i:.3f}, valid 1.0"
	assert not torch.allclose(fixed[0:1], fixed[1:2]), "and they must still differ"

def test_noise_rotate_noise_coordinate_round_trips():
	"""Exercises decompose -> reconstruct DIRECTLY, because theta=0 short-circuits.

	The short-circuit is deliberate (it keeps theta=0 bit-exact) but it means the
	usual control no longer touches this path, so the round trip is pinned here
	instead. Not bit-exact: it goes through two latent-format conversions and a
	division.
	"""
	lat, den, _, m, sigma = _flow_state()
	x, x0 = lat["samples"], den["samples"]
	ms = m.get_model_object("model_sampling")
	pin, pout = m.get_model_object("process_latent_in"), m.get_model_object("process_latent_out")
	xs, x0s = pin(x), pin(x0)
	S, G = _probe_noise_map(ms, torch.tensor(sigma), x0s)
	eps = (xs - S) / G
	back = pout(ms.noise_scaling(torch.tensor(sigma), eps, x0s, False))
	assert float((back - x).abs().max()) < 1e-5

def test_noise_rotate_probe_recovers_a_gain_that_is_not_sigma():
	# CONST applies an extra noise_scale. Assuming a gain of sigma is wrong by
	# exactly that factor, silently, as an under-denoise.
	x0 = torch.randn(1, 4, 8, 8)
	sigma = torch.tensor(0.875)
	_, G = _probe_noise_map(_FakeSampling("const", noise_scale=1.7), sigma, x0)
	assert abs(float(G.abs().mean()) - 0.875 * 1.7) < 1e-5

def test_noise_rotate_refuses_a_noise_map_it_cannot_invert():
	x0 = torch.randn(1, 4, 8, 8)
	sigma = torch.tensor(0.875)
	# Ignores its noise argument: perfectly affine, and perfectly useless. The
	# affinity check alone would PASS this, which is why the gain is checked too.
	try:
		_probe_noise_map(_FakeSampling("deaf"), sigma, x0)
	except RuntimeError as exc:
		assert "no recoverable noise component" in str(exc)
	else:
		raise AssertionError("a zero-gain noise map must be refused")
	# Linear at 0, 1 and 2 by construction, curved everywhere else: a three-point
	# probe accepts it with the right gain. The random probe is what catches it.
	try:
		_probe_noise_map(_FakeSampling("bent"), sigma, x0)
	except RuntimeError as exc:
		assert "not affine" in str(exc)
	else:
		raise AssertionError("a non-affine noise map must be refused")

def test_noise_rotate_model_and_sigma_are_a_pair():
	"""Half-wired must not fall back silently: the fallback is correct for
	additive models and wrong for flow, which is the whole bug."""
	lat, den, _, m, sigma = _flow_state()
	try:
		NoiseRotate().execute(lat, den, 45.0, 2, 900, model=m)
	except RuntimeError as exc:
		assert "carries no sigma" in str(exc)
	else:
		raise AssertionError("model without a sigma must raise")
	try:
		NoiseRotate().execute(lat, den, 45.0, 2, 900, current_sigma=sigma)
	except RuntimeError as exc:
		assert "`model` is not wired" in str(exc)
	else:
		raise AssertionError("a sigma without a model must raise")

def test_noise_rotate_takes_the_sigma_from_the_latents_stamp():
	"""The point of the stamp: nothing to wire, and it cannot be the WRONG sigma
	because the stage that produced this tensor is what wrote it."""
	lat, den, _, m, sigma = _flow_state()
	stamped = dict(lat)
	stamped["gte_sigma"] = sigma
	from_stamp = NoiseRotate().execute(stamped, den, 45.0, 2, 900, model=m)[0]["samples"]
	by_hand = NoiseRotate().execute(lat, den, 45.0, 2, 900,
									model=m, current_sigma=sigma)[0]["samples"]
	assert torch.allclose(from_stamp, by_hand, atol=0)

def test_noise_rotate_theta_zero_is_exact_on_every_path():
	lat, den, _, m, sigma = _flow_state()
	x = lat["samples"]
	for kw in ({}, {"model": m, "current_sigma": sigma}):
		out = NoiseRotate().execute(lat, den, 0.0, 3, 900, **kw)[0]["samples"]
		for k in range(3):
			assert float((out[k:k + 1] - x).abs().max()) == 0.0

def test_noise_rotate_drops_the_parents_batch_index():
	# Descendants are a new lineage; keeping the parent's slot would send anything
	# that regenerates noise to the wrong seed.
	lat, den = _parent()
	lat["batch_index"] = [3]
	assert "batch_index" not in NoiseRotate().execute(lat, den, 20.0, 2, 0)[0]

def test_noise_rotate_refuses_impossible_inputs():
	lat, den = _parent()
	# a finished latent has no residual left to turn
	try:
		NoiseRotate().execute({"samples": den["samples"].clone()}, den, 30.0, 2, 0)
		raise AssertionError("a zero residual must be refused")
	except RuntimeError as exc:
		assert "unresolved noise" in str(exc)
	# mismatched shapes mean the two inputs came from different samplers
	try:
		NoiseRotate().execute(lat, {"samples": torch.zeros(1, 4, 8, 8)}, 30.0, 2, 0)
		raise AssertionError("a shape mismatch must be refused")
	except RuntimeError as exc:
		assert "SAME sampler" in str(exc)
	# and a batch has no single parent to branch from
	try:
		NoiseRotate().execute({"samples": torch.zeros(3, 4, 16, 16)},
							  {"samples": torch.ones(3, 4, 16, 16)}, 30.0, 2, 0)
		raise AssertionError("a batched parent must be refused")
	except RuntimeError as exc:
		assert "Candidate Select" in str(exc)

# ------------------------------------------------------------------ runner
def _main():
	tests = [(n, f) for n, f in sorted(globals().items())
			if n.startswith("test_") and callable(f)]
	failed = []
	for name, fn in tests:
		try:
			fn()
			print(f"  PASS  {name}")
		except Exception as exc:  # noqa: BLE001 — this is the reporter
			failed.append((name, exc))
			print(f"  FAIL  {name}: {exc}")
	print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
	return 1 if failed else 0


if __name__ == "__main__":
	sys.exit(_main())
