"""Deterministic, stratified motion sampling for the v2 synthetic camera cache.

One motion per clip: family in {pan, tilt, zoom} with a direction, plus an
independent strength u in [u_min, u_max] (fraction of the trained maximum).
Static clips are handled by the cache builder (one per source) and are not drawn
here.

Balance is by construction, not by chance:

* directions: sources of each split are ordered by a seeded hash; consecutive blocks
  of six get a seeded permutation of the six directions, so every direction count
  differs by at most one per split and clip slot;
* later clips of a source are shifted by a fixed offset in that order, which moves
  them to a different *family* (for K <= 3) and keeps the shift bijective, so the
  balance property holds for every clip slot;
* strength: within each (split, direction) the available strata
  [u_min, u_max] are cut into n equal bins and assigned by a seeded permutation with
  a hash jitter, giving a near-uniform u marginal per direction.

Everything depends only on (seed, source population, source id, clip index), never
on call order, shard or process, so a restart or a different shard reproduces it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .flow_geometry import V2_MOTIONS, v2_mode

U_MIN = 0.15
U_MAX = 1.0


@dataclass(frozen=True)
class MotionRequest:
    """One sampled motion: family (pan/tilt/zoom), direction and strength u in [u_min, u_max]."""

    family: str
    direction: str | None
    u: float

    @property
    def mode(self):
        """v1-style mode string, e.g. ``pan_left``."""
        return v2_mode(self.family, self.direction)


STATIC = MotionRequest("static", None, 0.0)


def _digest(*parts) -> bytes:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()


def hash_uniform(*parts) -> float:
    """Deterministic uniform [0,1) from arbitrary parts."""
    return int.from_bytes(_digest(*parts)[:8], "big") / 2.0**64


def _rng(*parts):
    return np.random.default_rng(int.from_bytes(_digest(*parts)[:8], "big"))


def _clip_shift(clip_index: int) -> int:
    # +2 moves to the next family; every 3rd clip adds 1 to also change direction.
    return (2 * clip_index + clip_index // 3) % len(V2_MOTIONS)


def sample_motion(source_id, clip_index, seed, *, u_min=U_MIN, u_max=U_MAX):
    """Population-free fallback: independent hash draw (not balanced).

    Prefer :class:`MotionSampler`, which stratifies over the dataset.
    """
    family, direction = V2_MOTIONS[
        int(hash_uniform(seed, "dir", source_id, clip_index) * len(V2_MOTIONS))
    ]
    u = u_min + (u_max - u_min) * hash_uniform(seed, "u", source_id, clip_index)
    return MotionRequest(family, direction, round(float(u), 4))


class MotionSampler:
    """Balanced (family, direction, u) assignment for a fixed source population."""

    def __init__(
        self, entries, *, clips_per_source=2, seed=0, u_min=U_MIN, u_max=U_MAX
    ):
        """entries: iterable of (source_id, split). Balance is per split."""
        if not 0 <= u_min < u_max <= 1:
            raise ValueError("Need 0 <= u_min < u_max <= 1")
        entries = [(str(s), str(sp)) for s, sp in entries]
        if len({s for s, _ in entries}) != len(entries):
            raise ValueError("Duplicate source ids")
        self.seed, self.k = seed, int(clips_per_source)
        self.u_min, self.u_max = u_min, u_max
        n_dir = len(V2_MOTIONS)
        table = {}  # (source_id, clip) -> [direction_index, u]
        by_split = {}
        for source, split in entries:
            by_split.setdefault(split, []).append(source)
        for split, sources in sorted(by_split.items()):
            # balance by construction: order sources by a seeded hash, give each block of six consecutive sources a seeded
            # permutation of the six directions, then stratify u within each (split, direction) so the u marginal is near uniform.
            # Everything depends only on (seed, population, source id, clip index), never on call order or sharding.
            ordered = sorted(sources, key=lambda s: _digest(seed, "rank", s))
            for block, start in enumerate(range(0, len(ordered), n_dir)):
                perm = _rng(seed, "block", split, block).permutation(n_dir)
                for offset, source in enumerate(ordered[start : start + n_dir]):
                    base = int(perm[offset])
                    for c in range(self.k):
                        table[(source, c)] = [(base + _clip_shift(c)) % n_dir, 0.0]
            members = set(sources)
            for d in range(n_dir):
                keys = sorted(
                    (k for k, v in table.items() if k[0] in members and v[0] == d),
                    key=lambda k: (k[1], _digest(seed, "u-order", k[0])),
                )
                n = len(keys)
                if not n:
                    continue
                strata = _rng(seed, "u-perm", split, d).permutation(n)
                for key, stratum in zip(keys, strata):
                    jitter = hash_uniform(seed, "u-jit", key[0], key[1])
                    frac = (int(stratum) + jitter) / n
                    table[key][1] = round(u_min + (u_max - u_min) * frac, 4)
        self._table = table
        self._splits = dict(entries)

    def sample(self, source_id, clip_index) -> MotionRequest:
        """The pre-computed (family, direction, u) of ``source_id``'s ``clip_index``-th moving clip."""
        family, direction = V2_MOTIONS[self._table[(source_id, clip_index)][0]]
        u = self._table[(source_id, clip_index)][1]
        # rounding can nudge to the bounds only; clamp keeps the contract exact
        return MotionRequest(family, direction, min(max(u, self.u_min), self.u_max))

    def report(self):
        """Counts per split/direction and u summary, for logs and the data paper."""
        out = {}
        for (source, _), (d, u) in self._table.items():
            split = self._splits[source]
            row = out.setdefault(split, {"direction_counts": {}, "u": []})
            name = "_".join(V2_MOTIONS[d])
            row["direction_counts"][name] = row["direction_counts"].get(name, 0) + 1
            row["u"].append(u)
        for row in out.values():
            u = np.asarray(row.pop("u"))
            row["u_mean"], row["u_min"], row["u_max"] = (
                float(u.mean()),
                float(u.min()),
                float(u.max()),
            )
            row["u_hist_5bins"] = np.histogram(
                u, bins=5, range=(self.u_min, self.u_max)
            )[0].tolist()
        return out
