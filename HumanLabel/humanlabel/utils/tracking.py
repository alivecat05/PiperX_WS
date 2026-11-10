"""Offline EgoHand crop tracking within continuous capture-grid segments."""

import numpy as np


def contiguous_ranges(grid):
    grid = np.asarray(grid, dtype=np.int64)
    if len(grid) == 0:
        return []
    edges = np.r_[0, np.flatnonzero(np.diff(grid) != 1) + 1, len(grid)]
    return list(zip(edges[:-1], edges[1:]))


def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def track_boxes(detections, grid, max_gap=90):
    """Viterbi association uses both time directions and soft handedness costs.

    Candidate states contain one box per anatomical side or a missing state.
    A box cannot be assigned to both hands. Crossings are allowed.
    """
    n = len(detections)
    boxes = np.full((2, n, 4), np.nan, np.float32)
    scores = np.zeros((2, n), np.float32)
    for start, end in contiguous_ranges(grid):
        states, costs, links = [], [], []
        for t in range(start, end):
            d = detections[t]
            # Keep a bounded state space even when the detector sees background.
            if len(d) > 6:
                d = d[np.argsort(d[:, 4])[-6:]]
            options = [-1, *range(len(d))]
            st = [(l, r) for l in options for r in options if l != r or l == -1]
            unary = []
            for pair in st:
                cost = 0.0
                for side, j in enumerate(pair):
                    if j == -1:
                        cost += 1.65
                    else:
                        cost += -np.log(max(float(d[j, 4]), 0.001))
                        cost += 2.2 * (int(d[j, 5]) != side)
                unary.append(cost)
            if not states:
                costs.append(np.asarray(unary))
                links.append(None)
            else:
                pd = states[-1][1]
                transition = np.zeros((len(states[-1][0]), len(st)))
                for a, previous in enumerate(states[-1][0]):
                    for b, current in enumerate(st):
                        for j, k in zip(previous, current):
                            if j < 0 or k < 0:
                                transition[a, b] += 0.35 * (j != k)
                                continue
                            p, q = pd[j, :4], d[k, :4]
                            size = max(
                                np.max(p[2:] - p[:2]), np.max(q[2:] - q[:2]), 30.0
                            )
                            motion = (
                                np.linalg.norm((p[:2] + p[2:] - q[:2] - q[2:]) / 2)
                                / size
                            )
                            scale = abs(
                                np.log(
                                    max(np.max(q[2:] - q[:2]), 1.0)
                                    / max(np.max(p[2:] - p[:2]), 1.0)
                                )
                            )
                            transition[a, b] += min(5.0, motion * 3.0) + scale
                total = costs[-1][:, None] + transition
                links.append(total.argmin(axis=0))
                costs.append(np.asarray(unary) + total.min(axis=0))
            states.append((st, d))
        k = int(costs[-1].argmin())
        for local in range(end - start - 1, -1, -1):
            st, d = states[local]
            for side, j in enumerate(st[k]):
                if j >= 0:
                    boxes[side, start + local] = d[j, :4]
                    scores[side, start + local] = d[j, 4]
            if local:
                k = int(links[local][k])
    observed = np.isfinite(boxes).all(-1)
    supported = observed.copy()
    gap_length = np.zeros((2, n), np.int32)
    for start, end in contiguous_ranges(grid):
        for side in range(2):
            for a, b in runs(~observed[side, start:end]):
                a, b = a + start, b + start
                gap_length[side, a:b] = b - a
                if a == start or b == end or b - a > max_gap:
                    continue
                f = np.arange(1, b - a + 1)[:, None] / (b - a + 1)
                boxes[side, a:b] = boxes[side, a - 1] * (1 - f) + boxes[side, b] * f
                supported[side, a:b] = True
    return boxes, scores, observed, supported, gap_length
