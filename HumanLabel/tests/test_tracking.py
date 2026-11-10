import numpy as np

from humanlabel.utils.tracking import contiguous_ranges, track_boxes


def boxes(left=True, right=True):
    return np.array(
        ([20, 100, 100, 200, 0.95, 0],) * int(left)
        + ([400, 100, 480, 200, 0.95, 1],) * int(right),
        dtype=np.float32,
    ).reshape(-1, 6)


def test_bounded_missing_hand_is_completed_only_between_real_anchors():
    ds = [boxes(), boxes(False), boxes(False), boxes(), boxes(False)]
    b, score, observed, supported, length = track_boxes(ds, np.arange(5), max_gap=2)
    assert supported[0].tolist() == [True, True, True, True, False]
    assert observed[0].tolist() == [True, False, False, True, False]
    np.testing.assert_allclose(b[0, 1:3], np.tile(b[0, 0], (2, 1)))
    assert length[0, 1] == 2
    assert np.isnan(b[0, -1]).all()


def test_no_completion_crosses_a_capture_discontinuity():
    ds = [boxes(), boxes(False), boxes(False), boxes()]
    _, _, observed, supported, _ = track_boxes(ds, [0, 1, 8, 9], max_gap=90)
    np.testing.assert_array_equal(supported, observed)
    assert contiguous_ranges([0, 1, 8, 9]) == [(0, 2), (2, 4)]


def test_tracker_uses_future_and_past_to_correct_one_frame_class_flip():
    ds = [boxes() for _ in range(7)]
    ds[3][:, 5] = 1 - ds[3][:, 5]
    b, _, _, _, _ = track_boxes(ds, np.arange(7))
    assert b[0, 3, 0] < 200
    assert b[1, 3, 0] > 300


def test_context_crop_recovery_respects_elapsed_capture_time():
    from humanlabel.workflows.recover import crop_candidates

    boxes = np.zeros((2, 5, 4), np.float32)
    boxes[:, 1] = [10, 20, 30, 40]
    boxes[:, 3] = [30, 40, 50, 60]
    observed = np.zeros((2, 5), bool)
    observed[:, [1, 3]] = True
    grid = np.array([0, 1, 11, 21, 22])
    recovered, support, added = crop_candidates(
        boxes, observed, observed, grid, 30, edge_seconds=0.0
    )
    np.testing.assert_allclose(recovered[:, 2], [[20, 30, 40, 50]] * 2)
    assert added[:, 2].all() and not added[:, [0, 4]].any()
    assert not crop_candidates(
        boxes, observed, observed, grid * 10, 30, edge_seconds=0.0
    )[2].any()
    assert crop_candidates(boxes, observed, observed, grid, 30)[2][:, [0, 4]].all()
