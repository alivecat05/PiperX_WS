"""The live form must be the metadata used by headset B, without stale-status resets."""
import time

import pytest

from piperx_teleop.preview import Preview
from test_pipeline import cfg, make_controller, packet, step


@pytest.fixture(scope="module")
def form_view():
    # The application has one Tk interpreter for its lifetime; reuse it here too.
    view = Preview(" | FORM TEST")
    view.root.withdraw()
    yield view
    view.close()


@pytest.fixture
def window(cfg, form_view):
    control = make_controller(cfg)
    view = form_view
    view.seen_revision = -1
    view.form_dirty = False
    view.operation_handler = control.operator.request
    view.update_controls(control.status())
    try:
        yield control, view
    finally:
        control.close()


def neutral():
    data = packet()
    for side in ("left", "right"): data[side]["trigger"] = 0
    return data


def test_old_status_cannot_overwrite_current_task_edit(window):
    control, view = window
    control.capture.apply(control.capture.task, control.capture.index, True)
    old_status = control.status()
    view.task_name.set("新的任务名")
    view.update_controls(old_status)  # Delayed acknowledgement for the previous form contents.
    assert view.task_name.get() == "新的任务名" and view.form_dirty
    control.operator.apply_requests(time.perf_counter())
    view.update_controls(control.status())
    assert view.task_name.get() == control.capture.task == "新的任务名"
    assert not view.form_dirty


def test_headset_b_uses_edited_task_and_index_without_apply_click(window):
    control, view = window
    control.begin_input()
    step(control, neutral())
    view.task_name.set("用户刚修改的任务")
    view.capture_index.set("42")
    data = neutral()
    data["buttons"]["record_start"] = True
    step(control, data)
    assert control.recorder.path.parts[-2:] == ("用户刚修改的任务", "episode_042")
    view.update_controls(control.status())
    assert view.task_name.get() == "用户刚修改的任务"
    control.recorder.stop(complete=False)
    assert control.recorder.wait(3)


def test_invalid_task_edit_blocks_b_instead_of_recording_default(window):
    control, view = window
    control.begin_input()
    step(control, neutral())
    view.task_name.set("")
    data = neutral()
    data["buttons"]["record_start"] = True
    step(control, data)
    assert control.recorder.path is None and control.capture.error
    assert control.mode == "idle"


def test_editing_index_and_auto_increment_are_synchronized(window):
    control, view = window
    view.capture_index.set("+42")
    view.auto_increment.set(False)
    control.operator.apply_requests(time.perf_counter())
    view.update_controls(control.status())
    assert control.capture.index == 42 and not control.capture.auto_increment
    assert view.capture_index.get() == "42" and not view.form_dirty
