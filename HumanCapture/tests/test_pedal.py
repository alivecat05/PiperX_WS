import json

from human_capture.controller import Controller
from human_capture.pedal import Pedal, save_binding
from test_capture import add_group, cameras


def test_learning_press_and_held_key_never_toggle():
    pedal = Pedal()
    pedal.bind("F13")
    assert not pedal.press("F13", 10)
    pedal.release("F13")
    assert pedal.press("F13", 11)
    assert not pedal.press("F13", 12)  # Still held, even long after debounce time.
    pedal.release("F13")
    assert pedal.press("F13", 13)


def test_contact_bounce_and_other_keys_are_ignored():
    pedal = Pedal("space", debounce_ms=250)
    assert not pedal.press("a", 1)
    assert pedal.press("space", 1)
    pedal.release("space")
    assert not pedal.press("space", 1.05)
    assert not pedal.press("space", 1.5)  # Rejected bounce still requires a release.
    pedal.release("space")
    assert pedal.press("space", 1.6)


def test_binding_preserves_camera_config(tmp_path):
    path = tmp_path/"config.json"
    cfg = {"cameras": {"head": {"serial": "keep-this"}}, "task": "中文任务"}
    path.write_text(json.dumps(cfg), encoding="utf-8")
    save_binding(path, "F13")
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["cameras"] == cfg["cameras"] and result["task"] == cfg["task"]
    assert result["pedal"]["key"] == "F13"


def test_toggle_uses_current_controller_state_and_ignores_saving(cfg):
    cams = cameras()
    for i in range(10):
        add_group(cams, cfg, 100+i/30, i)
    control = Controller(cfg, cams, mock=True)
    selection = {"task": "pedal", "index": "42", "auto_increment": True}
    control.request("record_toggle", selection=selection)
    control.tick(100)
    assert control.recording
    control.request("record_toggle", selection=selection)
    control.tick(100.15)
    assert control.recording and control.end_time == 100.15
    control.request("record_toggle", selection=selection)
    control.tick(100.16)
    assert "本次踩踏已忽略" in control.message
    control.tick(100.25)
    assert control.recorder.wait(5)
    control.tick(100.26)
    assert control.recorder.result == "complete" and control.capture.index == 43
    control.tick(100.27)
    assert not control.recording  # Saving-time presses must never start a queued episode.


def test_toggle_with_invalid_form_never_uses_previous_task(cfg):
    cams = cameras()
    add_group(cams, cfg, 100, 1)
    control = Controller(cfg, cams, mock=True)
    control.request("record_toggle", selection={"task": "", "index": "1", "auto_increment": True})
    control.tick(100)
    assert not control.recording and control.recorder.path is None
    assert control.capture.error
