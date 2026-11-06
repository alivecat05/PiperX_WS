"""Exercise the actual Tk buttons and full-resolution camera preview together."""
import json
import time

from conftest import ROOT
from human_capture.cameras import Cameras
from human_capture.controller import Controller
from human_capture.preview import Preview


def test_full_resolution_gui_recording_and_form_sync(tmp_path):
    cfg = json.loads((ROOT/"config.example.json").read_text(encoding="utf-8"))
    cfg["output"] = str(tmp_path/"episodes")
    cameras = Cameras(cfg, mock=True)
    control = Controller(cfg, cameras, mock=True)
    config_path = tmp_path/"config.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    view = Preview(control, cameras, mock=True, config_path=config_path)
    try:
        cameras.start()
        control.start()
        view.refresh()

        def pump(duration):
            until = time.perf_counter()+duration
            while time.perf_counter() < until:
                view.root.update()
                time.sleep(.005)

        pump(.2)
        assert control.status()["ready"]
        view.task.set("真人采集界面验证")
        view.index.set("42")
        # A delayed snapshot from before edits must not reset the task name.
        view.update_controls(control.status())
        assert view.task.get() == "真人采集界面验证"
        pump(.1)
        view.start_button.invoke()
        pump(1.5)
        assert control.status()["recording"]
        assert view.heading.cget("text") == "● 采集中"
        assert len(view.photos) == 3
        view.stop_button.invoke()
        deadline = time.perf_counter()+5
        while (control.status()["busy"] or control.status()["result"] is None) and time.perf_counter() < deadline:
            pump(.05)
        pump(.1)
        assert not control.status()["busy"]
        assert control.recorder.count >= 40
        assert control.recorder.result == "complete", control.status()
        assert view.task.get() == "真人采集界面验证"
        assert view.index.get() == "43"
        assert view.heading.cget("text") == "● 就绪 · 未采集"
        # Explicit suggestion also updates the form after a user edit.
        view.index.set("999")
        view.next_button.invoke()
        pump(.1)
        assert view.index.get() == "43" and control.capture.index == 43
        # Identify an unknown keyboard pedal without starting a recording.
        view.learn_button.invoke()
        view.root.focus_force()
        view.root.event_generate("<KeyPress-space>")
        pump(.05)
        assert view.pedal.key == "space" and not control.status()["recording"]
        assert json.loads(config_path.read_text(encoding="utf-8"))["pedal"]["key"] == "space"
        view.root.event_generate("<KeyRelease-space>")
        pump(.07)
        task_before = view.task.get()
        view.task_entry.focus_force()
        view.task_entry.event_generate("<KeyPress-space>")
        pump(.3)
        assert control.status()["recording"]
        for _ in range(8):
            view.task_entry.event_generate("<KeyPress-space>")
            pump(.02)
        assert control.status()["recording"] and view.task.get() == task_before
        view.task_entry.event_generate("<KeyRelease-space>")
        pump(.07)
        view.task_entry.event_generate("<KeyPress-space>")
        pump(.2)
        view.task_entry.event_generate("<KeyRelease-space>")
        deadline = time.perf_counter()+5
        while control.status()["busy"] and time.perf_counter() < deadline:
            pump(.05)
        pump(.05)
        assert control.recorder.result == "complete" and view.index.get() == "44"
    finally:
        control.close()
        cameras.close()
        view.destroy()
