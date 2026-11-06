"""Keyboard-emulating pedal binding and press/release debouncing."""
import json
from pathlib import Path

MODIFIERS = {"shift_l", "shift_r", "control_l", "control_r", "alt_l", "alt_r",
             "meta_l", "meta_r", "super_l", "super_r", "caps_lock", "num_lock"}


class Pedal:
    def __init__(self, key=None, debounce_ms=250):
        self.key = key.lower() if key else None
        self.debounce = debounce_ms/1000
        self.down = False
        self.last_press = float("-inf")

    def bind(self, key):
        self.key = key.lower()
        self.down = True  # The identification press must be released before recording.
        self.last_press = float("-inf")

    def press(self, key, now):
        if key.lower() != self.key or self.down:
            return False
        self.down = True
        if now-self.last_press < self.debounce:
            return False
        self.last_press = now
        return True

    def release(self, key):
        if key.lower() == self.key:
            self.down = False


def save_binding(path, key):
    """Preserve all device settings when storing only the learned pedal key."""
    path = Path(path)
    source = path if path.exists() else Path(__file__).resolve().parents[1]/"config.example.json"
    cfg = json.loads(source.read_text(encoding="utf-8-sig"))
    cfg.setdefault("pedal", {})["key"] = key
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(cfg, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)
