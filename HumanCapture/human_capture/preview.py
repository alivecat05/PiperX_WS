"""White native UI with explicit recording controls and live three-camera preview."""
import tkinter as tk
from tkinter import ttk
import time

import cv2

from .core import CAMERAS
from .pedal import MODIFIERS, Pedal, save_binding

LABELS = {"head": "头部 · D555", "left_wrist": "左腕 · D405i", "right_wrist": "右腕 · D405i"}


class Preview:
    def __init__(self, controller, cameras, mock=False, config_path=None):
        self.controller, self.cameras = controller, cameras
        settings = controller.cfg.get("pedal", {})
        self.pedal = Pedal(settings.get("key"), settings.get("debounce_ms", 250))
        self.config_path = config_path
        self.learning_pedal = False
        self.pedal_release = None
        self.root = tk.Tk()
        self.root.title("HumanCapture | 人类三相机同步采集"+(" | 模拟模式" if mock else ""))
        self.root.geometry("1440x800")
        self.root.minsize(1080, 660)
        self.root.configure(background="white")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background="white", foreground="#253349", font=("Microsoft YaHei UI", 11))
        style.configure("TEntry", fieldbackground="white")
        style.configure("TSpinbox", fieldbackground="white")
        style.configure("TButton", padding=(12, 9), background="#f1f5f9")
        style.configure("Start.TButton", background="#e7f8ed", foreground="#13723a", font=("Microsoft YaHei UI", 14, "bold"))
        style.configure("Stop.TButton", background="#fff0f1", foreground="#b51d35", font=("Microsoft YaHei UI", 14, "bold"))
        self.closing = False
        self.syncing = self.dirty = False
        self.seen_revision = -1
        self.photos, self.last_frames = {}, {}
        self.stripe = tk.Frame(self.root, height=9, background="#617187")
        self.stripe.pack(fill="x")
        self.band = tk.Frame(self.root, background="#f4f7fb", padx=22, pady=18)
        self.band.pack(fill="x")
        self.heading = tk.Label(self.band, text="等待相机就绪", anchor="w",
                                font=("Microsoft YaHei UI", 27, "bold"))
        self.heading.pack(fill="x")
        self.summary = tk.Label(self.band, anchor="w", font=("Microsoft YaHei UI", 12))
        self.summary.pack(fill="x", pady=(7, 0))
        form = ttk.Frame(self.root, padding=(20, 16))
        form.pack(fill="x")
        ttk.Label(form, text="任务名").grid(row=0, column=0, padx=(0, 8))
        self.task = tk.StringVar(master=self.root)
        self.index = tk.StringVar(master=self.root, value="1")
        self.auto = tk.BooleanVar(master=self.root, value=True)
        self.task_entry = ttk.Entry(form, textvariable=self.task)
        self.task_entry.grid(row=0, column=1, sticky="ew", padx=(0, 18))
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="采集序号").grid(row=0, column=2, padx=(0, 8))
        self.index_entry = ttk.Spinbox(form, from_=1, to=999999999, textvariable=self.index, width=8)
        self.index_entry.grid(row=0, column=3, padx=(0, 14))
        self.auto_check = ttk.Checkbutton(form, text="自动递增", variable=self.auto)
        self.auto_check.grid(row=0, column=4, padx=(0, 14))
        self.next_button = ttk.Button(form, text="下一个可用序号", command=self.next_index)
        self.next_button.grid(row=0, column=5)
        actions = ttk.Frame(self.root, padding=(20, 0))
        actions.pack(fill="x")
        self.start_button = ttk.Button(actions, text="● 开始录制", style="Start.TButton", command=self.start_capture)
        self.start_button.pack(side="left", ipadx=30, ipady=7)
        self.stop_button = ttk.Button(actions, text="■ 结束录制并保存", style="Stop.TButton",
                                      command=lambda: self.controller.request("record_stop"))
        self.stop_button.pack(side="left", padx=16, ipadx=30, ipady=7)
        self.learn_button = ttk.Button(actions, text="识别踏板", command=self.learn_pedal)
        self.learn_button.pack(side="left", padx=(8, 12))
        self.pedal_note = tk.StringVar(master=self.root, value=self.pedal_description())
        ttk.Label(actions, textvariable=self.pedal_note, wraplength=350).pack(side="left", fill="x")
        self.note = ttk.Label(self.root, padding=(20, 12), wraplength=1350, justify="left")
        self.note.pack(fill="x")
        views = ttk.Frame(self.root, padding=(16, 4))
        views.pack(fill="both", expand=True)
        views.rowconfigure(1, weight=1)
        self.labels, self.images = {}, {}
        for col, name in enumerate(CAMERAS):
            views.columnconfigure(col, weight=1, uniform="camera")
            label = ttk.Label(views, text=LABELS[name], anchor="center", font=("Microsoft YaHei UI", 12, "bold"))
            label.grid(row=0, column=col, sticky="ew", padx=5, pady=8)
            image = tk.Label(views, text="等待实时视频…", background="#f4f7fb", foreground="#617187")
            image.grid(row=1, column=col, sticky="nsew", padx=5)
            self.labels[name], self.images[name] = label, image
        self.detail = ttk.Label(self.root, padding=(20, 12), wraplength=1350, justify="left")
        self.detail.pack(fill="x")
        for var in (self.task, self.index, self.auto):
            var.trace_add("write", self.mark_dirty)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda _: self.close())
        # Before Entry/Button class bindings: pedal Space/Return must not edit text
        # or also activate a focused button. Bindings are confined to this window.
        self.pedal_tag = f"HumanCapturePedal{id(self)}"
        self.root.bind_class(self.pedal_tag, "<KeyPress>", self.on_pedal_press)
        self.root.bind_class(self.pedal_tag, "<KeyRelease>", self.on_pedal_release)
        self._tag_pedal_widgets(self.root)

    def _tag_pedal_widgets(self, widget):
        widget.bindtags((self.pedal_tag, *widget.bindtags()))
        for child in widget.winfo_children():
            self._tag_pedal_widgets(child)

    def pedal_description(self):
        return (f"踏板键：{self.pedal.key} · 踩一下开始 / 再踩结束" if self.pedal.key
                else "踏板尚未识别 · 点击识别后踩一下")

    def learn_pedal(self):
        if self.controller.status()["busy"]:
            return
        self.learning_pedal = True
        self.pedal_note.set("请踩一下踏板（仅识别，不录制）；Esc 取消")
        self.learn_button.focus_set()

    def on_pedal_press(self, event):
        key = event.keysym.lower()
        if self.learning_pedal:
            if key == "escape":
                self.learning_pedal = False
                self.pedal_note.set(self.pedal_description())
            elif key not in MODIFIERS:
                self.learning_pedal = False
                if self.pedal_release is not None:
                    self.root.after_cancel(self.pedal_release)
                    self.pedal_release = None
                self.pedal.bind(key)
                self.controller.cfg.setdefault("pedal", {})["key"] = key
                self.pedal_note.set(self.pedal_description())
                if self.config_path is not None:
                    try:
                        save_binding(self.config_path, key)
                    except (OSError, ValueError) as error:
                        self.pedal_note.set(f"已绑定 {key}，本次可用；保存配置失败：{error}")
            return "break"
        if key != self.pedal.key:
            return None
        if self.pedal_release is not None:
            self.root.after_cancel(self.pedal_release)
            self.pedal_release = None
        if self.pedal.press(key, time.perf_counter()):
            self.controller.request("record_toggle", selection=self.selection())
        return "break"

    def on_pedal_release(self, event):
        key = event.keysym.lower()
        if key == self.pedal.key:
            if self.pedal_release is not None:
                self.root.after_cancel(self.pedal_release)
            # Some Tk platforms emit release/press pairs during auto-repeat.
            def release():
                self.pedal.release(key)
                self.pedal_release = None
            self.pedal_release = self.root.after(35, release)
            return "break"

    def selection(self):
        return {"task": self.task.get().strip(), "index": self.index.get(), "auto_increment": self.auto.get()}

    def mark_dirty(self, *_):
        if not self.syncing:
            self.dirty = True
            self.controller.request("capture_settings", **self.selection())

    def next_index(self):
        try:
            # Make the suggested index part of the visible form so stale replies cannot
            # leave a different index in the input and in the controller.
            self.index.set(str(self.controller.recorder.next_index(self.task.get().strip())))
        except ValueError as error:
            self.note.configure(text=str(error))

    def start_capture(self):
        self.controller.request("record_start", selection=self.selection())

    def update_controls(self, status):
        capture = status["capture"]
        shown = capture["active"] or capture
        selection = self.selection()
        try:
            index = int(selection["index"])
        except ValueError:
            index = None
        matches = (selection["task"] == shown["task"] and index == shown["index"]
                   and selection["auto_increment"] == capture["auto_increment"])
        if capture["revision"] != self.seen_revision and (capture["active"] or not self.dirty or matches):
            self.syncing = True
            self.task.set(shown["task"])
            self.index.set(str(shown["index"]))
            self.auto.set(capture["auto_increment"])
            self.syncing = self.dirty = False
            self.seen_revision = capture["revision"]
        if status["recording"]:
            color, surface, title = "#bf233c", "#fff0f3", "● 采集中"
        elif status["saving"]:
            color, surface, title = "#9d6414", "#fff9ed", "正在保存 · 请等待"
        elif status["ready"]:
            color, surface, title = "#168447", "#effaf3", "● 就绪 · 未采集"
        else:
            color, surface, title = "#9d6414", "#fff9ed", "等待就绪 · 请查看提示"
        self.stripe.configure(background=color)
        self.band.configure(background=surface)
        self.heading.configure(text=title, background=surface, foreground=color)
        self.summary.configure(background=surface, foreground="#334155",
            text=f"任务：{shown['task']}    |    {'当前' if capture['active'] else '下一段'}序号：{shown['index']:03d}"
                 f"    |    已保存 {status['frames']} 帧    |    丢帧 {status['dropped']}    |    写盘 {status['write_fps']:.1f} FPS")
        self.note.configure(text=capture["error"] or "；".join(status["problems"]) or status["message"])
        for widget in (self.task_entry, self.index_entry, self.auto_check, self.next_button):
            widget.state(["disabled"] if status["busy"] else ["!disabled"])
        self.start_button.state(["!disabled"] if status["ready"] and not status["busy"] else ["disabled"])
        self.stop_button.state(["!disabled"] if status["recording"] else ["disabled"])
        self.learn_button.state(["disabled"] if status["busy"] else ["!disabled"])
        for name in CAMERAS:
            camera = status["cameras"][name]
            clock = {"global_time": "SDK 全局时钟", "system_time": "SDK 系统时钟",
                     "host_monotonic": "模拟时钟", "host_receive": "未同步，不能录制"}.get(camera["clock"], camera["clock"])
            self.labels[name].configure(text=f"{LABELS[name]}   {camera['fps']:.1f} FPS   帧 {camera['frame_number']}\n{clock}")
        self.detail.configure(text="30 Hz 采集 · 三路来源时间戳差 ≤33.33 ms · 不重复帧补数\n"
                              +(f"最近输出：{status['path']}    {status['result'] or ''}    "
                                +str(status["drop_reasons"]) if status["path"] else "任务和序号编辑后自动同步；开始录制使用当前输入。"))

    def refresh(self):
        if self.closing:
            return
        self.update_controls(self.controller.status())
        for name in CAMERAS:
            frame = self.cameras.histories[name].latest()
            widget = self.images[name]
            width, height = max(1, widget.winfo_width()), max(1, widget.winfo_height())
            key = (frame.value["frame_number"], width, height) if frame else None
            if frame is None or self.last_frames.get(name) == key or width < 10:
                continue
            bgr = frame.value["bgr"]
            scale = min(width/bgr.shape[1], height/bgr.shape[0], 1.)
            rgb = cv2.cvtColor(cv2.resize(bgr, (max(1, int(bgr.shape[1]*scale)), max(1, int(bgr.shape[0]*scale)))), cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            self.photos[name] = tk.PhotoImage(master=self.root, data=f"P6\n{w} {h}\n255\n".encode()+rgb.tobytes(), format="PPM")
            widget.configure(image=self.photos[name], text="")
            self.last_frames[name] = key
        self.root.after(33, self.refresh)

    def close(self):
        self.closing = True
        self.root.quit()

    def run(self):
        self.refresh()
        self.root.mainloop()

    def destroy(self):
        if self.pedal_release is not None:
            self.root.after_cancel(self.pedal_release)
        self.root.destroy()
