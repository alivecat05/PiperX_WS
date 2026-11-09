"""Native capture window; Tk runs on the application's main thread."""
from collections import deque
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np

from .core import CAMERAS
from .operator_guidance import primary_message, recovery_guidance


def banner_style(status):
    if status["mode"] == "fault":
        return "#a85413", "故障 · 请查看状态"
    if status["mode"] == "recovery":
        return "#a85413", "保护保持 · 等待恢复"
    if status["mode"] == "replaying":
        return "#176bb0", "定位 / 回放中 · 可随时停止"
    if status["recording"]:
        return "#cf233a", "● 采集中"
    if status["saving"]:
        return "#b67a10", "保存中 · 请稍候"
    if status["capture"].get("error"):
        return "#a85413", "采集设置待处理"
    if status["ready"]:
        return "#168a46", "● 就绪 · 未采集"
    return "#46556b", "等待就绪 · 未采集"


class Preview:
    title = "PiperX | 双臂遥操与采集"

    def __init__(self, suffix=""):
        self.title += suffix
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.geometry("1440x820")
        self.root.minsize(1050, 650)
        self.root.configure(background="white")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background="white", foreground="#1f2937", font=("Microsoft YaHei UI", 10))
        style.configure("TFrame", background="white")
        style.configure("TLabel", background="white")
        style.configure("TEntry", fieldbackground="white", foreground="#172033")
        style.configure("TSpinbox", fieldbackground="white", foreground="#172033")
        style.configure("TButton", background="#f8fafc", padding=(8, 5))
        style.map("TButton", background=[("active", "#eef2f7"), ("disabled", "#f8fafc")],
                  foreground=[("disabled", "#94a3b8")])
        style.configure("Recover.TButton", background="#eff6ff", foreground="#1d4ed8",
                        font=("Microsoft YaHei UI", 12, "bold"), padding=(18, 12))
        self.exit_requested = False
        self.recover_requested = False
        self.commands = deque()
        self.photo = None
        self.seen_revision = -1
        self.form_dirty = False
        self.syncing_form = False
        self.was_replay_active = False
        self.band_mark = tk.Frame(self.root, background="#46556b", height=8)
        self.band_mark.pack(side="top", fill="x")
        self.band = tk.Frame(self.root, background="#f8fafc", padx=18, pady=10)
        self.band.pack(side="top", fill="x")
        self.band_title = tk.Label(self.band, text="等待就绪 · 未采集", background="#f8fafc",
                                   foreground="#46556b", font=("Microsoft YaHei UI", 24, "bold"), anchor="w")
        self.band_title.pack(fill="x")
        self.band_task = tk.Label(self.band, background="#f8fafc", foreground="#334155",
                                  font=("Microsoft YaHei UI", 12), anchor="w")
        self.band_task.pack(fill="x")
        self.recovery_card = ttk.Frame(self.root, padding=(18, 10))
        recovery_head = ttk.Frame(self.recovery_card)
        recovery_head.pack(fill="x")
        self.recovery_title = tk.StringVar(master=self.root)
        self.recovery_title_label = ttk.Label(recovery_head, textvariable=self.recovery_title, foreground="#9a4d0d",
                                              font=("Microsoft YaHei UI", 14, "bold"))
        self.recovery_title_label.pack(side="left", fill="x", expand=True)
        self.recover_button = ttk.Button(recovery_head, text="恢复遥操", style="Recover.TButton",
                                         command=self.request_recovery)
        self.recover_button.pack(side="right", padx=(12, 0))
        self.recovery_steps = tk.StringVar(master=self.root)
        self.recovery_steps_label = ttk.Label(self.recovery_card, textvariable=self.recovery_steps, justify="left",
                                               wraplength=1320)
        self.recovery_steps_label.pack(anchor="w", pady=(4, 4))
        self.hold_text = tk.StringVar(master=self.root)
        ttk.Label(self.recovery_card, textvariable=self.hold_text, foreground="#9a4d0d").pack(anchor="w")
        panel = ttk.Frame(self.root, padding=(12, 8))
        self.panel = panel
        panel.pack(side="top", fill="x")
        form = ttk.Frame(panel)
        form.pack(fill="x")
        self.task_name = tk.StringVar(master=self.root)
        self.capture_index = tk.StringVar(master=self.root, value="1")
        self.auto_increment = tk.BooleanVar(master=self.root, value=True)
        ttk.Label(form, text="任务名").pack(side="left")
        self.task_entry = ttk.Entry(form, textvariable=self.task_name, width=30)
        self.task_entry.pack(side="left", fill="x", expand=True, padx=(6, 12))
        ttk.Label(form, text="采集序号").pack(side="left")
        self.index_entry = ttk.Entry(form, textvariable=self.capture_index, width=8)
        self.index_entry.pack(side="left", padx=6)
        self.auto_check = ttk.Checkbutton(form, text="自动递增", variable=self.auto_increment)
        self.auto_check.pack(side="left", padx=6)
        self.apply_button = ttk.Button(form, text="应用设置", command=self.apply_capture)
        self.apply_button.pack(side="left", padx=4)
        self.next_button = ttk.Button(form, text="下一个可用序号", command=lambda: self.command(
            "suggest_index", task=self.task_name.get().strip(), auto_increment=self.auto_increment.get()))
        self.next_button.pack(side="left", padx=4)
        self.record_button = ttk.Button(form, text="开始采集 B", command=self.start_capture)
        self.record_button.pack(side="left", padx=(12, 4))
        self.stop_button = ttk.Button(form, text="停止并保存 A", command=lambda: self.command("record_stop"))
        self.stop_button.pack(side="left", padx=4)
        self.settings_note = tk.StringVar(master=self.root, value="任务和序号会自动同步；Quest B 使用当前有效设置。")
        ttk.Label(panel, textvariable=self.settings_note).pack(anchor="w", pady=(5, 8))
        replay_bar = ttk.Frame(panel)
        replay_bar.pack(fill="x")
        ttk.Label(replay_bar, text="Action CSV 路径").pack(side="left")
        self.csv_path = tk.StringVar(master=self.root)
        self.csv_entry = ttk.Entry(replay_bar, textvariable=self.csv_path, width=50)
        self.csv_entry.pack(side="left", fill="x", expand=True, padx=6)
        self.load_button = ttk.Button(replay_bar, text="加载 CSV", command=lambda: self.command(
            "load_replay", path=self.csv_path.get()))
        self.load_button.pack(side="left", padx=4)
        ttk.Label(replay_bar, text="回放倍率").pack(side="left", padx=(12, 3))
        self.replay_speed = tk.StringVar(master=self.root, value="1.0")
        self.speed_entry = ttk.Spinbox(replay_bar, from_=.1, to=1., increment=.1,
                                      textvariable=self.replay_speed, width=5)
        self.speed_entry.pack(side="left", padx=4)
        self.play_button = ttk.Button(replay_bar, text="到起点并回放", command=lambda: self.command(
            "replay_start", speed=self.replay_speed.get()))
        self.play_button.pack(side="left", padx=4)
        self.stop_replay_button = ttk.Button(replay_bar, text="停止回放（空格）",
                                             command=lambda: self.command("replay_stop"))
        self.stop_replay_button.pack(side="left", padx=4)
        self.replay_note = tk.StringVar(master=self.root, value="粘贴 CSV 完整路径后加载；加载不会启动运动。")
        ttk.Label(panel, textvariable=self.replay_note).pack(anchor="w", pady=(5, 0))
        ttk.Label(panel, text="启动追踪前，让头显正对机械臂基座 X 轴方向；接管时会自动记录当前零点。",
                  foreground="#005a9e").pack(anchor="w", pady=(5, 0))
        self.message = tk.StringVar(master=self.root)
        self.message_label = ttk.Label(panel, textvariable=self.message, wraplength=1350, foreground="#005a9e")
        self.message_label.pack(anchor="w", pady=(5, 0))
        footer = ttk.Frame(self.root, padding=(12, 6))
        footer.pack(side="bottom", fill="x")
        self.device_text = tk.StringVar(master=self.root)
        self.camera_warning_text = tk.StringVar(master=self.root)
        self.follow_text = tk.StringVar(master=self.root)
        self.calibration_text = tk.StringVar(master=self.root)
        self.motion_warning_text = tk.StringVar(master=self.root)
        ttk.Label(footer, textvariable=self.device_text).pack(anchor="w")
        ttk.Label(footer, textvariable=self.camera_warning_text, foreground="#9a4d0d", wraplength=1320).pack(anchor="w")
        ttk.Label(footer, textvariable=self.follow_text).pack(anchor="w", pady=3)
        ttk.Label(footer, textvariable=self.calibration_text, foreground="#005a9e").pack(anchor="w")
        ttk.Label(footer, textvariable=self.motion_warning_text, foreground="#9a4d0d", wraplength=1320).pack(anchor="w")
        self.show_details = tk.BooleanVar(master=self.root, value=False)
        ttk.Checkbutton(footer, text="显示详细诊断", variable=self.show_details,
                        command=self.toggle_details).pack(anchor="w")
        self.details_text = tk.StringVar(master=self.root)
        self.details_label = ttk.Label(footer, textvariable=self.details_text, wraplength=1320)
        headers = tk.Frame(self.root, background="white")
        headers.pack(side="top", fill="x", padx=12, pady=(8, 0))
        self.camera_labels = {}
        for index, (name, text) in enumerate(zip(CAMERAS, ("头部相机", "左腕相机", "右腕相机"))):
            headers.columnconfigure(index, weight=1, uniform="camera")
            label = tk.Label(headers, text=text, background="white", foreground="#334155",
                             anchor="w", font=("Microsoft YaHei UI", 12, "bold"))
            label.grid(row=0, column=index, sticky="ew", padx=6)
            self.camera_labels[name] = label
        self.views = tk.Label(self.root, background="white", borderwidth=0)
        self.views.pack(side="top", fill="both", expand=True)
        self.root.protocol("WM_DELETE_WINDOW", self.request_close)
        self.root.bind("<KeyPress>", self.on_key)
        for variable in (self.task_name, self.capture_index, self.auto_increment):
            variable.trace_add("write", self.mark_dirty)

    def command(self, operation, **arguments):
        handler = getattr(self, "operation_handler", None)
        if handler is not None:
            handler(operation, **arguments)
        else:
            self.commands.append((operation, arguments))

    def selection(self):
        return dict(task=self.task_name.get().strip(), index=self.capture_index.get(),
                    auto_increment=self.auto_increment.get())

    def apply_capture(self):
        self.command("capture_settings", **self.selection())

    def start_capture(self):
        self.command("record_start", selection=self.selection())

    def mark_dirty(self, *args):
        if not self.syncing_form:
            self.form_dirty = True
            self.settings_note.set("正在自动同步任务和序号；输入无效时会阻止开始采集。")
            self.command("capture_settings", **self.selection())

    def on_key(self, event):
        if isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Spinbox, ttk.Spinbox)):
            return
        key = event.keysym.lower()
        if key in ("q", "escape"):
            self.request_close()
        elif key == "r":
            self.recover_requested = True
        elif key == "space":
            self.command("replay_stop")
        elif key == "b":
            self.start_capture()
        elif key == "a":
            self.command("record_stop")

    def request_close(self):
        self.exit_requested = True

    def request_recovery(self):
        self.recover_requested = True

    def toggle_details(self):
        if self.show_details.get():
            self.details_label.pack(anchor="w", pady=4)
        else:
            self.details_label.pack_forget()

    def update_controls(self, status):
        capture, replay = status["capture"], status["replay"]
        color, text = banner_style(status)
        if status["mode"] == "replaying":
            text = "定位到轨迹起点" if replay.get("phase") != "playing" else f"轨迹回放中 · {replay['progress']*100:.1f}%"
        surface = {"#cf233a": "#fff4f5", "#168a46": "#f0fbf4", "#a85413": "#fffaf3",
                   "#b67a10": "#fffaf3", "#176bb0": "#f3f8ff"}.get(color, "#f8fafc")
        self.band_mark.configure(background=color)
        for widget in (self.band, self.band_title, self.band_task):
            widget.configure(background=surface)
        self.band_title.configure(text=text, foreground=color)
        if status["mode"] == "recovery":
            guide = recovery_guidance(status)
            if not self.recovery_card.winfo_manager():
                self.recovery_card.pack(before=self.panel, fill="x")
            self.recovery_title_label.configure(wraplength=max(600, self.root.winfo_width()-260))
            self.recovery_steps_label.configure(wraplength=max(700, self.root.winfo_width()-60))
            self.recovery_title.set(guide["title"])
            self.recovery_steps.set("原因："+guide["cause"]+"\n"+"\n".join(guide["steps"]))
            self.recover_button.state(["!disabled"] if guide["can_recover"] else ["disabled"])
            hold_labels = []
            for side, name in (("left", "左臂"), ("right", "右臂")):
                raw = status["recovery_hold"].get(side, "")
                label = ("示教中，程序暂停控制" if "teaching" in raw else
                         "原位保持已确认" if raw == "joint hold confirmed" else
                         "保持未确认，请扶稳" if "NOT confirmed" in raw else "正在等待保持反馈")
                hold_labels.append(f"{name}：{label}")
            self.hold_text.set("  |  ".join(hold_labels))
        else:
            if self.recovery_card.winfo_manager():
                self.recovery_card.pack_forget()
        shown = capture["active"] or capture
        label = "当前" if capture["active"] else "下一段"
        self.band_task.configure(text=f"任务：{shown['task']}    |    {label}采集序号：{shown['index']:03d}"
                                 f"    |    有效 {status['valid_frames']} 帧 / 占位 {status['placeholder_frames']} 帧")
        selection = self.selection()
        try:
            form_index = int(selection["index"])
        except (ValueError, TypeError):
            form_index = None
        matches_form = (selection["task"] == shown["task"] and
                        form_index == shown["index"] and
                        selection["auto_increment"] == capture["auto_increment"])
        if capture["revision"] != self.seen_revision and (capture["active"] or not self.form_dirty or matches_form):
            self.syncing_form = True
            self.task_name.set(shown["task"])
            self.capture_index.set(str(shown["index"]))
            self.auto_increment.set(capture["auto_increment"])
            self.syncing_form = False
            self.form_dirty = False
            self.seen_revision = capture["revision"]
            self.settings_note.set(capture["error"] or "设置已自动同步；Quest B / 桌面开始采集使用以上任务和序号。")
        if capture["error"]:
            self.settings_note.set(capture["error"])
        busy = status["recording"] or status["saving"] or replay["active"]
        for widget in (self.task_entry, self.index_entry, self.auto_check, self.apply_button, self.next_button):
            widget.state(["disabled"] if busy else ["!disabled"])
        self.record_button.state(["!disabled"] if status["ready"] and not busy else ["disabled"])
        self.stop_button.state(["!disabled"] if status["recording"] else ["disabled"])
        self.load_button.state(["disabled"] if replay["loading"] or busy else ["!disabled"])
        self.csv_entry.state(["disabled"] if replay["active"] else ["!disabled"])
        self.play_button.state(["!disabled"] if replay["loaded"] and not busy and not replay["loading"]
                               and status["mode"] not in ("fault", "recovery", "resetting") else ["disabled"])
        self.stop_replay_button.state(["!disabled"] if replay["active"] else ["disabled"])
        self.speed_entry.state(["disabled"] if replay["active"] else ["!disabled"])
        if replay["active"] and not self.was_replay_active:
            self.views.focus_set()  # Keep Space available while replay controls are locked.
        self.was_replay_active = replay["active"]
        note = ("正在后台校验 CSV…" if replay["loading"] else
                f"已加载 {replay['frames']} 帧 / {replay['duration_s']:.2f} 秒；{replay['path']}"
                if replay["loaded"] else "粘贴 CSV 完整路径后加载；加载不会启动运动。")
        self.replay_note.set(replay["error"] or note)
        self.message.set(primary_message(status))
        self.message_label.configure(wraplength=max(700, self.root.winfo_width()-40))
        rates, quest, scales = status["fps"], status["quest"], status["position_scale"]
        self.device_text.set(f"相机帧率：头部 {rates['cameras']['head']:.1f} / 左腕 {rates['cameras']['left_wrist']:.1f}"
                             f" / 右腕 {rates['cameras']['right_wrist']:.1f} Hz    |    采集写盘 {rates['recording']:.1f} FPS"
                             f"    |    手柄输入 {quest['hz']:.1f} Hz")
        camera_names = dict(zip(CAMERAS, ("头部", "左腕", "右腕")))
        warnings = status.get("camera_warnings", {})
        self.camera_warning_text.set(("相机提示（遥操与采集继续，缺帧有占位标记）："+
            "；".join(f"{camera_names[n]}：{reason}" for n, reason in warnings.items())) if warnings else "")
        for name, text in zip(CAMERAS, ("头部相机", "左腕相机", "右腕相机")):
            self.camera_labels[name].configure(text=f"{text}    {rates['cameras'][name]:.1f} FPS")
        follow = "    |    ".join(f"{name}跟踪误差 {status['follow'][side]['position_error_mm']} mm"
                                  + ("（触及限速）" if status['follow'][side]['rate_limited'] else "")
                                  for side, name in (("left", "左臂"), ("right", "右臂")))
        self.follow_text.set(f"位移倍率：左 {scales['left']:.2f} / 右 {scales['right']:.2f}    |    "+follow)
        calibration = status.get("calibration", {})
        self.calibration_text.set("接管参考："+"    |    ".join(
            name+("：本次已对齐" if calibration.get(side, {}).get("calibrated") else "：按中指记录当前零点")
            for side, name in (("left", "左臂"), ("right", "右臂"))))
        motion_notes = []
        for side, name in (("left", "左臂"), ("right", "右臂")):
            follow_info = status["follow"][side]
            guard = follow_info.get("joint_guard", {})
            if status.get("local_limits", {}).get(side):
                motion_notes.append(f"{name}目标被拒绝，已请求保持；松开中指，控制器正常后重新接管，另一臂及采集继续")
            elif guard.get("limited"):
                motion_notes.append(f"{name}边界/奇异位形软约束：{guard.get('nearest_joint')}余量 {guard.get('remaining_deg')}°；可向内收回")
            elif follow_info.get("lead_limited"):
                motion_notes.append(f"{name}目标领先量已限制，正在平顺跟随；采集继续")
        self.motion_warning_text.set("；".join(motion_notes))
        self.details_text.set("运行版本："+status.get("runtime_revision", "未知")+"\n"+
                              status["message"]+"\n"+str(status.get("robot_status", {}))+
                              "\n采集缺帧原因："+str(status["recording_drops"])+
                              "\n仅 RGB，曝光/增益锁定值（SDK 单位）："+str(status.get("camera_settings", {})))

    def update(self, cameras, status):
        if self.exit_requested:
            return False
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            return False
        if self.exit_requested:
            return False
        self.update_controls(status)
        tiles = []
        for name in CAMERAS:
            sample = cameras.histories[name].latest()
            tile = np.full((280, 480, 3), 255, np.uint8)
            if sample:
                image = sample.value["bgr"]
                scale = min(480 / image.shape[1], 280 / image.shape[0])
                image = cv2.resize(image, None, fx=scale, fy=scale)
                y, x = (280 - image.shape[0]) // 2, (480 - image.shape[1]) // 2
                tile[y:y + image.shape[0], x:x + image.shape[1]] = image
            tiles.append(tile)
        canvas = np.hstack(tiles)
        width, height = max(480, self.views.winfo_width()), max(200, self.views.winfo_height())
        scale = min(width/canvas.shape[1], height/canvas.shape[0])
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.photo = tk.PhotoImage(master=self.root, data=f"P6\n{w} {h}\n255\n".encode()+rgb.tobytes(), format="PPM")
        self.views.configure(image=self.photo)
        return True

    def close(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass

