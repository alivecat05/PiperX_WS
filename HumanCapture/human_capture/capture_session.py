"""Editable capture labels and collision-free episode numbering."""


class CaptureSelection:
    def __init__(self, recorder, task):
        self.recorder = recorder
        self.task = task
        self.index = recorder.next_index(task)
        self.auto_increment = True
        self.revision = 0
        self.active = None
        self.advance_pending = False
        self.error = ""

    def apply(self, task, index, auto_increment):
        if self.recorder.busy:
            raise ValueError("录制或保存期间不能修改任务和序号")
        self.recorder.task_directory(task)
        if isinstance(index, str):
            try:
                index = int(index)
            except ValueError:
                raise ValueError("采集序号必须是正整数") from None
        if type(index) is not int or index < 1:
            raise ValueError("采集序号必须是正整数")
        if type(auto_increment) is not bool:
            raise ValueError("自动递增选项无效")
        if (self.recorder.task_directory(task)/f"episode_{index:03d}").exists():
            raise ValueError("该任务下的采集序号已存在，请更换序号或使用下一个可用序号")
        self.task, self.index, self.auto_increment = task, index, auto_increment
        self.error = ""
        self.revision += 1

    def suggest(self, task, auto_increment=True):
        self.apply(task, self.recorder.next_index(task), auto_increment)

    def start(self, now):
        self.recorder.start(now, task=self.task, capture_index=self.index)
        self.active = {"task": self.task, "index": self.index}
        self.advance_pending = True
        self.error = ""
        self.revision += 1

    def refresh(self):
        if not self.advance_pending or self.recorder.busy or self.recorder.result is None:
            return
        self.advance_pending = False
        if self.auto_increment:
            index = self.index+1
            folder = self.recorder.task_directory(self.task)
            while (folder/f"episode_{index:03d}").exists():
                index += 1
            self.index = index
            self.error = ""
        else:
            self.error = "序号已使用；开始下一段前请修改序号"
        self.active = None
        self.revision += 1

    def snapshot(self):
        return {"task": self.task, "index": self.index, "auto_increment": self.auto_increment,
                "active": dict(self.active) if self.active else None,
                "revision": self.revision, "error": self.error}
