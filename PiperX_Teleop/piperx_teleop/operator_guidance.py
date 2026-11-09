"""Chinese instructions derived from live status; this module issues no robot commands."""
SIDES = {"left": "左臂", "right": "右臂"}


def recovery_guidance(status):
    info = status.get("recovery", {})
    arms = status.get("robot_status", {})
    teaching = [SIDES[s] for s in SIDES if arms.get(s, {}).get("teaching")]
    reason = info.get("reason", "")
    cause = ("相机画面超时或采集异常" if "RGB " in reason else
             "目标与实际位置偏差超过保护阈值" if "cannot follow" in reason else
             "关节接近限位" if "joint limit guard" in reason else
             "控制器报告目标越限" if "LIMIT" in reason or "limit" in reason else "进入保护保持")
    result = {"can_recover": False, "cause": cause}
    if info.get("exiting_teaching"):
        result.update(title="正在退出示教，等待控制器确认", steps=(
            "已发送结束示教并保持当前关节位置的请求；没有执行 reset、失能或回零。",
            "请扶稳机械臂并保持手柄松开；确认 CAN 正常反馈后继续恢复。"))
        return result
    if teaching:
        names = "、".join(teaching)
        result["cause"] = "检测到手动示教，程序已暂停遥操和采集"
        supported = all(arms.get(s, {}).get("teach_exit_supported") and not arms.get(s, {}).get("err_code")
                        for s in SIDES if arms.get(s, {}).get("teaching"))
        quest = status["quest"]
        held = (quest.get("age_ms") is not None and quest["age_ms"] <= 200 and
                (any(quest[s]["middle"] or quest[s]["trigger"] > .05 for s in SIDES)
                 or any(v for k, v in quest.get("buttons", {}).items() if k != "reset")))
        if not supported:
            result.update(title=f"{names}示教模式：当前固件需在本体退出", steps=(
                "当前无断电主动退出仅支持配置为 v189 的固件；不会自动 reset。",
                "请先扶稳机械臂，用本体示教按钮退出后再恢复。"))
        else:
            result.update(title=(f"{names}仍有示教标记，可点击恢复遥操退出" if not held else "请先松开手柄，再点击恢复遥操退出示教"),
                          can_recover=not held, steps=(
                f"1. 先扶稳{names}，松开两只手柄的中指键、食指扳机和面键。",
                "2. 点击「恢复遥操」或按电脑 R：程序主动结束示教并原位保持，不 reset、不失能、不回零。",
                "3. 等待新的正常反馈和松开持续 0.3 秒后，继续恢复；完成后重新按中指键接管。"))
            if info.get("teach_exit_error"):
                result["steps"] = (info["teach_exit_error"], *result["steps"])
        return result
    faulted = [SIDES[s] for s in SIDES if arms.get(s, {}).get("arm_status", 0) not in (0, None)
               or arms.get(s, {}).get("err_code", 0)]
    if faulted:
        result.update(title="、".join(faulted)+"控制器尚未恢复正常", steps=(
            "1. 先扶稳机械臂；若目标越限，可用本体示教模式把机械臂调整到正常工作姿态。",
            "2. 退出示教，等待控制器恢复正常；若故障持续，请检查本体状态，程序不会自动断电清故障。",
            "3. 故障解除后，按界面提示松开手柄并恢复。"))
        return result
    if info.get("hardware_missing"):
        cameras = {"head": "头部相机", "left_wrist": "左腕相机", "right_wrist": "右腕相机"}
        names = [label for key, label in cameras.items()
                 if any(item.startswith("RGB "+key) for item in info["hardware_missing"])]
        result.update(title=("、".join(names)+"未就绪，已暂停采集" if names else "设备尚未就绪，暂时不能恢复"), steps=(
            "1. 先确认两臂保持状态，再检查对应相机的 USB 连接、供电和画面更新。",
            "2. 相机正在自动重连，等待画面恢复；这属于相机问题，不需要对机械臂执行 reset。",
            "3. 相机正常后，松开手柄按键和扳机，再点击「恢复遥操」。"))
        return result
    quest = status["quest"]
    if not quest["connected"] or not quest["active"] or not all(quest[s]["tracked"] for s in SIDES):
        result.update(title="请先恢复 Quest 和双手柄追踪", steps=(
            "1. 在 Quest 浏览器中启动手柄追踪，让头显能够看到两只手柄。",
            "2. 退出机械臂示教后，松开两只手柄按键和扳机，再点击「恢复遥操」。"))
        return result
    if any(quest[s]["middle"] or quest[s]["trigger"] > .05 for s in SIDES):
        result.update(title="请松开两只手柄的中指键和食指扳机", steps=(
            "1. 保持机械臂稳定，完全松开两侧中指键、食指扳机和面键。",
            "2. 持续松开至少 0.3 秒，按钮可用后点击「恢复遥操」。"))
        return result
    if info.get("pose_issues"):
        names = "、".join(SIDES[s] for s in info["pose_issues"])
        result.update(title=f"{names}超出已启用的软件限位，请先调整姿态", steps=(
            f"1. 先扶稳{names}，用本体示教调整回已启用的软件限位内。",
            "2. 退出示教，松开手柄按键和扳机，再申请恢复。"))
        return result
    if info.get("enabling"):
        result.update(title="正在恢复，等待机械臂反馈确认", steps=(
            "请保持手柄按键和扳机松开；程序正在确认使能和正常状态。",))
        return result
    if not info.get("armed") or any(v for k, v in quest.get("buttons", {}).items() if k != "reset"):
        result.update(title="请持续松开全部按键和扳机至少 0.3 秒", steps=(
            "条件满足后「恢复遥操」按钮会变为可用。",))
        return result
    result.update(title="已满足恢复条件，可以恢复遥操", can_recover=True, steps=(
        "1. 点击「恢复遥操」（也可按电脑 R，或长按手柄 Y 1.5 秒）。",
        "2. 等待恢复完成，松开全部按键后，再重新按住中指键接管。"))
    return result


def primary_message(status):
    if any("\u4e00" <= ch <= "\u9fff" for ch in status["message"]):
        return status["message"]
    mode = status["mode"]
    if mode == "recovery":
        return ("保护期间采集和回放暂停；请按上方步骤操作。保持未确认时，请先扶稳机械臂。"+
                (" 上一段已标记为不完整，文件保留供检查。" if status.get("recording_result") == "incomplete" else ""))
    if mode == "fault":
        return "当前为故障状态，请先扶稳机械臂并排查设备；详细原因可展开下方诊断。"
    if mode == "replaying":
        return ("正在低速定位到轨迹起点；双臂到位后自动开始回放，随时可点击停止。"
                if status["replay"].get("phase") != "playing" else
                "正在按 CSV 时间轴回放；点击「停止回放」或按空格停止。")
    if mode == "resetting":
        return "双臂正在返回初始位；松开 Y 后再按一次 Y 可取消。"
    if status["saving"]:
        return "正在保存本段数据，请等待保存完成后再开始下一段。"
    if status["recording"]:
        return "正在采集；按 A 或点击「停止并保存」结束本段。"
    if status["ready"]:
        return "已就绪：按 B 开始采集；按住中指键遥操，松开后保留最后目标。"
    if status.get("replay", {}).get("result") in ("complete", "stopped"):
        return ("回放已完成，已保留最终目标。" if status["replay"]["result"] == "complete" else "回放已停止，已保留最后目标。") + \
               "可再次点击「到起点并回放」；要恢复遥操，请启动 Quest 追踪并松开全部按键。"
    if not status["quest"]["connected"]:
        return "等待 Quest 连接：在头显浏览器打开地址并启动手柄追踪。"
    if mode == "paused":
        return "遥操已暂停：恢复双手柄追踪，松开全部按键和扳机 0.3 秒，再重新接管。"
    return "正在等待相机、机械臂反馈和 Quest 追踪就绪；请检查设备连接。"
