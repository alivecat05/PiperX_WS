"""Recovery instructions identify the next action without bypassing any motion guard."""
from piperx_teleop.operator_guidance import primary_message, recovery_guidance
from test_motion_recovery import cfg, controller, enter_limit_recovery, arm_recovery, neutral
from test_pipeline import step


def ready_status(cfg):
    control = controller(cfg)
    control.begin_input()
    enter_limit_recovery(control)
    control.robots["right"].code = 0
    arm_recovery(control)
    return control, control.status()


def test_teaching_screen_explains_exit_teaching_then_release_then_recover(cfg):
    control, _ = ready_status(cfg)
    control.robots["left"].teaching = True
    control.robots["left"].code = 11
    step(control, neutral())
    info = recovery_guidance(control.status())
    assert "左臂" in info["title"] and "退出" in info["title"]
    assert "扶稳" in info["steps"][0] and "0.3" in info["steps"][-1]
    assert info["can_recover"]


def test_still_faulted_cannot_offer_recovery(cfg):
    control = controller(cfg)
    enter_limit_recovery(control)
    info = recovery_guidance(control.status())
    assert "右臂" in info["title"] and "尚未恢复正常" in info["title"]
    assert not info["can_recover"]


def test_missing_tracking_gives_headset_instructions(cfg):
    _, status = ready_status(cfg)
    status["quest"]["connected"] = False
    info = recovery_guidance(status)
    assert "追踪" in info["title"] and not info["can_recover"]


def test_pressed_trigger_requires_release_instead_of_enabling_recovery(cfg):
    _, status = ready_status(cfg)
    status["quest"]["left"]["trigger"] = .8
    info = recovery_guidance(status)
    assert "食指扳机" in info["title"] and not info["can_recover"]


def test_joint_limit_needs_manual_pose_adjustment(cfg):
    control, _ = ready_status(cfg)
    control.robots["left"].issue = "J3 near limit"
    step(control, neutral())
    info = recovery_guidance(control.status())
    assert "左臂" in info["title"] and "限位" in info["title"]
    assert not info["can_recover"]


def test_ready_screen_allows_explicit_recovery_only(cfg):
    control, status = ready_status(cfg)
    info = recovery_guidance(status)
    assert info["can_recover"] and "可以恢复" in info["title"]
    assert control.mode == "recovery"  # Rendering guidance itself does not resume anything.


def test_waiting_for_neutral_or_enable_has_specific_explanation(cfg):
    _, status = ready_status(cfg)
    status["recovery"]["armed"] = False
    assert "0.3" in recovery_guidance(status)["title"]
    status["recovery"]["enabling"] = True
    assert "反馈确认" in recovery_guidance(status)["title"]


def test_device_not_ready_prevents_recovery_button(cfg):
    _, status = ready_status(cfg)
    status["recovery"]["hardware_missing"] = ["RGB head"]
    assert not recovery_guidance(status)["can_recover"]


def test_primary_recovery_message_is_chinese_and_actionable(cfg):
    _, status = ready_status(cfg)
    assert "上方步骤" in primary_message(status)
