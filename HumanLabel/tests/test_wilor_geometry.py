import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from humanlabel.backends.wilor_model import (
    INDEX_TIP_VERTEX,
    THUMB_TIP_VERTEX,
    camera_translation,
    fit_perspective_camera,
    mano_geometry,
)


def model_geometry():
    rng = np.random.default_rng(92)
    joints = rng.normal(0, 0.025, (21, 3))
    joints[0] = [
        0.01,
        -0.02,
        0.005,
    ]  # The wrist root is deliberately not model translation.
    vertices = rng.normal(0, 0.03, (778, 3))
    vertices[THUMB_TIP_VERTEX] = joints[4]
    vertices[INDEX_TIP_VERTEX] = joints[8]
    return joints, vertices


@pytest.mark.parametrize("is_right", [True, False])
def test_mano_root_and_tips_use_one_mesh_and_proper_rotation(is_right):
    joints, vertices = model_geometry()
    root = np.array([0.3, -0.4, 0.2])
    translation = np.array([0.1, 0.05, 0.6])
    value, distance, camera_joints, camera_vertices, _, _ = mano_geometry(
        joints, vertices, root, np.zeros((15, 3)), translation, is_right, 1.2
    )
    reflection = np.diag([1, 1, 1] if is_right else [-1, 1, 1])
    expected_root = (reflection @ joints[0] + translation) * 1.2
    np.testing.assert_allclose(value[:3], expected_root)
    assert not np.allclose(value[:3], translation * 1.2)
    rotation = Rotation.from_quat(value[3:7]).as_matrix()
    np.testing.assert_allclose(
        rotation,
        reflection @ Rotation.from_rotvec(root).as_matrix() @ reflection,
        atol=1e-12,
    )
    assert np.linalg.det(rotation) > 0.999999
    np.testing.assert_allclose(
        distance,
        np.linalg.norm(vertices[THUMB_TIP_VERTEX] - vertices[INDEX_TIP_VERTEX]) * 1.2,
    )
    np.testing.assert_allclose(
        camera_joints[[4, 8]], camera_vertices[[THUMB_TIP_VERTEX, INDEX_TIP_VERTEX]]
    )


def test_mano_rejects_wrong_fingertip_layout():
    joints, vertices = model_geometry()
    vertices[THUMB_TIP_VERTEX] += 0.02
    with pytest.raises(ValueError, match="fingertip indexing"):
        mano_geometry(joints, vertices, [0, 0, 0], np.zeros((15, 3)), [0, 0, 0.5], True)


def test_camera_translation_uses_real_principal_point_and_focal():
    matrix = np.array([[800.0, 0, 610], [0, 1000.0, 390], [0, 0, 1.0]])
    z = 2 * np.sqrt(800 * 1000) / (400 * 2)
    expected = np.array(
        [(650 - 610) * z / 800 + 0.02, (420 - 390) * z / 1000 - 0.01, z]
    )
    np.testing.assert_allclose(
        camera_translation([2, 0.02, -0.01], [650, 420], 400, matrix, True), expected
    )
    expected[0] -= 0.04
    np.testing.assert_allclose(
        camera_translation([2, 0.02, -0.01], [650, 420], 400, matrix, False), expected
    )


@pytest.mark.parametrize("is_right", [True, False])
def test_nominal_to_perspective_conversion_preserves_projection_and_wrist_pivot(
    is_right,
):
    joints, vertices = model_geometry()
    matrix = np.array([[1280.0, 0, 640], [0, 1280.0, 400], [0, 0, 1.0]])
    nominal = matrix.copy()
    nominal[0, 0] = nominal[1, 1] = 25000.0
    camera, center, size = [4.0, 0.01, -0.02], [450, 650], 400
    translation, delta, residual = fit_perspective_camera(
        camera, center, size, matrix, (1280, 800), joints, is_right
    )
    value, distance, corrected, _, root, _ = mano_geometry(
        joints,
        vertices,
        [0.2, -0.1, 0.3],
        np.zeros((15, 3)),
        translation,
        is_right,
        camera_rotation=delta,
    )
    reflected = joints * ([1, 1, 1] if is_right else [-1, 1, 1])
    target = cv2.projectPoints(
        reflected,
        np.zeros(3),
        camera_translation(camera, center, size, nominal, is_right),
        nominal,
        None,
    )[0].reshape(-1, 2)
    old = cv2.projectPoints(
        reflected,
        np.zeros(3),
        camera_translation(camera, center, size, matrix, is_right),
        matrix,
        None,
    )[0].reshape(-1, 2)
    fitted = cv2.projectPoints(corrected, np.zeros(3), np.zeros(3), matrix, None)[
        0
    ].reshape(-1, 2)
    assert residual < np.sqrt(np.mean(np.sum((old - target) ** 2, axis=1))) / 3
    np.testing.assert_allclose(value[:3], corrected[0])
    np.testing.assert_allclose(corrected[0], reflected[0] + translation)
    np.testing.assert_allclose(distance, np.linalg.norm(joints[4] - joints[8]))
    np.testing.assert_allclose(
        np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))), residual
    )
    original_root = np.array([0.2, -0.1, 0.3]) * (
        [1, 1, 1] if is_right else [1, -1, -1]
    )
    np.testing.assert_allclose(
        Rotation.from_rotvec(root).as_matrix(),
        delta @ Rotation.from_rotvec(original_root).as_matrix(),
    )
