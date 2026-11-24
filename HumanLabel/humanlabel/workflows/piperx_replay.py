"""Replay and verify a dual Piper X joint trace with its matching Action CSV."""

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from humanlabel.export import SIDES
from humanlabel.utils.camera import load_head_calibration
from humanlabel.utils.paths import CONFIGS, MODEL
from humanlabel.workflows.piperx_retarget import load_base_transforms


CAMERA_WIDTH, SIM_WIDTH, ZOOM_WIDTH, HEIGHT, BAND = 624, 640, 300, 390, 86
POSE_FIELDS = ("x", "y", "z", "qx", "qy", "qz", "qw")
GREEN, CYAN, AMBER = "#8efa9d", "#7addf4", "#ffca74"
SIDE_COLORS = ("#76aff3", "#ffb571")
GRIPPER_MESHES = {
    "flange_link": "flange", "gripper_base": "gripper_base",
    "gripper_link1": "gripper_link1", "gripper_link2": "gripper_link2",
}
MESHES = {link: link for link in ("base_link", *(f"link{i}" for i in range(1, 7)))}
MESHES.update(GRIPPER_MESHES)
PIPERX_ASSETS = CONFIGS / "piperx"


def add_gripper(parent, side, assembly):
    bodies = {'link6': parent}
    for name in ('flange_joint', 'gripper_base_joint', 'gripper_joint1', 'gripper_joint2'):
        joint = assembly.find(f"joint[@name='{name}']")
        child = joint.find('child').get('link')
        origin = joint.find('origin')
        rotation = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy'), sep=' '))
        bodies[child] = ET.SubElement(
            bodies[joint.find('parent').get('link')], 'body', name=f'{side}_{child}',
            pos=origin.get('xyz'), quat=quat_wxyz(rotation),
        )
        if joint.get('type') == 'prismatic':
            limit = joint.find('limit')
            ET.SubElement(bodies[child], 'joint', name=f'{side}_{name}', type='slide',
                          axis=joint.find('axis').get('xyz'),
                          range=f'{limit.get("lower")} {limit.get("upper")}', limited='true')
        ET.SubElement(bodies[child], 'geom', type='mesh', mesh=child, contype='0', conaffinity='0')


def as_vector(vals):
    return ' '.join(f'{value:.10g}' for value in vals)


def quat_wxyz(rotation):
    return as_vector(np.roll(rotation.as_quat(), 1))


def build_scene(transforms, structure, assembly, mesh_dir):
    xml = ET.Element("mujoco", model="dual_piper_x")
    ET.SubElement(xml, "compiler", angle="radian", meshdir=str(mesh_dir))
    assets = ET.SubElement(xml, "asset")
    for link, mesh in MESHES.items():
        ET.SubElement(assets, "mesh", name=link, file=f"{mesh}.stl")

    world = ET.SubElement(xml, "worldbody")
    for side, transform in zip(SIDES, transforms):
        body = ET.SubElement(
            world, "body", name=f"{side}_base", pos=as_vector(transform[:3, 3]),
            quat=quat_wxyz(Rotation.from_matrix(transform[:3, :3])),
        )
        ET.SubElement(body, "geom", type="mesh", mesh="base_link", contype="0", conaffinity="0")
        for i in range(1, 7):
            joint = structure.find(f"joint[@name='joint{i}']")
            origin = joint.find("origin")
            orientation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" "))
            body = ET.SubElement(
                body, "body", name=f"{side}_link{i}", pos=origin.get("xyz"),
                quat=quat_wxyz(orientation),
            )
            limit = joint.find("limit").attrib
            ET.SubElement(
                body, "joint", name=f"{side}_joint{i}", type="hinge",
                axis=joint.find("axis").get("xyz"), range=f'{limit["lower"]} {limit["upper"]}',
                limited="true", armature=".01",
            )
            ET.SubElement(body, "geom", type="mesh", mesh=f"link{i}", contype="0", conaffinity="0")
        add_gripper(body, side, assembly)
    return mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))


def load_mesh_surfaces(mesh_dir):
    stl_type = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("flag", "<u2")])
    links = {}
    for link, mesh in MESHES.items():
        with (mesh_dir / f"{mesh}.stl").open("rb") as file:
            file.seek(80)
            count = int.from_bytes(file.read(4), "little")
            points = np.unique(
                np.fromfile(file, dtype=stl_type, count=count)["vertices"].reshape(-1, 3), axis=0,
            )
        samples = np.random.default_rng(42).choice(len(points), min(320, len(points)), replace=False)
        hull = ConvexHull(points[samples])
        links[link] = hull.points[hull.simplices], hull.equations[:, :3]
    return links


class CalibratedProjection:
    def __init__(self, configuration):
        self.intrinsics = np.asarray(configuration["camera_matrix"]["matrix_3x3"])
        distortion = configuration["distortion_coefficients"]
        if np.any(distortion[5:]):
            raise ValueError("Replay supports the D555 five-term Brown–Conrady model")
        self.coefficients = (distortion + [0.] * 5)[:5]
        self.rescale = [CAMERA_WIDTH / configuration["image_width"], HEIGHT / configuration["image_height"]]

    def __call__(self, points):
        x, y = np.moveaxis(points[..., :2] / points[..., 2:], -1, 0)
        r2 = x*x + y*y
        k1, k2, p1, p2, k3 = self.coefficients
        r = 1 + k1*r2 + k2*r2**2 + k3*r2**3
        distortion = np.stack((x*r + 2*p1*x*y + p2*(r2 + 2*x*x),
                               y*r + p1*(r2 + 2*y*y) + 2*p2*x*y), axis=-1)
        return (distortion * self.intrinsics.diagonal()[:2] + self.intrinsics[:2, 2]) * self.rescale


def circle(draw, point, color, radius=5, fill=None):
    x, y = point
    draw.ellipse((x-radius-1, y-radius-1, x+radius+1, y+radius+1), outline='#152119', width=3)
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, fill=fill, width=2)


def arrow(draw, start, end, color):
    direction = np.asarray(end) - start
    length = np.linalg.norm(direction)
    if length < 2:
        return
    base = end - direction * min(.30, 7 / length)
    normal = np.array([-direction[1], direction[0]]) * min(.14, 4 / length)
    draw.line([tuple(start), tuple(end)], fill='#122017', width=4)
    draw.line([tuple(start), tuple(end)], fill=color, width=2)
    draw.polygon([tuple(end), tuple(base+normal), tuple(base-normal)], fill=color)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class CenterView:
    def __init__(self, world_from_base, mesh_dir):
        self.links = load_mesh_surfaces(mesh_dir)
        self.light = np.array([-.4, .35, .845])
        base_centers = np.array([transform[:3,3] for transform in world_from_base])
        span = base_centers[1] - base_centers[0]
        separation = np.linalg.norm(span)
        if separation <= 1e-8:
            raise ValueError("The two robot base origins must be distinct")
        lateral = span / separation
        forward = np.array([1., 0., 0.])
        forward -= lateral * (forward @ lateral)
        forward /= np.linalg.norm(forward)
        up = np.cross(lateral, forward)
        elevation = np.deg2rad(27)
        look = np.cos(elevation)*forward - np.sin(elevation)*up
        vertical = np.cross(lateral, look)
        self.center = base_centers.mean(0) + .26*forward + .19*up
        self.eye = self.center - 3.0*look
        self.view = np.stack((lateral, vertical, look), axis=1)
        pixels, depth = self.project(base_centers)
        np.testing.assert_allclose(depth[0], depth[1], atol=1e-12)
        np.testing.assert_allclose(pixels[:,0].mean(), SIM_WIDTH/2, atol=1e-12)
        np.testing.assert_allclose(pixels[0,1], pixels[1,1], atol=1e-12)

    def project(self, world):
        coordinates = (world - self.center) @ self.view
        pixels = coordinates[..., :2] * [540,-540] + [SIM_WIDTH/2, HEIGHT/2]
        return pixels, coordinates[..., 2]


    def render(self, data):
        image = Image.new('RGB', (SIM_WIDTH, HEIGHT), '#202934')
        draw = ImageDraw.Draw(image)
        grid = np.arange(-.96, .97, .12)
        for x in grid:
            pts, _ = self.project(np.array([[x,-.95,-.018],[x,.45,-.018]]))
            draw.line(list(map(tuple,pts)), fill='#303c49')
        for y in np.arange(-.96,.46,.12):
            pts, _ = self.project(np.array([[-.6,y,-.018],[.9,y,-.018]]))
            draw.line(list(map(tuple,pts)), fill='#303c49')
        pieces = []
        for side, rgb in [('left',[60,155,224]),('right',[238,133,76])]:
            for link, (triangles, normals) in self.links.items():
                body = data.body(f'{side}_'+('base' if link == 'base_link' else link))
                rotation = body.xmat.reshape(3,3)
                faces = triangles @ rotation.T + body.xpos
                world_normals = normals @ rotation.T
                front = ((self.eye - faces[:,0]) * world_normals).sum(axis=-1) > 0
                pix, depths = self.project(faces[front])
                illumination = .52 + .43 * np.maximum(0, world_normals[front] @ self.light)
                tint = (224, 228, 232) if link.startswith('gripper_link') else (174, 180, 186) if link in GRIPPER_MESHES else rgb
                colors = np.asarray(illumination[:,None] * tint, dtype=int)
                pieces.extend(zip(depths.mean(axis=1),pix.tolist(),colors.tolist()))
        for _, verts, color in sorted(pieces, key=lambda item: item[0], reverse=True):
            draw.polygon(list(map(tuple,verts)), fill=tuple(color))
        return image


class JawCloseups:
    def __init__(self, view, head_from_world):
        self.meshes = {link: view.links[link] for link in GRIPPER_MESHES}
        self.rotation = head_from_world[:3,:3]
        self.translation = head_from_world[:3,3]
        self.light = np.array([-.40,-.40,-.82])

    def in_camera(self, point):
        return point @ self.rotation.T + self.translation

    def render(self, data, side, target, approach, jaws, width, color, font):
        panel_height = HEIGHT // 2
        panel = Image.new('RGB', (ZOOM_WIDTH,panel_height), '#232c38')
        painter = ImageDraw.Draw(panel)
        def project(world):
            return (world[..., :2]-target[:2])*1160 + [ZOOM_WIDTH*.48,panel_height*.43]

        housing, fingers = [], []
        for link, (triangles, normals) in self.meshes.items():
            body = data.body(f'{side}_{link}')
            axes = self.rotation @ body.xmat.reshape(3,3)
            camera = triangles @ axes.T + self.in_camera(body.xpos)
            normal = normals @ axes.T
            front = normal[:,2] < 0
            xy = project(camera[front])
            finger = link.startswith('gripper_link')
            tint = [247,222,181] if finger else [92,115,145]
            colors = ((.60+.40*np.maximum(normal[front]@self.light,0))[:,None]*tint).astype(int)
            layer = fingers if finger else housing
            layer.extend(zip(camera[front,:,2].mean(1),xy.tolist(),colors.tolist()))
        for layer in (housing,fingers):
            for _, pts, rgb in sorted(layer,key=lambda item:item[0],reverse=True):
                painter.polygon(list(map(tuple,pts)),fill=tuple(rgb))
        target_px, approach_px = project(np.stack((target,target+.038*approach)))
        arrow(painter,target_px,approach_px,GREEN)
        circle(painter,target_px,GREEN,5)
        tips = project(jaws)
        painter.line(list(map(tuple,tips)),fill=CYAN,width=3)
        for tip in tips:
            circle(painter,tip,CYAN,3,CYAN)
        painter.rectangle((0,0,ZOOM_WIDTH,26),fill='#1c2531')
        painter.text((10,5),f'{side.title()}  |  Opening {width*1e3:.1f} mm',font=font,fill=color)
        return panel


def replay(args):
    with np.load(args.joint_trace) as archive:
        trace = dict(archive)
    q = trace['joints']
    actions = read_csv(args.action)
    video_rows = read_csv(args.episode / 'timestamps.csv')
    if trace["sides"].tolist() != list(SIDES) or q.shape != (2, len(actions), 6):
        raise ValueError("Expected a left/right six-joint trajectory aligned to the Action CSV")
    if not actions or len(actions) != len(video_rows) or not np.isfinite(q).all():
        raise ValueError("Expected finite joint angles for all recorded frames")
    timestamps = [r['timestamp_unix_s'] for r in actions]
    if timestamps != [r["timestamp_unix_s"] for r in video_rows]:
        raise ValueError("Action and video timestamps differ")
    n = len(actions)
    commands = np.array([[[float(row[f'{side}.{key}']) for key in POSE_FIELDS]
                          for row in actions] for side in SIDES])
    widths = np.array([[float(row[f'{side}.gripper_m']) for row in actions] for side in SIDES])
    if not np.isfinite(commands).all() or not np.isfinite(widths).all() or np.any(widths < 0):
        raise ValueError("Invalid Action CSV pose or gripper opening")
    np.testing.assert_allclose(widths,trace['gripper_m'],atol=1e-10)

    base_from_head = load_base_transforms(args.extrinsics_yaml)
    head_from_world = np.linalg.inv(base_from_head["left"])
    world_from_base = (np.eye(4), base_from_head["left"] @ np.linalg.inv(base_from_head["right"]))
    structure = ET.parse(PIPERX_ASSETS / "piper_x_description.urdf")
    assembly = ET.parse(PIPERX_ASSETS / "piper_x_with_gripper_description.xacro")
    mesh_dir = args.mesh_dir.resolve()
    for mesh in MESHES.values():
        if not (mesh_dir / f"{mesh}.stl").is_file():
            raise FileNotFoundError(f"Missing Piper X mesh: {mesh_dir / (mesh + '.stl')}")
    model = build_scene(world_from_base, structure, assembly, mesh_dir)
    data = mujoco.MjData(model)
    sim_view = CenterView(world_from_base, mesh_dir)
    closeups = JawCloseups(sim_view, head_from_world)
    mimic = [assembly.find(f"joint[@name='gripper_joint{i}']/mimic") for i in (1, 2)]
    scales, offsets = np.array([(float(m.get("multiplier")), float(m.get("offset"))) for m in mimic]).T
    arm_joints = [[data.joint(f'{s}_joint{i}') for i in range(1,7)] for s in SIDES]
    jaw_joints = [[data.joint(f'{s}_gripper_joint{i}') for i in (1,2)] for s in SIDES]
    jaw_bodies = [[data.body(f'{s}_gripper_link{i}') for i in (1,2)] for s in SIDES]
    speed_limits = np.array([float(structure.find(f"joint[@name='joint{i}']/limit").get('velocity'))
                             for i in range(1,7)])
    source = imageio.get_reader(args.episode / "head.mp4", format="FFMPEG", input_params=["-threads", "2"])
    fps = source.get_meta_data()["fps"]
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid head video FPS")
    np.testing.assert_array_less(np.abs(np.diff(q, axis=1)) * fps, np.broadcast_to(speed_limits + 1e-6, (2, n - 1, 6)))
    for i,joints in enumerate(arm_joints):
        limits = np.array([model.joint(j.id).range for j in joints])
        if np.any(q[i] < limits[:, 0] - 1e-6) or np.any(q[i] > limits[:, 1] + 1e-6):
            raise ValueError(f"{SIDES[i].title()} joint angle exceeds the URDF limit")
        for finger, joint in enumerate(jaw_joints[i]):
            bounds = model.joint(joint.id).range
            finger_pos = scales[finger] * widths[i] + offsets[finger]
            if np.any(finger_pos < bounds[0] - 1e-8) or np.any(finger_pos > bounds[1] + 1e-8):
                raise ValueError(f"{SIDES[i].title()} gripper opening exceeds the Xacro limit")

    calibrated = CalibratedProjection(load_head_calibration(args.intrinsics_json))
    with np.load(args.soma) as soma:
        native = soma['native_joints25_camera_m']
    tips = np.stack([native[side,trace['source_frame'][side]][:,[4,9]] for side in range(2)])
    center,web = trace['target_camera'],trace['web_camera']
    approach = trace['smoothed_target_camera_rotation'][:,:,:,2]
    pixels = calibrated(np.concatenate((tips,center[:,:,None],web[:,:,None],(center+.045*approach)[:,:,None]),axis=2))
    world_targets = (center-head_from_world[:3,3])@head_from_world[:3,:3]
    pos_error = trace['position_error']*1e3
    raw_angle = trace['angle_error_deg']
    final_angle = trace['smoothed_angle_error_deg']
    actual_rotation = [Rotation.from_quat(a) for a in trace['quaternions']]
    omega = np.stack([(a[1:]*a[:-1].inv()).as_rotvec()*fps for a in actual_rotation])
    angular_accel = np.linalg.norm(np.diff(omega,axis=1)*fps,axis=2)
    font = ImageFont.truetype('DejaVuSans.ttf',14)
    title = ImageFont.truetype('DejaVuSans.ttf',18)
    frames = iter(source)
    try:
        with imageio.get_writer(args.output,fps=fps,codec='libx264',macro_block_size=2,
                ffmpeg_log_level='error',output_params=['-threads','4','-crf','18','-movflags','+faststart']) as writer:
            for i in range(n):
                for side in range(2):
                    for j,value in zip(arm_joints[side],q[side,i]):
                        j.qpos[0]=value
                    for j,value in zip(jaw_joints[side],scales*widths[side,i]+offsets):
                        j.qpos[0]=value
                mujoco.mj_forward(model,data)
                frame = next(frames, None)
                if frame is None:
                    raise ValueError(f"Head video ended at frame {i}")
                raw = Image.fromarray(frame).resize((CAMERA_WIDTH,HEIGHT),Image.Resampling.LANCZOS)
                draw_raw = ImageDraw.Draw(raw)
                sim = sim_view.render(data)
                draw_sim = ImageDraw.Draw(sim)
                sheet = Image.new('RGB',(CAMERA_WIDTH+SIM_WIDTH+ZOOM_WIDTH,HEIGHT+BAND),'#171e29')
                for side,name in enumerate(SIDES):
                    fl = data.body(name+'_link6')
                    t = world_from_base[side]
                    expected = commands[side,i]
                    np.testing.assert_allclose(fl.xpos,t[:3,:3]@expected[:3]+t[:3,3],atol=1e-8)
                    np.testing.assert_allclose(fl.xmat.reshape(3,3),t[:3,:3]@Rotation.from_quat(expected[3:]).as_matrix(),atol=1e-8)
                    jaws = np.array([b.xpos for b in jaw_bodies[side]])
                    jaw_cam = closeups.in_camera(jaws)
                    np.testing.assert_allclose(jaw_cam.mean(0),trace['jaw_camera'][side,i],atol=1e-8)
                    np.testing.assert_allclose(np.linalg.norm(jaws[0]-jaws[1]),widths[side,i],atol=1e-8)
                    thumb,index,target,patch,direction = pixels[side,i]
                    draw_raw.line([tuple(patch),tuple(target)],fill='#182d21',width=5)
                    draw_raw.line([tuple(patch),tuple(target)],fill=GREEN,width=2)
                    arrow(draw_raw,target,direction,GREEN)
                    circle(draw_raw,patch,AMBER,5)
                    for tip in (thumb,index):
                        circle(draw_raw,tip,SIDE_COLORS[side],4)
                    circle(draw_raw,target,GREEN,5)
                    target_px,jaw_px = sim_view.project(np.stack((world_targets[side,i],jaws.mean(0))))[0]
                    draw_sim.line([tuple(target_px),tuple(jaw_px)],fill=SIDE_COLORS[side],width=2)
                    circle(draw_sim,target_px,GREEN,6)
                    circle(draw_sim,jaw_px,SIDE_COLORS[side],3,SIDE_COLORS[side])
                    zoom = closeups.render(data,name,center[side,i],approach[side,i],jaw_cam,
                                           widths[side,i],SIDE_COLORS[side],font)
                    sheet.paste(zoom,(CAMERA_WIDTH+SIM_WIDTH,BAND+side*HEIGHT//2))
                draw_raw.text((12,HEIGHT-24),'Amber: thumb web   Green: grasp and filtered approach',
                              font=font,fill='white',stroke_width=2,stroke_fill='#182020')
                draw_sim.text((12,HEIGHT-24),'Blue: left arm   Orange: right arm   Green: SOMA target',
                              font=font,fill='white',stroke_width=2,stroke_fill='#182020')
                sheet.paste(raw,(0,BAND))
                sheet.paste(sim,(CAMERA_WIDTH,BAND))
                draw = ImageDraw.Draw(sheet)
                draw.text((14,11),'D555 | SOMA left and right hands',font=title,fill='white')
                draw.text((14,42),f'Frame {i+1:03d}/{n}  |  {i/fps:05.2f} s',font=font,fill=GREEN)
                draw.text((CAMERA_WIDTH+14,11),'MuJoCo | Dual Piper X centered view',font=title,fill='white')
                for side,name in enumerate(SIDES):
                    draw.text((CAMERA_WIDTH+14,39+side*19),
                              f'{name.title()}  |  Jaw center {pos_error[side,i]:.2f} mm  |  Filtered target {final_angle[side,i]:.1f} deg',
                              font=font,fill=SIDE_COLORS[side])
                draw.text((CAMERA_WIDTH+SIM_WIDTH+14,11),'Articulated grippers',font=title,fill='white')
                draw.text((CAMERA_WIDTH+SIM_WIDTH+14,42),'D555 axes | cutaway housings',font=font,fill='#b5c4d6')
                if i + 1 == args.snapshot_frame:
                    sheet.save(args.output.with_name(f"{args.output.stem}_frame_{i + 1:04d}.png"))
                writer.append_data(np.asarray(sheet))
                if (i+1)%200 == 0:
                    print(f'Rendered {i+1}/{n} frames',flush=True)
        if next(frames, None) is not None:
            raise ValueError("Head video has more frames than the Action CSV")
    finally:
        source.close()

    fields = ['frame_index','timestamp_unix_s']
    labels = ['geometry_frame_index','jaw_center_error_mm','raw_rotation_error_deg',
              'filtered_rotation_error_deg','gripper_opening_mm','peak_joint_speed_rad_s',
              'tcp_angular_speed_rad_s','tcp_angular_accel_rad_s2']
    speed = trace['speed'].max(axis=2)
    columns = [trace['source_frame'],pos_error,raw_angle,final_angle,widths*1000,
               np.pad(speed,((0,0),(1,0))),np.pad(np.linalg.norm(omega,axis=2),((0,0),(1,0))),
               np.pad(angular_accel, ((0, 0), (min(n, 2), 0)))]
    diagnostics = args.output.with_name(f"{args.output.stem}_diagnostics.csv")
    with diagnostics.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields+[f'{side}.{label}' for side in SIDES for label in labels])
        writer.writerows(zip(np.arange(n),timestamps,*(col[side] for side in range(2) for col in columns)))
    print(f'Validated synchronized rendering, both flange poses and both jaw openings on {n} frames.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="Directory containing head.mp4 and timestamps.csv")
    parser.add_argument("--action", type=Path, required=True, help="Dual-arm base-frame Action CSV")
    parser.add_argument("--joint-trace", type=Path, required=True, help="Matching dual-arm joint trace NPZ")
    parser.add_argument("--soma", type=Path, required=True, help="HumanLabel SOMA archive")
    parser.add_argument("--intrinsics-json", type=Path, required=True)
    parser.add_argument("--extrinsics-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output MP4")
    parser.add_argument("--mesh-dir", type=Path, default=MODEL / "piperx_replay", help="Piper X STL directory")
    parser.add_argument("--snapshot-frame", type=int, default=405, help="One-based frame to save as PNG")
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".mp4":
        parser.error("--output must be an MP4 path")
    if args.output.resolve() == (args.episode / "head.mp4").resolve():
        parser.error("--output must differ from the input video")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    replay(args)


if __name__ == "__main__":
    main()
