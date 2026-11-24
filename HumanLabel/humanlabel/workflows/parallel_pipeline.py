"""Label episodes in place with persistent CUDA workers and a single progress bar."""

import argparse
import fcntl
import multiprocessing as mp
import os
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack, closing
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from humanlabel.utils.paths import MODEL, ROOT


PIPERX_ASSETS = ROOT / "configs/piperx"
INTERMEDIATE_OUTPUTS = (
    ("soma.npz", "soma.npz"),
    ("epose.csv", "epose.csv"),
    ("wrist_epose.npz", "wrist_epose.npz"),
    ("pinch.csv", "pinch.csv"),
    ("quality.csv", "quality.csv"),
    ("preview.mp4", "preview.mp4"),
    ("manifest.json", "humanlabel_manifest.json"),
    ("validation.json", "humanlabel_validation.json"),
)


def find_episodes(root):
    if root.name.startswith("episode_") and (root / "head.mp4").is_file():
        return [root]
    return sorted(
        video.parent
        for pattern in ("episode_*/head.mp4", "*/episode_*/head.mp4")
        for video in root.glob(pattern)
        if video.is_file()
    )


def publish_action(
    source, destination, *, replace_existing=False, keep_intermediates=False
):
    with ExitStack() as cleanup:
        episode_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        cleanup.callback(os.close, episode_fd)
        fcntl.flock(episode_fd, fcntl.LOCK_EX)
        if destination.exists() and not replace_existing:
            return

        outputs = []
        if keep_intermediates:
            outputs.extend(
                (source.parent / name, destination.parent / target)
                for name, target in INTERMEDIATE_OUTPUTS
            )
        outputs.append((source, destination))
        staged = {}
        # Finish copying every file before publishing the action as the completion marker.
        for original, target in outputs:
            if target.is_dir():
                raise IsADirectoryError(f"Output path is a directory: {target}")
            with NamedTemporaryFile(
                dir=destination.parent, prefix=".action-", delete=False
            ) as file:
                temporary = Path(file.name)
            cleanup.callback(temporary.unlink, missing_ok=True)
            shutil.copy(original, temporary)
            staged[target] = temporary

        for target, temporary in staged.items():
            if target != destination:
                temporary.replace(target)
        if replace_existing:
            staged[destination].replace(destination)
        else:
            try:
                os.link(staged[destination], destination)
            except FileExistsError:
                pass


def fill_existing_actions(episodes):
    from humanlabel.export import (
        HEADER,
        frame_aligned_actions,
        read_action_rows,
        write_csv,
    )
    from humanlabel.utils.episode import read_episode

    updated = 0
    failures = []
    with TemporaryDirectory(prefix="humanlabel-fill-") as folder:
        output = Path(folder) / "action.csv"
        for episode in episodes:
            action = episode / "action.csv"
            if not action.is_file():
                continue
            try:
                _, source_rows, *_ = read_episode(episode)
                stamps = [row["timestamp_unix_s"] for row in source_rows]
                before = read_action_rows(action)
                aligned = frame_aligned_actions(before, stamps)
                if aligned != before:
                    write_csv(output, HEADER, aligned)
                    publish_action(output, action, replace_existing=True)
                    updated += 1
            except (OSError, ValueError) as exc:
                failures.append((episode, exc))
    print(f"已补齐 {updated} 个已有 action.csv。")
    if failures:
        episode, error = failures[0]
        raise SystemExit(f"{len(failures)} 个片段无法补齐。首个失败 {episode}: {error}")


class LabelWorker:
    def __init__(
        self, workspace, threads, intrinsics, relabel, piperx=None, *, keep_intermediates=False
    ):
        import cv2
        import torch

        from humanlabel.export import capture_contract
        from humanlabel.utils.episode import load_detector
        from humanlabel.workflows.egohand import load_predictor

        cv2.setNumThreads(1)
        torch.set_num_threads(threads)
        self.workspace = workspace
        self.intrinsics = intrinsics
        self.relabel = relabel
        self.keep_intermediates = keep_intermediates
        self.contract = capture_contract()
        self.retargeter = None
        if piperx is not None:
            from humanlabel.workflows.piperx_retarget import PiperXRetargeter

            self.retargeter = PiperXRetargeter(*piperx, self.contract["gripper_max_m"])
        self.detector = load_detector(
            MODEL / "weights/external/detector.pt", workspace / f"yolo-{os.getpid()}"
        )
        self.predictor = load_predictor()
        self.detector.to("cuda")

    def label(self, episode):
        from humanlabel.export import (
            HEADER,
            frame_aligned_actions,
            read_action_rows,
            update_episode,
            write_csv,
        )
        from humanlabel.validation import validate
        from humanlabel.workflows import egohand, recover

        destination = episode / "action.csv"
        if destination.exists() and not self.relabel:
            return
        with TemporaryDirectory(dir=self.workspace) as folder:
            work = Path(folder)
            native, output = work / "egohand", work / "result"
            egohand.run(
                episode,
                native,
                predictor=self.predictor,
                detector=self.detector,
                intrinsics=self.intrinsics,
            )
            recover.run(native, output, predictor=self.predictor)
            final = output / episode.name
            update_episode(final, self.contract)
            validate(final)
            if self.retargeter is None:
                raw = read_action_rows(final / "action_raw.csv")
                aligned = frame_aligned_actions(raw, [row[0] for row in raw])
            else:
                aligned = self.retargeter.actions(final, episode / "timestamps.csv")
            write_csv(final / "action.csv", HEADER, aligned)
            publish_action(
                final / "action.csv",
                destination,
                replace_existing=self.relabel,
                keep_intermediates=self.keep_intermediates,
            )


_worker = None


def initialize_worker(
    devices, workspace, threads, ready_workers, intrinsics, relabel, piperx, keep_intermediates
):
    global _worker

    device = devices.get()
    os.environ.update(
        CUDA_VISIBLE_DEVICES=device, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"
    )
    # Silence progress and library diagnostics at the descriptor level, including
    # native CUDA/OpenCV output. The parent owns terminal output.
    with open(os.devnull, "w") as quiet:
        os.dup2(quiet.fileno(), 1)
        os.dup2(quiet.fileno(), 2)
    try:
        _worker = LabelWorker(
            workspace, threads, intrinsics, relabel, piperx,
            keep_intermediates=keep_intermediates,
        )
    except Exception as exc:
        (workspace / f"startup-{os.getpid()}.error").write_text(
            f"GPU {device}: {type(exc).__name__}: {exc}"
        )
        raise
    with ready_workers.get_lock():
        ready_workers.value += 1


def label_episode(episode):
    _worker.label(episode)


def label_parallel(
    episodes, devices, workers_per_gpu, live, intrinsics=None, *, relabel=False,
    piperx=None, keep_intermediates=False,
):
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.spinner import Spinner

    slots = (devices * workers_per_gpu)[: len(episodes)]
    worker_config = f"{len(set(slots))} GPU · {len(slots)} workers"
    deploying = f"正在部署模型 · {worker_config}"
    live.update(Spinner("dots", text=deploying), refresh=True)
    if not live.console.is_terminal or live.console.is_dumb_terminal:
        live.console.print(deploying)
    threads = max(1, min(8, len(os.sched_getaffinity(0)) // len(slots)))
    context = mp.get_context("spawn")
    ready_workers = context.Value("i", 0)
    failures = []
    with (
        TemporaryDirectory(prefix="humanlabel-") as folder,
        closing(context.SimpleQueue()) as device_queue,
    ):
        workspace = Path(folder)
        for device in slots:
            device_queue.put(device)
        pool = ProcessPoolExecutor(
            max_workers=len(slots),
            mp_context=context,
            initializer=initialize_worker,
            initargs=(
                device_queue, workspace, threads, ready_workers, intrinsics,
                relabel, piperx, keep_intermediates,
            ),
        )
        task = None
        try:
            futures = {
                pool.submit(label_episode, episode): episode for episode in episodes
            }
            pending = set(futures)
            while pending and ready_workers.value < len(slots):
                _, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)

            progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TextColumn("ETA"),
                TimeRemainingColumn(),
                console=live.console,
            )
            if ready_workers.value == len(slots):
                stage = "正在标注" if piperx is None else "正在标注及优化双臂姿态"
                task = progress.add_task(
                    f"{stage} · {worker_config}", total=len(episodes)
                )
                live.update(progress, refresh=True)

            for future in as_completed(futures):
                try:
                    future.result()
                except BrokenProcessPool:
                    raise
                except Exception as exc:
                    failures.append((futures[future], exc))
                if task is not None:
                    progress.advance(task)
        except BrokenProcessPool as exc:
            if task is None:
                live.update("模型部署失败", refresh=True)
            error = next(workspace.glob("startup-*.error"), None)
            detail = error.read_text() if error else str(exc)
            raise SystemExit(f"标注 worker 中止。{detail}") from exc
        finally:
            pool.shutdown(cancel_futures=True)

    if failures:
        episode, error = failures[0]
        raise SystemExit(f"{len(failures)} 个片段失败。首个失败 {episode}: {error}")
    if task is None:
        live.update(f"标注完成 · {len(episodes)} 个片段", refresh=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Episode, task directory, or dataset root"
    )
    parser.add_argument(
        "--workers-per-gpu", type=int, default=2, help="Workers per visible GPU (default: 2)"
    )
    parser.add_argument("--intrinsics-json", type=Path, help="Head camera intrinsics JSON")
    parser.add_argument(
        "--extrinsics-yaml",
        type=Path,
        help="Head-to-base calibration. Enables SOMA-guided dual Piper X actions.",
    )
    parser.add_argument("--piperx-urdf", type=Path, help="Override the bundled Piper X arm URDF")
    parser.add_argument("--piperx-gripper-xacro", type=Path, help="Override the bundled gripper Xacro")
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Save SOMA, camera eposes, and preview directly in each episode directory",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fill-existing",
        action="store_true",
        help="Fill existing action.csv files to one finite action per frame on CPU",
    )
    mode.add_argument(
        "--relabel",
        action="store_true",
        help="Re-run inference and atomically replace each successfully validated action.csv",
    )
    args = parser.parse_args(argv)
    if args.workers_per_gpu < 1:
        parser.error("--workers-per-gpu must be at least 1")
    if args.fill_existing and args.keep_intermediates:
        parser.error("--keep-intermediates requires an inference run")
    if args.extrinsics_yaml:
        if not args.intrinsics_json:
            parser.error("--extrinsics-yaml requires --intrinsics-json")
    elif args.piperx_urdf or args.piperx_gripper_xacro:
        parser.error("Piper X model overrides require --extrinsics-yaml")

    source = args.source.resolve()
    episodes = find_episodes(source)
    if not episodes:
        parser.error(f"No head-view episodes in {source}")
    if args.fill_existing:
        fill_existing_actions(episodes)
        return
    pending = episodes if args.relabel else [
        episode for episode in episodes if not (episode / "action.csv").exists()
    ]
    if not pending:
        print("没有待标注的片段。")
        return

    intrinsics = None
    if args.intrinsics_json:
        from humanlabel.utils.camera import load_head_calibration

        try:
            intrinsics = load_head_calibration(args.intrinsics_json)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            parser.error(f"Invalid head calibration {args.intrinsics_json}: {exc}")

    piperx = None
    if args.extrinsics_yaml:
        try:
            from humanlabel.workflows.piperx_retarget import load_base_transforms
        except ImportError as exc:
            parser.error(f"Piper X dependencies are unavailable. See requirements-piperx.txt. {exc}")
        urdf = args.piperx_urdf or PIPERX_ASSETS / "piper_x_description.urdf"
        gripper = args.piperx_gripper_xacro or PIPERX_ASSETS / "piper_x_with_gripper_description.xacro"
        for model in (urdf, gripper):
            if not model.is_file():
                parser.error(f"Piper X model file not found: {model}")
        try:
            piperx = load_base_transforms(args.extrinsics_yaml), urdf.resolve(), gripper.resolve()
        except (KeyError, OSError, TypeError, ValueError) as exc:
            parser.error(f"Invalid base calibration {args.extrinsics_yaml}: {exc}")

    from rich.console import Console
    from rich.live import Live
    from rich.spinner import Spinner

    try:
        with Live(
            Spinner("dots", text="正在检查 CUDA 设备"), console=Console(stderr=True)
        ) as live:
            import torch

            count = torch.cuda.device_count()
            if not count:
                parser.error("No visible CUDA devices")
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            devices = (
                visible.split(",")[:count]
                if visible is not None
                else [str(index) for index in range(count)]
            )
            label_parallel(
                pending, devices, args.workers_per_gpu, live, intrinsics,
                relabel=args.relabel, piperx=piperx, keep_intermediates=args.keep_intermediates,
            )
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
