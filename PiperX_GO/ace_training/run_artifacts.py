"""Stage ACT artifacts on local disk; isolate remote storage in a subprocess."""
import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


FILES = ('training_config.json', 'dataset_stats.pkl', 'metrics.jsonl',
         'policy_latest.ckpt', 'last.pt')


def publish(source, destination):
    """Atomic remote replacements; remote I/O never runs in the training process."""
    copied = {}
    while True:
        stopping = (source / 'publish.stop').exists()
        try:
            destination.mkdir(parents=True, exist_ok=True)
            for name in FILES:
                src = source / name
                if not src.exists():
                    continue
                # Open before stat so a concurrent atomic checkpoint replacement
                # leaves this copy reading one complete, immutable inode.
                with src.open('rb') as reader:
                    st = os.fstat(reader.fileno())
                    identity = (st.st_ino, st.st_size, st.st_mtime_ns)
                    if copied.get(name) == identity:
                        continue
                    temporary = destination / (name + f'.publish-{os.getpid()}.tmp')
                    remaining = st.st_size
                    with temporary.open('wb') as writer:
                        while remaining:
                            chunk = reader.read(min(8 * 1024 * 1024, remaining))
                            if not chunk:
                                raise OSError(f'Short artifact read: {src}')
                            writer.write(chunk)
                            remaining -= len(chunk)
                    temporary.replace(destination / name)
                    copied[name] = identity
            (source / 'publish_status.json').write_text(json.dumps({
                'last_success_unix': time.time(), 'destination': str(destination)}))
        except OSError as error:
            (source / 'publish_status.json').write_text(json.dumps({'error': str(error)}))
            if stopping:
                raise
        if stopping:
            return
        # Poll only local storage. A stop request triggers a final publish.
        for _ in range(30):
            if (source / 'publish.stop').exists():
                break
            time.sleep(1)


def local_run_path(destination, local_root):
    destination = Path(destination).resolve()
    key = hashlib.sha256(str(destination).encode()).hexdigest()[:10]
    return Path(local_root) / (destination.name + '_' + key)


class RunArtifacts:
    def __init__(self, destination, local_root):
        self.destination = Path(destination).resolve()
        self.local = local_run_path(self.destination, local_root)
        self.local.mkdir(parents=True, exist_ok=True)
        (self.local / 'publish.stop').unlink(missing_ok=True)
        (self.local / 'destination.json').write_text(json.dumps({'output': str(self.destination)}))
        self.process = None

    def start(self):
        with (self.local / 'publisher.log').open('a') as log:
            self.process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                '--source', str(self.local), '--destination', str(self.destination)],
                stdout=log, stderr=log, start_new_session=True)

    def finish(self):
        if self.process is None:
            return
        (self.local / 'publish.stop').touch()
        try:
            code = self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f'Artifact sync still pending (pid={self.process.pid}); local files: {self.local}', flush=True)
            return
        if code:
            raise RuntimeError(f'Artifact sync failed; recover files from {self.local}; see publisher.log')


class TrainingWatchdog:
    """Record each rank's last phase and Python stacks if progress stops."""
    def __init__(self, directory, rank):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.state = ('startup', 0, time.monotonic())
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    def phase(self, name, step):
        self.state = (name, step, time.monotonic())

    def _watch(self):
        last_dump = 0
        while not self.stop.wait(10):
            phase, step, changed = self.state
            now = time.monotonic()
            (self.directory / f'rank-{self.rank}.json').write_text(json.dumps({
                'phase': phase, 'step': step, 'stalled_seconds': now - changed,
                'pid': os.getpid(), 'unix_time': time.time()}))
            if now - changed > 120 and now - last_dump > 60:
                with (self.directory / f'rank-{self.rank}-stacks.log').open('a') as log:
                    log.write(f'\nstep={step} phase={phase} stalled={now-changed:.1f}s\n')
                    log.flush()
                    faulthandler.dump_traceback(file=log, all_threads=True)
                last_dump = now


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    publish(args.source, args.destination)
