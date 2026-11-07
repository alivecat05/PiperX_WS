"""Pinned WiLoR assets and a one-time NumPy conversion of its MANO data."""

import argparse
import hashlib
import json
import pickle
import subprocess
import time
import urllib.request
from pathlib import Path

from humanlabel.utils.paths import ROOT

CODE_REVISION = "ebec42f94c389070cdd7dda6fd1bf0b4a659c960"
WEIGHTS_REVISION = "b00adea9a6843bbb4c9042109c5eb29ab2a59dea"
FILES = ("MANO_RIGHT.pkl", "mano_mean_params.npz", "detector.pt", "wilor_final.ckpt")


class ArrayState:
    def __setstate__(self, state):
        self.__dict__.update(state)


class ChumpyArray(ArrayState):
    pass


class ChumpySelect(ArrayState):
    pass


class SparseCSC(ArrayState):
    pass


class ManoReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return ChumpySelect if name == "Select" else ChumpyArray
        if module.startswith("scipy.sparse"):
            if name != "csc_matrix":
                raise ValueError(f"Unsupported MANO sparse data: {name}")
            return SparseCSC
        return super().find_class(module, name)


def convert_mano(source, target):
    """Resolve the legacy lazy arrays without importing obsolete chumpy."""
    import numpy as np

    def array(value):
        if isinstance(value, ChumpySelect):
            selected = np.asarray(array(value.a)).ravel()[np.asarray(value.idxs)]
            shape = getattr(value, "preferred_shape", None)
            return selected.reshape(shape) if shape is not None else selected
        if isinstance(value, ChumpyArray):
            return np.asarray(array(value.x))
        if isinstance(value, SparseCSC):
            dense = np.zeros(value._shape, dtype=np.asarray(value.data).dtype)
            for col in range(value._shape[1]):
                start, end = value.indptr[col : col + 2]
                dense[value.indices[start:end], col] = value.data[start:end]
            return dense
        return value

    with Path(source).open("rb") as stream:
        data = ManoReader(stream, encoding="latin1").load()
    converted = {key: array(value) for key, value in data.items()}
    if any(isinstance(value, ArrayState) for value in converted.values()):
        raise ValueError("Unresolved MANO data")
    for key, shape in (
        ("v_template", (778, 3)),
        ("J_regressor", (16, 778)),
        ("shapedirs", (778, 3, 10)),
    ):
        if np.asarray(converted[key]).shape != shape:
            raise ValueError(f"Unexpected MANO {key} dimensions")
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    with Path(target).open("wb") as stream:
        pickle.dump(converted, stream, protocol=4)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prepare(cache=ROOT / ".cache/wilor", repo=ROOT / ".cache/WiLoR-mini"):
    cache, repo = Path(cache), Path(repo)
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not (repo / ".git").exists():
        subprocess.run(
            ["git", "clone", "https://github.com/warmshao/WiLoR-mini.git", str(repo)],
            check=True,
        )
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != CODE_REVISION:
        subprocess.run(
            ["git", "-C", str(repo), "fetch", "origin", CODE_REVISION], check=True
        )
        subprocess.run(
            ["git", "-C", str(repo), "checkout", "--detach", CODE_REVISION], check=True
        )
    url = f"https://huggingface.co/api/models/warmshao/WiLoR-mini/revision/{WEIGHTS_REVISION}?blobs=true"
    info = json.load(urllib.request.urlopen(url, timeout=30))
    folder = cache / "pretrained_models"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for name in FILES:
        item = next(
            x for x in info["siblings"] if x["rfilename"] == f"pretrained_models/{name}"
        )
        expected = item.get("lfs", {}).get("sha256")
        target = folder / name
        if not target.exists() or (expected and digest(target) != expected):
            temporary = target.with_suffix(target.suffix + ".download")
            asset = f"https://huggingface.co/warmshao/WiLoR-mini/resolve/{WEIGHTS_REVISION}/pretrained_models/{name}"
            print(f"Downloading {name}...", flush=True)
            started = last = time.monotonic()
            done = 0
            with (
                urllib.request.urlopen(asset, timeout=60) as response,
                temporary.open("wb") as output,
            ):
                for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                    output.write(block)
                    done += len(block)
                    if time.monotonic() - last > 10:
                        print(
                            f"  {name}: {done / 1024**2:.0f} MiB ({done / (time.monotonic() - started) / 1024**2:.1f} MiB/s)",
                            flush=True,
                        )
                        last = time.monotonic()
            if expected and digest(temporary) != expected:
                raise ValueError(f"Checkpoint checksum mismatch: {name}")
            temporary.replace(target)
        records.append(
            {"name": name, "sha256": digest(target), "size": target.stat().st_size}
        )
    convert_mano(folder / "MANO_RIGHT.pkl", folder / "MANO_RIGHT_numpy.pkl")
    (cache / "weights.json").write_text(
        json.dumps(
            {
                "repository": "warmshao/WiLoR-mini",
                "revision": WEIGHTS_REVISION,
                "code_revision": CODE_REVISION,
                "files": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("MANO assets ready.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/wilor")
    parser.add_argument("--repo", type=Path, default=ROOT / ".cache/WiLoR-mini")
    options = parser.parse_args(argv)
    prepare(options.cache, options.repo)


if __name__ == "__main__":
    main()
