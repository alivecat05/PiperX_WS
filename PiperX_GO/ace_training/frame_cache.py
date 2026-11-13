"""Shared, resized uint8 frame cache: decode each training video once, mmap in workers."""
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import av
import numpy as np
from .preprocessing import letterbox_rgb


def signature(root, image_size):
    root = Path(root).resolve()
    digest = hashlib.sha256(str(root).encode())
    digest.update(json.dumps({"image_size":list(image_size),"preprocess":"rgb_letterbox_v1"}).encode())
    for name in ["info.json","splits.json","training_exclusions.json"]:
        p = root/"meta"/name
        if p.exists():
            digest.update(p.read_bytes())
    return digest.hexdigest()


def prepare_frame_cache(root, base, image_size, workers=4):
    from .data import metadata
    root = Path(root).resolve()
    image_size = tuple(image_size)
    key = signature(root,image_size)
    cache = Path(base).resolve()/f"{image_size[0]}x{image_size[1]}_{key[:12]}"
    cache.mkdir(parents=True,exist_ok=True)
    with (cache/"build.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        manifest_path = cache/"manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            if manifest["signature"] != key:
                raise ValueError("Frame cache identity mismatch")
            return cache
        info,splits,episodes = metadata(root)
        jobs = []
        for ep_id in splits["train"]:
            ep = episodes[ep_id]
            for camera in ["head","left_wrist","right_wrist"]:
                video_key = f"observation.images.{camera}"
                source = info["video_path"].format(video_key=video_key,
                    chunk_index=ep[f"videos/{video_key}/chunk_index"],file_index=ep[f"videos/{video_key}/file_index"])
                if ep[f"videos/{video_key}/from_timestamp"] != 0:
                    raise ValueError("The frame cache expects one full episode per video")
                destination = f"episode-{ep_id:06d}/{camera}.npy"
                jobs.append((source,destination,ep["length"]))
        print(f"Preparing shared frame cache: {len(jobs)} videos, {cache}",flush=True)
        start = time.perf_counter()

        def build(job):
            source,destination,n = job
            path = cache/destination
            expected = (n,*image_size,3)
            if path.exists():
                existing=np.load(path,mmap_mode="r")
                okay=existing.shape==expected and existing.dtype==np.uint8
                existing._mmap.close()
                if okay:
                    return source,destination,n
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_suffix(".npy.partial")
            array=np.lib.format.open_memmap(temporary,mode="w+",dtype=np.uint8,shape=expected)
            count=0
            with av.open(str(root/source)) as container:
                stream=container.streams.video[0]
                stream.thread_count=1
                for i,frame in enumerate(container.decode(stream)):
                    if i>=n or abs(float(frame.pts*stream.time_base)-i/info["fps"])>1e-3:
                        raise ValueError(f"Frame alignment mismatch in {source}")
                    array[i]=letterbox_rgb(frame,image_size)
                    count+=1
            if count!=n:
                raise ValueError(f"Frame count mismatch in {source}: {count}/{n}")
            array.flush(); array._mmap.close()
            temporary.replace(path)
            return source,destination,n

        mapping={}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(build,job) for job in jobs]
            for number,future in enumerate(as_completed(futures),1):
                source,destination,n=future.result()
                mapping[source]={"file":destination,"frames":n}
                if number%16==0 or number==len(jobs):
                    print(f"Frame cache: {number}/{len(jobs)} videos; {time.perf_counter()-start:.1f}s",flush=True)
        manifest={"signature":key,"dataset_root":str(root),"image_size":list(image_size),
            "dtype":"uint8","layout":"NHWC_RGB","training_episodes":splits["train"],"videos":mapping,
            "build_seconds":time.perf_counter()-start}
        temporary=manifest_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(manifest,indent=2))
        temporary.replace(manifest_path)
        print(f"Shared frame cache READY: {cache}",flush=True)
        return cache


class CachedFrames:
    def __init__(self, root, cache, image_size, return_uint8=False):
        self.root=Path(root).resolve();self.cache=Path(cache)
        self.manifest=json.loads((self.cache/"manifest.json").read_text())
        if self.manifest['signature']!=signature(self.root,image_size):
            raise ValueError("Frame cache does not match dataset/preprocessing")
        self.return_uint8=return_uint8
        self._pid=os.getpid();self._arrays=OrderedDict()

    def close(self):
        for value in self._arrays.values():
            value._mmap.close()
        self._arrays.clear()

    def __getstate__(self):
        state=self.__dict__.copy();state['_arrays']=OrderedDict()
        return state

    def read(self, path, indices, timestamp_offset=0.):
        if timestamp_offset!=0:
            raise ValueError("The frame cache expects zero video offset")
        if self._pid!=os.getpid():
            self.close();self._pid=os.getpid()
        relative=str(Path(path).relative_to(self.root))
        entry=self.manifest['videos'][relative]
        if relative not in self._arrays:
            value=np.load(self.cache/entry['file'],mmap_mode='r')
            if value.shape!=(entry['frames'],*self.manifest['image_size'],3) or value.dtype!=np.uint8:
                raise ValueError(f"Invalid cached frame array: {relative}")
            self._arrays[relative]=value
            if len(self._arrays)>32:
                self._arrays.popitem(last=False)[1]._mmap.close()
        self._arrays.move_to_end(relative)
        frames=np.ascontiguousarray(self._arrays[relative][np.asarray(indices,dtype=np.int64)].transpose(0,3,1,2))
        return frames if self.return_uint8 else frames.astype(np.float32)/255.


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',required=True)
    parser.add_argument('--cache-root',required=True)
    parser.add_argument('--image-size',type=int,nargs=2,default=[240,320])
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    prepare_frame_cache(args.dataset_root,args.cache_root,args.image_size,args.workers)
