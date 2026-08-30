# Running Frigate NVR on Jetson Orin Nano / JetPack 7.2 (unofficial build guide)

JetPack 7.2 is new enough that Frigate has no official image for it — the newest Jetson
build upstream ships is `-tensorrt-jp6` (JetPack 6). This guide covers building Frigate
from source against a hand-assembled JetPack 7.2 environment, with GPU-accelerated
object detection (ONNX Runtime + TensorRT) and NVDEC hardware video decode both working.

Two versions are covered:
- **Part 1 — Frigate 0.16.2**: the simpler, more stable path. Build this first regardless
  of which version you actually want to run day to day — Part 2 depends on it.
- **Part 2 — Frigate 0.18.0-beta3**: optional migration on top of Part 1, with several
  additional patches this newer version's build scripts need.

Tested hardware: Jetson Orin Nano, JetPack 7.2 (L4T r39.2), CUDA 13.2, TensorRT 10.16,
Ubuntu 24.04, sm_87.

---

## Before you start

### Check your JetPack version matches

Jetson Docker images are tightly coupled to the host's L4T/driver version — there's no
clean driver-abstraction layer the way `nvidia-docker` provides on x86 CUDA setups. If
you're not on **JetPack 7.2 / L4T r39.2** specifically, a pre-built base image tarball
(see below) may not work cleanly for you, and you may need to rebuild the base image
yourself following the same underlying approach.

### What you need before touching Frigate

**1. A CUDA 13.2 + TensorRT 10.16 + cuDNN base image for this platform.**
This doesn't exist prebuilt anywhere — it has to be assembled from the public Jetson apt
repo. [<a href="https://huggingface.co/ldalipis/jp7-base-cuda13.2-trt10.16/blob/main/jp7-base-cuda13.2-trt10.16.tar">jp7-base-cuda13.2-trt10.16.tar</a>, `docker load` it and tag it
`lab5/jp7-base:cuda13.2-trt10.16` to match the commands below, or adjust the tag
throughout.]

**2. `onnxruntime-gpu` compiled for CUDA 13.2 / aarch64 / sm_87 / Python 3.12.**
No prebuilt wheel exists for JetPack 7 anywhere — this has to be compiled from source,
and it's the single most time- and memory-intensive part of this whole project (budget
real time and make sure you have swap configured — 16GB was sufficient on an Orin Nano).
Not covered in this guide. [Attach: `onnxruntime_gpu-1.23.0-cp312-cp312-linux_aarch64.whl`]

Both of the above credit **straga's `jetson-jp7-onnxruntime`** project, which pioneered
this specific combination for JetPack 7.2.

---

## Part 1 — Frigate 0.16.2

### 1. Clone and set up the build context

```bash
cd ~
git clone --branch v0.16.2 https://github.com/blakeblackshear/frigate.git frigate-jp7
cd frigate-jp7
cp /path/to/onnxruntime_gpu-1.23.0-cp312-cp312-linux_aarch64.whl docker/main/
cp docker/main/Dockerfile docker/main/Dockerfile.jp7
```

### 2. Patch `docker/main/Dockerfile.jp7`

**Python 3.11 → 3.12** in the `wheels` stage (Ubuntu 24.04 ships 3.12 natively — no PPA
needed):
```dockerfile
# find:
    python3.11 \
    python3.11-dev \
# replace with:
    python3.12 \
    python3.12-dev \
```
```dockerfile
# find:
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
# replace with:
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
```

**Inject your onnxruntime wheel**, stripping the stock one first — find the block:
```dockerfile
COPY docker/main/requirements-wheels.txt /requirements-wheels.txt
RUN pip3 wheel --wheel-dir=/wheels -r /requirements-wheels.txt
```
Insert a `sed` strip before the build, and copy your wheel in after:
```dockerfile
COPY docker/main/requirements-wheels.txt /requirements-wheels.txt
RUN sed -i "/onnxruntime/d" /requirements-wheels.txt
RUN pip3 wheel --wheel-dir=/wheels -r /requirements-wheels.txt
COPY docker/main/onnxruntime_gpu-1.23.0-cp312-cp312-linux_aarch64.whl /wheels/
```
(The `deps` stage's existing `pip3 install -U /deps/wheels/*.whl` picks this up
automatically — no separate install step needed.)

**Strip `nvidia-pyindex`** from `requirements.txt` (it tries to build from source and
fails under Python 3.12 — you don't need NVIDIA's private pip index since CUDA/TensorRT
come from apt already):
```dockerfile
# find:
COPY docker/main/requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt
# replace with:
COPY docker/main/requirements.txt /requirements.txt
RUN sed -i "/nvidia-pyindex/d" /requirements.txt
RUN pip3 install -r /requirements.txt
```

### 3. Patch `docker/main/install_deps.sh`

Same Python swap, for the runtime stage:
```bash
sed -i 's/python3\.11/python3.12/g' docker/main/install_deps.sh
```

Add retry flags to the `yq` download (it can fail under concurrent build-stage load,
even with plenty of network bandwidth — this is a real, reproducible failure mode, not
a one-off):
```bash
# find the curl line for yq, add:
--retry 5 --retry-delay 3 --retry-connrefused --retry-all-errors
```

### 4. Patch the apt-source handling in `build_nginx.sh` and `build_sqlite_vec.sh`

Both scripts assume Debian's classic one-line apt sources format. Ubuntu 24.04 uses the
newer deb822 format (`/etc/apt/sources.list.d/ubuntu.sources`), which silently breaks
`apt-get build-dep`. In **both** files, find:
```bash
if [[ "$VERSION_ID" == "12" ]]; then
    sed -i '/^Types:/s/deb/& deb-src/' /etc/apt/sources.list.d/debian.sources
else
    cp /etc/apt/sources.list /etc/apt/sources.list.d/sources-src.list
    sed -i 's|deb http|deb-src http|g' /etc/apt/sources.list.d/sources-src.list
fi
```
and add an `elif` branch:
```bash
if [[ "$VERSION_ID" == "12" ]]; then
    sed -i '/^Types:/s/deb/& deb-src/' /etc/apt/sources.list.d/debian.sources
elif [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then
    sed -i '/^Types:/s/deb/& deb-src/' /etc/apt/sources.list.d/ubuntu.sources
else
    cp /etc/apt/sources.list /etc/apt/sources.list.d/sources-src.list
    sed -i 's|deb http|deb-src http|g' /etc/apt/sources.list.d/sources-src.list
fi
```

### 5. Wire the Makefile to your base image

Edit the top-level `Makefile`'s `local:` target:
```makefile
local: version
	docker buildx build --target=frigate --file docker/main/Dockerfile.jp7 . \
		--build-arg BASE_IMAGE=lab5/jp7-base:cuda13.2-trt10.16 \
		--build-arg SLIM_BASE=lab5/jp7-base:cuda13.2-trt10.16 \
		--tag frigate:latest \
		--load
```
`SLIM_BASE` matters most — it's what the final runtime image is built `FROM`, which is
why CUDA/TensorRT end up baked into the image with no host mounts needed at runtime.

### 6. Build

```bash
sudo make local
```

If it fails at any point, retrying the exact same command has resolved several
transient failures in testing (stale/incomplete apt indexes under concurrent build
load being the most common cause) — don't assume a repeat failure at the same spot
means something's fundamentally broken before trying at least once more.

### 7. Fix the host-level `nvidia-container-toolkit` regression (critical, do this now)

`nvidia-container-toolkit`/`libnvidia-container*` at **1.19.x** has a real bug — a new
"CUDA Forward Compatibility" hook for Tegra/Orin systems panics
(`cudacompat.trim: slice bounds out of range`) on non-official (hand-assembled) base
images, breaking **every** container using `--runtime nvidia`. Downgrade and hold:
```bash
sudo apt-get install --allow-downgrades \
  nvidia-container-toolkit=1.18.2-1 \
  nvidia-container-toolkit-base=1.18.2-1 \
  libnvidia-container-tools=1.18.2-1 \
  libnvidia-container1=1.18.2-1

sudo apt-mark hold nvidia-container-toolkit nvidia-container-toolkit-base \
  libnvidia-container-tools libnvidia-container1
```
Check this hold survives after every `apt upgrade` — `apt-mark showhold` should always
list all four. As of testing, 1.20.0 is available upstream and has **not** been
confirmed to fix this issue.

### 8. Run it

```bash
mkdir -p ~/frigate-jp7/config ~/frigate-jp7/storage

docker run -d \
  --name frigate \
  --runtime=nvidia \
  --network host \
  --ipc=host \
  --restart always \
  --shm-size=2g \
  --stop-timeout 30 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra:ro \
  -v /etc/localtime:/etc/localtime:ro \
  -v /home/YOUR_USER/frigate-jp7/config:/config \
  -v /home/YOUR_USER/frigate-jp7/storage:/media/frigate \
  frigate:latest
```

**Verify TensorRT/CUDA are actually active:**
```bash
docker run --rm --runtime nvidia --entrypoint python3 frigate:latest \
  -c "import onnxruntime as ort; print(ort.get_available_providers())"
# expect: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

### 9. Config essentials

```yaml
detectors:
  onnx:
    type: onnx

model:
  model_type: yolo-generic
  path: /config/model_cache/your-model.onnx
  width: 960       # must match your model's export imgsz exactly
  height: 960
  input_tensor: nchw
  input_pixel_format: rgb   # test bgr too if detection quality is poor
  input_dtype: float
  labelmap_path: /config/model_cache/coco-80.txt

cameras:
  your_camera:
    ffmpeg:
      hwaccel_args:
        - -c:v
        - h264_nvv4l2dec   # confirm this decoder name against your ffmpeg build; see note below
```

**Note on `h264_nvv4l2dec`**: this decoder name comes from NVIDIA's official L4T
Multimedia API. If your ffmpeg was instead patched with the separate, unofficial
`jocover/jetson-ffmpeg` project, the equivalent decoder is named `h264_nvmpi` — check
which one your build actually has:
```bash
docker exec frigate ffmpeg -hide_banner -decoders 2>&1 | grep -i "nvv4l2\|nvmpi"
```
The Frigate 0.18 UI's built-in "NVIDIA Jetson" hardware-acceleration preset assumes
`nvmpi` — if your build uses `nvv4l2dec` instead (as this base image's does), you'll
need to use **Manual arguments** in the UI rather than the preset, with
`-c:v h264_nvv4l2dec` entered explicitly.

---

## Part 2 — Migrating to Frigate 0.18.0-beta3 (optional)

Everything from Part 1 that's genuinely hard to build — the base image, the
onnxruntime wheel, the `nvidia-container-toolkit` fix — carries over completely
unchanged. Only the Frigate-side build scripts need re-patching, since 0.18's version
of them differs from 0.16.2's.

### 1. Clone fresh, in a separate directory

```bash
cd ~
git clone --branch v0.18.0-beta3 https://github.com/blakeblackshear/frigate.git frigate18
cd frigate18
cp /path/to/onnxruntime_gpu-1.23.0-cp312-cp312-linux_aarch64.whl docker/main/
cp docker/main/Dockerfile docker/main/Dockerfile.jp7
```

Keep your working Part 1 install running/intact throughout — this whole part is
designed to be fully reversible if anything goes wrong.

### 2. Same base patches as Part 1, applied to the 0.18 files

Apply the same categories of edits from Part 1 steps 2–4 (Python 3.12 swap, wheel
injection, apt-source deb822 fix) — but **verify exact line numbers/context first**,
since 0.18's files have real structural differences from 0.16.2's. One difference
worth knowing up front: in 0.18, `requirements.txt` no longer contains
`nvidia-pyindex` at all — that specific strip isn't needed this time.

### 3. New in 0.18 — additional cp311-only wheels that must be stripped

Two more packages are pinned to Python 3.11 wheels with no aarch64 build above cp311.
Neither is needed if you don't have Coral/Hailo hardware:

**`tflite_runtime`** — in `requirements-wheels.txt`, extend the earlier strip:
```dockerfile
RUN sed -i -e "/onnxruntime/d" -e "/tflite_runtime/d" /requirements-wheels.txt
```

**`hailort`** — staged directly into `/wheels/` by `install_hailort.sh`, not listed in
a requirements file. Remove the resulting wheel right after that script runs:
```dockerfile
RUN --mount=type=bind,source=docker/main/install_hailort.sh,target=/deps/install_hailort.sh \
    /deps/install_hailort.sh
RUN rm -f /wheels/hailort-*cp311*.whl
```

### 4. New in 0.18 — a new "Coral bookworm-backports" block breaks on this base image

`install_deps.sh`'s arm64 branch tries to add an unsigned Debian repo for
`mesa-teflon-delegate` (Coral EdgeTPU support). Extend its existing Nvidia-skip
condition to also cover this base image (you don't need Coral support either):
```bash
# find:
if [[ "${BASE_IMAGE}" == *"nvcr.io/nvidia/tensorrt"* ]]; then
    echo "Info: Skipping apt-get commands because BASE_IMAGE includes 'nvcr.io/nvidia/tensorrt' for arm64."
# replace with:
if [[ "${BASE_IMAGE}" == *"nvcr.io/nvidia/tensorrt"* ]] || [[ "${BASE_IMAGE}" == *"jp7-base"* ]]; then
    echo "Info: Skipping apt-get commands because BASE_IMAGE is a custom Nvidia/Jetson base for arm64."
```

### 5. New in 0.18 — `apt-get autoremove` strips real CUDA runtime libraries

Near the end of `install_deps.sh`, `apt-get autoremove --purge -y` removes the CUDA
toolkit meta-package (correct — you don't need the full dev toolkit at runtime) but
cascades into removing genuine runtime libraries (`cuda-cudart`, `libcublas`,
`libcufft`, `libcusolver`, etc.) that got dependency-orphaned in the process, even
though onnxruntime `dlopen`s them at runtime. This produces a build that succeeds but
silently has no working GPU inference. Skip autoremove for this base image:
```bash
# find:
apt-get purge gnupg apt-transport-https xz-utils -y
apt-get clean autoclean -y
apt-get autoremove --purge -y
rm -rf /var/lib/apt/lists/*
# replace with:
apt-get purge gnupg apt-transport-https xz-utils -y
apt-get clean autoclean -y
if [[ "${BASE_IMAGE}" == *"jp7-base"* ]]; then
    echo "Info: Skipping apt-get autoremove — would strip CUDA runtime libs needed by onnxruntime."
else
    apt-get autoremove --purge -y
fi
rm -rf /var/lib/apt/lists/*
```
This leaves some genuinely unneeded build tools in the final image (bigger, not
leaner) — a reasonable trade for not silently breaking GPU inference.

### 6. New in 0.18 — web UI build can OOM on 8GB boards

`npm run build` (Vite) can hit Node's default V8 heap ceiling under concurrent build
load, well before actual system memory/swap is exhausted. In `Dockerfile.jp7`'s
`web-build` stage:
```dockerfile
# find:
RUN npm run build \
    && mv dist/BASE_PATH/monacoeditorwork/* dist/assets/ \
    && rm -rf dist/BASE_PATH
# replace with:
RUN NODE_OPTIONS=--max-old-space-size=6144 npm run build \
    && mv dist/BASE_PATH/monacoeditorwork/* dist/assets/ \
    && rm -rf dist/BASE_PATH
```
Also make sure you actually have swap configured (16GB was sufficient) and that it
persists across reboots (`/etc/fstab`, not just a manual `swapon`).

### 7. New in 0.18 — a mixed-version onnxruntime install

Something in the build (likely bundled by `install_hailort.sh`) leaves a stray, older
`onnxruntime` (seen: 1.18.1) on disk via a raw file copy rather than `pip`, which
doesn't get cleanly overwritten by a normal reinstall — you can end up with a broken
mix of your wheel's Python files and the old version's compiled binary. Force a full
wipe before installing yours, positioned as the **very last thing** that touches
Python packages in the `deps` stage (after `COPY --from=deps-rootfs / /`, so nothing
later can overwrite it):
```dockerfile
COPY --from=deps-rootfs / /

RUN rm -rf /usr/local/lib/python3.12/dist-packages/onnxruntime*
RUN --mount=type=bind,from=wheels,source=/wheels,target=/deps/wheels \
    pip3 install --no-deps /deps/wheels/onnxruntime_gpu-1.23.0-cp312-cp312-linux_aarch64.whl

RUN ldconfig
```
Verify after building:
```bash
docker run --rm --entrypoint python3 frigate18:latest \
  -c "import onnxruntime; print(onnxruntime.__version__)"
# expect your actual version, e.g. 1.23.0 — not something older
```

### 8. New in 0.18 — bird classification crashes startup entirely if unavailable

`frigate/data_processing/real_time/bird.py` imports `tflite_runtime` (or falls back to
`ai_edge_litert`) **unconditionally at startup**, regardless of whether bird
classification is enabled in your config. Since neither has a working aarch64/cp312
wheel from their stable channels, install the nightly channel instead (same
importable module name):
```dockerfile
RUN pip3 install ai-edge-litert-nightly
```

### 9. New in 0.18 — arm64 ffmpeg has no NVDEC support at all

0.18's `install_deps.sh` downloads generic community ffmpeg binaries for arm64, with no
Jetson-specific decode support whatsoever — a real regression from whatever 0.16.2's
own arm64 path did (never fully identified; possibly a locally-provided `.deb` baked
into the base image assembly rather than anything Frigate's own scripts did). Rather
than re-deriving that, reuse the known-good binary from your **already-built Part 1**
image — this is why Part 1 has to happen first:

In `install_deps.sh`, guard out the generic tarball downloads for this base image
(skip straight to nothing — we'll get ffmpeg a different way):
```bash
if [[ "${TARGETARCH}" == "arm64" ]]; then
    if [[ "${BASE_IMAGE}" == *"jp7-base"* ]]; then
        echo "Info: ffmpeg for jp7-base is supplied via Dockerfile COPY from frigate:latest, not downloaded here."
    else
        mkdir -p /usr/lib/ffmpeg/5.0
        # ... (existing tarball download blocks, unchanged)
    fi
fi
```

In `Dockerfile.jp7`, right after the `install_deps.sh` RUN block:
```dockerfile
RUN --mount=type=bind,source=docker/main/install_deps.sh,target=/deps/install_deps.sh \
    /deps/install_deps.sh

COPY --from=frigate:latest /usr/bin/ffmpeg /usr/bin/ffmpeg
COPY --from=frigate:latest /usr/bin/ffprobe /usr/bin/ffprobe
COPY --from=frigate:latest /lib/aarch64-linux-gnu/libavdevice.so.62 /lib/aarch64-linux-gnu/libavdevice.so.62
COPY --from=frigate:latest /lib/aarch64-linux-gnu/libavfilter.so.11 /lib/aarch64-linux-gnu/libavfilter.so.11
COPY --from=frigate:latest /lib/aarch64-linux-gnu/libavformat.so.62 /lib/aarch64-linux-gnu/libavformat.so.62
COPY --from=frigate:latest /lib/aarch64-linux-gnu/libavcodec.so.62 /lib/aarch64-linux-gnu/libavcodec.so.62
COPY --from=frigate:latest /lib/aarch64-linux-gnu/libswresample.so.6 /lib/aarch64-linux-gnu/libswresample.so.6
COPY --from=frigate:latest /lib/aarch64-linux-gnu/libswscale.so.9 /lib/aarch64-linux-gnu/libswscale.so.9
COPY --from=frigate:latest /lib/aarch64-linux-gnu/libavutil.so.60 /lib/aarch64-linux-gnu/libavutil.so.60
RUN ldconfig
RUN mkdir -p /usr/lib/ffmpeg/8.0/bin \
    && ln -sf /usr/bin/ffmpeg /usr/lib/ffmpeg/8.0/bin/ffmpeg \
    && ln -sf /usr/bin/ffprobe /usr/lib/ffmpeg/8.0/bin/ffprobe
```
Only these 7 libraries are missing — everything else in ffmpeg's full dependency tree
(150+ libraries) already resolves against packages already present from other build
steps. Verify with `ldd` (`docker run --rm --entrypoint bash frigate:latest -c "ldd
/usr/bin/ffmpeg | grep 'not found'"` should be empty on the source image before you
trust the copy) — and confirm on the built image too:
```bash
docker run --rm --entrypoint bash frigate18:latest -c "ldd /usr/bin/ffmpeg | grep 'not found'"
# expect: empty
docker run --rm --entrypoint bash frigate18:latest -c "ffmpeg -version 2>&1 | grep -i nvv4l2"
# expect: a line showing --enable-nvv4l2dec
```

### 10. Wire the Makefile and build

Same as Part 1 step 5, but tag it separately so you can run both side by side and roll
back instantly if anything's wrong:
```makefile
local: version
	docker buildx build --target=frigate --file docker/main/Dockerfile.jp7 . \
		--build-arg BASE_IMAGE=lab5/jp7-base:cuda13.2-trt10.16 \
		--build-arg SLIM_BASE=lab5/jp7-base:cuda13.2-trt10.16 \
		--tag frigate18:latest \
		--load
```
```bash
sudo make local
```

### 11. Run it — test in isolation first

```bash
docker stop frigate   # stop Part 1's container while testing, avoid double camera connections

mkdir -p ~/frigate18/config ~/frigate18/storage
docker run -d \
  --name frigate18 \
  --runtime=nvidia \
  --network host \
  --ipc=host \
  --restart always \
  --shm-size=2g \
  --stop-timeout 30 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra:ro \
  -v /etc/localtime:/etc/localtime:ro \
  -v /home/YOUR_USER/frigate18/config:/config \
  -v /home/YOUR_USER/frigate18/storage:/media/frigate \
  frigate18:latest

docker logs -f frigate18
```

If anything's wrong: `docker rm -f frigate18` and `docker start frigate` gets you back
to exactly where Part 1 left off, untouched.

---

## Quick troubleshooting reference

| Symptom | Cause | Fix section |
|---|---|---|
| `E: Unable to locate package python3.12` | Transient apt index issue under concurrent build load | Just retry the build |
| `curl: (23) Failure writing output to destination` (yq) | Same — resource contention during concurrent build stages | Retry flags (Part 1 §3); retry the build |
| `E: The repository '...bookworm-backports...' is not signed` | Debian repo added onto Ubuntu base | Part 1 §4 or Part 2 §4, depending on which script |
| `TypeError: Unable to load from type 'NoneType'` (onnx detector) | `model:` nested under `detectors.onnx` instead of top-level | Move `model:` back to top level of config |
| `... is not a supported wheel on this platform` | A cp311-only wheel (onnxruntime, tflite_runtime, hailort) | Part 1 §2 / Part 2 §3 |
| `ModuleNotFoundError: No module named 'ai_edge_litert'` crashing startup | Unconditional bird.py import | Part 2 §8 |
| `AzureExecutionProvider` instead of Tensorrt/CUDA | Wrong/stock onnxruntime won the install | Part 2 §7 |
| `ImportError: ... cannot import name 'LoraAdapter'` | Mixed-version onnxruntime install | Part 2 §7 (full wipe before install) |
| `Unknown decoder 'h264_nvv4l2dec'` or `'h264_nvmpi'` | ffmpeg build doesn't match hwaccel_args / UI preset assumption | Check with `ffmpeg -decoders`, use Manual arguments |
| Container panics with `cudacompat.trim: slice bounds out of range` | `nvidia-container-toolkit` 1.19.x bug on Tegra | Part 1 §7 |
| Object masks seem to "hide" everything | Masks are **exclusion-only** in Frigate, not inclusion | Delete/redraw the mask |

---

## Credits

Base image and onnxruntime wheel: **straga/jetson-jp7-onnxruntime**.
Frigate: **blakeblackshear/frigate**.
