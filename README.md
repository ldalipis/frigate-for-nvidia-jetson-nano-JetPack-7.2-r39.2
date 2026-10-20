# Running Frigate NVR on Jetson Orin Nano / JetPack 7.2 (unofficial build guide)

JetPack 7.2 is new enough that Frigate has no official image for it — the newest Jetson
build upstream ships is `-tensorrt-jp6` (JetPack 6). This guide covers building Frigate
**0.18.2** from source against a hand-assembled JetPack 7.2 environment, with
GPU-accelerated object detection (ONNX Runtime + TensorRT) and NVDEC hardware video decode
both working. 0.18.2 is a security release (go2rtc exec-filter RCE, websocket viewer RCE,
`/auth` path handling), so use it rather than the 0.18.0 betas.

- **Part 1** is the complete guide: from a fresh JetPack 7.2 install to a running
  Frigate 0.18.2 container.
- **Appendix A** covers building Frigate 0.16.2 the same way. It is optional, and is also
  one way to obtain the NVDEC ffmpeg that Part 1 needs.

The patched build files that Part 1 uses are in [`frigate-0.18-docker-files/`](frigate-0.18-docker-files/).

Tested hardware: Jetson Orin Nano (Super dev kit), JetPack 7.2 and 7.2.1 (L4T r39.2),
CUDA 13.2, TensorRT 10.16, Ubuntu 24.04, sm_87.

---

## Before you start

### Check your JetPack version matches

Jetson Docker images are tightly coupled to the host's L4T/driver version — there's no
clean driver-abstraction layer the way `nvidia-docker` provides on x86 CUDA setups. If
you're not on **JetPack 7.2 / L4T r39.2** specifically, the base image below may not work
cleanly for you, and you may need to adapt it.

### What you need before touching Frigate

**1. A CUDA 13.2 + TensorRT 10.16 + cuDNN base image for this platform.**
This doesn't exist prebuilt anywhere, and it can't be shared as a download because it
contains NVIDIA's CUDA/TensorRT/cuDNN packages. Build it yourself from the public Jetson
apt repo with straga's recipe (`docker/base/Dockerfile`):
```bash
git clone https://github.com/straga/jetson-jp7-onnxruntime
cd jetson-jp7-onnxruntime
sudo make base
```
Tag the result `lab5/jp7-base:cuda13.2-trt10.16` to match the commands below, or adjust
the tag throughout. The author's image was ~27 GB on disk. Once built, back it up with
`docker save` and record its `sha256sum` (see "Keep your artifacts safe").

**2. `onnxruntime-gpu` compiled for CUDA 13.2 / aarch64 / sm_87 / Python 3.12.**
No prebuilt wheel exists for JetPack 7 on PyPI. Compiling it from source is the most
time- and memory-intensive part of this project (budget real time and make sure you have
swap configured, 16GB was sufficient on an Orin Nano); straga's repo has the build
scripts. Or use the wheel the author built from that recipe (onnxruntime is MIT licensed
and the wheel bundles no NVIDIA libraries):
[onnxruntime_gpu-1.23.0-cp312-cp312-linux_aarch64.whl](https://huggingface.co/ldalipis/jp7-onnxruntime-gpu/blob/main/onnxruntime_gpu-1.23.0-cp312-cp312-linux_aarch64.whl)
(SHA256 in the table under "Keep your artifacts safe").

**3. An NVDEC-capable ffmpeg bundle (`jp7-ffmpeg-nvdec.tgz`).**
Frigate 0.18 ships only generic arm64 ffmpeg builds with no Jetson hardware decode, so
you supply your own. See Part 1 step 4 for what it must contain and how to produce it.
It cannot be published here (see "Redistribution").

Straga's `jetson-jp7-onnxruntime` project (MIT licensed) pioneered the base image and
wheel combination for JetPack 7.2.

---

## Part 1 — Build and run Frigate 0.18.2

### 1. Check `nvidia-container-toolkit`

Early on JetPack 7.2.0, `nvidia-container-toolkit` 1.19.x panicked
(`cudacompat.trim: slice bounds out of range`) on hand-assembled base images, and older
versions of this guide said to downgrade to 1.18.2 and hold it. **That is no longer
needed:** 1.19.1-1 works on JetPack 7.2.1 with no holds (1.20.0 exists upstream and is
untested here). Test before relying on it:
```bash
sudo docker run --rm --runtime nvidia lab5/jp7-base:cuda13.2-trt10.16 python3 --version
```
If that prints a Python version, you are fine. If it panics with the `cudacompat` error,
fall back to the downgrade-and-hold of the four packages (`nvidia-container-toolkit`,
`nvidia-container-toolkit-base`, `libnvidia-container-tools`, `libnvidia-container1`) at
`1.18.2-1`.

### 2. Clone Frigate and drop in the patched files

```bash
cd ~
git clone --branch v0.18.2 https://github.com/blakeblackshear/frigate.git frigate18
cd frigate18
```
Copy these from this repository's `frigate-0.18-docker-files/` into `docker/main/`:

| File in this repo | Copy to |
|---|---|
| `Dockerfile.jp7` | `docker/main/Dockerfile.jp7` |
| `install_deps.sh` | `docker/main/install_deps.sh` (overwrite) |
| `build_nginx.sh` | `docker/main/build_nginx.sh` (overwrite) |
| `build_sqlite_vec.sh` | `docker/main/build_sqlite_vec.sh` (overwrite) |

Also copy the onnxruntime wheel and the ffmpeg bundle (step 4) into `docker/main/`:
```bash
cp /path/to/onnxruntime_gpu-1.23.0-cp312-cp312-linux_aarch64.whl docker/main/
cp /path/to/jp7-ffmpeg-nvdec.tgz docker/main/
```
These files were written against 0.18.0-beta3 and applied unchanged to 0.18.0 and 0.18.2
(the upstream diff between 0.18.0 and 0.18.2 touched only `Makefile` and three nginx files
under `docker/main/rootfs/`). `Dockerfile.jp7` is a patched copy of upstream's
`docker/main/Dockerfile`; if you would rather patch the upstream files yourself, or you
are moving to a newer release, step 3 lists every change and why.

### 3. What the patches do (and how to check a newer release)

Before moving to any newer Frigate release, see what upstream touched:
```bash
git fetch --tags
git diff --stat v<old> v<new> -- docker/ Makefile
```
If a file listed below (or `requirements*.txt`, `install_hailort.sh`) shows up, read that
diff before building.

**a. Python 3.11 → 3.12** (Ubuntu 24.04 ships 3.12 natively — no PPA needed). In the
`wheels` stage of the Dockerfile:
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
Plus, in `install_deps.sh`: `sed -i 's/python3\.11/python3.12/g' docker/main/install_deps.sh`.

**b. Inject your onnxruntime wheel and strip the cp311-only wheels.** Three packages are
pinned to Python 3.11 wheels with no aarch64 build above cp311: `onnxruntime`,
`tflite_runtime` (Coral) and `hailort` (Hailo). The stock onnxruntime and `tflite_runtime`
are stripped from `requirements-wheels.txt` and your wheel is copied in after the build:
```dockerfile
COPY docker/main/requirements-wheels.txt /requirements-wheels.txt
RUN sed -i -e "/onnxruntime/d" -e "/tflite_runtime/d" /requirements-wheels.txt
RUN pip3 wheel --wheel-dir=/wheels -r /requirements-wheels.txt
COPY docker/main/onnxruntime_gpu-1.23.0-cp312-cp312-linux_aarch64.whl /wheels/
```
`hailort` is staged straight into `/wheels/` by `install_hailort.sh`, so remove its wheel
right after that script runs (you don't need Coral/Hailo support):
```dockerfile
RUN --mount=type=bind,source=docker/main/install_hailort.sh,target=/deps/install_hailort.sh \
    /deps/install_hailort.sh
RUN rm -f /wheels/hailort-*cp311*.whl
```
(The `deps` stage's existing `pip3 install -U /deps/wheels/*.whl` picks the wheel up.)

**c. `yq` download retries.** In `install_deps.sh`, add to the `yq` curl line:
`--retry 5 --retry-delay 3 --retry-connrefused --retry-all-errors`. The download can fail
under concurrent build-stage load, even with plenty of bandwidth.

**d. deb822 apt sources** in `build_nginx.sh` and `build_sqlite_vec.sh`. Both assume
Debian's one-line sources format; Ubuntu 24.04 uses `/etc/apt/sources.list.d/ubuntu.sources`,
which silently breaks `apt-get build-dep`. Add an `elif` branch:
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

**e. Skip the Coral "bookworm-backports" block.** `install_deps.sh`'s arm64 branch adds
an unsigned Debian repo for `mesa-teflon-delegate`. Extend its existing Nvidia-skip
condition to cover this base image:
```bash
# find:
if [[ "${BASE_IMAGE}" == *"nvcr.io/nvidia/tensorrt"* ]]; then
# replace with:
if [[ "${BASE_IMAGE}" == *"nvcr.io/nvidia/tensorrt"* ]] || [[ "${BASE_IMAGE}" == *"jp7-base"* ]]; then
```

**f. Skip `apt-get autoremove`.** Near the end of `install_deps.sh`,
`apt-get autoremove --purge -y` removes the CUDA toolkit meta-package (correct) but
cascades into removing real runtime libraries (`cuda-cudart`, `libcublas`, `libcufft`,
`libcusolver`, …) that onnxruntime `dlopen`s. The build succeeds but silently has no GPU
inference. Wrap it:
```bash
if [[ "${BASE_IMAGE}" == *"jp7-base"* ]]; then
    echo "Info: Skipping apt-get autoremove — would strip CUDA runtime libs needed by onnxruntime."
else
    apt-get autoremove --purge -y
fi
```
This leaves some unneeded build tools in the final image (bigger, not leaner) — a
reasonable trade.

**g. Web UI build can OOM on 8GB boards.** `npm run build` (Vite) can hit Node's default
V8 heap ceiling well before system memory is exhausted. In the `web-build` stage:
```dockerfile
RUN NODE_OPTIONS=--max-old-space-size=6144 npm run build \
    && mv dist/BASE_PATH/monacoeditorwork/* dist/assets/ \
    && rm -rf dist/BASE_PATH
```
Also make sure swap is configured (16GB was sufficient) and persists across reboots
(`/etc/fstab`, not just a manual `swapon`).

**h. A mixed-version onnxruntime install.** A stray, older `onnxruntime` (seen: 1.18.1)
ends up on disk via a raw file copy rather than `pip`, which a normal reinstall does not
overwrite cleanly — you get your wheel's Python files with the old compiled binary. Wipe
it before installing yours, as the **very last thing** touching Python packages in the
`deps` stage (after `COPY --from=deps-rootfs / /`):
```dockerfile
COPY --from=deps-rootfs / /

RUN rm -rf /usr/local/lib/python3.12/dist-packages/onnxruntime*
RUN --mount=type=bind,from=wheels,source=/wheels,target=/deps/wheels \
    pip3 install --no-deps /deps/wheels/onnxruntime_gpu-1.23.0-cp312-cp312-linux_aarch64.whl

RUN ldconfig
```

**i. Bird classification crashes startup if its library is missing.**
`frigate/data_processing/real_time/bird.py` imports `tflite_runtime` (or falls back to
`ai_edge_litert`) **unconditionally**, whether or not bird classification is enabled.
Neither has a stable aarch64/cp312 wheel, so install the nightly channel (same module
name):
```dockerfile
RUN pip3 install ai-edge-litert-nightly
```

**j. ffmpeg** — step 4.

### 4. The ffmpeg bundle (NVDEC)

0.18's `install_deps.sh` downloads generic community ffmpeg binaries for arm64, with no
Jetson-specific decode support. You need an NVDEC-capable ffmpeg 8.0.x
(`--enable-nvv4l2dec`, decoders `h264_nvv4l2dec` / `hevc_nvv4l2dec`, and the `rtsp` and
`sdp` demuxers), packaged as `jp7-ffmpeg-nvdec.tgz` so the build does not depend on any
local image tag.

**Contents.** `usr/bin/ffmpeg`, `usr/bin/ffprobe` and, in `usr/lib/aarch64-linux-gnu/`,
`libavdevice.so.62`, `libavfilter.so.11`, `libavformat.so.62`, `libavcodec.so.62`,
`libswresample.so.6`, `libswscale.so.9`, `libavutil.so.60`.

**Where to get one.**
- **Option A — extract it from a Frigate 0.16.2 image built per Appendix A.** This is how
  the author's bundle was produced. On the Orin, with that image built:
  ```bash
  docker run --rm --entrypoint tar frigate:latest -czhf - \
    /usr/bin/ffmpeg /usr/bin/ffprobe \
    /usr/lib/aarch64-linux-gnu/libavdevice.so.62 /usr/lib/aarch64-linux-gnu/libavfilter.so.11 \
    /usr/lib/aarch64-linux-gnu/libavformat.so.62 /usr/lib/aarch64-linux-gnu/libavcodec.so.62 \
    /usr/lib/aarch64-linux-gnu/libswresample.so.6 /usr/lib/aarch64-linux-gnu/libswscale.so.9 \
    /usr/lib/aarch64-linux-gnu/libavutil.so.60 > jp7-ffmpeg-nvdec.tgz
  ```
  (`-h` in `-czhf` stores real files instead of symlinks.) Before trusting the result,
  check `ldd /usr/bin/ffmpeg | grep 'not found'` inside that image is empty.
- **Option B — build your own ffmpeg 8.0.x** with NVIDIA's L4T Multimedia API decoder
  patches and RTSP support. The community builds by **grossimd**
  (the `selfffmpeg` variants of his `ghcr.io/grossimd/frigate` images) are a worked
  example of a self-built ffmpeg; note that NVIDIA's own `8.0.1-nvidia1` apt package has RTSP stripped
  out, so it is not usable for IP cameras.

Either way, lay the files out as described above, put the archive at
`docker/main/jp7-ffmpeg-nvdec.tgz`, and see "Redistribution" before sharing it.

**How the Dockerfile uses it.** Right after the `install_deps.sh` RUN block (already in
this repo's `Dockerfile.jp7`):
```dockerfile
RUN --mount=type=bind,source=docker/main/install_deps.sh,target=/deps/install_deps.sh \
    /deps/install_deps.sh

ADD docker/main/jp7-ffmpeg-nvdec.tgz /
RUN ldconfig
RUN mkdir -p /usr/lib/ffmpeg/8.0/bin \
    && ln -sf /usr/bin/ffmpeg /usr/lib/ffmpeg/8.0/bin/ffmpeg \
    && ln -sf /usr/bin/ffprobe /usr/lib/ffmpeg/8.0/bin/ffprobe
```
`ADD` auto-extracts a local tar archive. The symlinks matter: Frigate's
`resolve_ffmpeg_path('default')` expects `/usr/lib/ffmpeg/8.0/bin/ffmpeg`. Only these 7
libraries are missing; everything else in ffmpeg's dependency tree already resolves. In
`install_deps.sh` the arm64 branch for this base image must not download the generic
tarballs (this repo's file apt-installs stock ffmpeg plus symlinks; the Dockerfile
overwrites those files in the next step, so either is fine).

### 5. Wire the Makefile and build

Edit the stock `local:` target **in place on each fresh clone**; don't copy an old
Makefile over it, because its `VERSION` line would make the UI report the wrong version.
Use a versioned image tag so you can run two versions side by side and roll back:
```makefile
local: version
	docker buildx build --target=frigate --file docker/main/Dockerfile.jp7 . \
		--build-arg BASE_IMAGE=lab5/jp7-base:cuda13.2-trt10.16 \
		--build-arg SLIM_BASE=lab5/jp7-base:cuda13.2-trt10.16 \
		--tag frigate18:0.18.2 \
		--load
```
`SLIM_BASE` matters most — it's what the final runtime image is built `FROM`, which is
why CUDA/TensorRT end up baked into the image with no host mounts needed at runtime.
```bash
sudo make local 2>&1 | tee ~/build.log
```
A cold build takes about 20 minutes on an Orin Nano. If it fails, retrying the exact same
command has resolved several transient failures in testing (stale apt indexes under
concurrent build load, or a failed `yq` download); try once more before debugging.

### 6. Verify the image before running it

```bash
# onnxruntime with TensorRT + CUDA providers
sudo docker run --rm --runtime nvidia --entrypoint python3 frigate18:0.18.2 \
  -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
# expect: 1.23.0 ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

# ffmpeg with NVDEC and RTSP
sudo docker run --rm --entrypoint bash frigate18:0.18.2 -c "ffmpeg -version | head -1; ffmpeg -hide_banner -decoders | grep -c nvv4l2; ffmpeg -hide_banner -demuxers | grep -Ec ' rtsp| sdp'; ldd /usr/bin/ffmpeg | grep -c 'not found'"
# expect: n8.0.1-..., 6, 2, 0
```
(Always test images with `--entrypoint …`: the image's own ENTRYPOINT is the s6 `/init`,
which boots the whole stack and buries your command's output.)

### 7. Run it

```bash
mkdir -p ~/frigate18/config ~/frigate18/storage

docker run -d \
  --name frigate18 \
  --runtime=nvidia \
  --network host \
  --ipc=host \
  --restart always \
  --shm-size=2g \
  --stop-timeout 30 \
  --tmpfs /tmp/cache:size=1000000000 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra:ro \
  -v /etc/localtime:/etc/localtime:ro \
  -v /home/YOUR_USER/frigate18/config:/config \
  -v /home/YOUR_USER/frigate18/storage:/media/frigate \
  frigate18:0.18.2

docker logs -f frigate18
```
The container uses `--network host`, so no `-p` mappings are needed. Port **5000** is the
plain-HTTP, unauthenticated UI/API (what Home Assistant's integration uses); port **8971**
is the authenticated UI/API with TLS (`https://<jetson-ip>:8971`, login required; the
admin password is printed once in the first-run log). Other ports: 8554 RTSP restream,
8555 WebRTC, 1984 go2rtc API. In 0.18 the config file is `config.yaml`, and the UI writes
back into it, so always `docker restart frigate18` after editing it by hand.

### 8. Config essentials

```yaml
detectors:
  onnx:
    type: onnx

model:                       # top level — never nested under the detector
  model_type: yolo-generic
  path: /config/model_cache/your-model.onnx
  width: 960                 # must match your model's export imgsz exactly
  height: 960
  input_tensor: nchw
  input_pixel_format: rgb    # test bgr too if detection quality is poor
  input_dtype: float
  labelmap_path: /config/model_cache/coco-80.txt

cameras:
  your_camera:
    ffmpeg:
      hwaccel_args:
        - -c:v
        - h264_nvv4l2dec
```

**Set `hwaccel_args` manually.** The 0.18 UI preset "NVIDIA Jetson (H.264)" uses
`h264_nvmpi`, which comes from the separate, unofficial `jocover/jetson-ffmpeg` project
and which this ffmpeg does not have (`Unknown decoder`). The decoder here is
`h264_nvv4l2dec`, from NVIDIA's L4T Multimedia API; enter it as manual arguments and never
switch back to the preset. Check what your build has with:
```bash
docker exec frigate18 ffmpeg -hide_banner -decoders 2>&1 | grep -i "nvv4l2\|nvmpi"
```

**Camera limits.** Many cameras (e.g. TP-Link Tapo) accept only 3 concurrent RTSP
sessions. Use one record/view stream and one low-resolution detect stream per camera
through go2rtc; clients connecting to Frigate's restream (`rtsp://<jetson-ip>:8554/<name>`)
don't count against the camera.

### 9. Upgrading to a newer Frigate version

1. Clone the new tag next to the old one and run `git diff --stat v<old> v<new> -- docker/ Makefile`
   (step 3). If a patched file appears, read that diff first.
2. Copy the patched files, the wheel and the ffmpeg bundle into `docker/main/` and edit the
   new `Makefile`'s `local:` target in place with a new versioned tag. `sudo make local`,
   then run the step 6 checks.
3. Cut over: `docker stop frigate18 && docker update --restart=no frigate18`, then
   `sudo cp -a ~/frigate18/config ~/frigate18/config.bak-<old>` (with sudo, while stopped,
   so the SQLite DB is consistent). Start the new image under a temporary name with the
   same mounts, check `curl -s http://localhost:5000/api/stats` (version, camera fps,
   inference speed), then `docker rm frigate18`, rename the new container, and tag it.
   `--restart=no` matters: a manually stopped `--restart always` container comes back
   after a reboot and would open double RTSP sessions to your cameras.
4. Rollback: `docker rm -f <new>`, restore `config.bak-<old>` only if the old build rejects
   the database, and run the step 7 command with the old tag.

---

## Keep your artifacts safe

**Never run `docker system prune -a` or `docker image prune -a`.** They delete every image
not used by a running container, including your base image (this happened once and the
base image had to be reloaded from its tarball). Use `docker builder prune` or plain
`docker image prune` only.

Back up, and record `sha256sum` for, the three expensive artifacts:

| Artifact | Checksum |
|---|---|
| `jp7-base-cuda13.2-trt10.16.tar` (`docker save` of **your own** base build; not published, so your hash will differ from the author's) | author's: `e961ba5f778b30bb0e9012ff394a57e79b482faec7deca66ee3a60a199923ad0` |
| `jp7-ffmpeg-nvdec.tgz` | author's: `0a16c872461de05f71c3adced1b3f68bb671a682fe0568fb7813c4678d2c07bd` |
| `onnxruntime_gpu-1.23.0-cp312-cp312-linux_aarch64.whl` (95.7 MiB) | `5569e79a347e65c04e73413da1ebdbb306c7d6d5dfe46e4d097d52f5cda74b75` |

Keep your `config.yaml` backed up too, but never commit it: it holds camera and MQTT
credentials. The `.gitignore` in this repo excludes `config.yaml`, keys, certificates and
all large binaries.

## Optional: trusted HTTPS on port 8971

Frigate serves 8971 with whatever is in `/etc/letsencrypt/live/frigate/{fullchain.pem,privkey.pem}`
inside the container and reloads nginx when the fingerprint changes. Create a local CA
and certificate with [mkcert](https://github.com/FiloSottile/mkcert) on your PC:
```
mkcert -install
mkcert -cert-file fullchain.pem -key-file privkey.pem frigate.local frigate <jetson-ip>
```
Upload both files to e.g. `~/frigate18/tls/` on the Jetson (`chmod 600 privkey.pem`) and
add `-v /home/YOUR_USER/frigate18/tls:/etc/letsencrypt/live/frigate:ro` to the `docker run`.
Install the mkcert root CA (`rootCA.pem`/`.crt`, from `mkcert -CAROOT`) on each client
device and enable full trust (on iOS: Settings → General → About → Certificate Trust
Settings). Never publish `rootCA-key.pem` or `privkey.pem`.

## Optional: upgrading JetPack 7.2.0 → 7.2.1 in place

Record state first (`apt-cache policy nvidia-jetpack`, `apt-mark showhold`,
`docker images`, `docker ps -a`). If you still hold `nvidia-container-toolkit` at 1.18.2,
`sudo apt install nvidia-jetpack` fails because `nvidia-container` depends on exactly
toolkit 1.19.1-1: unhold it, install 1.19.1-1, test
`docker run --rm --runtime nvidia <image> python3 --version`, then
`sudo apt install nvidia-jetpack` and check `docker --version`. **Warning:** a community
report says this upgrade can remove Docker CE on some units; if so, run
`sudo apt-get install -y docker-ce docker-ce-cli containerd.io` and
`sudo systemctl disable nv-install-docker.service`. It did not happen on the author's unit.

## Redistribution

The ffmpeg bundle was configured with `--enable-nonfree --enable-gpl`, which FFmpeg
documents as not redistributable. The base-image tarball contains NVIDIA CUDA, TensorRT,
cuDNN and L4T packages whose licences may also restrict redistribution. This is not legal
advice; check the terms before publishing or downloading binaries from anyone. That is why
neither is published here: build the base image yourself (straga's `make base`) and obtain
or build your own ffmpeg (Part 1 step 4). Only the onnxruntime wheel, which is MIT
licensed and contains no NVIDIA libraries, is shared.

---

## Appendix A — Frigate 0.16.2 (optional)

A simpler, older build of the same idea. Use it if you want 0.16.2 itself, or as one way
to obtain the ffmpeg bundle for Part 1 (Option A in step 4). It needs the same base image
and wheel from "Before you start", and the `nvidia-container-toolkit` check from Part 1
step 1.

### A1. Clone and set up the build context

```bash
cd ~
git clone --branch v0.16.2 https://github.com/blakeblackshear/frigate.git frigate-jp7
cd frigate-jp7
cp /path/to/onnxruntime_gpu-1.23.0-cp312-cp312-linux_aarch64.whl docker/main/
cp docker/main/Dockerfile docker/main/Dockerfile.jp7
```

### A2. Patch the build files

Apply these edits (each is explained in Part 1 step 3):
- **Python 3.11 → 3.12** in the `wheels` stage of `Dockerfile.jp7`, and
  `sed -i 's/python3\.11/python3.12/g' docker/main/install_deps.sh` (3a).
- **Inject your wheel**: add `RUN sed -i "/onnxruntime/d" /requirements-wheels.txt` before
  the `pip3 wheel` line and `COPY docker/main/onnxruntime_gpu-…whl /wheels/` after it (3b).
  In 0.16.2 `tflite_runtime` and `hailort` are not an issue.
- **Strip `nvidia-pyindex`** from `requirements.txt` (it fails to build under Python 3.12,
  and you don't need NVIDIA's private pip index because CUDA/TensorRT come from apt):
  ```dockerfile
  COPY docker/main/requirements.txt /requirements.txt
  RUN sed -i "/nvidia-pyindex/d" /requirements.txt
  RUN pip3 install -r /requirements.txt
  ```
- **`yq` retry flags** in `install_deps.sh` (3c).
- **deb822 apt sources** in `build_nginx.sh` and `build_sqlite_vec.sh` (3d).

### A3. Makefile, build and run

Edit the `local:` target as in Part 1 step 5 but with `--tag frigate:latest`, then
`sudo make local`. Run it like Part 1 step 7, with these differences: container name
`frigate`, image `frigate:latest`, and data in `~/frigate-jp7/config` and
`~/frigate-jp7/storage`. Verify TensorRT/CUDA:
```bash
docker run --rm --runtime nvidia --entrypoint python3 frigate:latest \
  -c "import onnxruntime as ort; print(ort.get_available_providers())"
# expect: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```
The config essentials are the same as Part 1 step 8, except that 0.16.2 uses `config.yml`.

---

## Quick troubleshooting reference

| Symptom | Cause | Fix section |
|---|---|---|
| `E: Unable to locate package python3.12` | Transient apt index issue under concurrent build load | Just retry the build |
| `curl: (23) Failure writing output to destination` (yq) | Same — resource contention during concurrent build stages | Part 1 §3c; retry the build |
| `E: The repository '...bookworm-backports...' is not signed` | Debian repo added onto Ubuntu base | Part 1 §3e (or §3d for the apt-source scripts) |
| `TypeError: Unable to load from type 'NoneType'` (onnx detector) | `model:` nested under `detectors.onnx` instead of top-level | Move `model:` back to top level of config |
| `... is not a supported wheel on this platform` | A cp311-only wheel (onnxruntime, tflite_runtime, hailort) | Part 1 §3b |
| `ModuleNotFoundError: No module named 'ai_edge_litert'` crashing startup | Unconditional bird.py import | Part 1 §3i |
| `AzureExecutionProvider` instead of Tensorrt/CUDA | Wrong/stock onnxruntime won the install | Part 1 §3h |
| `ImportError: ... cannot import name 'LoraAdapter'` | Mixed-version onnxruntime install | Part 1 §3h (full wipe before install) |
| Build succeeds but GPU inference silently missing | `apt-get autoremove` stripped CUDA runtime libs | Part 1 §3f |
| Vite/npm "heap out of memory" in `web-build` | Node's default heap too small on 8GB | Part 1 §3g |
| `Unknown decoder 'h264_nvv4l2dec'` or `'h264_nvmpi'` | ffmpeg build doesn't match hwaccel_args / UI preset assumption | Part 1 §8 (manual arguments) |
| Container panics with `cudacompat.trim: slice bounds out of range` | Old `nvidia-container-toolkit` 1.19.x bug on JetPack 7.2.0 | Part 1 §1 |
| UI shows the wrong Frigate version | An old Makefile's `VERSION` line was copied over | Part 1 §5 |
| Object masks seem to "hide" everything | Masks are **exclusion-only** in Frigate, not inclusion | Delete/redraw the mask |

---

## Credits

Base image and onnxruntime wheel: **straga/jetson-jp7-onnxruntime**.
Frigate: **blakeblackshear/frigate**.
Community Jetson builds and the self-built ffmpeg work: **grossimd**
(`ghcr.io/grossimd/frigate:0.18.2-tensorrt-jp7`).
