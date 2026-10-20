# Environment

The runtime image is `mem-x`. Docker tags are lowercase, so the name is `mem-x`, not `Mem-x`. Scripts use `MEMX_IMAGE`, which defaults to `mem-x:latest`.

`mem-x:latest` is the slim image, about 27 GB. The previous 35 GB image is kept as `mem-x:full` and `lavira-oss:dev`. The slim image removes the Habitat-sim build tree, Nsight Compute, the conda package cache, and unused GroundingDINO and segment-anything sources. It can be uploaded. A registry must accept an image of this size. Docker Hub, GitHub Container Registry, and a private registry all work after `docker login`.

The image has one conda env, `/opt/conda/envs/lavira`. It does not contain Janus weights or a `streamvln` environment. lavira can load JanusVLN_Extra: `einops`, `qwen-vl-utils`, and `accelerate` are installed there, and `baselines/adapt/janus_policy.py` registers `qwen2_5_vl` for transformers 4.46. A one-step load on that interpreter returned `TURN_LEFT`. There is no `flash-attn` wheel in lavira, so that path uses SDPA. The level scripts still start Janus with `/janus/envs/streamvln/bin/python` and still require `JANUS_ENV`.

## What is inside the image

Two Pythons share this image. They are not one process.

| Python | Used by |
|---|---|
| `/opt/conda/envs/lavira/bin/python` | ETPNav, CMA, Uni-LaViRA, scene graph, Habitat, and a Janus load |
| `/janus/envs/streamvln/bin/python` | JanusVLN in the level scripts. This interpreter is mounted, not stored in the image |

Uni-LaViRA and ETPNav both ship a package named `vlnce_baselines`. Do not import both in one process.

## What stays outside the image

These are mounted at run time. They are not in `mem-x`.

| Variable | Contents |
|---|---|
| `ETPNAV_ROOT` | ETPNav code, Habitat configs, and the ETPNav checkpoint |
| `JANUS_ROOT` | JanusVLN code. Weights stay outside the image. `JANUS_ENV` is the optional streamvln mount |
| `CMA_ROOT` | CMA code and `CMA_PM_DA_Aug.pth` |
| `UNI_ROOT` or `UNILAVIRA_ROOT` | Uni-LaViRA habitat directory |
| `SCENES_DIR` | Parent of `hm3dsem/val/...` |
| `MODELS_DIR` | SAM3, DINOv3, MiniLM, BERT |
| `VLM_ENV_FILE` | `VA_BASE_URL`, `VA_MODEL_NAME`, `VA_API_KEY` |

## Pull and run

```bash
docker pull <registry>/mem-x:latest
docker tag <registry>/mem-x:latest mem-x:latest

export MEMX_IMAGE=mem-x:latest
export ETPNAV_ROOT=/path/to/ETPNav
export SCENES_DIR=/path/to/scenes
export VLM_ENV_FILE=/path/to/.env
export CMA_ROOT=/path/to/CMA

bash scripts/level1/level1.sh
```

`DRY_RUN` defaults to `0`, so this starts containers. Set `DRY_RUN=1` to print the plan only.

## Upload

From the machine that already has the image:

```bash
docker tag mem-x:latest <registry>/mem-x:latest
docker push <registry>/mem-x:latest
```

`<registry>` is the registry host and repository, for example `ghcr.io/<user>/mem-x` or `docker.io/<user>/mem-x`. The push is large. Do not commit weights, scenes, or `.env` into the repository.
