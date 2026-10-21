<div align="center">

# Mem-X

**A Memory-Dependent Cross-Episode Benchmark for Vision-and-Language Navigation**

<p align="center">
  <img alt="Paper: coming soon" src="https://img.shields.io/badge/Paper-Coming%20Soon-b31b1b">
  <a href="https://github.com/Zhikang-Shi/Mem-X"><img alt="Code" src="https://img.shields.io/badge/Code-GitHub-181717?logo=github"></a>
  <a href="https://huggingface.co/datasets/KangThomas/Mem-X/tree/main"><img alt="Dataset" src="https://img.shields.io/badge/Dataset-Hugging%20Face-FFD21E"></a>
  <a href="assets/Mem-X_teaser.pdf"><img alt="Teaser" src="https://img.shields.io/badge/Teaser-PDF-red"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue"></a>
</p>

*The paper is under review. A public paper link will be added when available.*

<img src="assets/teaser.png" alt="Overview of the Mem-X benchmark" width="100%">

</div>

Mem-X evaluates whether an embodied agent can use memory across episodes to recall a past target, localize against an earlier episode, and follow a historical route.

The benchmark evaluates four navigation policies on the same episodes: [ETPNav](https://github.com/MarSaKi/ETPNav), [CMA](https://github.com/jacobkrantz/VLN-CE), [Uni-LaViRA](https://github.com/NJU-R-L-Group-Embodied-Lab/uni-lavira-code), and [JanusVLN](https://github.com/MIV-XJTU/JanusVLN). Each policy can run with MemBridge or through its native navigation pipeline.

**Contents:** [Setup](#setup) · [Running](#running) · [Repository layout](#repository-layout) · [Contributing](#contributing) · [License](#license)

## Setup

Run the commands below from the repository root. Model checkouts, datasets, scene meshes, and model weights are stored locally and are not included in this repository.

### 1. Conda environment

Create the Conda environment using the pinned dependencies in [environment.yml](environment.yml):

```bash
conda env create -f environment.yml
conda activate memx
```

The environment uses Python **3.8.20**, PyTorch **2.4.1+cu118**, and Transformers **4.46.3**. Habitat is installed separately:

- [Habitat-Sim v0.1.7](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7)
- [Habitat-Lab v0.1.7](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7)

Follow the official installation instructions linked from [aihabitat.org](https://aihabitat.org/). The packages are imported as `habitat_sim` and `habitat`. The Habitat-Sim build also provides the Magnum Python bindings.

### 2. Baseline models

Clone the baseline repositories into `baselines/models/`:

```text
baselines/models/
├── ETPNav/
├── CMA/
├── JanusVLN/
└── Uni-LaViRA/
```

```bash
mkdir -p baselines/models
git clone https://github.com/MarSaKi/ETPNav.git baselines/models/ETPNav
git clone https://github.com/jacobkrantz/VLN-CE.git baselines/models/CMA
git clone https://github.com/MIV-XJTU/JanusVLN.git baselines/models/JanusVLN
git clone https://github.com/NJU-R-L-Group-Embodied-Lab/uni-lavira-code.git baselines/models/uni-lavira-code
mv baselines/models/uni-lavira-code/sim-code/habitat baselines/models/Uni-LaViRA
```

For Uni-LaViRA, the launcher uses the `sim-code/habitat` directory, moved to `baselines/models/Uni-LaViRA` above. Model paths can be configured with:

| Policy | Path variable |
| --- | --- |
| ETPNav | `ETPNAV_ROOT` |
| CMA | `CMA_ROOT` |
| Uni-LaViRA | `UNI_ROOT` or `UNILAVIRA_ROOT` |
| JanusVLN | `JANUS_ROOT` |

ETPNav and Uni-LaViRA both provide a package named `vlnce_baselines`; import only one of them per process.

Download the required checkpoints separately, including the ETPNav checkpoint and JanusVLN_Extra weights. The Janus launchers also require a `streamvln` environment, provided through `JANUS_ENV`. Set `MODELS_DIR` to the directory containing the shared auxiliary models (SAM3, DINOv3, MiniLM, and BERT).

<a id="dataset"></a>

### 3. Benchmark dataset

Download the episodes from [KangThomas/Mem-X](https://huggingface.co/datasets/KangThomas/Mem-X/tree/main) into `datasets/` using the Hugging Face CLI:

```bash
hf download KangThomas/Mem-X --repo-type dataset --local-dir datasets
```

Each level contains episodes, ground truth, and an evaluation protocol:

```text
datasets/
├── level1/
│   ├── level1.json
│   ├── level1_gt.json
│   └── protocol_level1.json
├── level2/
│   ├── level2.json
│   ├── level2_gt.json
│   └── protocol_level2.json
└── level3/
    ├── level3.json
    ├── level3_gt.json
    └── protocol_level3.json
```

| File | Contents |
| --- | --- |
| `levelN.json` | Episode records |
| `levelN_gt.json` | Ground truth keyed by `episode_id`, with `locations`, `actions`, and `forward_steps` |
| `protocol_levelN.json` | Evaluation protocol for the level |
| `EPISODE_SCHEMA.json` | Shared episode field definitions |

The episode counts below follow the role order in the second column:

| Level | Episode roles | Episodes per role |
| --- | --- | --- |
| 1 | ep0–ep4 | 315 / 315 / 221 / 34 / 1 |
| 2 | ep0–ep2 | 238 / 238 / 238 |
| 3 | ep0–ep2 | 157 / 157 / 157 |

Habitat launchers read per-role splits from `datasets/levelN/habitat/`. Generate or rebuild them with [scripts/materialize_habitat_splits.py](scripts/materialize_habitat_splits.py):

```bash
for level in 1 2 3; do
  python scripts/materialize_habitat_splits.py --root "$PWD" --level "$level"
done
```

Each episode has a unique `episode_id`, such as `level1_00800_002_ep1` or `level3_000_ep0`. The `info.task_type` field is `basic` or `memory`. Basic tasks build memory and are shared by the native and +MemBridge pipelines. An episode succeeds when its final Habitat geodesic distance to the goal is **at most 3 m**; a STOP action is not required.

### 4. Scene meshes

Mem-X uses the **HM3D v0.2 validation split** with semantic annotations. Download the Habitat archive, semantic annotations, and semantic configs from the **val** rows in the official [Matterport HM3D repository](https://github.com/matterport/habitat-matterport-3dresearch). Access is subject to its academic and non-commercial terms; accept those terms before downloading.

Extract the files so that `SCENES_DIR` contains the `hm3dsem/val/` tree. For example:

```text
scenes/hm3dsem/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb
```

```bash
export SCENES_DIR="$PWD/scenes"
```

Every episode's `scene_id` is relative to `SCENES_DIR`. The local `scenes/` directory is gitignored.

### 5. API

Configure the API base URL, model name, and API key in a local `vlm.env` file:

```bash

BASE_URL=https://your-endpoint.example/v1
MODEL_NAME=your-model-name
API_KEY=your-api-key
```

Our experiments used the following API base URL and model:

```bash
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen3.5-27b
```

```bash
export VLM_ENV_FILE="$PWD/vlm.env"
```

| Variable | Purpose |
| --- | --- |
| `BASE_URL` | API base URL |
| `MODEL_NAME` | Model identifier |
| `API_KEY` | API key |



## Running

After completing setup, configure the paths and GPU workers:

```bash
export SCENES_DIR="$PWD/scenes"
export VLM_ENV_FILE="$PWD/vlm.env"
export MODELS_DIR=/path/to/models
export ETPNAV_ROOT="$PWD/baselines/models/ETPNav"
export GPU_IDS="0"
```

Choose a level, pipeline, and policy:

```bash
# Level 1: ETPNav + MemBridge
PIPELINE=membridge MODEL=etpnav bash scripts/level1/level1.sh

# Level 2: native JanusVLN (requires JANUS_ENV)
JANUS_ENV=/path/to/streamvln GPU_PAIRS="0,1" \
  PIPELINE=native MODEL=janus bash scripts/level2/level2.sh

# Level 3: CMA + MemBridge; the policy can also be a positional argument
PIPELINE=membridge bash scripts/level3/level3.sh cma
```

| Setting | Values / behavior |
| --- | --- |
| `PIPELINE` | `membridge` (default; aliases: `+membridge`, `plus`) or `native` (alias: `original`) |
| `MODEL` | `etpnav` (default), `cma`, `unilavira`, or `janus`; the first positional argument overrides this variable |
| `GPU_IDS` | Space-separated GPU IDs for ETPNav, CMA, and Uni-LaViRA, e.g. `"0 1"` |
| `GPU_PAIRS` | Space-separated GPU pairs for JanusVLN, e.g. `"0,1 2,3"` |
| `DRY_RUN` | `0` (default) runs evaluation; `1` prints the launch plan without running evaluation |

To inspect a launch plan after setup:

```bash
DRY_RUN=1 PIPELINE=membridge MODEL=etpnav bash scripts/level1/level1.sh
```

Results are written under `experiments_results/<policy>/levelN/` by default. The launchers print the run directory at startup; set `RUN_ROOT` to override it.

## Repository layout

```text
Mem-X/
├── baselines/
│   ├── run.py          # Shared Python entry point
│   ├── adapt/          # Policy adapters
│   ├── runners/        # Level runners and stage logic
│   └── models/         # Local baseline checkouts
├── membridge/          # Scene graphs, maps, retrieval, navigation, VLM, exploration, localization
├── scripts/
│   ├── level1/         # Level 1 launchers
│   ├── level2/         # Level 2 launchers
│   └── level3/         # Level 3 launchers
├── datasets/           # Local episodes, ground truth, protocols, and Habitat splits
├── scenes/             # Local HM3D scene meshes
├── experiments_results/ # Evaluation outputs
├── tests/              # Test suite
└── environment.yml     # Pinned Conda environment
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and [SECURITY.md](SECURITY.md) to report security issues. Run the test suite with:

```bash
python -m unittest discover -s tests -v
```

## License

Mem-X is released under the [Apache License 2.0](LICENSE).

Upstream code, including ETPNav, CMA, Uni-LaViRA, JanusVLN, and CLIP, remains subject to the licenses provided in its respective directories.
