# Environment setup

Mem-X provides a Conda environment definition in [environment.yml](environment.yml). Run the commands below from the repository root.

## Create the Conda environment

```bash
conda env create -f environment.yml
conda activate memx
```

The main package versions are:

| Package | Version |
| --- | --- |
| Python | 3.8.20 |
| PyTorch | 2.4.1+cu118 |
| Transformers | 4.46.3 |

GPU evaluation requires an NVIDIA driver compatible with the CUDA 11.8 PyTorch build.

## Install Habitat

Habitat is installed separately from the Conda environment definition. Follow the official installation instructions for:

- [Habitat-Sim v0.1.7](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7)
- [Habitat-Lab v0.1.7](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7)

Install both into the active `memx` environment. They are imported as `habitat_sim` and `habitat`. The Habitat-Sim build also provides the Magnum Python bindings.

Check the installation with:

```bash
python -c "import torch, transformers; print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import habitat_sim, habitat, magnum; print('Habitat imports succeeded')"
```

## Prepare model code and local assets

Baseline repositories, checkpoints, auxiliary models, and scene meshes must be downloaded separately. See the [README setup instructions](README.md#setup) for model repositories and dataset preparation.

| Variable | Local path |
| --- | --- |
| `ETPNAV_ROOT` | ETPNav checkout, including its configs and checkpoint |
| `CMA_ROOT` | CMA checkout, including `CMA_PM_DA_Aug.pth` |
| `UNI_ROOT` or `UNILAVIRA_ROOT` | Uni-LaViRA Habitat code directory |
| `JANUS_ROOT` | JanusVLN checkout |
| `JANUS_ENV` | `streamvln` environment directory required by the Janus launchers |
| `SCENES_DIR` | Scene directory containing `hm3dsem/val/` |
| `MODELS_DIR` | Auxiliary model directory containing SAM3, DINOv3, MiniLM, and BERT |
| `VLM_ENV_FILE` | Local API configuration file |

The default baseline checkout locations are under `baselines/models/`. Set the corresponding path variables when needed by the selected launcher. JanusVLN_Extra weights must also be prepared separately.

ETPNav and Uni-LaViRA both provide a Python package named `vlnce_baselines`. Use them in separate processes to avoid import conflicts.

## Configure the API

Create a local `vlm.env` file with your API base URL, model name, and API key:

```bash
BASE_URL=https://your-endpoint.example/v1
MODEL_NAME=your-model-name
API_KEY=your-api-key
```

Set the file path:

```bash
export VLM_ENV_FILE="$PWD/vlm.env"
```

The [README API section](README.md#5-api) lists the URL and model used in our experiments. Keep real API keys out of version control; see [SECURITY.md](SECURITY.md).

For evaluation commands and GPU settings, see [Running](README.md#running).
