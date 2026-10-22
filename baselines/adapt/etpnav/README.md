# ETPNav adaptation for Mem-X

`baselines/models/ETPNav` holds the upstream repository and model assets.
This directory holds the Mem-X configuration files and source patches needed
for benchmark sharding, pose7 trajectory export, and GraphMap persistence.
These interfaces are additions to ETPNav.

The patches were recovered from the locally validated ETPNav adaptation,
based on upstream commit `160e65fe5d23ad83d024d57d710e47be6892fc68`.
`manifest.json` records the hashes of each original and adapted file. Both
versions are accepted, so an existing adapted checkout is also usable. An
unrecognized version fails with the affected filename before navigation starts.

The Level 1–3 launchers call `../etpnav_runtime.py` automatically. It creates
`RUN_ROOT/.runtime/etpnav/<fingerprint>/`, applies the patches to copies, and
generates the configs there. It never patches the upstream checkout. Model
assets are linked from the checkout's `data/` directory; generated logs are
written into the runtime. Each runtime records its source in
`memx_adapter.json`. Preparation is locked and cached for concurrent workers.

Task configs read datasets from Mem-X's `datasets/levelN/habitat/` and scenes
from `SCENES_DIR`. The default semantic scene config is
`$SCENES_DIR/hm3dsem/hm3d_annotated_basis.scene_dataset_config.json`; set
`SCENE_DATASET_CONFIG` to override it.

To prepare a runtime manually, without importing Habitat or PyTorch:

```bash
python3 baselines/adapt/etpnav_runtime.py \
  --source "$ETPNAV_ROOT" --runtime-root /tmp/memx-etpnav-runtime
```

The command prints the runtime path. Direct `baselines/run.py --nav-model
etpnav` calls also prepare a runtime and rebase the benchmark config paths.
Custom config filenames are preserved.

The adapter also accepts public Mem-X string episode IDs, derives their
session IDs consistently for resume and export, and tokenizes instructions
with the checkout's local BERT vocabulary when tokens are absent.

Run the adapter regression tests with `python -m unittest discover -s tests`.
To include CPU integration tests against the real ETPNav dataset loader,
configs, and sharding, run the same command in `memx` with
`ETPNAV_TEST_SOURCE="$ETPNAV_ROOT"` set.

Required assets retain ETPNav's relative paths:

```text
data/logs/checkpoints/release_r2r/ckpt.iter12000.pth
data/wp_pred/check_cwp_bestdist_hfov90
data/ddppo-models/gibson-2plus-resnet50.pth
third_party/CLIP/
```

The official ETPNav and CLIP code retain their upstream licenses. The patches
also preserve the earlier adaptation's GraphMap continuation behavior; this
migration does not change the recovered navigation algorithm.
