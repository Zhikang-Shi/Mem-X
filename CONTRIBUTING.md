# Contributing

LifelongHomeNav is the three-level benchmark, the MemBridge code, and the adapters that call ETPNav, CMA, Uni-LaViRA, and JanusVLN.

## What to leave alone

`scripts/level1/level1.sh`, `scripts/level2/level2.sh`, and `scripts/level3/level3.sh` are the public entry points. They call `baselines/run.py`. Change a runner behind that entry rather than the shell flags those scripts already pass.

Upstream trees under `baselines/models/` keep their own layout and licenses.

## Checks

```bash
python -m unittest discover -s tests -v
bash -n scripts/level1/level1.sh scripts/level2/level2.sh scripts/level3/level3.sh
```

`DRY_RUN=1` prints a launch plan and does not start Docker.

## License

Contributions are accepted under the Apache License 2.0. See [LICENSE](LICENSE).
