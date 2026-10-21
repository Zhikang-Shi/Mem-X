# Contributing

Contributions to Mem-X can include bug fixes, documentation improvements, benchmark tooling, MemBridge modules, and adapters for navigation policies.

## Getting started

Follow [ENVIRONMENT.md](ENVIRONMENT.md) to create the Conda environment and install Habitat. See [README.md](README.md) for datasets, model preparation, API configuration, and evaluation commands.

For bug reports, include the affected commit, the level and policy, reproduction steps, and the expected and observed behavior. Remove API keys and private data from logs before sharing them. Report security vulnerabilities as described in [SECURITY.md](SECURITY.md).

## Making changes

Keep each pull request focused on a specific change. Describe the problem, the resulting behavior, and the checks you ran. Update the documentation when setup steps, configuration variables, or usage change.

Use the existing project structure:

| Location | Purpose |
| --- | --- |
| `baselines/adapt/` | Policy adapters |
| `baselines/runners/` | Evaluation runners and level logic |
| `membridge/` | Memory, retrieval, localization, and navigation modules |
| `scripts/level1/`, `scripts/level2/`, `scripts/level3/` | Public evaluation launchers |

Preserve existing launcher arguments and configuration variables where possible. Explain any interface changes in the pull request. For changes to episode splits, success criteria, or evaluation protocols, describe how they affect comparisons with existing results.

Upstream checkouts under `baselines/models/` retain their own layout and licenses. Keep local datasets, scene meshes, model weights, experiment outputs, and credential files out of commits.

## Checks

Run the unit tests and check launcher syntax from the repository root:

```bash
python -m unittest discover -s tests -v

for level in 1 2 3; do
  bash -n "scripts/level${level}/level${level}.sh" \
    "scripts/level${level}/level${level}_membridge.sh" \
    "scripts/level${level}/level${level}_native.sh"
done
```

After preparing the required local files, use `DRY_RUN=1` to inspect a launch plan without running evaluation. This checks launch configuration; it does not validate navigation results.

For changes to evaluation behavior, also report the level, policy, configuration, and outcome of any evaluation you ran. State any checks you could not run.

## License

Contributions to Mem-X are made under the [Apache License 2.0](LICENSE). Third-party code remains subject to its original license.
