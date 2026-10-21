# Security policy

## API credentials

Keep real API keys in a local, untracked configuration file such as `vlm.env`, and provide its path through `VLM_ENV_FILE`. The file may be stored locally in the checkout; `.env` and `*.env` files are ignored by Git by default.

Use `BASE_URL`, `MODEL_NAME`, and `API_KEY` for the API configuration. Examples in the documentation use placeholder keys. Do not include real keys in commits, issues, logs, screenshots, or shared experiment outputs. If a key is exposed, revoke it with the provider and replace it.

API calls may send navigation instructions and scene images to the configured service. Use an endpoint suitable for the data in your experiments.

## Reporting a vulnerability

Report security vulnerabilities in Mem-X privately to the repository maintainers. If available, use **Report a vulnerability** on the repository's [Security page](https://github.com/Zhikang-Shi/Mem-X/security).

Include:

- The affected commit or version and relevant files.
- A description of the issue and its potential impact.
- Reproduction steps or a minimal example, with credentials and private data removed.

If private reporting is unavailable, open an issue requesting a private contact channel without disclosing vulnerability details. Keep credentials and details of unresolved vulnerabilities out of public issues and pull requests.
