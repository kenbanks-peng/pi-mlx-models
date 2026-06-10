# pi-mlx-models

Local MLX model launcher for [Pi](https://pi.dev) on Apple Silicon.

<p align="center">
  <img src="./assets/preset-selector.png" alt="Preset selector" width="780" />
</p>

`pi-mlx-models` adds a local OpenAI-compatible provider to Pi, with a guided startup flow for MLX models.

## What you get

- Interactive cached-model selector via `/mlx-start`
- Local runtime bootstrap via `/mlx-init` (venv + `mlx-lm`)
- Startup progress with stage/status updates and warm-up timer
- Provider models only appear when the server is actually ready
- Stop/cleanup command via `/mlx-stop`

## Install

### From npm

```bash
pi install npm:pi-mlx-models
```

### From GitHub

```bash
pi install git:github.com/vmarinogg/pi-mlx-models
```

## Quick start

```bash
/mlx-init
/mlx-start
```

When `/mlx-start` is used with no args, a selector opens and you can pick an MLX model already present in the Hugging Face cache with arrow keys + Enter.

You can also start directly with a full model id:

```bash
/mlx-start <hf-model-id>
```

## Commands

- `/mlx-init` — initialize local MLX runtime
- `/mlx-start [hf-model-id]` — open cached-model selector (no args) or start with full model id
- `/mlx-stop` — stop server and clear active provider models

## Model selection

The selector is populated from:

```bash
hf cache ls --format json
```

Cached model repos that look MLX-compatible are shown (`mlx-community/*` or ids containing `mlx`). If no cached MLX models are found, start one directly by Hugging Face model id; MLX will download it into the package cache and it will appear in the selector next time.

> [!TIP]
> Want to try other models? Browse MLX Community collections on Hugging Face:  
> https://huggingface.co/mlx-community/collections

## Requirements

- macOS on Apple Silicon
- Python 3.10–3.13

## Configuration

Environment variables supported by the extension:

- `PI_MLX_MODELS_PORT` (default: `11434`)
- `PI_MLX_MODELS_HOST` (default: `127.0.0.1`)
- `PI_MLX_MODELS_BASE_URL` (default: `http://<host>:<port>/v1`)
- `PI_MLX_MODELS_DEFAULT_MODEL` (optional cached model id or generated cache key)
- `PI_MLX_MODELS_HF_CLI` (optional path to the `hf` CLI)
- `HF_HOME` (default: `~/.cache/huggingface`; Hugging Face stores repos under its `hub/` subdirectory)

## Local data paths

Runtime artifacts are stored under:

- `~/.pi/agent/pi-mlx-models/venv`
- Hugging Face model cache: `$HF_HOME/hub` or `~/.cache/huggingface/hub` when `HF_HOME` is unset

## Development

```bash
bun install
bun run build
```

Pi package manifest is in `package.json`:

- `pi.extensions: ["dist/index.js"]`

## Screenshots

### Startup progress

![Startup progress](./assets/start-progress.png)

### Command menu

![Commands](./assets/commands.png)

### Model selection

![Model selection](./assets/model-selection.png)
