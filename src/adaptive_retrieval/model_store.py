"""Explicit downloads; normal demo commands use cached model files only."""

from pathlib import Path

from .paths import PROCESSED_DIR


EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
GENERATION_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MODEL_CACHE = PROCESSED_DIR / "models"
MODEL_REVISIONS = {
    EMBEDDING_MODEL: "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    GENERATION_MODEL: "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b",
}


def model_snapshot(model_name: str, *, download: bool = False, revision: str | None = None) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return Path(snapshot_download(
            repo_id=model_name,
            revision=revision or MODEL_REVISIONS.get(model_name),
            cache_dir=str(MODEL_CACHE),
            local_files_only=not download,
            # Never fetch alternative formats, training artifacts, or executable code.
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "LICENSE"],
        ))
    except LocalEntryNotFoundError as exc:
        raise FileNotFoundError(
            f"Model {model_name} is not downloaded. Run 'python src/runner.py download-models'."
        ) from exc


def download_models() -> None:
    for name in (EMBEDDING_MODEL, GENERATION_MODEL):
        print(f"Downloading {name}", flush=True)
        path = model_snapshot(name, download=True)
        print(f"Ready: {name} (revision {path.name})", flush=True)
