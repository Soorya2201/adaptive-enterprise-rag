"""Repository locations independent of the current working directory."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FULL_DOCUMENTS_DIR = DATA_DIR / "all_documents"
SUBSET_DOCUMENTS_DIR = DATA_DIR / "subset_documents"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
