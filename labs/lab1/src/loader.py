from pathlib import Path
import pandas as pd
from huggingface_hub import snapshot_download

data_dir = Path(__file__).resolve().parent.parent / "data" / "rurebus" / "data"

def load_dataset(split="train"):
    return pd.read_json(data_dir / f"{split}.jsonl", lines=True)