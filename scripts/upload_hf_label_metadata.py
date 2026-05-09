"""
Ad hoc helper to upload label metadata needed for standalone inference from the
Hugging Face adapter repo.
"""

# packages

## standard library
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

## Hugging Face
from huggingface_hub import HfApi
from huggingface_hub import get_full_repo_name

## project modules
from src.config import TASK, OUTPUT_DIR
from src.data_gathering.scryfall_dataset import ScryfallDataset


def _repo_path(config_path:str):
    """
    Convert notebook-relative config paths such as ../data/... into
    repo-root absolute paths for script execution.
    """
    return str((PROJECT_ROOT / config_path.replace('../', '')).resolve())


def _build_metadata():
    """
    Reconstruct label metadata from the local dataset splits.
    """
    sf = ScryfallDataset(task = TASK)
    sf.load_hf_dataset(
        train_path = _repo_path(f'../data/scryfall_{TASK}_train.json'),
        val_path = _repo_path(f'../data/scryfall_{TASK}_val.json'),
        test_path = _repo_path(f'../data/scryfall_{TASK}_test.json'),
        verbose = False
    )

    metadata = {
        'task': TASK,
        'n_labels': len(sf.unique_tags),
        'unique_tags': sf.unique_tags,
        'id2label': {str(k): v for k, v in sf.id2label.items()},
        'label2id': sf.label2id
    }
    return metadata


def main():
    if TASK != 'multi_label_classification':
        raise ValueError(
            f'This script only supports TASK="multi_label_classification". Received {TASK!r}.'
        )

    metadata = _build_metadata()

    out_dir = PROJECT_ROOT / 'models' / 'scryfall_auto_tagger'
    out_dir.mkdir(parents = True, exist_ok = True)
    metadata_path = out_dir / 'label_metadata.json'

    with open(metadata_path, 'w', encoding = 'utf-8') as f:
        json.dump(metadata, f, indent = 4)

    api = HfApi()
    repo_id = get_full_repo_name(OUTPUT_DIR)
    api.create_repo(
        repo_id = repo_id,
        repo_type = 'model',
        exist_ok = True
    )
    api.upload_file(
        path_or_fileobj = str(metadata_path),
        path_in_repo = 'label_metadata.json',
        repo_id = repo_id,
        repo_type = 'model',
        commit_message = 'Add label metadata for standalone inference.'
    )

    print(f'Wrote local metadata to {metadata_path}')
    print(f'Uploaded label_metadata.json to {repo_id}')


if __name__ == '__main__':
    main()
