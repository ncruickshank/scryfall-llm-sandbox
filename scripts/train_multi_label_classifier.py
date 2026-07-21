"""
Mirror of notebooks/scryfall_tag_multi-lab_class.ipynb for the active
multi-label classification path.

NOTE
- DATASET_SOURCE ['build_from_scryfall', 'load_from_scryfall'] are intended
    to be used to generate the fine tuned model based on *structured scryfall
    text*. Wherease DATASET_SOURCE 'load_from_ocr' is intended to be used t
    to generated a fine tuned model based on *ocr-sourced text*
"""

# packages

## project directory
from pathlib import Path
import sys

## general
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

## data gathering
from src.config import DATASET_SOURCE, TASK, TAG_SIZE, DATASET_SIZE_N, TEST_SIZE_N
from src.config import MAX_INPUT_LENGTH

## modeling
from src.config import MODEL_NAME, TRAIN_MODEL
from src.config import BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS

## save model
from src.config import OUTPUT_DIR

## project modules
from src.data_gathering.scryfall_dataset import ScryfallDataset
from src.fine_tuning.fine_tune_multi_lab import FineTuneLLM
from src.modeling.auto_tagger_multi_lab import ScryfallTaggerFromPretrained


def _repo_path(config_path:str):
    """
    Convert notebook-relative config paths such as ../data/... into
    repo-root absolute paths for script execution.
    """
    return str((PROJECT_ROOT / config_path.replace('../', '')).resolve())


def _save_training_log(tagger_ft):
    """
    The trainer stores log history as a list of dicts.
    Convert it to a DataFrame before writing to disk.
    """
    out_path = PROJECT_ROOT / 'reports' / f'fine_tune_{MODEL_NAME.replace("/", "__")}_training_log.csv'
    pd.DataFrame(tagger_ft.training_log_history).to_csv(out_path, index = False)
    print(f'Saved training log to {out_path}')


def main():
    if TASK != 'multi_label_classification':
        raise ValueError(
            f'This script only supports TASK="multi_label_classification". Received {TASK!r}.'
        )

    # get data
    sf = ScryfallDataset(task = TASK)

    
    if DATASET_SOURCE == 'build_from_scryfall':
        # build dataset as needed
        sf.build_dataset(
            card_path = _repo_path('../data/oracle-cards.json'),
            tag_path = _repo_path('../reports/scryfall_tags.json'),
            train_size_pct = 0.8,
            truncate_dataset = DATASET_SIZE_N,
            test_size_n = TEST_SIZE_N,
            top_n_tags = TAG_SIZE
        )

    elif DATASET_SOURCE == 'load_from_scryfall':
        # assumes we have previously used DATASET_SOURCE == 'build_from_scryfall'
        sf.load_hf_dataset(
            train_path = _repo_path(f'../data/scryfall_{TASK}_train.json'),
            val_path = _repo_path(f'../data/scryfall_{TASK}_val.json'),
            test_path = _repo_path(f'../data/scryfall_{TASK}_test.json')
        )

    if TRAIN_MODEL:
        tagger_ft = FineTuneLLM(
            model_name = MODEL_NAME,
            dataset = sf.dataset,
            n_labels = len(sf.unique_tags),
            label2id = sf.label2id,
            id2label = sf.id2label,
            class_weights = sf.class_weights
        )

        tagger_ft.prepare_data(
            max_input_length = MAX_INPUT_LENGTH
        )

        tagger_ft.train(
            batch_size = BATCH_SIZE,
            n_epochs = NUM_EPOCHS,
            learning_rate = LEARNING_RATE,
            weight_decay = WEIGHT_DECAY,
            output_dir = _repo_path('../models/scryfall_auto_tagger')
        )

        _save_training_log(tagger_ft)

    # load model and validate the saved adapter path
    tagger = ScryfallTaggerFromPretrained(
        base_model_name = MODEL_NAME,
        n_labels = len(sf.unique_tags),
        output_dir = OUTPUT_DIR,
        id2label = sf.id2label,
        label2id = sf.label2id
    )
    print(f'Loaded inference model from Hugging Face repo for {OUTPUT_DIR}.')
    print(f'Unique Tags = {len(sf.unique_tags)}')
    print(f'Test Records = {sf.dataset["test"].num_rows}')
    del tagger


if __name__ == '__main__':
    main()
