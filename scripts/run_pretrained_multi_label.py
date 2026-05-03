"""
Mirror of the pretrained inference section from
notebooks/scryfall_tag_multi-lab_class.ipynb.

By default, this script loads the withheld test split and samples cards for
prediction. It can also accept a single structured card text input or a JSON
file containing a batch of structured card texts.
"""

# packages

## standard library
import argparse
import json
import random
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

## data gathering
from src.config import TASK

## modeling
from src.config import MODEL_NAME, OUTPUT_DIR

## project modules
from src.data_gathering.scryfall_dataset import ScryfallDataset
from src.modeling.auto_tagger_multi_lab import ScryfallTaggerFromPretrained


def _repo_path(config_path:str):
    """
    Convert notebook-relative config paths such as ../data/... into
    repo-root absolute paths for script execution.
    """
    return str((PROJECT_ROOT / config_path.replace('../', '')).resolve())


def _str_to_bool(value):
    """
    Allow intuitive CLI booleans such as true/false, yes/no, 1/0.
    """
    if isinstance(value, bool):
        return value

    value = value.strip().lower()
    if value in {'true', 't', 'yes', 'y', '1'}:
        return True
    if value in {'false', 'f', 'no', 'n', '0'}:
        return False

    raise argparse.ArgumentTypeError(f'Cannot interpret {value!r} as boolean.')


def _load_dataset():
    """
    Load the Hugging Face dataset object and label mappings used by the
    pretrained multi-label classifier.
    """
    sf = ScryfallDataset(task = TASK)
    sf.load_hf_dataset(
        train_path = _repo_path(f'../data/scryfall_{TASK}_train.json'),
        val_path = _repo_path(f'../data/scryfall_{TASK}_val.json'),
        test_path = _repo_path(f'../data/scryfall_{TASK}_test.json')
    )
    return sf


def _build_tagger_from_dataset(sf):
    """
    Instantiate the pretrained tagger using the dataset-derived label mappings.
    """
    return ScryfallTaggerFromPretrained(
        base_model_name = MODEL_NAME,
        n_labels = len(sf.unique_tags),
        output_dir = OUTPUT_DIR,
        id2label = sf.id2label,
        label2id = sf.label2id
    )


def _build_tagger_from_repo():
    """
    Instantiate the pretrained tagger using label metadata stored in the
    Hugging Face repo.
    """
    return ScryfallTaggerFromPretrained(
        base_model_name = MODEL_NAME,
        output_dir = OUTPUT_DIR
    )


def _predict_one(tagger, card_text, threshold, top_k, cap_tags):
    return tagger.generate_tags(
        card_text = card_text,
        threshold = threshold,
        top_k = top_k,
        cap_tags = cap_tags
    )


def _run_test_set_mode(sf, tagger, sample_size, threshold, top_k, cap_tags, seed):
    """
    Sample cards from the withheld test split and compare predictions against
    ground-truth tags.
    """
    rng = random.Random(seed)
    test_records = list(sf.dataset['test'])
    sample_size = min(sample_size, len(test_records))
    sampled_cards = rng.sample(test_records, sample_size)

    print(f'Test Records Available = {len(test_records)}')
    print(f'Sampled Records = {sample_size}')
    print(f'Threshold = {threshold}')
    print(f'Top K = {top_k}')
    print(f'Cap Tags = {cap_tags}')
    print('')

    for card in sampled_cards:
        pred = _predict_one(
            tagger = tagger,
            card_text = card['document'],
            threshold = threshold,
            top_k = top_k,
            cap_tags = cap_tags
        )
        print(f'Card ID {card["id"]}')
        print(f'\tActual Tags = {sorted(card["tags"])}')
        print(f'\tPredicted Tags = {sorted(pred)}')


def _run_single_card_mode(tagger, card_text, threshold, top_k, cap_tags):
    """
    Predict tags for a single structured card text string supplied by the user.
    """
    pred = _predict_one(
        tagger = tagger,
        card_text = card_text,
        threshold = threshold,
        top_k = top_k,
        cap_tags = cap_tags
    )

    print('Input Card Text:')
    print(card_text)
    print('')
    print(f'Predicted Tags = {sorted(pred)}')


def _run_batch_mode(tagger, batch_path, threshold, top_k, cap_tags):
    """
    Predict tags for a batch JSON file containing either:
    - a list of strings, or
    - a list of dicts with a document field
    """
    with open(batch_path, 'r', encoding = 'utf-8') as f:
        records = json.load(f)

    assert isinstance(records, list), 'Batch file must contain a JSON list.'

    print(f'Batch Records = {len(records)}')
    print('')

    for i, record in enumerate(records, start = 1):
        if isinstance(record, str):
            card_text = record
            label = f'Record {i}'
        elif isinstance(record, dict) and ('document' in record):
            card_text = record['document']
            label = record.get('id', f'Record {i}')
        else:
            raise ValueError(
                'Each batch record must be either a string or a dict with a document field.'
            )

        pred = _predict_one(
            tagger = tagger,
            card_text = card_text,
            threshold = threshold,
            top_k = top_k,
            cap_tags = cap_tags
        )
        print(f'{label}')
        print(f'\tPredicted Tags = {sorted(pred)}')


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description = 'Run the pretrained Scryfall multi-label tagger.'
    )
    parser.add_argument(
        '--sample-size',
        type = int,
        default = 5,
        help = 'Number of withheld test cards to sample when using test-set mode.'
    )
    parser.add_argument(
        '--threshold',
        type = float,
        default = 0.7,
        help = 'Confidence threshold per predicted tag.'
    )
    parser.add_argument(
        '--top-k',
        type = int,
        default = 5,
        help = 'Fallback number of tags when no thresholded labels are found.'
    )
    parser.add_argument(
        '--cap-tags',
        type = _str_to_bool,
        default = False,
        help = 'If true, cap predictions to at most top-k tags.'
    )
    parser.add_argument(
        '--seed',
        type = int,
        default = 42,
        help = 'Random seed used when sampling the withheld test set.'
    )
    parser.add_argument(
        '--card-text',
        type = str,
        default = None,
        help = 'Structured card text to score directly.'
    )
    parser.add_argument(
        '--batch-json',
        type = str,
        default = None,
        help = 'Path to a JSON file containing a batch of structured card texts.'
    )
    return parser


def main():
    if TASK != 'multi_label_classification':
        raise ValueError(
            f'This script only supports TASK="multi_label_classification". Received {TASK!r}.'
        )

    args = _build_arg_parser().parse_args()

    if (args.card_text is not None) and (args.batch_json is not None):
        raise ValueError('Use either --card-text or --batch-json, not both.')

    if args.card_text is not None:
        tagger = _build_tagger_from_repo()
        _run_single_card_mode(
            tagger = tagger,
            card_text = args.card_text,
            threshold = args.threshold,
            top_k = args.top_k,
            cap_tags = args.cap_tags
        )
    elif args.batch_json is not None:
        tagger = _build_tagger_from_repo()
        _run_batch_mode(
            tagger = tagger,
            batch_path = args.batch_json,
            threshold = args.threshold,
            top_k = args.top_k,
            cap_tags = args.cap_tags
        )
    else:
        sf = _load_dataset()
        tagger = _build_tagger_from_dataset(sf)
        _run_test_set_mode(
            sf = sf,
            tagger = tagger,
            sample_size = args.sample_size,
            threshold = args.threshold,
            top_k = args.top_k,
            cap_tags = args.cap_tags,
            seed = args.seed
        )


if __name__ == '__main__':
    main()
