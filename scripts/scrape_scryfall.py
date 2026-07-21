"""
Mirror of notebooks/scrape_scryfall.ipynb.

This script keeps the same high-level workflow as the notebook while providing
an executable entrypoint for reproducible runs outside Jupyter.
"""

# packages

## project directory
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

## from project directory
from src.config import TOTAL_CARDS
from src.config import RATE_LIMIT_SECONDS, MAX_LOAD_TIME
from src.config import SAVE_TAGS_EVERY, GET_IMAGES, OUTPUT_PATH, IMAGE_FOLDER, IMAGE_TYPE, SCRAPE_MODE
from src.config import IMAGE_DOWNLOAD_DELAY_SECONDS
from src.data_gathering.scryfall import Scryfall
from src.data_gathering.scryfall_scraper import ScryfallScraper


def _repo_path(config_path:str):
    """
    Convert notebook-relative config paths such as ../data/... into
    repo-root absolute paths for script execution.
    """
    return str((PROJECT_ROOT / config_path.replace('../', '')).resolve())


def _build_parser():
    parser = argparse.ArgumentParser(description = 'Scrape Scryfall tags and card images.')
    parser.add_argument(
        '--mode',
        choices = ['tags', 'dataset-images', 'dataset-image-manifest'],
        default = SCRAPE_MODE,
        help = 'Run the original tag scrape, only download dataset card images, or build an image-text manifest.'
    )
    parser.add_argument(
        '--card-data-path',
        default = _repo_path('../data/oracle-cards.json'),
        help = 'Path to the Scryfall oracle cards bulk JSON.'
    )
    parser.add_argument(
        '--tag-output-path',
        default = _repo_path(OUTPUT_PATH),
        help = 'Where the scraped tags JSON is stored or should be written.'
    )
    parser.add_argument(
        '--image-folder',
        default = _repo_path(IMAGE_FOLDER),
        help = 'Output folder for downloaded dataset card images.'
    )
    parser.add_argument(
        '--image-type',
        default = IMAGE_TYPE,
        help = 'Scryfall image type to download for dataset-images mode.'
    )
    parser.add_argument(
        '--manifest-path',
        default = _repo_path('../data/card_image_text_manifest.jsonl'),
        help = 'Output path for the image-text manifest.'
    )
    parser.add_argument(
        '--train-path',
        default = _repo_path('../data/scryfall_multi_label_classification_train.json'),
        help = 'Path to the multi-label classification train split.'
    )
    parser.add_argument(
        '--val-path',
        default = _repo_path('../data/scryfall_multi_label_classification_val.json'),
        help = 'Path to the multi-label classification validation split.'
    )
    parser.add_argument(
        '--test-path',
        default = _repo_path('../data/scryfall_multi_label_classification_test.json'),
        help = 'Path to the multi-label classification test split.'
    )
    return parser


def main():
    args = _build_parser().parse_args()

    print(f'Total Cards = {TOTAL_CARDS}')

    # read scryfall data
    sf = Scryfall()
    sf.read_data(
        filepath = args.card_data_path
    )

    # instantiate the scraper
    scraper = ScryfallScraper()

    if args.mode == 'tags':
        # scrape tagger.scryfall.com for each card's tags
        scraper.scrape_all_cards(
            data = sf.data,
            total_cards = TOTAL_CARDS,
            rate_limit_seconds = RATE_LIMIT_SECONDS,
            max_load_time = MAX_LOAD_TIME,
            get_images = GET_IMAGES,
            save_tags_every = SAVE_TAGS_EVERY,
            output_path = args.tag_output_path,
            image_folder = _repo_path(IMAGE_FOLDER)
        )
        print(f'Total Tags Scraped = {len(scraper.data)}')
        return

    if args.mode == 'dataset-images':
        scraper.scrape_card_images_for_dataset(
            card_data = sf.data,
            split_paths = [args.train_path, args.val_path, args.test_path],
            tag_path = args.tag_output_path,
            output_folder = args.image_folder,
            image_type = args.image_type,
            max_download_time = MAX_LOAD_TIME,
            request_delay_seconds = IMAGE_DOWNLOAD_DELAY_SECONDS
        )
        return

    scraper.build_dataset_image_manifest(
        card_data = sf.data,
        split_paths = {
            'train': args.train_path,
            'val': args.val_path,
            'test': args.test_path
        },
        tag_path = args.tag_output_path,
        output_path = args.manifest_path,
        image_folder = args.image_folder,
        image_type = args.image_type,
        project_root = str(PROJECT_ROOT)
    )


if __name__ == '__main__':
    main()
