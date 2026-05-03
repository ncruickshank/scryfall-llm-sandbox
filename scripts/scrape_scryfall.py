"""
Mirror of notebooks/scrape_scryfall.ipynb.

This script keeps the same high-level workflow as the notebook while providing
an executable entrypoint for reproducible runs outside Jupyter.
"""

# packages

## project directory
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

## from project directory
from src.config import TOTAL_CARDS
from src.config import RATE_LIMIT_SECONDS, MAX_LOAD_TIME
from src.config import SAVE_TAGS_EVERY, GET_IMAGES, OUTPUT_PATH, IMAGE_FOLDER
from src.data_gathering.scryfall import Scryfall
from src.data_gathering.scryfall_scraper import ScryfallScraper


def _repo_path(config_path:str):
    """
    Convert notebook-relative config paths such as ../data/... into
    repo-root absolute paths for script execution.
    """
    return str((PROJECT_ROOT / config_path.replace('../', '')).resolve())


def main():
    print(f'Total Cards = {TOTAL_CARDS}')

    # read scryfall data
    sf = Scryfall()
    sf.read_data(
        filepath = _repo_path('../data/oracle-cards.json')
    )

    # instantiate the scraper
    scraper = ScryfallScraper()

    # scrape tagger.scryfall.com for each card's tags
    scraper.scrape_all_cards(
        data = sf.data,
        total_cards = TOTAL_CARDS,
        rate_limit_seconds = RATE_LIMIT_SECONDS,
        max_load_time = MAX_LOAD_TIME,
        get_images = GET_IMAGES,
        save_tags_every = SAVE_TAGS_EVERY,
        output_path = _repo_path(OUTPUT_PATH),
        image_folder = _repo_path(IMAGE_FOLDER)
    )
    print(f'Total Tags Scraped = {len(scraper.data)}')


if __name__ == '__main__':
    main()
