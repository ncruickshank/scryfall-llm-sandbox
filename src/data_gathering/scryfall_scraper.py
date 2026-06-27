# packages

## general
import time
from tqdm import tqdm
import json
from pathlib import Path

## internet browsing
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import requests

## local directory navigation
import os

## project directory
from ..utils.retry import retry

SCRYFALL_DOWNLOAD_HEADERS = {
    'User-Agent': 'scryfall-llm-sandbox/0.1 (local dataset image downloader)',
    'Accept': 'image/*,*/*;q=0.8'
}

# class
class ScryfallScraper():
    """
    Description
    ----------
    This class contains the methods to iterate through the scryfall data, using Selenium to navigate
    to the Scryfall Tagger page, and extract the tags for each card. These tags represent the 
    elements of our card summaries.

    Inputs
    ----------
    None
    """
    def __init__(self):
        super().__init__()

        # objects to be filled in iteratively
        self.data = []
        self._data_index = {}

    # === Main Methods ===

    def scrape_all_cards(
        self, 
        data:list[dict],
        total_cards:int = None,
        rate_limit_seconds:float = 3.0,
        max_load_time:float = 15.0,
        get_images:bool = True,
        save_tags_every:int = 100,
        output_path:str = '../reports/scryfall_tags.json',
        image_folder:str = '../data/images',
        verbose:bool = True
    ):
        """
        Description
        ----------
        This method iterates through the entire card database and navigates to the Scryfall Tagger page
        to retrieve the relevant tags for each card.

        Inputs
        ----------
        data = A list of dictionaries containing the card data. Sourced from the Scryfall() class.
        total_cards = The total number of cards we want to return. If None, we instead retrieve
            all cards.
        rate_limit_seconds = The number of seconds to wait before proceeding
            through the page.
        max_load_time = Max load time between steps
        get_images = If true, also retrieves the card image crop
        save_tags_every = The frequency we want to save the outputs to ensure we don't lose
            info if we run into issues.
        output_path = Where to save.
        image_folder = Where to save images
        verbose = If true, prints useful intermediates

        Returns
        ---------
        None, but self.data will be populated as dict where keys are card oracle-ids and 
        values are the sorted set of tags.
        """
        if total_cards is None:
            total_cards = len(data)

        # make sure the image directory exists if we need it
        if get_images:
            os.makedirs(image_folder, exist_ok = True)
            
        # instantiate browser
        driver = webdriver.Chrome()
        # wait = WebDriverWait(driver, 5)

        # iterate through all cards
        data = data[:total_cards] # to limit run time.
        progress_bar = tqdm(total = len(data))
        # card_tags = []
        processed_cards = 0
        for i in range(len(data)):
            # retrieve card
            card = data[i]

            # perform initial filtration

            ## card types that don't matter
            base_type = card['type_line'].split(' — ', 1)[0]
            bad_card_types = {
                'Token', 'Card', 'Conspiracy', 'Emblem',
                'Phenomenon', 'Plane', 'Scheme', 'Stickers', 'Vanguard'
            }
            if any(bad in base_type for bad in bad_card_types):
                continue

            ## Art Series Cards Don't Matter
            if 'Art Series' in card['set_name']:
                continue

            # scrape the cards tags
            try:
                out = {}
                out[card['oracle_id']] = self._scrape(
                    card = card,
                    driver = driver,
                    rate_limit_seconds = rate_limit_seconds,
                    max_load_time = max_load_time,
                    get_images = get_images,
                    image_folder = image_folder
                )

                # store our
                self.data.append(out)
                self._data_index[card['oracle_id']] = out[card['oracle_id']]

                # update progress bar and processed_cards
                progress_bar.update(1)
                processed_cards += 1

                # save every save_every iterations
                if i % save_tags_every == 0:
                    with open(output_path, 'w') as f:
                        json.dump(self.data, f, indent = 4)
            
            except Exception as e:
                if verbose:
                    print(f'FAILED: {card["name"]} - {e}')
                continue # move onto the next one


        # # quit the browser
        driver.quit()

        # self.data = card_tags 
        # del card_tags

        if verbose:
            print(f'Processed {processed_cards} cards Scryfall tags.')

        # one final save
        with open(output_path, 'w') as f:
            json.dump(self.data, f, indent = 4)
            print(f'Data successfully saved to {output_path}')

    def scrape_card_images_for_dataset(
        self,
        card_data:list[dict],
        split_paths:list[str],
        tag_path:str,
        output_folder:str = '../data/card_images',
        image_type:str = 'png',
        max_download_time:float = 30.0,
        request_delay_seconds:float = 0.1,
        verbose:bool = True
    ):
        """
        Description
        ----------
        Download card images only for the cards represented in the existing
        train/val/test multi-label classification dataset files.

        Inputs
        ----------
        card_data = Bulk Scryfall oracle card data.
        split_paths = Paths to train/val/test dataset JSON files.
        tag_path = Path to the previously scraped Scryfall tags JSON.
        output_folder = Folder where downloaded card images will be stored.
        image_type = Scryfall image type to download. Common values include
            'png', 'jpg', 'large', 'normal', and 'small'.
        max_download_time = Timeout in seconds for each image request.
        request_delay_seconds = Delay between image requests.
        verbose = If true, prints useful intermediates.

        Returns
        ----------
        A dict summarizing download counts.
        """
        output_dir = Path(output_folder)
        output_dir.mkdir(parents = True, exist_ok = True)

        target_oracle_ids = self._get_dataset_oracle_ids(
            card_data = card_data,
            split_paths = split_paths,
            tag_path = tag_path
        )
        cards_to_download = [
            card for card in card_data
            if card.get('oracle_id') in target_oracle_ids
        ]

        downloaded = 0
        skipped_existing = 0
        missing_image_url = 0
        failed = 0

        progress_bar = tqdm(cards_to_download, desc = 'Downloading card images')
        session = requests.Session()
        session.headers.update(SCRYFALL_DOWNLOAD_HEADERS)

        for card in progress_bar:
            oracle_id = card['oracle_id']
            image_url = self._get_card_image_url(card = card, image_type = image_type)
            if image_url is None:
                missing_image_url += 1
                continue

            file_ext = self._infer_image_extension(image_url = image_url, image_type = image_type)
            image_path = output_dir / f'{oracle_id}.{file_ext}'

            if image_path.exists():
                skipped_existing += 1
                continue

            try:
                if request_delay_seconds > 0:
                    time.sleep(request_delay_seconds)

                response = retry(
                    lambda: session.get(image_url, timeout = max_download_time),
                    retries = 3,
                    delay = 2,
                    allowed_exceptions = (requests.RequestException,),
                    context = f'card image: {card["name"]}'
                )
                response.raise_for_status()

                with open(image_path, 'wb') as f:
                    f.write(response.content)

                downloaded += 1
            except Exception as e:
                failed += 1
                if verbose:
                    details = self._get_response_error_details(e)
                    print(f'FAILED IMAGE: {card["name"]} ({oracle_id}) - {e}{details}')

        summary = {
            'target_cards': len(cards_to_download),
            'downloaded': downloaded,
            'skipped_existing': skipped_existing,
            'missing_image_url': missing_image_url,
            'failed': failed,
            'output_folder': str(output_dir.resolve())
        }

        if verbose:
            print('Card image scrape complete:')
            for key, val in summary.items():
                print(f'\t{key} = {val}')

        return summary

    def _get_response_error_details(self, error:Exception):
        """
        Return a short Scryfall error detail when a failed response includes one.
        """
        response = getattr(error, 'response', None)
        if response is None:
            return ''

        try:
            payload = response.json()
        except ValueError:
            detail = response.text[:200].strip()
            return f' - {detail}' if detail else ''

        detail = payload.get('details') or payload.get('code')
        return f' - {detail}' if detail else ''

    def build_dataset_image_manifest(
        self,
        card_data:list[dict],
        split_paths:dict[str, str],
        tag_path:str,
        output_path:str = '../data/card_image_text_manifest.jsonl',
        image_folder:str = '../data/card_images',
        image_type:str = 'png',
        project_root:str = None,
        verbose:bool = True
    ):
        """
        Description
        ----------
        Build a manifest pairing dataset target text with the expected local
        card image path for image-to-text fine-tuning.

        Inputs
        ----------
        card_data = Bulk Scryfall oracle card data.
        split_paths = Mapping of split name to dataset JSON path.
        tag_path = Path to the previously scraped Scryfall tags JSON.
        output_path = Where to save the manifest file.
        image_folder = Folder containing or expected to contain the images.
        image_type = Scryfall image type used for the image download workflow.
        project_root = Base directory used to store project-relative image paths.
        verbose = If true, prints useful intermediates.

        Returns
        ----------
        A dict summarizing manifest contents.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents = True, exist_ok = True)

        image_dir = Path(image_folder)
        project_root_path = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
        split_data = self._load_split_records(split_paths = split_paths)
        idx_to_oracle_id = self._build_dataset_index_to_oracle_id(
            card_data = card_data,
            split_paths = list(split_paths.values()),
            tag_path = tag_path
        )
        oracle_id_to_card = {
            card['oracle_id']: card for card in card_data
            if 'oracle_id' in card
        }

        manifest_rows = []
        missing_cards = 0
        existing_images = 0
        missing_images = 0

        for split_name, records in split_data.items():
            for record in records:
                oracle_id = idx_to_oracle_id.get(record['id'])
                card = oracle_id_to_card.get(oracle_id)
                if (oracle_id is None) or (card is None):
                    missing_cards += 1
                    continue

                image_url = self._get_card_image_url(card = card, image_type = image_type)
                file_ext = self._infer_image_extension(image_url = image_url, image_type = image_type)
                image_path = (image_dir / f'{oracle_id}.{file_ext}').resolve()
                image_exists = image_path.exists()
                try:
                    relative_image_path = image_path.relative_to(project_root_path)
                except ValueError:
                    relative_image_path = image_path

                if image_exists:
                    existing_images += 1
                else:
                    missing_images += 1

                manifest_rows.append({
                    'id': record['id'],
                    'oracle_id': oracle_id,
                    'card_name': card['name'],
                    'split': split_name,
                    'image_path': str(relative_image_path).replace('\\', '/'),
                    'image_type': image_type,
                    'image_exists': image_exists,
                    'target_text': record['document'],
                    'tags': record['tags']
                })

        with open(output_file, 'w', encoding = 'utf-8') as f:
            for row in manifest_rows:
                f.write(json.dumps(row) + '\n')

        summary = {
            'manifest_path': str(output_file.resolve()),
            'records': len(manifest_rows),
            'existing_images': existing_images,
            'missing_images': missing_images,
            'missing_cards': missing_cards
        }

        if verbose:
            print('Card image manifest built:')
            for key, val in summary.items():
                print(f'\t{key} = {val}')

        return summary
    
    # === Internal Methods ===
    
    def _scrape(
        self,
        card:dict,
        driver,
        get_images:bool = True,
        image_folder:str = '../data/images',
        rate_limit_seconds:float = 3.0,
        max_load_time:float = 15.0
    ):
        """
        Description
        ----------
        This method describes how we will scrape the tags for one card.

        Inputs
        ---------
        card = A dictionary containing all key card attributes
        driver = Our selenium web driver
        get_images = If true, also retrieves the card image crop
        image_folder = Where to save images
        rate_limit_seconds = The number of seconds to wait before proceeding
            through the page.
        max_load_time = Max load time between steps

        Returns
        ----------
        None, but self.data will be appended with a sorted list of unique tags
        for the associated card.
        """
        # Helper Functions
        def load_scryfall_page():
            driver.get(card['scryfall_uri'])
            return wait.until(
                EC.presence_of_element_located((
                    By.XPATH,
                    "//a[contains(@href, 'tagger.scryfall.com')]"
                ))
            )
        
        def download_image():
            r = requests.get(img_url, timeout = max_load_time)
            r.raise_for_status()
            return r.content

        # ---------- Step 1: Cache short-circuit ----------
        if card['oracle_id'] in self._data_index:
            return self._data_index[card['oracle_id']]

        
        wait = WebDriverWait(driver, max_load_time)
        
        # ---------- Step 2: Visit scryfall page (lookup only) ----------
        time.sleep(rate_limit_seconds) # to not overload the servers

        # tagger link is typically an <a> pointing to tagger.scryfall.com
        tagger_link_elem = retry(
            load_scryfall_page,
            retries = 3,
            delay = rate_limit_seconds,
            context = f'scryfall page: {card["name"]}'
        )
        tagger_url = tagger_link_elem.get_attribute('href')

        # (optional) get URL for art crop
        if get_images:
            img_link_elem = wait.until(
                EC.presence_of_element_located((
                    By.XPATH,
                    "//a[contains(@href, 'cards.scryfall.io/art_crop')]"
                ))
            )
            img_url = img_link_elem.get_attribute('href')
            img_url = img_link_elem.get_attribute("href").split("?")[0]


        # ---------- Step 3: Visit tagger page ----------
        # time.sleep(rate_limit_seconds) # not necessary due to our Scryfall > Tagger loop and the first sleep.
        driver.get(tagger_url)

        card_container = wait.until(
            EC.presence_of_element_located((
                By.XPATH,
                "//section[.//h2[normalize-space()='Card']]"
                "//div[contains(@class,'taggings')]"
            ))
        )

        rows = card_container.find_elements(By.CSS_SELECTOR, 'div.tag-row')

        # ---------- Step 4. Extract Tag Names ----------
        tags = []
        for row in rows:
            # skip relation reference rows
            if row.find_elements(By.CSS_SELECTOR, '.icon-rel-references'):
                continue

            text = row.text.strip()
            if text:
                tags.append(text)

        # ---------- Step 5. Download Art Crop (Optional) ----------
        if get_images:
            # get the image bytes
            image_bytes = retry(
                download_image,
                retries = 3,
                delay = 2,
                allowed_exceptions = (requests.RequestException,),
                context = f'image: {card["name"]}'
            )

            # download it if we don't already have it
            image_path = f'{image_folder}/{card["oracle_id"]}.jpg'
            if not os.path.exists(image_path):
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)

        # store the output
        normalized_tags = list(sorted(set(tags)))
        # self.data[card['oracle_id']] = normalized_tags
        return normalized_tags

    def _get_dataset_oracle_ids(
        self,
        card_data:list[dict],
        split_paths:list[str],
        tag_path:str
    ):
        """
        Reconstruct the oracle IDs represented in the saved multi-label
        classification dataset splits by replaying the original dataset index
        assignment.
        """
        split_record_ids = set()
        split_data = self._load_split_records(
            split_paths = {str(i): split_path for i, split_path in enumerate(split_paths)}
        )
        for records in split_data.values():
            for record in records:
                split_record_ids.add(record['id'])

        idx_to_oracle_id = self._build_dataset_index_to_oracle_id(
            card_data = card_data,
            split_paths = split_paths,
            tag_path = tag_path
        )

        missing_ids = [record_id for record_id in split_record_ids if record_id not in idx_to_oracle_id]
        if len(missing_ids) > 0:
            raise ValueError(
                f'Failed to reconstruct {len(missing_ids)} dataset ids from the saved splits.'
            )

        return {idx_to_oracle_id[record_id] for record_id in split_record_ids}

    def _get_card_image_url(self, card:dict, image_type:str):
        """
        Retrieve the direct Scryfall image URL for a card record.
        """
        if ('image_uris' in card) and (image_type in card['image_uris']):
            return card['image_uris'][image_type]

        if 'image_uris' in card:
            fallback_order = ['png', 'large', 'normal', 'small', 'art_crop', 'border_crop']
            for fallback_type in fallback_order:
                if fallback_type in card['image_uris']:
                    return card['image_uris'][fallback_type]

        return None

    def _infer_image_extension(self, image_url:str, image_type:str):
        """
        Infer the output extension from the Scryfall URL and requested image type.
        """
        suffix = Path(image_url.split('?')[0]).suffix.lower().lstrip('.')
        if suffix:
            return suffix

        if image_type == 'png':
            return 'png'

        return 'jpg'

    def _load_split_records(self, split_paths:dict[str, str]):
        """
        Load dataset split records keyed by split name.
        """
        out = {}
        for split_name, split_path in split_paths.items():
            with open(split_path, 'r', encoding = 'utf-8') as f:
                out[split_name] = json.load(f)
        return out

    def _build_dataset_index_to_oracle_id(
        self,
        card_data:list[dict],
        split_paths:list[str],
        tag_path:str
    ):
        """
        Replay the original dataset build order so saved dataset ids can be
        mapped back to oracle ids.
        """
        dataset_unique_tags = set()
        split_data = self._load_split_records(
            split_paths = {str(i): split_path for i, split_path in enumerate(split_paths)}
        )
        for records in split_data.values():
            for record in records:
                dataset_unique_tags.update(record['tags'])

        with open(tag_path, 'r', encoding = 'utf-8') as f:
            card_tags_raw = json.load(f)

        card_tags = {}
        for card_tag_record in card_tags_raw:
            for oracle_id, tags in card_tag_record.items():
                trimmed_tags = [tag for tag in tags if tag in dataset_unique_tags]
                if len(trimmed_tags) > 0:
                    card_tags[oracle_id] = trimmed_tags

        idx_to_oracle_id = {}
        idx = 0
        for card in card_data:
            oracle_id = card.get('oracle_id')
            if (oracle_id in card_tags) and ('oracle_text' in card):
                idx_to_oracle_id[idx] = oracle_id
                idx += 1

        return idx_to_oracle_id
