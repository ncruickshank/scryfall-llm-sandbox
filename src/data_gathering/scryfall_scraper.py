# packages

## general
import time
from tqdm import tqdm
import json

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