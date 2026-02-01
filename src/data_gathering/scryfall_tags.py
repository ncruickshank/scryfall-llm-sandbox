# packages

## general
import time
from tqdm import tqdm

## internet browsing
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

# class
class ScryfallTags():
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

        # objects to be used for later
        self.cache = {}

        # objects to be created later
        self.data = None


    # === Main Methods ===

    def scrape_all_cards(
        self, 
        data:list[dict],
        total_cards:int = None,
        rate_limit_seconds:float = 3.0,
        max_load_time:float = 15.0,
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
        verbose = If true, prints useful intermediates

        Returns
        ---------
        None, but self.data will be populated as dict where keys are card oracle-ids and 
        values are the sorted set of tags.
        """
        if total_cards is None:
            total_cards = len(data)
            
        # instantiate browser
        driver = webdriver.Chrome()
        wait = WebDriverWait(driver, 5)

        # iterate through all cards
        data = data[:total_cards] # to limit run time.
        progress_bar = tqdm(len(data))
        card_tags = []
        for i in range(len(data)):
            # retrieve card
            card = data[i]

            # scrape the cards tags
            try:
                out = {}
                out[card['oracle_id']] = self._scrape(
                    card = card,
                    driver = driver,
                    rate_limit_seconds = rate_limit_seconds,
                    max_load_time = max_load_time
                )

                # store our
                card_tags.append(out)

                # update progress bar 
                progress_bar.update(1)
            
            except Exception as e:
                # print(f'FAILED: {card["name"]} - {e}')
                # print(f'\tScryfall URI = {card["scryfall_uri"]}')
                return card['scryfall_uri']


        # # quit the browser
        driver.quit()

        self.data = card_tags 
        del card_tags
    
    # === Internal Methods ===
    
    def _scrape(
        self,
        card:dict,
        driver,
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
        rate_limit_seconds = The number of seconds to wait before proceeding
            through the page.
        max_load_time = Max load time between steps

        Returns
        ----------
        normalized_tags = A sorted list of unique tags
        """
        # ---------- Step 1: Cache short-circuit ----------
        if card['oracle_id'] in self.cache:
            return self.cache[card['oracle_id']]
        
        wait = WebDriverWait(driver, max_load_time)
        
        # ---------- Step 2: Visit scryfall page (lookup only) ----------
        time.sleep(rate_limit_seconds)
        driver.get(card['scryfall_uri'])

        # tagger link is typically an <a> pointing to tagger.scryfall.com
        tagger_link_elem = wait.until(
            EC.presence_of_element_located((
                By.XPATH,
                "//a[contains(@href, 'tagger.scryfall.com')]"
            ))
        )

        tagger_url = tagger_link_elem.get_attribute('href')

        # ---------- Step 3: Visit tagger page ----------
        time.sleep(rate_limit_seconds)
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
        tags = [
            row.text.strip() 
            for row in rows 
            if row.text and row.text.strip()
        ]
        normalized_tags = list(sorted(set(tags)))
        self.cache[card['oracle_id']] = normalized_tags

        return normalized_tags