
# packages
import json
from datetime import datetime

# class
class Scryfall():
    """
    Description
    ----------
    This class contains the methods to read cards downloaded from Scryfall into
    a manipulateable object. 

    Source:
    https://scryfall.com/docs/api/bulk-data

    Inputs
    ----------
    None
    """
    def __init__(self):
        super().__init__()

        # create objects to be populated later
        self.data = None
 
    
    def read_data(self, filepath:str = '../data/oracle-cards.json', verbose:bool = True):
        """
        Description
        ----------
        This method reads in the card info from the designated `filepath`

        Inputs
        ---------
        filepath = The location where the card data has been stored.

        Returns
        ----------
        None, but self.data will be populated with the Scryfall data, which is a list
        of dicts for each card.
        """
        # read data
        with open(filepath, 'r', encoding = 'utf-8') as f:
            self.data = json.load(f)

        if verbose:
            print('Scryfall Cards')
            print(f'\tSource = {filepath}\n\tCard Count = {len(self.data)}\n\tRead On = {datetime.now().strftime("%Y-%m-%d")}') 