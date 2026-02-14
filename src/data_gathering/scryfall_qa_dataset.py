import json
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset

class ScryfallQADataset():
    """
    Description
    ----------
    This class contains the methods to reshape the scryfall API bulk data
    and the scraped scryfall tagger community tags into a structured dataset
    intended for fine tunining a LLM towards a Question Answering task.

    Inputs

    ----------
    """
    def __init__(self):
        super().__init__()

        # create objects to be populated lated
        self.train = []
        self.val = []
        self.test = []
        self.dataset = None

    def build_dataset(
        self,
        card_path:str = '../data/oracle-cards.json',
        tag_path:str = '../reports/scryfall_tags.json', 
        train_size_pct:float = 0.7,
        test_size_n:int = 500,
        random_seed:int = 42,
        verbose:bool = True,
        save_data:bool = True
    ):
        """
        Description
        ----------
        This method retrieves the relevant datasets from the local project directory
        and transforms it into a standard question answering format. Inspiration 
        drawn from: https://huggingface.co/learn/llm-course/en/chapter7/7

        Inputs
        ----------
        card_path = The path to where the bulk API data for the cards are stored.
        tag_path = The path to where the scryfall tags are saved
        train_size_pct = The percent of the records that aren't withheld for testing
            purposes that should be allocated to the training dataset.
        test_size_n = The number of records that should be withheld for testing
            purposes.
        random_seed = For controlling random elements.
        verbose = If true, prints useful intermediates
        save_data = If true, saves the dataset as json to the ../data folder

        Returns
        ----------
        None, but the following objects will be created:
        self.train = A list of dicts containing approximately train_size_pct of our tagged
            cards.
        self.val = A list of dicts containing approximately (1 - train_size_pct) of our
            tagged cards
        self.test = A list of dicts containing exactly train_size_n of our tagged cards.
        """
        # === initial data loading ===

        ## load bulk API data
        with open(card_path, 'r') as f:
            card_data = json.load(f)

        ## read tags
        with open(tag_path, 'r') as f:
            card_tags_raw = json.load(f)

        ## reshape tags to dict
        card_tags = {}
        for c in card_tags_raw:
            ## make sure there is only one k,v pair in c
            assert len(c) == 1, f'Somehow {c} has more then one k,v pair'
            for oid, tags in c.items():
                card_tags[oid] = tags

        # === build dataset ===

        ## iterate through card_data to build dataset using only oids that
        ## are also in card_tags
        dataset = []
        i = 0
        for card in card_data:
            if (card['oracle_id'] in card_tags.keys()) and ('oracle_text' in card.keys()):
                # define context
                mv_clause = f'Mana Cost = {card["mana_cost"]}\nMana Value = {card["cmc"]}\n' if 'mana_cost' in card.keys() else ''
                pt_clause = f'Power = {card["power"]}\nToughness = {card["toughness"]}\n' if 'power' in card.keys() else ''
                ly_clause = f'Loyalty = {card["loyalty"]}\n' if 'loyalty' in card.keys() else ''

                context = f"""
                {mv_clause}
                Type Line = {card['type_line']}\n
                Rules Text = {card['oracle_text']}\n 
                {pt_clause}
                {ly_clause}
                Color Identity = {card['color_identity']}\n
                Rarity = {card['rarity']}
                """

                # create output
                out = {}
                out['id'] = i
                out['title'] = card['oracle_id']
                out['context'] = context 
                out['question'] = f'How would you functionally tag the Magic the Gathering card [[{card["name"]}]]?'
                out['answer'] = ', '.join(t for t in card_tags[card['oracle_id']])

                # store output
                dataset.append(out)
                i += 1

        # === slice dataset into [train, val, test] ===

        ## select withholdind for test
        rng = np.random.default_rng(seed = random_seed)
        choices = list(range(len(dataset)))
        test_ids = rng.choice(choices, size = test_size_n, replace = False)
        nontest = []
        for record in dataset:
            if record['id'] in test_ids:
                self.test.append(record)
            else:
                nontest.append(record)
        
        ## perform a standard train test split on the remained
        self.train, self.val = train_test_split(
            nontest,
            train_size = train_size_pct, 
            random_state = random_seed
        )

        # === save data and clean up ===
        storage_paths = {
            '../data/scryfall_qa_train.json': self.train,
            '../data/scryfall_qa_val.json': self.val,
            '../data/scryfall_qa_test.json': self.test
        }
        
        for path, data in storage_paths.items():
            with open(path, 'w') as f:
                json.dump(data, f, indent = 4)

        if verbose:
            print(f'Scryfall Tag Question Answering Dataset Built')
            print(f'\tTrain Records = {len(self.train)}')
            print(f'\tValidation Records = {len(self.val)}')
            print(f'\tTest Records = {len(self.test)}')
            if save_data:
                print('\tRecords saved to...')
                for path, _ in storage_paths.items():
                    print(f'\t\t{path}')
            print(f'\tNOTE: This method does not create the huggingface dataset object. Run load_dataset() for that.')

        del card_data, card_tags_raw, card_tags, dataset, storage_paths

    def load_hf_dataset(
        self, 
        train_path:str, val_path:str, test_path:str, 
        verbose:bool = True
    ):
        """
        Description
        ---------
        This method reads in the dataset as a HuggingFace dataset object

        Inputs
        ----------
        train_path, val_path, test_path = Where the key objects are stored
        verbose = If true, prints useful intermediates

        Returns
        ----------

        """
        # load dataset
        self.dataset = load_dataset(
            'json',
            data_files = {
                'train': train_path,
                'test': val_path
            }
        )

        with open(test_path, 'r') as f:
            self.test = json.load(f)

        if verbose:
            print('Scryfall Tag Question Answering Dataset Loaded')
            print(f'\tTrain Records = {self.dataset['train'].num_rows}')
            print(f'\tVal Records = {self.dataset['test'].num_rows}')
            print(f'\tTest Records = {len(self.test)}')
    
