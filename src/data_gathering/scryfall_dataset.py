# packages

## general use
import json
from collections import defaultdict

## data wrangling
import numpy as np
import heapq

## modeling
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

class ScryfallDataset():
    """
    Description
    ----------
    This class contains the methods to reshape the scryfall API bulk data
    and the scraped scryfall tagger community tags into a structured dataset
    intended for fine tunining a LLM towards a Question Answering task.

    Inputs
    ----------
    task = The modeling task we wish to perform
    """
    def __init__(self, task:str):
        super().__init__()
        expected_tasks = ['question-answering', 'summarization', 'multi_label_classification', 'seq2seq']
        assert task in expected_tasks, f'task must be one of {expected_tasks}, not {task}'

        # store params as objects
        self.task = task

        # create objects to be populated lated
        self.dataset = None

        ## (optional) if our task is multi_label_classification
        self.unique_tags = None
        self.label2id = None
        self.id2label = None
        self.class_weights = None

    def build_dataset(
        self,
        card_path:str = '../data/oracle-cards.json',
        tag_path:str = '../reports/scryfall_tags.json', 
        train_size_pct:float = 0.7,
        test_size_n:int = 500,
        top_n_tags:int = None,
        truncate_dataset:int = None,
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
        task = The modeling task we wish to construct a dataset for
        card_path = The path to where the bulk API data for the cards are stored.
        tag_path = The path to where the scryfall tags are saved
        train_size_pct = The percent of the records that aren't withheld for testing
            purposes that should be allocated to the training dataset.
        test_size_n = The number of records that should be withheld for testing
            purposes.
        truncate_dataset = If not None, it reduces the dataset to the specified size 
            *before* any train-val-test splits. 
        top_n_tags = If not None, this is a number to dictate how many of the most 
            common tags we want to train on.
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

        If our task is multi_label_classification, the following additional objects
        will be populated:
        - self.unique_tags = A list of unique tags across the dataset
        - self.label2id = A mapping from label (tag) to id
        - self.id2label = A mapping from id to label (tag)
        """
        if truncate_dataset is not None:
            assert test_size_n < truncate_dataset, 'Test size must be smaller than dataset.'

        # === initial data loading ===

        ## load bulk API data
        with open(card_path, 'r') as f:
            card_data = json.load(f)

        ## read tags
        with open(tag_path, 'r') as f:
            card_tags_raw = json.load(f)

        ## get tag counts and use that to define unique tags (limited if necessary)
        tag_counts = defaultdict(int)
        for c in card_tags_raw:
            ## make sure there is only one k,v pair in c
            assert len(c) == 1, f'Somehow {c} has more then one k,v pair'
            for _, tags in c.items():
                for t in tags:
                    tag_counts[t] += 1
        if top_n_tags is not None:
            self.unique_tags = heapq.nlargest(top_n_tags, tag_counts.items(), key = lambda item: item[1])
            self.unique_tags = set(dict(self.unique_tags).keys())
        
        # save tag counts for QC purposes
        self.tag_counts = tag_counts

        ## reshape tags to dict
        card_tags = {}
        for c in card_tags_raw:
            ## make sure there is only one k,v pair in c
            assert len(c) == 1, f'Somehow {c} has more then one k,v pair'
            for oid, tags in c.items():
                tags_trimmed = [t for t in tags if t in self.unique_tags]
                # print(f'tags = {tags}')
                # print(f'tags_trimmed = {tags_trimmed}')
                if (tags_trimmed is not None) and (len(tags_trimmed) > 0):
                    card_tags[oid] = tags_trimmed # tags

        # (optional) truncate dataset for testing purposes
        if truncate_dataset is not None:
            card_data = card_data[:truncate_dataset]

        # === build dataset ===

        ## iterate through card_data to build dataset using only oids that
        ## are also in card_tags
        dataset = []
        i = 0
        for card in card_data:
            if (card['oracle_id'] in card_tags.keys()) and ('oracle_text' in card.keys()):
                if self.task == 'question-answering':
                    out = self._reshape_to_question_answering(
                        idx = i,
                        card = card, 
                        tags = card_tags[card['oracle_id']]
                    )
                
                elif self.task in ['summarization', 'seq2seq']:
                    out = self._reshape_to_seq2seq(
                        idx = i, 
                        card = card,
                        tags = card_tags[card['oracle_id']]
                    )

                elif self.task == 'multi_label_classification': 
                    ## clean up the dataset
                    out = self._reshape_to_multi_label_classification(
                        idx = i,
                        card = card,
                        tags = card_tags[card['oracle_id']]
                    )

                # store output
                dataset.append(out)
                i += 1

        # === slice dataset into [train, val, test] ===

        ## select withholdind for test
        rng = np.random.default_rng(seed = random_seed)
        choices = list(range(len(dataset)))
        test_ids = rng.choice(choices, size = test_size_n, replace = False)
        test = []
        nontest = []
        for record in dataset:
            if record['id'] in test_ids:
                test.append(record)
            else:
                nontest.append(record)
        
        ## perform a standard train test split on the remained
        train, val = train_test_split(
            nontest,
            train_size = train_size_pct, 
            random_state = random_seed
        )

        # === save data and clean up ===
        storage_paths = {
            f'../data/scryfall_{self.task}_train.json': train,
            f'../data/scryfall_{self.task}_val.json': val,
            f'../data/scryfall_{self.task}_test.json': test
        }
        
        for path, data in storage_paths.items():
            with open(path, 'w') as f:
                json.dump(data, f, indent = 4)

        if verbose:
            print(f'Scryfall Tag Question Answering Dataset Built')
            print(f'\tTrain Records = {len(train)}')
            print(f'\tValidation Records = {len(val)}')
            print(f'\tTest Records = {len(test)}')
            if save_data:
                print('\tRecords saved to...')
                for path, _ in storage_paths.items():
                    print(f'\t\t{path}')
            print(f'\tNOTE: This method does not create the huggingface dataset object. Run load_dataset() for that.')

        del card_data, card_tags_raw, card_tags, dataset, storage_paths

    def load_hf_dataset(
        self, 
        train_path:str, val_path:str, test_path:str, 
        class_weight_clipping:tuple[float] = (1.0, 10.0),
        verbose:bool = True
    ):
        """
        Description
        ---------
        This method reads in the dataset as a HuggingFace dataset object

        Inputs
        ----------
        train_path, val_path, test_path = Where the key objects are stored
        class_weight_clipping = The values to clip our class weights between
        verbose = If true, prints useful intermediates

        Returns
        ----------

        """
        # load datasets
        data_paths = {'train': train_path, 'val': val_path, 'test': test_path}
        output = {}
        all_tags = [] # used if our task is multi_label_classification
        for name, path in data_paths.items():
            # read in the dataset
            with open(path, 'r') as f:
                dataset_split = json.load(f)

            # [if task == multi_label_classification] define additional objects
            if self.task == 'multi_label_classification':
                for card in dataset_split:
                    ## extend the collection of tags
                    all_tags.extend(card['tags']) # we will create labels as multi-hot vectors later

            ## store the dataset
            output[name] = dataset_split
        
        ## create label mapping objects
        self.unique_tags = sorted(set(all_tags))
        self.label2id = {tag: i for i, tag in enumerate(self.unique_tags)}
        self.id2label = {i: tag for tag, i in self.label2id.items()}

        # store the output as a huggingface dataset
        dataset_dict = DatasetDict()
        for name, data in output.items():
            dataset_dict[name] = Dataset.from_list(data)
        self.dataset = dataset_dict

        ## compute class weights as needed
        if self.task == 'multi_label_classification':
            self._compute_class_weights(clip_vals = class_weight_clipping)

        if verbose:
            print(f'Scryfall Tag {self.task.replace("_", " ").title()} Dataset Loaded')
            print(f'\tTrain Records = {self.dataset['train'].num_rows}')
            print(f'\tVal Records = {self.dataset['val'].num_rows}')
            print(f'\tTest Records = {self.dataset['test'].num_rows}')
            if self.unique_tags is not None:
                print(f'\tCount Unique Tags = {len(self.unique_tags)}')

        del output, dataset_dict

    # === Internal Methods ===

    def _compute_class_weights(self, clip_vals):
        """
        Description
        ----------
        Computes class weights for a multi-label classification problem.

        Inputs
        ----------
        clip_vals = The values to clip our weights between

        Returns
        ----------
        None, but self.class_weights will be populated.
        """
        counts = np.zeros(len(self.label2id))

        for tags in self.dataset['train']['tags']:
            for tag in tags:
                if tag in self.label2id:
                    counts[self.label2id[tag]] += 1

        total = len(self.dataset['train'])

        # inverse frequency (standard approach)
        self.class_weights = (total - counts) / (counts + 1e-6)
        self.class_weights = np.clip(
            self.class_weights, 
            clip_vals[0], clip_vals[1]
        ) # clamp weights

    def _reshape_to_question_answering(self, idx, card, tags):
        """
        Description
        ----------
        This method reshapes a card into a question-answering framework.
        Specifically, with columns ['id', 'title', 'question', 'context', 'answer']

        Inputs
        ----------
        idx = The arbitrary incremental number
        card = The dictionary of card details
        tags = The tags for the associated card.

        Returns
        ----------
        out = A dict containing the requisite keys.
        """
        # define context

        ## subclauses that are not always present
        mv_clause = f'Mana Cost = {card["mana_cost"]}\nMana Value = {card["cmc"]}\n' if 'mana_cost' in card.keys() else ''
        pt_clause = f'Power = {card["power"]}\nToughness = {card["toughness"]}\n' if 'power' in card.keys() else ''
        ly_clause = f'Loyalty = {card["loyalty"]}\n' if 'loyalty' in card.keys() else ''

        ## compile the context
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
        out['id'] = idx
        out['title'] = card['oracle_id']
        out['context'] = context 
        out['question'] = f'How would you functionally tag the Magic the Gathering card [[{card["name"]}]]?'
        out['answer'] = ', '.join(t for t in tags)

        return out
    
    def _reshape_to_seq2seq(self, idx, card, tags):
        """
        Description
        ----------
        This method reshapes a card into a summarization or seq2seq framework

        Inputs
        ----------
        idx = The arbitrary incremental number
        card = The dictionary of card details
        tags = The tags for the associated card.

        Returns
        ----------
        out = A dict containing the requisite keys.
        """
        # clean up the document
        
        ## subclauses that are not always present
        mv_clause = f'Mana Cost = {card["mana_cost"]}\nMana Value = {card["cmc"]}\n' if 'mana_cost' in card.keys() else ''
        pt_clause = f'Power = {card["power"]}\nToughness = {card["toughness"]}\n' if 'power' in card.keys() else ''
        ly_clause = f'Loyalty = {card["loyalty"]}\n' if 'loyalty' in card.keys() else ''

        ## compile the document
        ## choosing not to have the mv_clause
        ## CONSIDER dropping the "Type Line = " formatting as it may trick the model into a summarization task
        doc = f"""
        mtg card scryfall tags task:
        Return tags for the following card using this format:
        <tag> example tag </tag>

        ----------
        Card = {card['name']}
        {mv_clause}
        Type Line = {card['type_line']}\n
        Rules Text = {card['oracle_text']}\n 
        {pt_clause}
        {ly_clause}
        Color Identity = {card['color_identity']}\n
        Rarity = {card['rarity']}
        ----------
        """

        # --- Efficient Tag Trimming ---
        # We truncate each tag at common 'noise' markers: 'annotation:', ' via', or '('
        # This turns "group slugannotation: via..." into just "group slug"
        cleaned_tags = set()
        for t in tags:
            # Split and take the first part
            # ' via' (with space) prevents accidental trimming of words like 'viaduct' (if any)
            core_tag = t.split('annotation:')[0].split(' via')[0].split('(')[0].strip().lower()
            if core_tag:
                cleaned_tags.add(core_tag)
                
        # reshape tags to text
        structured_tags = [f'<tag> {t.lower()} </tag>' for t in cleaned_tags]
        tag_text = 'tags: ' + ' '.join(structured_tags)

        # creat output
        out = {}
        out['id'] = idx
        out['document'] = doc
        out['tags'] = tag_text

        return out 

    def _reshape_to_multi_label_classification(self, idx, card, tags):
        """
        Description
        ----------
        This method reshapes a card into a summarization framework

        Inputs
        ----------
        idx = The arbitrary incremental number
        card = The dictionary of card details
        tags = The tags for the associated card.

        Returns
        ----------
        out = A dict containing the requisite keys.
        """
        # clean up the document
        
        ## subclauses that are not always present
        mv_clause = f'Mana Cost = {card["mana_cost"]}\nMana Value = {card["cmc"]}\n' if 'mana_cost' in card.keys() else ''
        pt_clause = f'Power = {card["power"]}\nToughness = {card["toughness"]}\n' if 'power' in card.keys() else ''
        ly_clause = f'Loyalty = {card["loyalty"]}\n' if 'loyalty' in card.keys() else ''

        ## compile the document
        ## choosing not to have the mv_clause
        doc = f"""
        {card['name']}
        {mv_clause}
        Type Line = {card['type_line']}\n
        Rules Text = {card['oracle_text']}\n 
        {pt_clause}
        {ly_clause}
        Color Identity = {card['color_identity']}\n
        Rarity = {card['rarity']}
        """

        # creat output
        out = {}
        out['id'] = idx
        out['document'] = doc
        out['tags'] = tags

        return out
    
