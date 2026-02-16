# packages

## hugging face
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import evaluate
from torch.optim import AdamW
from accelerate import Accelerator

## from project directory
from .preprocess import preprocess

# class
class FineTuneLLM():
    """
    Description
    ----------
    This class contains the methods to fine tune a model using the huggingface interface.
    Source = https://huggingface.co/learn/llm-course/en/chapter7/5#fine-tuning-mt5-with-accelerate

    Inputs
    ----------
    model_name = The name of the model we want to start from
    """
    def __init__(self, model_name:str):
        # store params as objects
        self.model_name = model_name

        # create objects for later use
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # create objects to be made later
        # self.tokenized_datasets = None
        # self.data_collator = None
        # self.train_dataloader = None
        # self.eval_dataloader = None

    def prepare_data(
        self, 
        dataset,
        max_input_length:int = 252,
        max_target_length:int = 64,
        batch_size:int = 8
    ):
        """
        Description
        ----------
        This method performs the preprocessing on our dataset and stores it to 
        a class object. The following steps are performed.
        1. Tokenize Dataset
        2. Define data collator
        3. Populate dataloaders

        Inputs
        ----------
        dataset = A huggingface dataset to perform preprocessing on
        max_input_length = The maximum input length
        max_output_length = The maximum output length
        batch_size = The desired batch size

        Returns
        ----------
        None, but the following objects will be populated:
        self.tokenized_dataset
        self.data_collator
        self.train_dataloader
        self.eval_dataloader
        """
        # === 1. Tokenize Dataset ===
        self.tokenized_datasets = dataset.map(
            lambda ex: preprocess(
                tokenizer = self.tokenizer, 
                examples = ex, 
                max_input_length = max_input_length, 
                max_target_length = max_target_length
            ),
            batched = True,
            remove_columns = dataset['train'].column_names
        )
        # self.tokenized_datasets = self.tokenized_datasets.set_format('torch')

        # === 2. Define Data Collator ===
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, 
            model = self.model, 
            padding = True
        )

        # === 3. Populate Data Loaders ===
        self.train_dataloader = DataLoader(
            self.tokenized_datasets['train'],
            shuffle = True,
            collate_fn = self.data_collator,
            batch_size = batch_size
        )

        self.eval_dataloader = DataLoader(
            self.tokenized_datasets['test'],
            collate_fn = self.data_collator,
            batch_size = batch_size
        )

    def train(
        self,
        learning_rate:float = 2e-4
    ):
        """
        Description
        -----------
        This method performs the training loop for the model we want to fine tune

        Inputs
        ----------
        learning_rate = The starting learning rate (we will be using a scheduler
            to adjust throughout training).)

        Returns
        ----------

        """
        # define the objective function we want to improve
        rouge_score = evaluate.load('rouge')

        # define our optimizer
        optimizer = AdamW(summarizer.parameters(), lr = learning_rate)        

