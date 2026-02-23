# packages

## torch
import torch

## hugging face
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq

## other
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# class
class FineTuneLLM():
    """
    Description
    ----------
    This class contains the necessary methods to fine tune a LLM for multi-label classification

    Inputs
    ----------
    model_name = The name of the model we want to start from
    dataset = The dataset object from ScryfallDataset
    """
    def __init__(
        self, 
        model_name:str,
        dataset
    ):
        super().__init__()

        # store params as objects
        self.model_name = model_name
        self.dataset = dataset
        
        # initialize objects
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_input_length = None
        self.max_target_length = None
        # print(f'Tokenizer Pad Token = {self.tokenizer.pad_token_id}')

        # placeholders for objects to be created later

    # === Main Methods ===

    def prepare_data(
        self,
        max_input_length:int = 256,
        max_target_length:int = 64
    ):
        """
        Description
        ----------
        This method performs all necessary preprocessing on our dataset

        Inputs
        -----------
        max_input_length, max_target_length = The max lengths for our data
        """
        # store params as objects
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        # preprocess
        og_col_names = self.dataset.column_names
        self.dataset = self.dataset.map(
            lambda x: self._tokenize_function(
                example = x,
                max_input_length = max_input_length,
                max_target_length = max_target_length
            )
        )
        self.dataset = self.dataset.map(remove_columns = og_col_names['train'])

        self.dataset.set_format('torch')

    def train(
        self,
        batch_size:int,
        n_epochs:int,
        learning_rate:float,
        weight_decay:float,
        generation_max_length:int,
        generation_num_beams:int
    ):
        """
        Description
        ----------
        This method defines the training arguments and then trains the model based on
        those arguments.

        Inputs
        ----------
        batch_size = The batch size for training
        n_epochs = The number of epochs to train
        learning_rate = The learning rate for learning
        weight_decay = The weight decay rate
        generation_max_length = The max desired length for the desired output
        generation_num_beams = For beam search

        Returns
        ----------
        None, but self.model will be trained
        """
        # define the training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir = '../models/scryfall_auto_tagger',
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size = batch_size,
            num_train_epochs = n_epochs,
            eval_strategy  = 'epoch',
            save_strategy = 'epoch',
            load_best_model_at_end = True,
            metric_for_best_model = 'eval_loss',
            # logging_dir = '../logs',
            logging_steps = 1,
            learning_rate = learning_rate,
            weight_decay = weight_decay,
            fp16 = False, # True,

            # specific to Seq2SeqTrainer
            predict_with_generate = True,
            generation_max_length = generation_max_length,
            generation_num_beams = generation_num_beams
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer = self.tokenizer,
            model = self.model,
            label_pad_token_id = -100,
            pad_to_multiple_of = 8 if training_args.fp16 else None
        )

        # train the model
        trainer = Seq2SeqTrainer(
            model = self.model,
            args = training_args,
            train_dataset = self.dataset['train'],
            eval_dataset = self.dataset['val'],
            data_collator = data_collator,
            processing_class = self.tokenizer,
            compute_metrics = self._compute_metrics
        )
        trainer.train()

    def generate_tags(self, card_text, max_length:int = 256, num_beams:int = 4, early_stopping:bool = True):
        """
        Description
        -----------
        Once we have a tuned model, this function will create a collection of tags
        for the provided card_text

        Inputs
        ----------
        card_text = The text data (as structed in scryfall_dataset.py)

        Returns
        ----------
        card_tags = A generated set of tags for that card
        """
        inputs = self.tokenizer(card_text, return_tensors = 'pt').to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_length = max_length,
            num_beams = num_beams,
            early_stopping = True
        )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens = True)
        card_tags = self._parse_text(decoded)

        return card_tags

    # === Internal Methods ===

    def _tokenize_function(self, example, max_input_length, max_target_length):
        """
        Description
        ----------
        This method performs the tokenization preprocessing for our records

        Inputs
        ----------
        example = The record we want to tokenize
        max_input_length, max_target_length = The max length for our inputs and targets

        Returns
        ---------
        model_inputs = The tokenized example
        """
        model_inputs = self.tokenizer(
            example['document'],
            max_length = max_input_length,
            truncation = True,
            padding = 'max_length'
        )

        labels = self.tokenizer(
            example['tags'],
            max_length = max_target_length,
            truncation = True,
            padding = 'max_length'
        )

        # replace padding token id's in labels with -100
        labels['input_ids'] = [
            (l if l != self.tokenizer.pad_token_id else -100)
            for l in labels['input_ids']
        ]

        model_inputs['labels'] = labels['input_ids']

        return model_inputs
    
    def _parse_text(self, text):
        return set(t.strip().lower() for t in text.split(',') if t.strip())

    def _compute_metrics(self, eval_preds):
        """
        Description
        ----------
        This method dictates which metrics we will compute as part of the training loop

        Inputs
        ----------
        eval_preds = The evaluation prediction we want to compute metrics for

        Returns
        ----------
        evals = A dict containing our evaluated metrics
        """
        # # define predictions
        # if isinstance(eval_preds, tuple):
        #     preds, labels = eval_preds
        # else:
        #     preds = eval_preds.predictions
        #     labels = eval_preds.label_ids

        # if isinstance(preds, tuple):
        #     preds = preds[0]

        # # replace -100 in labels
        # labels = labels.copy()
        # labels[labels == -100] = self.tokenizer.pad_token_id

        # print(labels.dtype)
        # print(np.min(labels))
        # print(type(preds))
        # print(len(preds))
        # print(type(preds[0]))

        # decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens = True)
        # # labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        # decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens = True)

        if hasattr(eval_preds, "predictions"):
            preds = eval_preds.predictions
            labels = eval_preds.label_ids
        else:
            preds, labels = eval_preds

        # Unwrap tuple
        if isinstance(preds, tuple):
            preds = preds[0]

        # # Convert list-of-arrays → stacked array
        # if isinstance(preds, list):
        #     preds = np.stack(preds)

        # if isinstance(labels, list):
        #     labels = np.stack(labels)

        # convert preds and labels to numpy arrays
        preds = np.array(preds)
        labels = np.array(labels)

        # replace -100 in BOTH labels and predictions
        # labels = labels.copy()
        # labels[labels == -100] = self.tokenizer.pad_token_id
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)

        # print(preds.shape)

        decoded_preds = self.tokenizer.batch_decode(
            preds.tolist(),
            skip_special_tokens=True
        )

        decoded_labels = self.tokenizer.batch_decode(
            labels.tolist(),
            skip_special_tokens=True
        )

        all_pred, all_true = [], []

        for pred, true in zip(decoded_preds, decoded_labels):
            pred_set = self._parse_text(pred)
            true_set = self._parse_text(true)

            # convert to binary vectors over union
            union = list(pred_set | true_set)
            y_pred = [1 if t in pred_set else 0 for t in union]
            y_true = [1 if t in true_set else 0 for t in union]

            all_pred.extend(y_pred)
            all_true.extend(y_true)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true, all_pred, average = 'binary', zero_division = 0
        )

        out = {
            'micro_precision': precision,
            'micro_recall': recall,
            'micro_f1': f1
        }

        return out