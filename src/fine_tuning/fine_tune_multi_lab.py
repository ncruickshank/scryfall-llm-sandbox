# packages

## torch
import torch
from torch.nn import BCEWithLogitsLoss

## hugging face
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
# from transformers import default_data_collator
from transformers import DataCollatorWithPadding
from transformers import Trainer
from transformers import EarlyStoppingCallback

## lora
from peft import LoraConfig, get_peft_model

## other
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from scipy.special import expit 

## project directory
from .multi_lab_evaluator import MultiLabelEvaluator

# classes
class MultiLabelTrainer(Trainer):
    """
    Provided by ChatGPT
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.pos_weight = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.pos_weight = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.pos_weight is not None:
            loss_fct = BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(logits.device)
            )
        else:
            loss_fct = BCEWithLogitsLoss()

        loss = loss_fct(logits, labels.float())

        return (loss, outputs) if return_outputs else loss

class FineTuneLLM():
    """
    Description
    ----------
    This class contains the necessary methods to fine tune a LLM for multi-label classification

    Inputs
    ----------
    model_name = The name of the model we want to start from
    dataset = The dataset object from ScryfallDataset
    n_labels = The number of labels in our dataset
    label2id, id2label = Necessary mappings for multi-hot encoding. Sourced from scryfall_dataset
    class_weights = The class weights for BCEWithLogitsLoss
    """
    def __init__(
        self, 
        model_name:str,
        dataset,
        n_labels:int,
        label2id:dict,
        id2label:dict,
        class_weights
    ):
        super().__init__()

        # store params as objects
        self.model_name = model_name
        self.dataset = dataset
        self.label2id = label2id
        self.id2label = id2label

        # initialize objects
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels = n_labels,
            problem_type = 'multi_label_classification'
        )

        # update model multi-lab components
        self.model.config.id2label = id2label
        self.model.config.label2id = label2id

        # below handled in MultiLabelTrainer
        self.class_weights = torch.tensor(class_weights, dtype = torch.float32)
        # self.model.loss_fct = BCEWithLogitsLoss(
        #     pos_weight = self.class_weights.to(self.device)
        # )

        # load model to lora
        if model_name == 'distilbert-base-uncased':
            tgt_mods = ['q_lin', 'v_lin']
            r = 8 # lower representional capacity, smaller adapter
        elif model_name == 'microsoft/deberta-v3-base':
            tgt_mods = ["query_proj", "key_proj", "value_proj"] # "value_proj" <- for extra capacity
            r = 16 # larger representational capacity, larger adapter

        lora_config = LoraConfig(
            task_type = 'SEQ_CLS',
            r = r,
            lora_alpha = 32,
            target_modules = tgt_mods,
            lora_dropout = 0.1
        )
        self.model = get_peft_model(self.model, lora_config, 'default')
        self.model.print_trainable_parameters()

        # connect to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # instantiate objects to be populated later
        self.max_input_length = None

    # === Main Methods ===

    def prepare_data(
        self,
        max_input_length:int = 256
    ):
        """
        Description
        ----------
        This method performs all necessary preprocessing on our dataset

        Inputs
        -----------
        max_input_length = The max lengths for our data
        """
        # store params as objects
        self.max_input_length = max_input_length

        # preprocess
        og_col_names = self.dataset.column_names
        self.dataset = self.dataset.map(
            lambda x: self._tokenize_function(
                example = x,
                max_input_length = max_input_length
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
        patience:int = 5,
        output_dir:str = '../models/scryfall_auto_tagger'
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
        patience = The amount of epochs we are willing to wait for improvements to the 
            validation loss before early stopping.
        generation_max_length = The max desired length for the desired output. Only used
            if we are in a seq2seq paradigm
        generation_num_beams = For beam search. Only used if we are in a seq2seq paradigm
        device = The device we are training on. Either 'cpu' or 'cuda'

        Returns
        ----------
        None, but self.model will be trained
        """
        # define the training arguments
        training_args = TrainingArguments(
            output_dir = output_dir,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size = batch_size,
            num_train_epochs = n_epochs,
            
            # logging_dir = '../logs',
            logging_steps = 1,
            learning_rate = learning_rate,
            weight_decay = weight_decay,
            fp16 = False, # torch.cuda.is_available(),
            
            # gradient checkpointing
            gradient_checkpointing = True, # trades compute for memory (allows larger batch sizes)
            
            # evaluation
            eval_strategy = 'epoch',
            
            # checkpointing and saving
            save_strategy = 'best',
            save_total_limit = 1,
            
            # best model tracking
            load_best_model_at_end = True,
            metric_for_best_model = 'eval_loss',
        )

        # data collator
        # data_collator = default_data_collator
        data_collator = DataCollatorWithPadding(tokenizer = self.tokenizer)

        # train the model
        print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
        trainer = MultiLabelTrainer(
            model = self.model,
            args = training_args,
            train_dataset = self.dataset['train'],
            eval_dataset = self.dataset['val'],
            data_collator = data_collator,
            compute_metrics = self._compute_metrics,
            class_weights = self.class_weights,
            callbacks = [EarlyStoppingCallback(early_stopping_patience = patience)]
        )
        
        trainer.train()
        self.training_log_history = trainer.state.log_history

    def generate_tags(self, card_text:str, threshold = 0.4, top_k:int = 5):
        """
        Description
        ----------
        Once we have fine tuned the model, this function will createa collection of tags
        for the provided card_text

        Inputs
        ----------
        card_text = The text data (as structured in scryfall_dataset.py)
        threshold = Confidence per tag predicition required to output
        top_k = The number of tags we want to return, in order of confidence

        Returns
        ----------
        card_tags = A generated set of tags for that card
        """
        self.model.eval()
        inputs = self.tokenizer(card_text, return_tensors = 'pt').to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)

        # hybrid threshold + top-k floor
        preds = (probs > threshold).squeeze(0).nonzero(as_tuple = True)[0]
        if len(preds) == 0:
            preds = torch.topk(probs, k = top_k).indices.squeeze(0)

        # OPTIONAL: also cap max predictions
        if len(preds) > top_k:
            preds = torch.topk(probs, k=top_k).indices.squeeze(0)

        tags = [self.id2label[i.item()] for i in preds]
        
        return tags

    # === Internal Methods ===

    def _tokenize_function(
        self, 
        example, 
        max_input_length
    ):
        """
        Description
        ----------
        This method performs the tokenization preprocessing for our records

        Inputs
        ----------
        example = The record we want to tokenize
        max_input_length = The max length for our inputs and targets

        Returns
        ---------
        model_inputs = The tokenized example
        """
        model_inputs = self.tokenizer(
            example['document'],
            max_length = max_input_length,
            truncation = True,
            padding = False
        )

        label_vec = np.zeros(len(self.label2id), dtype=float)

        for tag in example['tags']:
            if tag in self.label2id:
                label_vec[self.label2id[tag]] = 1.0

        model_inputs["labels"] = label_vec.tolist()

        return model_inputs
    
    # def _compute_metrics(self, eval_preds):
    #     """
    #     Description
    #     ----------
    #     Computes the metrics for each training and validation loop

    #     Inputs
    #     ----------
    #     eval_preds = The outputs of the model we want to compute metrics for

    #     Returns
    #     ----------
    #     micro_precision, micro_recall, micro_f1
    #     """
    #     logits, labels = eval_preds

    #     if isinstance(logits, tuple):
    #         logits = logits[0]

    #     logits = np.array(logits)
    #     logits = np.nan_to_num(logits) # defensive clip
    #     labels = np.array(labels)

    #     # probs = 1 / (1 + np.exp(-logits))  # sigmoid
    #     probs = expit(logits) # safer logits calc
    #     preds = (probs > 0.5).astype(int)

    #     precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
    #         labels, preds, average = 'micro', zero_division = 0
    #     )

    #     precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    #         labels, preds, average = 'macro', zero_division = 0 
    #     )

    #     return {
    #         "micro_precision": precision_micro,
    #         "micro_recall": recall_micro,
    #         "micro_f1": f1_micro,
    #         'macro_precision': precision_macro,
    #         'macro_recall': recall_macro,
    #         'macro_f1': f1_macro
    #     }

    def _compute_metrics(self, eval_preds):
        """
        provided by ChatGPT to pair with MutliLabelEvaluator
        """
        logits, labels = eval_preds

        if isinstance(logits, tuple):
            logits = logits[0]

        evaluator = MultiLabelEvaluator(
            thresholds=np.linspace(0.1, 0.9, 5),
            k_values=[3, 5]
        )

        return evaluator.evaluate(logits, labels)