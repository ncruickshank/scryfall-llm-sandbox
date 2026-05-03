# packages

## torch
import torch

## hugging face
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from transformers import GenerationConfig

## lora
from peft import LoraConfig, get_peft_model

## other
import re
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

        # load model to lora
        lora_config = LoraConfig(
            task_type = 'SEQ_2_SEQ_LM',
            r = 8,
            lora_alpha = 32, 
            target_modules = ['q', 'v'],
            lora_dropout = 0.1
        )
        self.model = get_peft_model(self.model, lora_config, 'default')
        self.model.print_trainable_parameters()
        
        # # Ensure generation start tokens exist
        # if self.model.config.decoder_start_token_id is None:
        #     self.model.config.decoder_start_token_id = self.tokenizer.pad_token_id
            
        # explicitly define the structural tags
        special_token_dict = {'additional_special_tokens': ['<tag>', '</tag>']}
        self.tokenizer.add_special_tokens(special_token_dict)
        
        # CRITICAL: resize the model embeddings to accomodate the new tokens
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
            
        # # Force decoder to start with 'tags:'
        # tags_token = self.tokenizer.encode('tags:', add_special_tokens = False)[0]
        # self.model.config.decoder_start_token_id = tags_token
        
        # if self.model.config.bos_token_id is None:
        #     # prefer tokenizer BOS if available, otherwise fall back to PAD
        #     bos = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.pad_token_id
        #     self.model.config.bos_token_id = bos

        # placeholders for objects to be created later
        self.max_input_length = None
        self.max_target_length = None

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
        generation_max_length:int = None,
        generation_num_beams:int = None,
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
        generation_max_length = The max desired length for the desired output. Only used
            if we are in a seq2seq paradigm
        generation_num_beams = For beam search. Only used if we are in a seq2seq paradigm
        device = The device we are training on. Either 'cpu' or 'cuda'

        Returns
        ----------
        None, but self.model will be trained
        """
        # define generation arguments
        generation_config = GenerationConfig(
            max_length = generation_max_length,
            num_beams = generation_num_beams,
            length_penalty = 0.8,
            no_repeat_ngram_size = 2,
            early_stopping = True,
            decoder_start_token_id = self.model.config.decoder_start_token_id # ,
            # bos_token_id = self.model.config.bos_token_id
        )
        self.model.generation_config = generation_config
        
        # define the training arguments
        training_args = Seq2SeqTrainingArguments(
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

            # specific to Seq2SeqTrainer
            predict_with_generate = True,
            # generation_max_length = generation_max_length,
            # generation_num_beams = generation_num_beams,
            generation_config = generation_config
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer = self.tokenizer,
            model = self.model,
            label_pad_token_id = -100,
            pad_to_multiple_of = 8 if training_args.fp16 else None
        )

        # train the model
        print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
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
        inputs = self.tokenizer(card_text, return_tensors = 'pt').to(self.device)

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
            padding = False # 'max_length'
        )

        labels = self.tokenizer(
            example['tags'],
            max_length = max_target_length,
            truncation = True,
            padding = False # 'max_length'
        )

        # replace padding token id's in labels with -100
        labels['input_ids'] = [
            (l if l != self.tokenizer.pad_token_id else -100)
            for l in labels['input_ids']
        ]
        
        # added defensive safety step
        if len(labels['input_ids']) == 0:
            labels['input_ids'] = [self.tokenizer.pad_token_id]

        model_inputs['labels'] = labels['input_ids']

        return model_inputs
    
    def _parse_text(self, text):
        # 1. Clean up noise but keep potential tag content
        text = text.lower().replace("<pad>", "").replace("</s>", "").strip()
        
        # 2. Look for ANYTHING inside brackets, or just words following 'tag'
        # This captures <tag>flying</tag> AND things like tag-flying-
        found = re.findall(r"tag[>\s\-]*([^<\s\-]+)", text)
        
        return set(t for t in found if t not in ['s', 'tags', 'tag'])
    
    def _compute_metrics(self, eval_preds):
        """
        Computes metrics and prints 'Golden Samples' to debug generation quality.
        """
        if hasattr(eval_preds, "predictions"):
            preds = eval_preds.predictions
            labels = eval_preds.label_ids
        else:
            preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.array(preds)
        labels = np.array(labels)
        
        # Convert logits -> token ids if needed
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)
        
        # Replace masked labels (-100) with pad_token_id for decoding
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        
        # Decode everything
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False)
        
        # # --- Golden Sample Printing ---
        # print("\n" + "="*50)
        # print("EVALUATION GOLDEN SAMPLES")
        # print("="*50)
        # num_to_print = min(3, len(decoded_preds))
        # for i in range(num_to_print):
        #     p_raw = decoded_preds[i]
        #     l_raw = decoded_labels[i]
        #     p_parsed = self._parse_text(p_raw)
        #     l_parsed = self._parse_text(l_raw)
            
        #     print(f"\nSample {i+1}:")
        #     print(f"  RAW GT:    {l_raw}")
        #     print(f"  RAW PRED:  {p_raw}")
        #     print(f"  PARSED GT:   {l_parsed}")
        #     print(f"  PARSED PRED: {p_parsed}")
        # print("="*50 + "\n")

        # Metric calculation
        TP = FP = FN = 0
        for pred, true in zip(decoded_preds, decoded_labels):
            pred_set = self._parse_text(pred)
            true_set = self._parse_text(true)
        
            TP += len(pred_set & true_set)
            FP += len(pred_set - true_set)
            FN += len(true_set - pred_set)
        
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        return {
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": f1
        }