# packages

## hugging face
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import evaluate
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler

## other neural network functions
import torch

## data wrangling
import numpy as np

## other
from tqdm.auto import tqdm

## from project directory
from .preprocess import preprocess
from .postprocess import postprocess

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
        self.tokenized_datasets = None
        self.data_collator = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.accelerator = None

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
        learning_rate:float = 2e-4,
        accelerator_mixed_precision:float = None,
        accelerator_force_cpu:bool = True,
        accelerator_gradient_steps:int = None,
        num_train_epochs:int = 10
    ):
        """
        Description
        -----------
        This method performs the training loop for the model we want to fine tune

        Inputs
        ----------
        learning_rate = The starting learning rate (we will be using a scheduler
            to adjust throughout training).
        accelerator_mixed_precision = Keep as default for full training with sufficient
            RAM. Otherwise use this to minimize RAM cost (in which case the recommmendation
            is to set it to "fp16").
        accelerator_force_cpu = If true, requires the model be trained on CPU
        accelerator_gradient_steps = Keep as default (None) if sufficient RAM means we don't
            have to throttle batch size. If RAM is limited, this param (plus batch_size = 1)
            can approximate a regular batch size.
        num_train_epochs = The number of epochs to train for

        Returns
        ----------
        None, but the following objects will be populated
        self.accelerator

        Also, self.model will be fine tuned
        """
        # === pre training loop set up ===

        ## define the objective function and optimizer
        rouge_score = evaluate.load('rouge')
        optimizer = AdamW(self.model.parameters(), lr = learning_rate)    

        ## set up the accelerator
        self.accelerator = Accelerator(
            mixed_precision = accelerator_mixed_precision, # if training on local machine
            cpu = accelerator_force_cpu, # if on local machine,
            gradient_accumulation_steps = accelerator_gradient_steps
        )
        self.model, optimizer, self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
            self.model, optimizer, self.train_dataloader, self.eval_dataloader
        )

        ## define learning rate scheduler
        num_update_steps_per_epoch = len(self.train_dataloader)
        num_training_steps = num_train_epochs * num_update_steps_per_epoch
        lr_scheduler = get_scheduler(
            'linear',
            optimizer = optimizer,
            num_warmup_steps = 0,
            num_training_steps = num_training_steps
        )

        # === training loop ===
        progress_bar = tqdm(range(num_training_steps))
        for epoch in range(num_train_epochs):
            # Training
            self.model.train()
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    self.accelerator.backward(loss)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)

            # Evaluation
            self.model.eval()
            for step, batch in enumerate(self.eval_dataloader):
                with torch.no_grad():
                    generated_tokens = self.accelerator.unwrap_model(self.model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )

                    generated_tokens = self.accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index = self.tokenizer.pad_token_id
                    )
                    labels = batch["labels"]

                    # If we did not pad to max length, we need to pad the labels too
                    labels = self.accelerator.pad_across_processes(
                        batch["labels"], dim=1, pad_index = self.tokenizer.pad_token_id
                    )

                    generated_tokens = self.accelerator.gather(generated_tokens).cpu().numpy()
                    labels = self.accelerator.gather(labels).cpu().numpy()

                    # Replace -100 in the labels as we can't decode them
                    labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                    if isinstance(generated_tokens, tuple):
                        generated_tokens = generated_tokens[0]
                    decoded_preds = self.tokenizer.batch_decode(
                        generated_tokens, skip_special_tokens=True
                    )
                    decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                    decoded_preds, decoded_labels = postprocess(
                        decoded_preds, decoded_labels
                    )

                    rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

            # Compute metrics
            result = rouge_score.compute()

            ## Extract the median ROUGE scores
            result = {key: value * 100 for key, value in result.items()}
            result = {k: round(v, 4) for k, v in result.items()}
            print(f"Epoch {epoch}:", result)

    def save_to_huggingface_hub(
        self,
        output_dir:str,
        repo,
        commit_message:str
    ):
        """
        Description
        ----------
        This method saves the model to the huggingface hub. Unlike the tutorial this class 
        is based on, here we only intend to upload to the hub once the full model is trained.

        Inputs
        ----------
        output_dir = The directory we want to save to
        repo = A local instance of the model path
        commit_message = What to say when we upload to the hub.

        Returns
        ----------
        None, but the model should be uploaded to the huggingface hub
        """
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(output_dir, save_function = self.accelerator.save)

        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(output_dir)
            repo.push_to_hub(commit_message = commit_message, blocking = False)

