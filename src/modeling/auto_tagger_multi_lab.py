# src/modeling/auto_tagger_multi_lab.py

## packages
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
from huggingface_hub import get_full_repo_name, hf_hub_download

## constants
from ..config import MAX_INPUT_LENGTH

def get_device_and_dtype():
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32
DEVICE, DTYPE = get_device_and_dtype()

## class
class ScryfallTaggerFromPretrained():
    """
    Description
    ----------
    This class retrieves a *pre trained fine tuned* model from the 
    huggingface hub and uses it to generate tags.

    Inputs
    ---------
    base_model_name = The name of the model our LoRA model was based on
    n_labels = The number of labels in our classification problem.
    output_dir = The output directory for the model. Used in tandem
        with get_full_repo_name to get the full path.
    id2label, label2id = From the ScryfallDataset class
    """
    def __init__(
        self,
        base_model_name:str,
        output_dir:str,
        n_labels:int = None,
        id2label:dict = None,
        label2id:dict = None
    ):
        super().__init__()

        # create objects to be used throughout
        self.repo_id = get_full_repo_name(output_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id)

        if (n_labels is None) or (id2label is None) or (label2id is None):
            n_labels, id2label, label2id = self._load_label_metadata()

        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels = n_labels,
            problem_type = 'multi_label_classification'
        )
        self.model = PeftModel.from_pretrained(base_model, self.repo_id)
        self.device = DEVICE # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.config.id2label = id2label
        self.model.config.label2id = label2id

        self.id2label = id2label
        self.label2id = label2id

        self.model.to(self.device)
        self.model.eval()

    def _load_label_metadata(self):
        """
        Load label metadata from the Hugging Face repo when it has been uploaded
        as a standalone artifact.
        """
        metadata_path = hf_hub_download(
            repo_id = self.repo_id,
            filename = 'label_metadata.json'
        )

        with open(metadata_path, 'r', encoding = 'utf-8') as f:
            metadata = json.load(f)

        n_labels = metadata['n_labels']
        id2label = {int(k): v for k, v in metadata['id2label'].items()}
        label2id = metadata['label2id']

        return n_labels, id2label, label2id

    def generate_tags(
        self, 
        card_text:str, 
        threshold = 0.7, 
        top_k = 5,
        cap_tags:bool = False
    ):
        """
        Description
        ----------
        Once we have fine tuned the model, this function will createa collection of tags
        for the provided card_text.

        Directly copy-pasted from modeling_multi_lab FineTuneLLM. It is recommended to
        consolidate both classes to call a separate py file which just contains 
        generate_tags().

        CONSIDERATIONS
        1. Adjust card_text argument to receive batches of cards.

        Inputs
        ----------
        card_text = The text data (as structured in scryfall_dataset.py)
        threshold = Confidence per tag predicition required to output
        top_k = The number of tags we want to return, in order of confidence
        cap_tags  If true, throttles the model to only provide top_k or less 
            tags. Do this if we suspect the trained model on average generates
            way more tags than reality. Since our training demonstrated that
            this is not true, we are safe to default this to False.
            Average True = 4.485119. Average Pred @ Thresh0.7 = 5.280952

        Returns
        ----------
        card_tags = A generated set of tags for that card, sorted in descending
            order of confidence.
        """
        inputs = self.tokenizer(
            card_text, 
            return_tensors = 'pt',
            truncation = True,
            max_length = MAX_INPUT_LENGTH
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)

        # hybrid threshold + top-k floor
        preds = (probs > threshold).squeeze(0).nonzero(as_tuple = True)[0]
        if len(preds) == 0:
            preds = torch.topk(probs, k = top_k, dim = 1).indices.squeeze(0)

        # OPTIONAL: also cap max predictions
        if (cap_tags) and (len(preds) > top_k):
            preds = torch.topk(probs, k = top_k, dim = 1).indices.squeeze(0)

        # sort by confidence
        preds = preds[torch.argsort(probs[0, preds], descending=True)]

        # tags = [
        #     self.id2label[i.item()] 
        #     if isinstance(self.id2label, dict) else self.id2label[i.item()] 
        #     for i in preds
        # ]
        tags = [self.id2label[i.item()] for i in preds]
        
        return tags

