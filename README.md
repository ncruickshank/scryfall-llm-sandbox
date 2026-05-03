# scryfall-llm-sandbox

Experimental code for building a Magic: The Gathering card auto-tagger from Scryfall data and Scryfall Community tags.

## Overview

This repository is a sandbox for training a Hugging Face model to infer functional tags for Magic cards.

The current workflow is:

1. Download or load a large Scryfall bulk card dataset.
2. Scrape the related Scryfall Community tag pages for card-level tags.
3. Reshape each card into a structured text format expected by the model.
4. Build train, validation, and test datasets for a multi-label classification task.
5. Fine-tune a transformer model, currently centered on DistilBERT, using LoRA adapters.
6. Load the trained adapter and predict tags for unseen cards.

In practice, the project is focused less on generic LLM experimentation and more on a specific supervised tagging pipeline:

- source data: Scryfall bulk card data
- target labels: Scryfall Community tags
- task: multi-label classification
- primary model family: Hugging Face transformers
- current base model: `distilbert-base-uncased`

## What The Model Learns

The model is trained to read a deliberately structured representation of a Magic card and predict a set of community-style functional tags.

That card representation is assembled from fields such as:

- card name
- mana cost
- mana value
- type line
- oracle text
- power/toughness
- loyalty
- color identity
- rarity

The exact formatting matters. The data pipeline in [`src/data_gathering/`](C:/Documents/GitHub/scryfall-llm-sandbox/src/data_gathering) constructs the text in a consistent layout so training and inference use the same schema.

## Repository Structure

### Core code

- [`src/config.py`](C:/Documents/GitHub/scryfall-llm-sandbox/src/config.py): central experiment settings for scraping, dataset building, task selection, model choice, and training parameters
- [`src/data_gathering/scryfall.py`](C:/Documents/GitHub/scryfall-llm-sandbox/src/data_gathering/scryfall.py): loads Scryfall bulk card data
- [`src/data_gathering/scryfall_scraper.py`](C:/Documents/GitHub/scryfall-llm-sandbox/src/data_gathering/scryfall_scraper.py): scrapes Scryfall Community tags and optionally downloads card art crops
- [`src/data_gathering/scryfall_dataset.py`](C:/Documents/GitHub/scryfall-llm-sandbox/src/data_gathering/scryfall_dataset.py): builds structured datasets for model training
- [`src/fine_tuning/fine_tune_multi_lab.py`](C:/Documents/GitHub/scryfall-llm-sandbox/src/fine_tuning/fine_tune_multi_lab.py): fine-tunes a transformer for multi-label classification
- [`src/fine_tuning/multi_lab_evaluator.py`](C:/Documents/GitHub/scryfall-llm-sandbox/src/fine_tuning/multi_lab_evaluator.py): evaluation helpers for threshold sweeps, top-k metrics, and hybrid scoring
- [`src/modeling/auto_tagger_multi_lab.py`](C:/Documents/GitHub/scryfall-llm-sandbox/src/modeling/auto_tagger_multi_lab.py): loads a fine-tuned adapter and predicts tags for new cards
- [`src/utils/`](C:/Documents/GitHub/scryfall-llm-sandbox/src/utils): support utilities such as retry logic and prediction debugging

### Data and artifacts

- [`data/`](C:/Documents/GitHub/scryfall-llm-sandbox/data): raw Scryfall data, generated datasets, and cached images
- [`reports/`](C:/Documents/GitHub/scryfall-llm-sandbox/reports): scraped tag outputs and training logs
- [`models/`](C:/Documents/GitHub/scryfall-llm-sandbox/models): saved fine-tuning outputs and checkpoints
- [`notebooks/`](C:/Documents/GitHub/scryfall-llm-sandbox/notebooks): exploratory and workflow notebooks

## Pipeline

### 1. Load card data

The pipeline starts with the Scryfall bulk card dump, typically `oracle-cards.json`.

[`src/data_gathering/scryfall.py`](C:/Documents/GitHub/scryfall-llm-sandbox/src/data_gathering/scryfall.py) reads that data into Python objects for downstream processing.

### 2. Scrape community tags

[`src/data_gathering/scryfall_scraper.py`](C:/Documents/GitHub/scryfall-llm-sandbox/src/data_gathering/scryfall_scraper.py) uses Selenium to:

- visit the Scryfall card page
- follow the link to the Scryfall Community tagger page
- extract the visible tags for the card
- optionally download the art crop image

The scraper filters out card types that are not relevant to the training objective, such as tokens and emblems.

### 3. Build the training dataset

[`src/data_gathering/scryfall_dataset.py`](C:/Documents/GitHub/scryfall-llm-sandbox/src/data_gathering/scryfall_dataset.py) merges:

- Scryfall bulk card data
- scraped community tags

It then reshapes each card into the text format expected by the model and writes train, validation, and test splits to `data/`.

While the repo contains code paths for several task formulations, the current intended use is:

- task: `multi_label_classification`

For that task, each example contains:

- `document`: the structured card text
- `tags`: the list of target labels

### 4. Fine-tune the model

[`src/fine_tuning/fine_tune_multi_lab.py`](C:/Documents/GitHub/scryfall-llm-sandbox/src/fine_tuning/fine_tune_multi_lab.py) handles:

- tokenization
- multi-hot label encoding
- class weighting
- LoRA adapter configuration
- Hugging Face training
- validation metrics

The current setup is aimed at fine-tuning `distilbert-base-uncased` for multi-label card tagging.

### 5. Run inference

[`src/modeling/auto_tagger_multi_lab.py`](C:/Documents/GitHub/scryfall-llm-sandbox/src/modeling/auto_tagger_multi_lab.py) loads:

- the base classification model
- the saved LoRA adapter
- the label mappings

It then predicts a ranked set of tags for a card based on the same structured text format used during training.

## Current Task Focus

Although some older or alternate code paths remain in the repo, the main project direction is:

- scrape a large sample of Magic cards
- pair those cards with Scryfall Community tags
- fine-tune a Hugging Face classifier on that data
- use the result as an automatic Magic card tagger

The most important implementation detail is consistency between dataset construction and inference formatting. The model is not learning from raw card JSON directly. It is learning from the specific textual representation produced in the data gathering pipeline.

## Outputs

Typical generated outputs include:

- scraped tag JSON files in `reports/`
- train/validation/test dataset JSON files in `data/`
- downloaded art crops in `data/images/`
- training logs in `reports/`
- model checkpoints and adapters in `models/`

## Notes

- This repo is currently organized as an experiment workspace rather than a polished package.
- Notebooks are part of the working process.
- Large generated artifacts may exist locally even if they are not intended to be committed long term.
- Some archived code remains in `src/_archives/` for reference.

## Future Improvements

Possible next steps for the project include:

- formalizing environment and dependency setup
- adding a reproducible training script or CLI entrypoint
- documenting dataset assumptions and tag cleanup rules in more detail
- adding evaluation summaries and example predictions to the repo
- comparing DistilBERT against larger encoder models
