"""
This config file contains all the necessary parameters for the various scripts
throughout this repository. Specifically, it contains parameters for three 
essential steps of the process:
1. Data Scraping: Done to link card oracle IDs with scryfall tags (not natively
    linked as scryfall tagging is community driven).
2. Image-To-Text OCR: Done using docling to convert card images to text. Often
    times the most accessible format for evaluating magic cards is via the 
    card image.
3. Scryfall Auto Tagger: This is the real emphasis of the project, where we 
    fine tune a model to predict the scryfall tags based on the card text.
"""
# =============================================================================
# ========== Data Scraping ====================================================
# =============================================================================

# --- Scryfall Tag Scraping Params ---
SCRAPE_MODE = 'dataset-images' # ['tags', 'dataset-images', 'dataset-image-manifest']
TOTAL_CARDS = 10000 # Use None for all cards
RATE_LIMIT_SECONDS = 2.5 # number of seconds to wait after loading a page
MAX_LOAD_TIME = 15 # max seconds to wait for any given operation
SAVE_TAGS_EVERY = 100 # how often to save outputs
OUTPUT_PATH = '../reports/scryfall_tags.json'
GET_IMAGES = False
IMAGE_FOLDER = '../data/card_images'
IMAGE_TYPE = 'png'
IMAGE_DOWNLOAD_DELAY_SECONDS = 0.1

# =============================================================================
# ========== Image-To-Text OCR ================================================
# =============================================================================

# --- OCR Parameters ---
USE_GRANITE = False # default to False
OCR_FILENAME = 'card_image_ocr_text_tags.json'

# =============================================================================
# ========== Scryfall Auto Tagger =============================================
# =============================================================================

# --- Modeling Params ---
DATASET_SOURCE = 'load_from_ocr' # ['build_from_scryfall', 'load_from_scryfall', 'load_from_ocr']
TAG_SIZE = 300 # top n tags only, and only cards which contain at least one of those tags
DATASET_SIZE_N = None # Default = None, choose for max datasize
TEST_SIZE_N = 50
TASK = 'multi_label_classification' # ['question_answering', 'summarization', 'multi_label_classification', 'seq2seq']

# --- Multi-Label Classification ---
# --- https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
MODEL_NAME = 'distilbert-base-uncased' # smaller
# MODEL_NAME = 'microsoft/deberta-v3-base' # larger

# --- Training ---
TEXT_COLUMN = 'card_text_ocr' # 'document' for structured scryfall text, 'card_text_ocr' for ocr-derived text
TRAIN_MODEL = True
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 48 # 1 for local training, 8 for GPU

LEARNING_RATE = 5e-4 # distilbert is less sensitive (5e-4 okay), deberta is sensitive (3e-5 works well)
WEIGHT_DECAY = 0.01
GRAD_ACCUMULATION_STEPS = 8 # useful if using accelerate
NUM_EPOCHS = 100

GENERATION_MAX_LENGTH = 128
GENERATION_NUM_BEAMS = 4

# be sure to create on the hub first
# OUTPUT_DIR = 'scryfall-auto-tagger' # for the structured card-text model version
OUTPUT_DIR = 'scryfall-auto-tagger-ocr' # for the OCR-generated card-text model version

# =============================================================================
# ========== Graveyard ========================================================
# =============================================================================

# --- Modeling Params ---
# BUILD_DATASET = False

# --- Question Asnwering ---
# # documentation = https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad
# MODEL = 'distilbert-base-cased-distilled-squad'

# --- Summarization ---

# smaller model
# documentation = https://huggingface.co/google/flan-t5-small
# MODEL_NAME = 'google/mt5-small' # smaller model
# MODEL_NAME = 'google/flan-t5-small' # smaller model
# MODEL_NAME = 'google/flan-t5-base' # larger model

# # larger model
# # documentation = https://huggingface.co/facebook/bart-large-cnn
# MODEL_NAME = 'facebook/bart-large-cnn'