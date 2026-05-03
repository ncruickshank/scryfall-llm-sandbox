# === Scryfall Tag Scraping Params ===
TOTAL_CARDS = 10000 # Use None for all cards
RATE_LIMIT_SECONDS = 2.5 # number of seconds to wait after loading a page
MAX_LOAD_TIME = 15 # max seconds to wait for any given operation
SAVE_TAGS_EVERY = 100 # how often to save outputs
OUTPUT_PATH = '../reports/scryfall_tags.json'
GET_IMAGES = True
IMAGE_FOLDER = '../data/images'

# === Modeling Params ===
BUILD_DATASET = False
TAG_SIZE = 300 # top n tags only, and only cards which contain at least one of those tags
DATASET_SIZE_N = None # Default = None, choose for max datasize
TEST_SIZE_N = 50
TASK = 'multi_label_classification' # ['question_answering', 'summarization', 'multi_label_classification', 'seq2seq']

# === Question Asnwering ===
# # documentation = https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad
# MODEL = 'distilbert-base-cased-distilled-squad'

# === Summarization ===

# smaller model
# documentation = https://huggingface.co/google/flan-t5-small
# MODEL_NAME = 'google/mt5-small' # smaller model
# MODEL_NAME = 'google/flan-t5-small' # smaller model
# MODEL_NAME = 'google/flan-t5-base' # larger model

# # larger model
# # documentation = https://huggingface.co/facebook/bart-large-cnn
# MODEL_NAME = 'facebook/bart-large-cnn'

# === Multi-Label Classification ===
# documentation = https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
MODEL_NAME = 'distilbert-base-uncased' # smaller
# MODEL_NAME = 'microsoft/deberta-v3-base' # larger

# === Training ===
TRAIN_MODEL = False
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 48 # 1 for local training, 8 for GPU

LEARNING_RATE = 5e-4 # distilbert is less sensitive (5e-4 okay), deberta is sensitive (3e-5 works well)
WEIGHT_DECAY = 0.01
GRAD_ACCUMULATION_STEPS = 8 # useful if using accelerate
NUM_EPOCHS = 100

GENERATION_MAX_LENGTH = 128
GENERATION_NUM_BEAMS = 4

OUTPUT_DIR = 'scryfall-auto-tagger' # be sure to create on the hub first