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
TASK = 'summarization'

# === Question Asnwering ===
# # documentation = https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad
# MODEL = 'distilbert-base-cased-distilled-squad'

# === Summarization ===

# smaller model
# documentation = 
MODEL = 'google/mt5-small' # smaller model

# # larger model
# # documentation = https://huggingface.co/facebook/bart-large-cnn
# MODEL = 'facebook/bart-large-cnn'