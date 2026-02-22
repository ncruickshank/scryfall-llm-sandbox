# packages
import nltk

# functions
def postprocess(preds, labels):
    # initial cleaning
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUTE expects a new line after each sentence
    preds = ['\n'.join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ['\n'.join(nltk.sent_tokenize(label)) for label in labels]
    
    return preds, labels