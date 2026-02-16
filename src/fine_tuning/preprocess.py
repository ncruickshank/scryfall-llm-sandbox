def preprocess(
    tokenizer,
    examples,
    max_input_length:int = 256, # 512 for full training, 256 for local
    max_target_length:int = 64 # 128 for full training, 64 for local
):
    """
    Description
    ----------
    Performs necessary preprocessing for the dataset before modeling.
    In this case, specifically ensuring proper length.
    Source = https://huggingface.co/learn/llm-course/en/chapter7/5

    Inputs
    ----------
    tokenizer = Our models tokenizer
    examples = The records we want to preprocess
    max_input_length = The maximum input length
    max_output_length = The maximum output length

    Returns
    ---------
    model_inputs = A cleaned object for the modeling inputs
    """
    model_inputs = tokenizer(
        examples['document'],
        max_length = max_input_length,
        truncation = True # ,
        # padding = 'max_length' # ignored to allow dynamic padding
    )

    labels = tokenizer(
        examples['summary'],
        max_length = max_target_length,
        truncation = True # ,
        # padding = 'max_length' # ignored to allow dynamic padding
    )

    model_inputs['labels'] = labels['input_ids']

    return model_inputs