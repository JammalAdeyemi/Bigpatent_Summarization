# data_loading.py
from transformers import AutoTokenizer

def preprocess_function(examples, tokenizer, max_input_length, min_target_length, max_target_length, prefix):
    inputs = [prefix + inp for inp in examples["description"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    targets = examples["abstract"]

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_dataset(dataset, tokenizer, max_input_length, min_target_length, max_target_length, prefix):
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer, max_input_length, min_target_length, max_target_length, prefix), batched=True)
    return tokenized_dataset
