import numpy as np
from datasets import load_metric
import nltk


def evaluate_model(tokenizer, metric):
    metric_fn = load_metric(metric)

    def metric_fn_wrapper(eval_predictions):
        predictions, labels = eval_predictions
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        for label in labels:
            label[label < 0] = tokenizer.pad_token_id  # Replace masked label tokens
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Rouge expects a newline after each sentence
        decoded_predictions = [
            "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_predictions
        ]
        decoded_labels = [
            "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
        ]
        result = metric_fn.compute(
            predictions=decoded_predictions, references=decoded_labels, use_stemmer=True
        )
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        # Add mean generated length
        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
        ]
        result["gen_len"] = np.mean(prediction_lens)

        return result
    
    return metric_fn_wrapper