from transformers import DataCollatorForSeq2Seq
import matplotlib.pyplot as plt

def prepare_tf_datasets(tokenized_train_dataset, tokenized_val_dataset, batch_size, tokenizer, model, generation_data_collator=None):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")
    
    tf_train_set = model.prepare_tf_dataset(
        tokenized_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )

    tf_val_set = model.prepare_tf_dataset(
        tokenized_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    tf_generation_set = None
    if generation_data_collator:
        tf_generation_set = model.prepare_tf_dataset(
            tokenized_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=generation_data_collator,
        )

    return tf_train_set, tf_val_set, tf_generation_set

def compile_model(model, optimizer):
    model.compile(optimizer=optimizer)

def plot_loss(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
