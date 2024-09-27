from transformers import Trainer, TrainingArguments


def train_model(
    tokenized_dataset,
    tokenizer,
    model,
    output_path="./data/outputs/results",
    logging_dir="./data/outputs/logs",
):
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=8,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_dir=logging_dir,
        logging_steps=100,
        warmup_steps=500,
        weight_decay=0.01,
    )

    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,  # Use the dataset directly
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    return model
