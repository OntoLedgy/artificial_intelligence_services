from transformers import Trainer, TrainingArguments


def train_model(
        tokenized_dataset,
        tokenizer,
        model):
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=100,
        warmup_steps=500,
        weight_decay=0.01
    )

    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,  # Use the dataset directly
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

    return model
