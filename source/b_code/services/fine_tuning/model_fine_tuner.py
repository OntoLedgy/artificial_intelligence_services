from nf_common_source.code.configurations.datastructure.logging_inspection_level_b_enums import LoggingInspectionLevelBEnums
from nf_common_source.code.services.reporting_service.reporters.inspection_message_logger import log_inspection_message
from nf_common_source.code.services.reporting_service.wrappers.run_and_log_function_wrapper import run_and_log_function
from transformers import Trainer, TrainingArguments


@run_and_log_function
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
    log_inspection_message(
            message='Training model...',
            logging_inspection_level_b_enum=LoggingInspectionLevelBEnums.INFO)
    
    trainer.train()

    return model