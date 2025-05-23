from bclearer_core.configurations.datastructure.logging_inspection_level_b_enums import (
    LoggingInspectionLevelBEnums,
)
from bclearer_orchestration_services.reporting_service.reporters.inspection_message_logger import (
    log_inspection_message,
)
from bclearer_orchestration_services.reporting_service.wrappers.run_and_log_function_wrapper_latest import (
    run_and_log_function,
)
from transformers import Trainer, TrainingArguments


@run_and_log_function()
def fine_tune_model(
    tokenized_dataset,
    tokenizer,
    model,
    output_path: str = "./data/outputs/results",
    logging_dir: str = "./data/outputs/logs",
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
    log_inspection_message(
        message="Training model...",
        logging_inspection_level_b_enum=LoggingInspectionLevelBEnums.INFO,
    )

    trainer.train()

    return model
