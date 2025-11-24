import os
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from src.model.model_loader import load_model_and_tokenizer
from src.model.metrics import compute_metrics
from src.training.callbacks import build_callbacks

def build_training_arguments(cfg: dict) -> TrainingArguments:
    return TrainingArguments(
        output_dir=cfg["output_dir"],
        evaluation_strategy=cfg.get("evaluation_strategy", "epoch"),
        save_strategy=cfg.get("save_strategy", "epoch"),
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["epochs"],
        weight_decay=cfg.get("weight_decay", 0.01),
        logging_steps=cfg.get("logging_steps", 20),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "f1"),
        greater_is_better=cfg.get("greater_is_better", True),
        push_to_hub=cfg.get("push_to_hub", False), 
    )

def train_model(cfg: dict, tokenized_datasets: dict):
    '''
    Train the model:
    - load model
    - load tokenizer 
    - create DataCollator 
    - build Trainer
    - start Trainer
    '''

    num_labels = cfg["num_labels"]
    model_name = cfg["model_name"]

    model, tokenizer = load_model_and_tokenizer(model_name, num_labels)

    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = build_callbacks(cfg)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()

    if cfg.get("safe_model", True):
        save_path = os.path.join(cfg["output_dir"], "best_model")
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)

    return trainer
