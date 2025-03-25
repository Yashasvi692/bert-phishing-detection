import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from codecarbon import EmissionsTracker
import evaluate
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset
dataset_dict = load_dataset("shawhin/phishing-site-classification")

# Load tokenizer and model
model_path = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)
id2label = {0: "Safe", 1: "Not Safe"}
label2id = {"Safe": 0, "Not Safe": 1}
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, num_labels=2, id2label=id2label, label2id=label2id
)

# Freeze base model parameters, unfreeze pooler
for name, param in model.base_model.named_parameters():
    param.requires_grad = False
for name, param in model.base_model.named_parameters():
    if "pooler" in name:
        param.requires_grad = True

# Preprocessing function
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

# Tokenize dataset
tokenized_data = dataset_dict.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define metrics
accuracy = evaluate.load("accuracy")
auc_score = evaluate.load("roc_auc")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    probabilities = np.exp(predictions) / np.exp(predictions).sum(-1, keepdims=True)
    positive_class_probs = probabilities[:, 1]
    auc = np.round(auc_score.compute(prediction_scores=positive_class_probs, references=labels)["roc_auc"], 3)
    predicted_classes = np.argmax(predictions, axis=1)
    acc = np.round(accuracy.compute(predictions=predicted_classes, references=labels)["accuracy"], 3)
    return {"Accuracy": acc, "AUC": auc}

# Training hyperparameters
lr = 2e-4
batch_size = 8
num_epochs = 10

training_args = TrainingArguments(
    output_dir="bert-phishing-classifier_teacher",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Set up trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train with emissions tracking
def train_with_profiling():
    tracker = EmissionsTracker(output_dir="./", output_file="emissions.csv", allow_multiple_runs=True)
    tracker.start()
    try:
        trainer.train()
    finally:
        tracker.stop()

    # Read emissions
    emissions_df = pd.read_csv("emissions.csv")
    latest_emissions = emissions_df["emissions"].iloc[-1]
    print(f"Total emissions: {latest_emissions} kgCO2eq")

    # Evaluate on validation set
    predictions = trainer.predict(tokenized_data["validation"])
    metrics = compute_metrics((predictions.predictions, predictions.label_ids))
    print("Validation Metrics:")
    print(metrics)

if __name__ == "__main__":
    print("Starting training and profiling...")
    train_with_profiling()
