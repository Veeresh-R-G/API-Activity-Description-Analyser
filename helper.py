from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)
import torch

MODEL_PATH = "./model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading DistilBERT from {MODEL_PATH}...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.push_to_hub("veeresh11/workout-activity-classifier")
tokenizer.push_to_hub("veeresh11/workout-activity-classifier")
print("Model and Tokenizers are loaded")
