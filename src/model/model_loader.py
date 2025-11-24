from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)
import torch

def load_tokenizer(model_name: str):
    '''
    Load Tokenizer for a pretrained model 
    '''
    return AutoModelForSequenceClassification(model_name)

def load_model(model_name: str, num_labels: int):
    '''
    Load SequenceClassification Model with dynamic lable amount 
    '''
    model = AutoModelForSequenceClassification(
        model_name,
        num_labels=num_labels
    )
    return model

def load_model_and_tokenizer(model_name: str, num_labels: int):
    '''
    Loads both Model and Tokenizer
    '''
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, num_labels)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, tokenizer
