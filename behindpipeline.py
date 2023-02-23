from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

### Select Checkpoint with its own tokenizer and head 

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)


inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

outputs = model(**inputs)

### Output gives us the logits
print(outputs.logits.shape)
print(outputs.logits)

### Feed through softmax layer for predictions
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(model.config.id2label)
print(predictions)


