from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."
model_inputs = tokenizer(sequence)

model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])
print(tokenizer.decode(model_inputs["input_ids"]))

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(tokenizer.decode(ids))
print(ids)


model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
print((model))

tokens_1 = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens_1)
print(output.logits)
