from transformers import BertConfig, BertModel, BertTokenizer, AutoTokenizer, AutoModel

# Building the config, 
# The configuration contains many attributes that are used to build the model
config = BertConfig()
print(config)
# Building the model from the config
# Model is randomly initialized!
model_0 = BertModel(config)

# Model is pretrained
model_1 = BertModel.from_pretrained("bert-base-cased")
# or model_1 = AutoModel.from_pretrained("bert-base-cased")

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
# Or tokenizer = AutoTokenizer("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
print(tokens)

# Encoding tokens -> #s
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)

# Decoding #s -> tokens
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)