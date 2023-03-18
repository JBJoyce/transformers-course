#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW


# In[2]:


# Download dataset, select pretrained model and associated tokenizer
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# In[3]:


# Implement custom tokenizer function to handle two sentence structure
# Tokenizer handles truncation and custom Collator handles padding 
# inputs in the batch have the same size

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# In[4]:


# Strip datasets down to essentials, transformer expects 'labels' col
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names


# In[5]:


# Implement custom dataloaders
train_dataloader = DataLoader(
    tokenized_datasets['train'], shuffle=True, batch_size=8, collate_fn=data_collator
)

eval_dataloader = DataLoader(
    tokenized_datasets['validation'], batch_size=8, collate_fn=data_collator
)


# In[6]:


# Inspect first batch, shape should be the same
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}


# In[7]:


# create model from spec
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


# In[8]:


# run example batch through model to ensure proper function
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)


# In[9]:


# Implement standard loss optimizer plus learning rate scheduler

optimizer = AdamW(model.parameters(), lr=5e-5)

NUM_EPOCHS = 3
num_training_steps = NUM_EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)


# In[11]:


### Code to implement GPU acceleration if GPU available
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
device


# In[ ]:


from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(NUM_EPOCHS):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        
        # Calculate loss and gradient
        loss = outputs.loss
        loss.backward()
        
        #Update parameters
        optimizer.step()
        lr_scheduler.step()
        # Zero out gradient to prevent accumulation
        optimizer.zero_grad()
        progress_bar.update(1)


# In[ ]:




