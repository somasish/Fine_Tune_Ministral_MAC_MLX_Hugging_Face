# data_create.py
# This script prepares and splits a dataset for fine-tuning a language model using custom medical data.
# It loads a CSV file of symptom-diagnosis pairs, reformats them into prompt-response text format,
# shuffles and splits the data into train, test, and validation sets, and saves them as JSONL files.
# Intended for use with MLX and Hugging Face on macOS for LLM fine-tuning.
#
# Usage:
#   python data_create.py
#
# Output:
#   train.jsonl, test.jsonl, valid.jsonl in the current directory.
#
# Author: [Your Name]
# Date: [YYYY-MM-DD]
#
# For more details, see the project README.md.

import pandas as pd
import random
import json

# Load CSV dataset
file_path = 'symptom_dig.csv'
df = pd.read_csv(file_path)
print(df.head())

# reformat the csv data into text data
csv_data = []
for _, row in df.iterrows():
    diagnosis = row['label']
    symptoms = row['text']
    prompt = f"You are a medical diagnosis expert. You will give answer to patient's question based on the symptoms they have. Symptoms: '{symptoms}'. Question: 'What is the diagnosis I have?'. Response: You may be diagnosed with {diagnosis}."
    csv_data.append({"text": prompt})
random.shuffle(csv_data)
csv_data[:10]

# calculate split indices
total_records = len(csv_data)
train_split = int(total_records * 2 / 3)
test_split = int(total_records * 1 / 6)
print(train_split, test_split)

# split the data
train_data = csv_data[:train_split]
test_data = csv_data[train_split:train_split + test_split]
valid_data = csv_data[train_split + test_split:]

# save the split data to jsonl files
with open('train.jsonl', 'w') as train_file:
    for entry in train_data:
        train_file.write(json.dumps(entry) + '\n')

with open('test.jsonl', 'w') as test_file:
    for entry in test_data:
        test_file.write(json.dumps(entry) + '\n')

with open('valid.jsonl', 'w') as valid_file:
    for entry in valid_data:
        valid_file.write(json.dumps(entry) + '\n')