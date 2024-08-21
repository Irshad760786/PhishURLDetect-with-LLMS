from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import RobertaTokenizerFast
import pandas as pd
import os

# Load your CSV data into a pandas DataFrame
df = pd.read_csv('path of csv dataset')

# Assuming the text column in your CSV is named 'url'
texts = df['url'].tolist()

# Define a custom tokenizer model for RoBERTa
bpe_model = models.BPE(unk_token="<unk>")
custom_tokenizer = Tokenizer(bpe_model)
custom_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
custom_tokenizer.decoder = decoders.ByteLevel()

# Configure the trainer for the custom tokenizer
trainer = trainers.BpeTrainer(
    vocab_size=50256,  # Ensure this is the same as RoBERTa's vocab size
    min_frequency=2,
    special_tokens=[
        "<s>", "<pad>", "</s>", "<unk>", "<mask>"
    ]
)

# Train the tokenizer on your dataset
custom_tokenizer.train_from_iterator(texts, trainer=trainer)

# Post-process to add special tokens for RoBERTa
custom_tokenizer.post_processor = processors.RobertaProcessing(
    ("</s>", custom_tokenizer.token_to_id("</s>")),
    ("<s>", custom_tokenizer.token_to_id("<s>")),
)

# Save the tokenizer to a directory
output_dir = 'path where you want to save the fine tuned tokenizer'
os.makedirs(output_dir, exist_ok=True)
custom_tokenizer.save(f'{output_dir}/tokenizer.json')

