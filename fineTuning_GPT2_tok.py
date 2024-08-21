from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
import pandas as pd
import os

# Load your CSV data into a pandas DataFrame
df = pd.read_csv('path of csv dataset')

# Extract text data from the DataFrame
# Assuming the text column in your CSV is named 'text'
texts = df['url'].tolist()

# Define a custom tokenizer model
bpe_model = models.BPE()
custom_tokenizer = Tokenizer(bpe_model)
custom_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
custom_tokenizer.decoder = decoders.ByteLevel()

# Configure the trainer for the custom tokenizer
trainer = trainers.BpeTrainer(
    vocab_size=50256,  # Ensure this is the same as your model's vocab size
    min_frequency=2,
    special_tokens=[
        "<s>", "<pad>", "</s>", "<unk>", "<mask>"
    ]
)

# Train the tokenizer on your dataset
custom_tokenizer.train_from_iterator(texts, trainer=trainer)

# Save the tokenizer to a directory
output_dir = 'path where you want to save the fine tuned tokenizer'
os.makedirs(output_dir, exist_ok=True)
custom_tokenizer.save(f'{output_dir}/tokenizer.json')

