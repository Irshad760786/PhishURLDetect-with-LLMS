import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from transformers import GPT2Tokenizer, GPT2Model
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from peft import LoraConfig, get_peft_model, LoraModel


def load_data():
    df = pd.read_csv('path of csv datset')
    X_train, X_test, y_train, y_test = train_test_split(
        df['url'],
        df['label'],
        test_size=0.1,
        random_state=42,
        stratify=df['label']
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    # Create training DataFrame
    train_df = pd.DataFrame({
        'url': X_train,
        'label': y_train
    })
    
    # Create validation DataFrame
    val_df = pd.DataFrame({
        'url': X_val,
        'label': y_val
    })
    
    # Create test DataFrame
    test_df = pd.DataFrame({
        'url': X_test,
        'label': y_test
    })

    return train_df, val_df, test_df


def tokenize_function(text):
    return tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')

def create_dataset(dataframe):
    input_ids = []
    attention_masks = []
    labels = []

    for _, row in dataframe.iterrows():
        encoding = tokenize_function(row['url'])
        input_ids.append(encoding['input_ids'].squeeze())
        attention_masks.append(encoding['attention_mask'].squeeze())
        labels.append(row['label'])

    return TensorDataset(torch.stack(input_ids), torch.stack(attention_masks), torch.tensor(labels, dtype=torch.float))

class GPT2Classifier(nn.Module):
    def __init__(self, hidden_size, dense_size):
        super(GPT2Classifier, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.dense = nn.Linear(hidden_size, dense_size)
        self.fc = nn.Linear(dense_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        cls_embedding = hidden_states[:, -1, :]
        x = self.relu(self.dense(cls_embedding))
        logits = self.fc(x)
        probs = self.sigmoid(logits)
        return probs

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device).view(-1, 1)  # Reshape labels to [batch_size, 1]
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)  # No need to squeeze here
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device).view(-1, 1)  # Ensure labels are [batch_size, 1]

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * input_ids.size(0)

            # Convert outputs to 1D for metrics calculation
            preds = outputs.view(-1).cpu().numpy()
            all_preds.extend(preds)

            # Ensure labels are also 1D
            all_labels.extend(labels.view(-1).cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    
    # Convert predictions to binary values for metric calculation
    all_preds_bin = (np.array(all_preds) > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, all_preds_bin)
    precision = precision_score(all_labels, all_preds_bin)
    recall = recall_score(all_labels, all_preds_bin)
    f1 = f1_score(all_labels, all_preds_bin)
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    return avg_loss, accuracy, precision, recall, f1, fpr, tpr, roc_auc


if __name__ == "__main__":
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df, val_df, test_df = load_data()
    
    # Load the GPT-2 tokenizer and model
    print('Loading tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Define padding token
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_dataset(train_df)
    val_dataset = create_dataset(val_df)
    test_dataset = create_dataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Initialize model and move to GPU
    #model = GPT2Classifier(hidden_size=768, dense_size=128).to(device)
    # Initialize the base model
    print('Model Loading.....')
    base_model = GPT2Classifier(hidden_size=768, dense_size=128).to(device)

    # LoRA configuration
    lora_config = LoraConfig(
        r=8,                    # Low-rank dimension
        lora_alpha=16,          # Alpha value for scaling
        lora_dropout=0.1,       # Dropout rate for LoRA layers
        bias="none",            # No additional bias
        target_modules=["c_attn", "c_proj"]  # Target GPT-2 layers for LoRA
    )

    # Apply LoRA to the model
    model = get_peft_model(base_model, lora_config)
    model = model.to(device)
    print('Model Loaded!!')

    import torch.optim as optim

    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.BCELoss()

    # Initialize lists to store losses and metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []
    fpr_list = []
    tpr_list = []
    roc_auc_list = []

    # Training loop
    num_epochs = 15
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_accuracy, val_precision, val_recall, val_f1, fpr, tpr, roc_auc = evaluate(model, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1_scores.append(val_f1)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(roc_auc)
        
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {train_loss:.6f} - '
              f'Val Loss: {val_loss:.6f} - '
              f'Val Accuracy: {val_accuracy:.6f} - '
              f'Val Precision: {val_precision:.6f} - '
              f'Val Recall: {val_recall:.6f} - '
              f'Val F1 Score: {val_f1:.6f}')

    # Plot training and validation loss
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_Curve_gpt.pdf', format='pdf')
    plt.show()

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr_list[0], tpr_list[0], color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_list[0])
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig('roc_Curve_gpt.pdf', format='pdf')
    plt.show()

    # Evaluate on the test set
    test_loss, test_accuracy, test_precision, test_recall, test_f1, test_fpr, test_tpr, test_roc_auc = evaluate(model, test_loader, criterion)
    print(f'Test Loss: {test_loss:.6f} - '
          f'Test Accuracy: {test_accuracy:.6f} - '
          f'Test Precision: {test_precision:.6f} - '
          f'Test Recall: {test_recall:.6f} - '
          f'Test F1 Score: {test_f1:.6f}')

    # Save the model and tokenizer
    print("Saving the model...")
    torch.save(model.state_dict(), 'fine_tuned_gpt2.pt')
    tokenizer.save_pretrained('fine_tuned_gpt2_tokenizer')
    print("Model and tokenizer saved successfully!")

