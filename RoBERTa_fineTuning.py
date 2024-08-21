import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model
import torch.optim as optim

def load_data():
    df = pd.read_csv('path of csv dataset')
    X_train, X_test, y_train, y_test = train_test_split(
        df['url'],
        df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        random_state=42,
        stratify=y_train
    )

    train_df = pd.DataFrame({
        'url': X_train,
        'label': y_train
    })
    
    val_df = pd.DataFrame({
        'url': X_val,
        'label': y_val
    })
    
    test_df = pd.DataFrame({
        'url': X_test,
        'label': y_test
    })

    return train_df, val_df, test_df

def tokenize_function(text, tokenizer):
    return tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')

def create_dataset(dataframe, tokenizer, device):
    input_ids = []
    attention_masks = []
    labels = []

    for _, row in dataframe.iterrows():
        encoding = tokenize_function(row['url'], tokenizer)
        input_ids.append(encoding['input_ids'].squeeze().to(device))
        attention_masks.append(encoding['attention_mask'].squeeze().to(device))
        labels.append(torch.tensor(row['label'], dtype=torch.float).to(device))

    return TensorDataset(torch.stack(input_ids), torch.stack(attention_masks), torch.stack(labels))

class RobertaClassifier(nn.Module):
    def __init__(self, hidden_size, dense_size):
        super(RobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dense = nn.Linear(hidden_size, dense_size)
        self.fc = nn.Linear(dense_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        x = self.relu(self.dense(cls_embedding))
        logits = self.fc(x)
        probs = self.sigmoid(logits)
        return probs

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device).view(-1, 1)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device).view(-1, 1)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * input_ids.size(0)

            preds = outputs.view(-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.view(-1).cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    all_preds_bin = (np.array(all_preds) > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, all_preds_bin)
    precision = precision_score(all_labels, all_preds_bin)
    recall = recall_score(all_labels, all_preds_bin)
    f1 = f1_score(all_labels, all_preds_bin)
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    return avg_loss, accuracy, precision, recall, f1, fpr, tpr, roc_auc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df, val_df, test_df = load_data()
    print('Loading tokenizer...')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_dataset(train_df, tokenizer, device)
    val_dataset = create_dataset(val_df, tokenizer, device)
    test_dataset = create_dataset(test_df, tokenizer, device)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    print('Loading model...')
    base_model = RobertaClassifier(hidden_size=768, dense_size=128).to(device)
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query", "key", "value"]
    )

    model = get_peft_model(base_model, lora_config)
    model = model.to(device)
    print('Model loaded!')

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []
    fpr_list = []
    tpr_list = []
    roc_auc_list = []

    num_epochs = 15
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy, val_precision, val_recall, val_f1, fpr, tpr, roc_auc = evaluate(model, val_loader, criterion, device)
        
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
              f'Train Loss: {train_loss:.4f} - '
              f'Val Loss: {val_loss:.4f} - '
              f'Val Accuracy: {val_accuracy:.4f} - '
              f'Val Precision: {val_precision:.4f} - '
              f'Val Recall: {val_recall:.4f} - '
              f'Val F1 Score: {val_f1:.4f}')

    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(fpr_list[0], tpr_list[0], color='blue', lw=2, label='ROC curve (area = %0.4f)' % roc_auc_list[0])
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    main()

