# Multi-Task Classification of The Blog Authorship Corpus using CNNs in PyTorch

## Project Overview

This project implements an AI-driven multi-task classification system to analyze blog posts and predict:
- Gender of the author (Male/Female)
- Age group of the author
- Topic category of the blog post

The system is trained using The Blog Authorship Corpus and leverages Convolutional Neural Networks (CNNs) in PyTorch.

## Key Features

- Multi-task learning approach
- CNN-based text classification
- Preprocessing and tokenization pipeline
- Customizable model architecture

## Dataset Preprocessing

### Data Extraction
- Extracts blog text from XML files
- Parses metadata from filenames
- Preprocesses text by:
  - Lowercasing
  - Removing special characters
  - Tokenization
  - Removing stopwords
  - Padding/truncating input

## Model Architecture

The multi-task CNN model consists of:
- Embedding Layer
- Convolutional Layers
- Max-Pooling Layers
- Fully Connected Layers

### Model Implementation

```python
class MultiTaskCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, hidden_dim, output_dims, dropout=0.5):
        super(MultiTaskCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in filter_sizes])
        self.fc_gender = nn.Linear(num_filters * len(filter_sizes), output_dims["gender"])
        self.fc_age = nn.Linear(num_filters * len(filter_sizes), output_dims["age"])
        self.fc_topic = nn.Linear(num_filters * len(filter_sizes), output_dims["topic"])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        conv_results = [F.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(conv_results, dim=1)
        x = self.dropout(x)
        return self.fc_gender(x), self.fc_age(x), self.fc_topic(x)
```

## Training Process

### Loss Functions
- Gender Classification: CrossEntropyLoss
- Age Classification: CrossEntropyLoss
- Topic Classification: CrossEntropyLoss

### Training Loop

```python
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for texts, gender_labels, age_labels, topic_labels in train_loader:
        texts, gender_labels, age_labels, topic_labels = texts.to(device), gender_labels.to(device), age_labels.to(device), topic_labels.to(device)
        optimizer.zero_grad()
        gender_out, age_out, topic_out = model(texts)
        loss = criterion_gender(gender_out, gender_labels) + criterion_age(age_out, age_labels) + criterion_topic(topic_out, topic_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
```

## Model Evaluation

### Evaluation Metrics
- Accuracy for gender classification
- Accuracy for age classification
- Accuracy for topic classification

```python
def evaluate_model(model, dataloader):
    model.eval()
    total_correct_gender, total_correct_age, total_correct_topic = 0, 0, 0
    with torch.no_grad():
        for texts, gender_labels, age_labels, topic_labels in dataloader:
            texts, gender_labels, age_labels, topic_labels = texts.to(device), gender_labels.to(device), age_labels.to(device), topic_labels.to(device)
            gender_out, age_out, topic_out = model(texts)
            total_correct_gender += (gender_out.argmax(1) == gender_labels).sum().item()
            total_correct_age += (age_out.argmax(1) == age_labels).sum().item()
            total_correct_topic += (topic_out.argmax(1) == topic_labels).sum().item()
    print(f"Gender Accuracy: {total_correct_gender/len(val_dataset):.4f}")
    print(f"Age Accuracy: {total_correct_age/len(val_dataset):.4f}")
    print(f"Topic Accuracy: {total_correct_topic/len(val_dataset):.4f}")
```

## Prediction

```python
def predict_blog_post(model, text):
    text = clean_text(text)
    tokens = tokenizer(text)
    numerical_text = [vocab_obj[word] for word in tokens]
    padded_text = pad_sequence(numerical_text)
    input_tensor = torch.tensor(padded_text, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        gender_out, age_out, topic_out = model(input_tensor)
    return gender_out.argmax().item(), age_out.argmax().item(), topic_out.argmax().item()
```

## Future Work

- Fine-tune hyperparameters
- Explore Transformer-based models (BERT, RoBERTa)
- Improve multi-task learning strategies

## Requirements

- PyTorch
- TorchText
- NumPy
- pandas
- XML parsing libraries

