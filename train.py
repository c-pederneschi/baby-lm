# train.py
# Treinamento do modelo BabyLM

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import ConversationDataset
from tokenizer import Tokenizer
from model import BabyLM
import os

MEMORY_PATH = "baby-lm/memory/conversations.txt"
CHECKPOINT_PATH = "baby-lm/checkpoints/model.pt"
SEQ_LEN = 4
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-2

FEEDBACK_PATH = "baby-lm/memory/feedback.txt"

# Carrega dados e tokenizer
tokenizer = Tokenizer()
dataset = ConversationDataset(MEMORY_PATH, tokenizer, seq_len=SEQ_LEN)

# Prepara batches
from torch.utils.data import DataLoader

def collate_fn(batch):
    # Preenche as sequências para o mesmo tamanho
    max_len = max(len(x[0]) for x in batch)
    inputs = []
    targets = []
    for inp, tgt in batch:
        pad = [tokenizer.vocab[tokenizer.pad_token]] * (max_len - len(inp))
        inputs.append(pad + inp)
        targets.append(tgt)
    return torch.tensor(inputs), torch.tensor(targets)

negativas = set()
if os.path.exists(FEEDBACK_PATH):
    with open(FEEDBACK_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3 and parts[2] == 'negativo':
                negativas.add(parts[1])  # resposta

# Carrega dados e tokenizer
lines = []
if os.path.exists(MEMORY_PATH):
    with open(MEMORY_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            l = line.strip()
            if l and l not in negativas:
                lines.append(l)
dataset = ConversationDataset(MEMORY_PATH, tokenizer, seq_len=SEQ_LEN)
dataset.data = [pair for pair in dataset.data if tokenizer.decode([pair[1]]) not in negativas]

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Cria modelo
model = BabyLM(tokenizer.vocab_size())
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.NLLLoss()

# Treinamento
for epoch in range(EPOCHS):
    total_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(dataloader):.4f}")
    # Salva checkpoint
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)

print("Treinamento finalizado. Modelo salvo em", CHECKPOINT_PATH)
