# model.py
# Implementa um modelo de linguagem neural simples usando PyTorch

import torch
import torch.nn as nn

class BabyLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids):
        # input_ids: [batch, seq_len]
        embeds = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        # Faz média dos embeddings para obter um vetor fixo
        x = embeds.mean(dim=1)  # [batch, embed_dim]
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x  # [batch, vocab_size]

# Exemplo de uso:
# model = BabyLM(vocab_size=100)
# input_ids = torch.tensor([[1,2,3,4]])
# output = model(input_ids)
# print(output.shape)  # Deve ser [1, vocab_size]
