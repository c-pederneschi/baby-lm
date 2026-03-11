# dataset.py
# Responsável por ler conversas e gerar pares de treino (entrada, alvo)

from tokenizer import Tokenizer

class ConversationDataset:
    def __init__(self, filepath, tokenizer, seq_len=4):
        self.data = []
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self._load_data(filepath)

    def _load_data(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        self.tokenizer.build_vocab(lines)
        for line in lines:
            tokens = self.tokenizer.encode(line)
            for i in range(1, len(tokens)):
                input_seq = tokens[max(0, i-self.seq_len):i]
                target = tokens[i]
                self.data.append((input_seq, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
