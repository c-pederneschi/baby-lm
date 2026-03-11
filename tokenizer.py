# tokenizer.py
# Responsável por tokenizar e detokenizar textos, além de criar o vocabulário.

class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.special_tokens = [self.unk_token, self.pad_token]

    def build_vocab(self, texts):
        tokens = set()
        for text in texts:
            tokens.update(text.strip().split())
        tokens = self.special_tokens + sorted(list(tokens))
        self.vocab = {tok: idx for idx, tok in enumerate(tokens)}
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}

    def encode(self, text):
        return [self.vocab.get(tok, self.vocab[self.unk_token]) for tok in text.strip().split()]

    def decode(self, ids):
        return " ".join([self.inv_vocab.get(idx, self.unk_token) for idx in ids])

    def vocab_size(self):
        return len(self.vocab)
