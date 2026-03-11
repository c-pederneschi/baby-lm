# chat.py
# Interface de chat no terminal para o baby-LM

from tokenizer import Tokenizer
from model import BabyLM
import torch
import os

MEMORY_PATH = "baby-lm/memory/conversations.txt"
CHECKPOINT_PATH = "baby-lm/checkpoints/model.pt"
SEQ_LEN = 4

# Carrega conversas para vocabulário
def load_lines(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

lines = load_lines(MEMORY_PATH)
tokenizer = Tokenizer()
tokenizer.build_vocab(lines)

# Carrega modelo
device = torch.device('cpu')
model = BabyLM(tokenizer.vocab_size())
if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

# Função para prever próximo token
def predict_next(input_text):
    tokens = tokenizer.encode(input_text)
    if len(tokens) == 0:
        return ""
    input_seq = tokens[-SEQ_LEN:]
    pad = [tokenizer.vocab[tokenizer.pad_token]] * (SEQ_LEN - len(input_seq))
    input_ids = torch.tensor([pad + input_seq])
    temperature = 1.0  # pode ajustar para mais criatividade
    with torch.no_grad():
        logits = model(input_ids)
        probs = torch.softmax(logits / temperature, dim=-1)
        pred_id = torch.multinomial(probs, num_samples=1).item()
        return tokenizer.inv_vocab.get(pred_id, tokenizer.unk_token)

# Loop de chat
print("baby-LM (digite 'sair' para encerrar)")
while True:
    user = input("Você: ").strip()
    if user.lower() == 'sair':
        break
    # Salva conversa
    with open(MEMORY_PATH, 'a', encoding='utf-8') as f:
        f.write(user + "\n")
    # Gera resposta
    resposta = predict_next(user)
    print("baby-LM:", resposta)
    # Salva resposta
    with open(MEMORY_PATH, 'a', encoding='utf-8') as f:
        f.write(resposta + "\n")
    # Treina automaticamente após cada mensagem
    print("Treinando modelo...")
    os.system("python3 baby-lm/train.py")
    print("Modelo atualizado!")
