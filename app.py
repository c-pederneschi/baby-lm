from fastapi import FastAPI, Request, Body
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import os
import torch
from tokenizer import Tokenizer
from model import BabyLM

FEEDBACK_PATH = "baby-lm/memory/feedback.txt"
MEMORY_PATH = "baby-lm/memory/conversations.txt"
CHECKPOINT_PATH = "baby-lm/checkpoints/model.pt"
SEQ_LEN = 4
app = FastAPI()
app.mount("/web", StaticFiles(directory=os.path.join(os.getcwd(), "web")), name="web")

class FeedbackRequest(BaseModel):
    user_text: str
    response: str
    feedback: str  # 'positivo' ou 'negativo'


@app.post("/feedback")
def feedback_endpoint(req: FeedbackRequest):
    # Salva feedback
    with open(FEEDBACK_PATH, 'a', encoding='utf-8') as f:
        f.write(f"{req.user_text}\t{req.response}\t{req.feedback}\n")
    return {"status": "ok"}

class ChatRequest(BaseModel):
    text: str

# Carrega conversas para vocabulário
def load_lines(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

lines = load_lines(MEMORY_PATH)
tokenizer = Tokenizer()
tokenizer.build_vocab(lines)

device = torch.device('cpu')
model = BabyLM(tokenizer.vocab_size())
if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    user_text = req.text.strip()
    # Garante que a pasta memory existe
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    # Salva conversa
    with open(MEMORY_PATH, 'a', encoding='utf-8') as f:
        f.write(user_text + "\n")
    # Prever próximo token
    tokens = tokenizer.encode(user_text)
    if len(tokens) == 0:
        resposta = ""
    else:
        input_seq = tokens[-SEQ_LEN:]
        pad = [tokenizer.vocab[tokenizer.pad_token]] * (SEQ_LEN - len(input_seq))
        input_ids = torch.tensor([pad + input_seq])
        with torch.no_grad():
            output = model(input_ids)
            pred_id = output.argmax(dim=-1).item()
            resposta = tokenizer.inv_vocab.get(pred_id, tokenizer.unk_token)
    # Salva resposta
    with open(MEMORY_PATH, 'a', encoding='utf-8') as f:
        f.write(resposta + "\n")
    # Treina automaticamente após cada mensagem
    result = os.system("python3 train.py")
    print("[LOG] Treinamento executado após mensagem. Resultado:", result)
    return {"response": resposta}
