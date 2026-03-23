from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import os
import time

import httpx
import torch
from tokenizer import Tokenizer
from model import BabyLM

FEEDBACK_PATH = "baby-lm/memory/feedback.txt"
MEMORY_PATH = "baby-lm/memory/conversations.txt"
CHECKPOINT_PATH = "baby-lm/checkpoints/model.pt"
SEQ_LEN = 4
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "").rstrip("/")
LM_TIMEOUT_SECONDS = float(os.getenv("LM_TIMEOUT_SECONDS", "120"))
LM_MAX_RETRIES = int(os.getenv("LM_MAX_RETRIES", "3"))
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


class LMStudioMessage(BaseModel):
    role: str
    content: str


class LMStudioChatRequest(BaseModel):
    model: str = "local-model"
    messages: list[LMStudioMessage]
    temperature: float = 0.7
    max_tokens: int | None = None


class LMStudioProxyResponse(BaseModel):
    raw: dict

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


def _call_lm_studio(payload: dict) -> dict:
    if not LM_STUDIO_URL:
        raise HTTPException(
            status_code=500,
            detail="LM_STUDIO_URL não configurada. Defina para o IP Tailscale da sua máquina local.",
        )

    endpoint = f"{LM_STUDIO_URL}/v1/chat/completions"
    last_error = None

    for attempt in range(1, LM_MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=LM_TIMEOUT_SECONDS) as client:
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()
        except Exception as err:
            last_error = str(err)
            if attempt < LM_MAX_RETRIES:
                time.sleep(min(2 ** (attempt - 1), 5))

    raise HTTPException(
        status_code=502,
        detail=f"Falha ao chamar LM Studio em {endpoint}: {last_error}",
    )

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


@app.get("/health")
def healthcheck():
    return {
        "status": "ok",
        "lm_studio_configured": bool(LM_STUDIO_URL),
        "lm_timeout_seconds": LM_TIMEOUT_SECONDS,
        "lm_max_retries": LM_MAX_RETRIES,
    }


@app.post("/lmstudio/chat/completions", response_model=LMStudioProxyResponse)
def lmstudio_chat(req: LMStudioChatRequest):
    payload = {
        "model": req.model,
        "messages": [m.model_dump() for m in req.messages],
        "temperature": req.temperature,
    }
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens

    result = _call_lm_studio(payload)
    return LMStudioProxyResponse(raw=result)
