## Como iniciar o servidor

Para rodar o servidor FastAPI localmente, execute:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

O servidor ficará disponível em http://localhost:8000
# baby-LM

A tiny experimental language model that **starts knowing nothing** and learns gradually through interaction.

This project is meant for **learning, experimentation, and curiosity**, not performance or production.
The goal is to understand how a language model can be built **from scratch**, trained on small data, and gradually improved.

The system should be small enough to run on a **normal laptop CPU** and simple enough to understand fully.

---

# Project Goals

The project should implement a minimal language model that:

1. Starts with **no knowledge**
2. Learns from **text provided by the user**
3. Improves gradually through **training**
4. Allows the user to **teach it via chat**
5. Can run as a **local program or web application**

The focus is **clarity and educational value**, not optimization.

---

# Core Concept

The model learns by predicting the **next token** in a sequence.

Example:

Input:

```
gato é um
```

Expected prediction:

```
animal
```

Training example:

```
gato é um animal
```

The model gradually learns statistical patterns in language.

---

# Requirements

Python version:

```
Python 3.10+
```

Libraries allowed:

```
numpy
torch
fastapi
uvicorn
```

Dependencies should remain **minimal**.

---

# Project Architecture

Expected project structure:

```
baby-lm/
│
├─ README.md
├─ requirements.txt
│
├─ app.py
├─ train.py
├─ chat.py
│
├─ tokenizer.py
├─ dataset.py
├─ model.py
│
├─ memory/
│  └─ conversations.txt
│
├─ checkpoints/
│  └─ model.pt
│
└─ web/
   └─ index.html
```

---

# Component Responsibilities

## tokenizer.py

Responsible for converting text into tokens and tokens back into text.

Minimum features:

* whitespace tokenizer
* vocabulary creation
* token to id mapping
* id to token mapping

Example:

```
"gato é um animal"
→ ["gato","é","um","animal"]
```

---

## dataset.py

Handles training data.

Responsibilities:

* read conversation memory
* convert text into token sequences
* generate training pairs

Example training pair:

```
input:  "gato é um"
target: "animal"
```

---

## model.py

Implements a **very small neural language model**.

Suggested architecture:

```
Embedding
↓
Linear
↓
ReLU
↓
Linear
↓
Softmax
```

Input:

```
sequence of token IDs
```

Output:

```
probability distribution over vocabulary
```

The model predicts the **next token**.

The implementation should be:

* small
* readable
* heavily commented

---

## train.py

Handles the training loop.

Responsibilities:

* load dataset
* train the model
* compute loss
* update weights
* save checkpoints

Training flow:

```
text
→ tokens
→ model prediction
→ cross entropy loss
→ backpropagation
→ update weights
```

Save trained model to:

```
checkpoints/model.pt
```

---

## chat.py

Local terminal chat interface.

Workflow:

```
user input
↓
tokenize
↓
model generates response
↓
print response
↓
save conversation
```

Conversation history is stored in:

```
memory/conversations.txt
```

This file becomes future training data.

---

# Web Application

The project should also support running as a **web app**.

The web interface communicates with the model through a simple API.

Architecture:

```
Browser
↓
HTML / JS UI
↓
FastAPI backend
↓
baby-LM model
```

---

## app.py

Implements a **FastAPI server**.

Responsibilities:

* load trained model
* expose API endpoints
* handle chat requests

Example endpoint:

```
POST /chat
```

Request:

```
{
 "text": "gato é um animal"
}
```

Response:

```
{
 "response": "animal"
}
```

The server should also save conversations to:

```
memory/conversations.txt
```

---

## web/index.html

Simple chat interface.

Features:

* text input
* send button
* conversation display
* calls `/chat` API

Basic JavaScript using `fetch()` is sufficient.

The UI should remain **minimal and simple**.

---

# Learning Loop

The system supports a **learn-from-chat loop**.

Example interaction:

User:

```
gato é um animal
```

The system stores this text.

Later training allows predictions like:

```
gato é um → animal
```

The more text the user provides, the more patterns the model learns.

---

# Training Strategy

Training should be:

* simple
* small batches
* CPU-friendly
* easy to restart

Frequent checkpoint saving is recommended.

---

# Running the Project

Install dependencies:

```
pip install -r requirements.txt
```

Train the model:

```
python train.py
```

Run terminal chat:

```
python chat.py
```

Run web server:

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open the web interface:

```
http://localhost:8000
```

---

# Deploying to Render

The project should be deployable on Render.

Example configuration:

Build command:

```
pip install -r requirements.txt
```

Start command:

```
uvicorn app:app --host 0.0.0.0 --port 10000
```

This allows baby-LM to run as a **public web app**.

---

# Future Improvements

Possible extensions:

* better tokenizer
* transformer architecture
* conversation context memory
* online training
* reinforcement learning
* improved web UI

---

# Philosophy

baby-LM is a **toy brain**.

It begins with **zero knowledge**.

Over time it learns language patterns from the user and stored conversations.

The project exists to explore:

* how language models learn
* how data shapes model behavior
* how simple neural architectures can produce interesting results

---

# Final Note

Keep the code:

* simple
* readable
* educational
* hackable

The goal is **understanding**, not optimization.