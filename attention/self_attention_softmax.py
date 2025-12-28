import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

torch.set_printoptions(precision=3, sci_mode=False)

sentence = "I love pizza"

# =========================
# 2. Tokens + model
# =========================
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# =========================
# 3. Tokens
# =========================
inputs = tokenizer(sentence, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

print("TOKENS:")
print(tokens)

# =========================
# 4. Embeddings (last hidden state)
# =========================
with torch.no_grad():
    outputs = model(**inputs)

# shape: [tokens, hidden_dim]
embeddings = outputs.last_hidden_state[0]

print("\nEMBEDDINGS SHAPE:", embeddings.shape)

# =========================
# 5. Q, K, V (1 head, identity)
# =========================
Q = embeddings
K = embeddings
V = embeddings

# =========================
# 6. Scaled Dot-Product Attention
# =========================
dk = Q.shape[-1]
scores = torch.matmul(Q, K.T) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

attention_weights = F.softmax(scores, dim=-1)

# =========================
# 7. Output attention
# =========================
attention_output = torch.matmul(attention_weights, V)

def pretty(t):
    return torch.round(t * 1000) / 1000

print("\nATTENTION SCORES (Q·Kᵀ / √d):")
print(pretty(scores))

print("\nATTENTION WEIGHTS (softmax):")
print(pretty(attention_weights))

print("\nROW SUMS (should be 1.0):")
print(attention_weights.sum(dim=-1))

print("\nATTENTION OUTPUT (contextual embeddings):")
print(pretty(attention_output))
