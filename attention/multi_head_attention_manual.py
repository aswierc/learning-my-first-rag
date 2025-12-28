import torch
import torch.nn.functional as F

torch.set_printoptions(precision=3, sci_mode=False)

# =========================
# Setup
# =========================
sentence = ["I", "love", "pizza"]
seq_len = len(sentence)
d_model = 6          # total embedding size
num_heads = 2
d_head = d_model // num_heads

# Fake token embeddings (normally from a model)
torch.manual_seed(42)
embeddings = torch.randn(seq_len, d_model)

print("TOKEN EMBEDDINGS:")
print(embeddings)

# =========================
# Split into heads
# =========================
# shape: [tokens, heads, d_head]
emb_split = embeddings.view(seq_len, num_heads, d_head)

print("\nSPLIT INTO HEADS:")
print(emb_split)

# =========================
# Attention per head
# =========================
def attention(Q, K, V):
    dk = Q.shape[-1]
    scores = torch.matmul(Q, K.T) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return weights, output

head_outputs = []

for h in range(num_heads):
    Q = emb_split[:, h, :]
    K = emb_split[:, h, :]
    V = emb_split[:, h, :]

    weights, out = attention(Q, K, V)
    head_outputs.append(out)

    print(f"\nHEAD {h} ATTENTION WEIGHTS:")
    print(weights)

# =========================
# Concatenate heads
# =========================
# shape: [tokens, d_model]
multi_head_output = torch.cat(head_outputs, dim=-1)

print("\nCONCATENATED OUTPUT:")
print(multi_head_output)
