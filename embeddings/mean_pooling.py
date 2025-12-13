import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mask-aware mean pooling.

    token_embeddings: [num_tokens, hidden_dim]
    attention_mask:   [num_tokens]
    """
    mask = attention_mask.unsqueeze(-1)  # [num_tokens, 1]
    masked_embeddings = token_embeddings * mask
    pooled = masked_embeddings.sum(dim=0) / mask.sum()
    return pooled


def embed_text(text: str) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Token-level embeddings
    token_embeddings = outputs.last_hidden_state[0]  # [num_tokens, hidden_dim]
    attention_mask = inputs["attention_mask"][0]  # [num_tokens]

    # Sentence embedding via pooling
    sentence_embedding = mean_pooling(token_embeddings, attention_mask)

    # L2 normalization (required for cosine similarity)
    sentence_embedding = torch.nn.functional.normalize(
        sentence_embedding, p=2, dim=0
    )

    return sentence_embedding


if __name__ == "__main__":
    text = "Import VAT depends on the customs value of the goods."

    embedding = embed_text(text)

    print("Sentence embedding shape:", embedding.shape)
    print("First 10 values:", embedding[:10])
