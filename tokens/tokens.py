from transformers import AutoTokenizer

# BERT uses WordPiece tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "How to cook pasta al dente?"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Text:", text)
print("Tokens:", tokens)
print("Token IDs:", token_ids)

print("Vocabulary size:", len(tokenizer.vocab))
print("First 50 vocab entries:", list(tokenizer.vocab.keys())[:50])

print("-" * 40)

text_pl = "Import VAT depends on the customs value of goods."
tokens_pl = tokenizer.tokenize(text_pl)
token_ids_pl = tokenizer.convert_tokens_to_ids(tokens_pl)

print("Text:", text_pl)
print("Tokens:", tokens_pl)
print("Token IDs:", token_ids_pl)

