import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

pl_prompt = "Dlaczego zwraca te wyniki?"
en_prompt = "Why does it return the results?"

def show_tokens(label: str, text: str):
    tokens = enc.encode(text)
    decoded = [enc.decode([t]) for t in tokens]

    print(f"\n=== {label} ===")
    print(f"Text: {text}")
    print(f"Token count: {len(tokens)}")
    print("Tokens:")
    for i, tok in enumerate(decoded):
        print(f"{i:02d}: {repr(tok)}")

show_tokens("PL", pl_prompt)
show_tokens("EN", en_prompt)
