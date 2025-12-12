### tokenization

- Language models do not operate on text directly.  
  All input text must first be converted into tokens represented as numbers.

- Tokenizers are model-specific.  
  A BERT tokenizer is different from LLaMA or GPT tokenizers and produces different tokens for the same text.

- Tokens are **not words**.  
  They are often subword units (e.g. "import", "##ing"), punctuation, or special symbols.

- A token ID does **not** represent meaning by itself.  
  It is simply an index into the model’s vocabulary.

- Semantic meaning emerges later, at the embedding level,  
  not at the token or token ID level.

- Different languages may express the same concept,  
  but they usually map to different tokens and different token IDs.

- Models learn relationships between tokens through training,  
  not because token IDs inherently encode meaning.

