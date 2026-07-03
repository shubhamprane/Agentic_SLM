import os
import torch
from tokenizers import Tokenizer
from phase3_model import ModelArgs, AgenticSLM

# --- 1. CONFIGURATION ---
BATCH_SIZE = 16          # How many sequences to train on at once
LEARNING_RATE = 3e-4     # How fast the model updates its weights
MAX_ITERS = 2000         # Number of training steps (increase to 5000+ for actual final training)
EVAL_INTERVAL = 200      # How often to print the loss
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # Use GPU if you have one, else CPU

print(f"Using device: {DEVICE}")

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "synthetic_dataset.txt")
    tok_path = os.path.join(script_dir, "..", "data", "custom_tokenizer.json")
    
    if not os.path.exists(data_path) or not os.path.exists(tok_path):
        raise FileNotFoundError("Missing data or tokenizer. Run Phase 1 & 2 first.")

    # Load our custom BPE tokenizer
    tokenizer = Tokenizer.from_file(tok_path)
    
    # Read the text dataset
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
        
    print("Encoding dataset... (This might take a few seconds)")
    encoded = tokenizer.encode(raw_text)
    data = torch.tensor(encoded.ids, dtype=torch.long)
    print(f"Dataset has {len(data):,} total tokens.")
    
    return data, tokenizer

# --- 2. THE DATA LOADER ---
def get_batch(data, config):
    # We pick random starting indices in the dataset
    ix = torch.randint(len(data) - config.max_seq_len - 1, (BATCH_SIZE,))
    
    # Input x is the sequence. Target y is the sequence shifted by 1 token.
    # The model learns to predict the next token.
    x = torch.stack([data[i:i+config.max_seq_len] for i in ix])
    y = torch.stack([data[i+1:i+config.max_seq_len+1] for i in ix])
    
    # Move batches to GPU if available
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

# --- 3. THE TRAINING LOOP ---
def train():
    data, tokenizer = load_data()
    
    config = ModelArgs()
    # Let's ensure our vocabulary size matches the tokenizer exactly
    config.vocab_size = tokenizer.get_vocab_size()
    
    model = AgenticSLM(config).to(DEVICE)
    
    # Standard optimizer for Transformers
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print("\n--- Starting Training ---")
    for iter in range(MAX_ITERS):
        # 1. Get a batch of data
        xb, yb = get_batch(data, config)
        
        # 2. Forward pass: evaluate the loss
        logits, loss = model(xb, targets=yb)
        
        # 3. Backward pass: calculate gradients
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # 4. Update the weights
        optimizer.step()
        
        # 5. Print progress
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            print(f"Step {iter:4d} | Training Loss: {loss.item():.4f}")

    # --- 4. SAVE THE BRAIN ---
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "agentic_janitor.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining Complete! Model weights saved to {save_path}")

if __name__ == "__main__":
    train()
