import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- 1. HYPERPARAMETERS ---
# Keeping the model small (~3M parameters) so it trains fast on a laptop or single GPU
class ModelArgs:
    vocab_size: int = 5000     # Must match the tokenizer we built in Phase 2
    d_model: int = 256         # The size of our embedding vectors
    n_layers: int = 4          # Number of Transformer blocks
    n_heads: int = 8           # Number of attention heads (d_model must be divisible by this)
    max_seq_len: int = 256     # Max tokens the model can look at once
    dropout: float = 0.1

# --- 2. CAUSAL SELF-ATTENTION ---
class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        
        # Key, Query, Value projections combined into one batch
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask to ensure attention is only applied to the left (past tokens)
        self.register_buffer("bias", torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
                                     .view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence length, d_model (Channels)
        
        # Calculate Q, K, V for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # Reshape to (Batch, Heads, SeqLen, Head_Dim)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # Attention calculation: Softmax(Q*K^T / sqrt(d_k)) * V
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply the causal mask (replace upper triangle with -infinity)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v # (B, Heads, T, Head_Dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble all head outputs
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

# --- 3. FEED-FORWARD NETWORK ---
class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        # Standard MLP: expand dimension by 4, apply non-linearity, project back
        self.c_fc    = nn.Linear(config.d_model, 4 * config.d_model, bias=False)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# --- 4. TRANSFORMER BLOCK ---
class Block(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(self, x):
        # Residual connections: x = x + Sublayer(LayerNorm(x))
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# --- 5. THE MAIN AGENTIC SLM ---
class AgenticSLM(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model), # Token embeddings
            wpe = nn.Embedding(config.max_seq_len, config.d_model), # Positional embeddings
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.d_model),
        ))
        
        # Language modeling head (predicts the next token)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying: share weights between token embeddings and output head (saves parameters)
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max {self.config.max_seq_len}"

        # Create positional indices (0, 1, 2, ..., T-1)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Forward pass through embeddings
        tok_emb = self.transformer.wte(idx) # (Batch, SeqLen, d_model)
        pos_emb = self.transformer.wpe(pos) # (SeqLen, d_model)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Forward pass through Transformer blocks
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)

        if targets is not None:
            # If we are training, compute the loss
            logits = self.lm_head(x)
            # Flatten logits and targets to use CrossEntropyLoss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        else:
            # If we are just generating text (inference), get logits for the last token only
            logits = self.lm_head(x[:, [-1], :]) 
            return logits, None

if __name__ == "__main__":
    # Let's test if the brain compiles!
    print("Initializing AgenticSLM...")
    config = ModelArgs()
    model = AgenticSLM(config)
    
    # Print total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model built successfully! Total parameters: {total_params / 1e6:.2f} M")
    
    # Create a dummy batch of integer tokens (Batch Size: 2, Sequence Length: 10)
    dummy_input = torch.randint(0, config.vocab_size, (2, 10))
    dummy_targets = torch.randint(0, config.vocab_size, (2, 10))
    
    print("\nRunning a dummy forward pass...")
    logits, loss = model(dummy_input, targets=dummy_targets)
    
    print(f"Output logits shape: {logits.shape} (Expected: 2, 10, 5000)")
    print(f"Calculated Loss: {loss.item():.4f}")
    print("\nPhase 3 complete: The brain is ready to learn.")
