import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def train_custom_tokenizer():
    # 1. Bulletproof Pathing
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "synthetic_dataset.txt")
    output_path = os.path.join(script_dir, "..", "data", "custom_tokenizer.json")
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}. Did you run Phase 1?")
        return

    print("Initializing BPE Tokenizer...")
    
    # 2. Initialize a blank Byte-Pair Encoding model
    # We define [UNK] for any weird characters it hasn't seen
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    
    # Pre-tokenizer splits by whitespace before applying BPE
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # 3. Define our Agentic Special Tokens
    # Order matters: padding/unknown/EOS first, then our custom tags
    special_tokens = [
        "<|endoftext|>", # Marks the end of a sequence
        "[UNK]",         # Unknown token
        "<OBSERVE>", "</OBSERVE>",
        "<THINK>", "</THINK>",
        "<ACT>", "</ACT>",
        "<SUCCESS>", "<ERROR>" # We will use these later in the real sandbox!
    ]

    # 4. Configure the Trainer
    # 5000 is a great vocab size for our restricted dictionary.
    # It forces the model to learn semantic chunks without memorizing whole sentences.
    trainer = trainers.BpeTrainer(
        vocab_size=5000,
        special_tokens=special_tokens
    )

    # 5. Train the Tokenizer on our generated dataset
    print(f"Training tokenizer on {data_path}...")
    tokenizer.train(files=[data_path], trainer=trainer)

    # 6. Save it for Phase 3
    tokenizer.save(output_path)
    print(f"Tokenizer successfully saved to {output_path}!")

    # --- TEST THE TOKENIZER ---
    print("\n--- Testing the Tokenizer ---")
    test_string = "<OBSERVE> Env_ls: [/logs]. File: 'app_42.log'. </OBSERVE>\n<THINK> Target is /logs. </THINK>\n<ACT> MOVE app_42.log /logs </ACT><|endoftext|>"
    
    encoded = tokenizer.encode(test_string)
    
    print(f"Original String:\n{test_string}\n")
    print(f"Token IDs:\n{encoded.ids}\n")
    print(f"Tokens (Notice how the tags stayed intact!):\n{encoded.tokens}")

if __name__ == "__main__":
    train_custom_tokenizer()