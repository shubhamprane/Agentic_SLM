import random
import os

# --- 1. THE DICTIONARIES (Expanded Action Space) ---
FOLDERS = ["/logs", "/src", "/media", "/docs", "/data"]

FILE_TYPES = {
    "log": {
        "exts": [".log", ".txt", ""],
        "signals": ["[ERROR]", "Traceback", "kernel panic", "timeout", "WARN"],
        "target": "/logs"
    },
    "code": {
        "exts": [".py", ".js", ".html", ""],
        "signals": ["def main():", "import os", "console.log", "function()"],
        "target": "/src"
    },
    "image": {
        "exts": [".png", ".jpg", ""],
        "signals": ["Resolution: 1920x1080", "RGB", "EXIF Data", "Pixels"],
        "target": "/media"
    },
    "document": {
        "exts": [".md", ".pdf", ".txt", ""],
        "signals": ["Meeting notes", "Dear team,", "Chapter 1", "Summary:"],
        "target": "/docs"
    },
    "junk": { # NEW: Teaches the DELETE action
        "exts": [".tmp", ".bak", ".cache"],
        "signals": ["temp data", "cache dump", "backup corrupted", "ignore"],
        "target": "DELETE"
    }
}

FILE_NAMES = ["app", "server", "user", "backup", "data", "draft", "final", "config", "test"]
DISTRACTORS = ["the", "and", "this", "is", "a", "quick", "test", "of", "system"]

# --- 2. THE GENERATOR LOGIC ---
def generate_sequence():
    true_type = random.choice(list(FILE_TYPES.keys()))
    target_dir = FILE_TYPES[true_type]["target"]
    
    name_base = random.choice(FILE_NAMES) + "_" + str(random.randint(1, 99))
    ext = random.choice([e for e in FILE_TYPES[true_type]["exts"] if e != ""])
    file_name = f"{name_base}{ext}"
    
    signal_word = random.choice(FILE_TYPES[true_type]["signals"])
    noise = " ".join(random.choices(DISTRACTORS, k=random.randint(3, 8)))
    content = f"{noise} {signal_word} {noise}"
    
    # Generate the Environment State (ls simulation)
    # We randomly decide if the required target folder is missing to teach MKDIR
    env_state = []
    other_dirs = [d for d in FOLDERS if d != target_dir]
    env_state.extend(random.sample(other_dirs, k=random.randint(1, 3)))
    
    if target_dir == "DELETE":
        # Junk files don't need a folder, they just get deleted.
        thought = f"This file has a {ext} extension and contains junk/cache signals. It is not needed."
        act = f"<ACT> DELETE {file_name} </ACT>"
        
    else:
        # Decide if the folder exists in the environment
        folder_exists = random.choice([True, False])
        
        if folder_exists:
            env_state.append(target_dir)
            random.shuffle(env_state) # Shuffle so it doesn't memorize positions
            thought = f"Content indicates this is a {true_type} file. Target is {target_dir}. The folder {target_dir} exists in the environment. I can move it."
            act = f"<ACT> MOVE {file_name} {target_dir} </ACT>"
        else:
            random.shuffle(env_state)
            thought = f"Content indicates this is a {true_type} file. It belongs in {target_dir}. Checking environment... {target_dir} does not exist. I must create the folder first."
            act = f"<ACT> MKDIR {target_dir} </ACT>"

    # Format the environment to look like a list of current directories
    env_str = f"[{', '.join(env_state)}]" if env_state else "[]"

    observe_block = f"<OBSERVE> Env_ls: {env_str}. File: '{file_name}'. Content: '{content.strip()}' </OBSERVE>"
    think_block = f"<THINK> {thought} </THINK>"
    
    return f"{observe_block}\n{think_block}\n{act}\n"

# --- 3. THE EXECUTION ---
def build_dataset(num_samples, output_path):
    print(f"Generating {num_samples} advanced agentic sequences...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(num_samples):
            sequence = generate_sequence()
            f.write(sequence + "<|endoftext|>\n")
            if (i + 1) % 10000 == 0:
                print(f"Generated {i + 1} / {num_samples} sequences...")

    print(f"Dataset successfully saved to {output_path}!")

if __name__ == "__main__":
    # Bulletproof pathing: Saves to the data folder relative to THIS script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(script_dir, "..", "data", "synthetic_dataset.txt")
    
    # Generate 20 test sequences so you can read the new logic
    build_dataset(100000, DATA_PATH)
