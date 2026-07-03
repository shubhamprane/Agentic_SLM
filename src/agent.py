import os
import shutil
import torch
import re
from tokenizers import Tokenizer
from phase3_model import ModelArgs, AgenticSLM

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. SETUP THE JAIL (SANDBOX) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SANDBOX_DIR = os.path.join(SCRIPT_DIR, "..", "sandbox")

# --- 1. SETUP THE JAIL (SANDBOX) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SANDBOX_DIR = os.path.join(SCRIPT_DIR, "..", "sandbox")

def reset_sandbox():
    if os.path.exists(SANDBOX_DIR):
        shutil.rmtree(SANDBOX_DIR)
    os.makedirs(SANDBOX_DIR)
    
    files_to_create = {
        "server_42.log": "This is a test file. [ERROR] connection timeout on port 8080.",
        "script_99": "import os\ndef main():\n    print('Hello World')", 
        "vacation.bak": "cache dump ignore this corrupted data",         
    }
    
    for filename, content in files_to_create.items():
        with open(os.path.join(SANDBOX_DIR, filename), "w") as f:
            f.write(content)
            
    # We removed the pre-created /logs folder. 
    # The agent must now create all necessary infrastructure itself.
    print(f"Sandbox reset and populated with 3 raw files at: {SANDBOX_DIR}")

# --- 2. LOAD THE BRAIN ---
def load_agent():
    tok_path = os.path.join(SCRIPT_DIR, "..", "data", "custom_tokenizer.json")
    model_path = os.path.join(SCRIPT_DIR, "..", "data", "agentic_janitor.pt")
    
    tokenizer = Tokenizer.from_file(tok_path)
    config = ModelArgs()
    config.vocab_size = tokenizer.get_vocab_size()
    
    model = AgenticSLM(config)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval() 
    return model, tokenizer

# --- 3. GENERATION LOOP ---
def generate_action(model, tokenizer, prompt, max_tokens=100):
    input_ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    stop_tokens = [tokenizer.token_to_id("</ACT>"), tokenizer.token_to_id("<|endoftext|>")]
    
    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond = idx[:, -model.config.max_seq_len:]
            logits, _ = model(idx_cond)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)
            if next_token.item() in stop_tokens:
                break
                
    return tokenizer.decode(idx[0].tolist(), skip_special_tokens=False)

# --- 4. THE HANDS (Bulletproof Parsing) ---
def execute_command(generated_text, target_file_name):
    match = re.search(r'<ACT>\s*(.*?)\s*</ACT>', generated_text)
    if not match: return "ERROR: Could not find valid <ACT> tags."
        
    cmd_string = match.group(1).strip()
    parts = cmd_string.split()
    if not parts: return "ERROR: Empty command."
    
    # Clean out any standalone slashes caused by tokenizer spaces
    clean_parts = [p.strip('/') for p in parts if p.strip('/') != '']
    action = clean_parts[0].upper()
    
    try:
        if action == "MOVE" and len(clean_parts) >= 2:
            target_dir = clean_parts[-1] 
            src = os.path.join(SANDBOX_DIR, target_file_name)
            dst = os.path.join(SANDBOX_DIR, target_dir)
            
            if not os.path.exists(src): return f"ERROR: File {target_file_name} not found."
            if not os.path.exists(dst): return f"ERROR: Directory {target_dir} does not exist."
            
            shutil.move(src, os.path.join(dst, target_file_name))
            return f"SUCCESS: Moved {target_file_name} to {target_dir}"
            
        elif action == "MKDIR" and len(clean_parts) >= 2:
            target_dir = clean_parts[-1]
            dst = os.path.join(SANDBOX_DIR, target_dir)
            os.makedirs(dst, exist_ok=True)
            return f"SUCCESS: Created directory {target_dir}"
            
        elif action == "DELETE":
            src = os.path.join(SANDBOX_DIR, target_file_name)
            if os.path.exists(src):
                os.remove(src)
            return f"SUCCESS: Deleted {target_file_name}"
            
        else:
            return f"ERROR: Unrecognized command syntax -> {cmd_string}"
            
    except Exception as e:
        return f"ERROR: OS level exception - {str(e)}"

# --- 5. TRUE MULTI-STEP AGENTIC LOOP ---
def run_agent():
    reset_sandbox()
    model, tokenizer = load_agent()
    print("\n--- Agent Initialized. Scanning Sandbox ---")
    
    # Outer Loop: Scans the whole directory
    for _ in range(10): 
        items = os.listdir(SANDBOX_DIR)
        folders = [f"/{item}" for item in items if os.path.isdir(os.path.join(SANDBOX_DIR, item))]
        files = [item for item in items if os.path.isfile(os.path.join(SANDBOX_DIR, item))]
        
        if not files:
            print("\n[✓] Sandbox is entirely clean! Agent finished its job.")
            break
            
        target_file = files[0]
        print(f"\n==============================================")
        print(f"Targeting File: {target_file}")
        print(f"==============================================")
        
        # Inner Loop: Multi-step actions for ONE specific file
        for step in range(3): 
            # 1. Refresh environment (in case we just created a directory!)
            items = os.listdir(SANDBOX_DIR)
            current_folders = [f"/{item}" for item in items if os.path.isdir(os.path.join(SANDBOX_DIR, item))]
            env_str = f"[{', '.join(current_folders)}]" if current_folders else "[]"
            
            # 2. Safety check: Did we already move/delete it in the last step?
            if not os.path.exists(os.path.join(SANDBOX_DIR, target_file)):
                break
                
            with open(os.path.join(SANDBOX_DIR, target_file), "r") as f:
                content_snippet = f.read(50).strip() 
                
            prompt = f"<OBSERVE> Env_ls: {env_str}. File: '{target_file}'. Content: '{content_snippet}' </OBSERVE>\n"
            
            print(f"\n  [Step {step+1}]")
            print(f"  Observation: Env has {env_str}")
            
            # 3. Generate Action
            output = generate_action(model, tokenizer, prompt)
            
            # 4. Clean up the printed text so it doesn't look like gibberish
            response = output.replace(prompt, "").strip()
            clean_print = response.replace(" .", ".").replace(" / ", "/").replace(" '", "'").replace("' ", "'")
            print(f"  Agent Thinks & Acts:\n  {clean_print}")
            
            # 5. Execute
            result = execute_command(output, target_file)
            print(f"  System Execution: {result}")
            
            # 6. Evaluate Result
            if "SUCCESS: Created" in result:
                print("  [+] Directory created. Looping back to check environment again...")
                continue # This triggers the multi-step loop you asked for!
            elif "SUCCESS: Moved" in result or "SUCCESS: Deleted" in result:
                print("  [✓] File successfully handled.")
                break # Break inner loop, move to next file
            else:
                print("  [!] Error encountered. Breaking to avoid infinite loop.")
                break

if __name__ == "__main__":
    run_agent()
