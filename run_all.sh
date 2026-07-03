#!/bin/bash

# Exit immediately if any script fails
set -e

echo "=========================================="
echo "  Agentic Janitor - Pipeline Execution    "
echo "=========================================="

# Check to make sure we are in the right directory
if [ ! -d "src" ]; then
  echo "Error: 'src' directory not found. Please run this from the root of your project."
  exit 1
fi

echo -e "\n---> [1/5] Generating Synthetic Data..."
python3 src/data_gen.py

echo -e "\n---> [2/5] Training Tokenizer..."
python3 src/tokenizer.py

echo -e "\n---> [3/5] Compiling and Testing Model..."
python3 src/model.py

echo -e "\n---> [4/5] Training the SLM..."
python3 src/train.py

echo -e "\n---> [5/5] Deploying Agent to Sandbox..."
python3 src/agent.py

echo -e "\n=========================================="
echo "  Pipeline Completed Successfully!        "
echo "=========================================="
