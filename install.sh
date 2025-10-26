#!/bin/bash

if command -v "ollama" &>/dev/null; then
  echo -e "Ollama is installed.."
else
  echo -e "Installing Ollama"
  curl -fsSL https://ollama.com/install.sh | sh
fi

# Pulling the necessary models for this project
ollama pull codellama:7b

# Create the Python virtual environment
VENV_DIR=".venv"

# Check if the virtual environment already exists
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment '$VENV_DIR'..."
  python3 -m venv "$VENV_DIR"
  echo "Virtual environment created."
else
  echo "Virtual environment '$VENV_DIR' already exists."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install the necessary dependencies
pip install requests chromadb sentence-transformers torch 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118