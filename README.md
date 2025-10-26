# Local AI Code Assistant - Demo Guide

## Overview

This is a proof-of-concept demonstration for the IEEE Computer Society talk on "Getting Started with Local AI Agents - Code-Assist Copilot-like". It shows how to build a functional code assistant that runs entirely on your local machine using open-source LLMs.

## Features

- **Code Generation**: Generate code from natural language descriptions
- **Code Explanation**: Understand what existing code does
- **Debugging**: Get help fixing bugs with detailed analysis
- **Documentation**: Auto-generate docstrings and comments
- **Refactoring**: Improve code quality and structure
- **Code Review**: Get feedback on code quality and best practices
- **Optimization**: Improve performance and efficiency
- **Test Generation**: Create comprehensive unit tests

## Prerequisites

### Hardware Requirements

**Minimum (7B models):**
- 8GB RAM
- 10GB free disk space
- CPU-based inference (slower)

**Recommended (13B+ models):**
- 16GB+ RAM
- GPU with 8GB+ VRAM (NVIDIA CUDA or Apple Metal)
- 20GB+ free disk space

### Software Requirements

- Python 3.7 or higher
- Ollama (see installation below)
- `requests` library

## Installation

### Step 1: Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download the installer from https://ollama.com/download

**Verify installation:**
```bash
ollama --version
```

### Step 2: Pull a Code Model

Start with a smaller model for testing:
```bash
# Fast, good for most tasks (recommended for demo)
ollama pull codellama:7b

# Alternative: Better coding capabilities
ollama pull deepseek-coder:6.7b

# For better quality (requires 16GB RAM)
ollama pull codellama:13b
```

### Step 3: Install Python Dependencies

```bash
pip install requests
```

### Step 4: Download the Demo Code

Save the `local_code_assistant.py` file from the presentation, or copy it from the slides.

## 🚀 Usage

### Starting Ollama

Ollama runs as a service, but you can start it manually:
```bash
ollama serve
```

Leave this running in a terminal window.

### Running the Code Assistant

In a new terminal:
```bash
python local_code_assistant.py
```

Or specify a different model:
```bash
python local_code_assistant.py deepseek-coder:6.7b
```

### Available Commands

Once running, you can use these commands:

- `generate` - Generate code from a description
- `explain` - Explain what code does
- `debug` - Debug code with optional error message
- `document` - Add documentation to code
- `refactor` - Refactor for better quality
- `review` - Perform code review
- `optimize` - Optimize for performance
- `test` - Generate unit tests
- `help` - Show help message
- `quit` - Exit

## 📖 Example Session

```
🔧 Command> generate
  📝 Description: binary search algorithm with type hints
  💻 Language (default: python): 
🔄 Generating code...

def binary_search(arr: list[int], target: int) -> int:
    """
    Perform binary search on a sorted array.
    
    Args:
        arr: Sorted list of integers
        target: Value to search for
    
    Returns:
        Index of target if found, -1 otherwise
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

🔧 Command> explain
  📋 Paste your code:
[paste the binary search code]

🔄 Analyzing code...

This function implements the binary search algorithm:
1. Initialize two pointers: left at start, right at end
2. Loop while search space is valid
3. Calculate middle index
4. Compare middle element with target
5. Adjust search space based on comparison
6. Return index if found, -1 if not found

Time complexity: O(log n)
Space complexity: O(1)
```

## Demo Flow for Presentation

### Demo 1: Code Generation
```
Command: generate
Description: Python function to check if a string is a palindrome
Language: python
```

### Demo 2: Add Documentation
```
Command: document
[Paste the generated palindrome function]
```

### Demo 3: Code Review
```
Command: review
[Paste any code with potential issues]
```

### Demo 4: Generate Tests
```
Command: test
[Paste a function to test]
```

## 🔍 Troubleshooting

### "Cannot connect to Ollama"
- Make sure Ollama is running: `ollama serve`
- Check if it's listening on port 11434: `curl http://localhost:11434`

### "Model not found"
- Pull the model first: `ollama pull codellama:7b`
- Check available models: `ollama list`

### Slow responses
- First request loads model into memory (10-15 seconds)
- Subsequent requests are faster (1-5 seconds)
- Consider using a smaller model or GPU acceleration

### Out of memory errors
- Use smaller models (7B instead of 13B)
- Close other applications
- Use quantized models: `ollama pull codellama:7b-q4`

## 📊 Model Comparison

| Model | Size | RAM Needed | Strengths | Best For |
|-------|------|------------|-----------|----------|
| CodeLlama 7B | 3.8GB | 8GB | Fast, good snippets | Demos, quick tasks |
| DeepSeek Coder 6.7B | 3.8GB | 8GB | Strong coding | General development |
| CodeLlama 13B | 7.4GB | 16GB | Better reasoning | Complex tasks |
| DeepSeek Coder 33B | 19GB | 32GB | Production quality | Enterprise use |
| Qwen2.5-Coder 7B | 4.7GB | 8GB | Multilingual | International teams |

## 🚀 Advanced Usage

### Using Different Models

```bash
# List available models
ollama list

# Pull a specific model
ollama pull qwen2.5-coder:7b

# Use in the assistant
python local_code_assistant.py qwen2.5-coder:7b
```

### GPU Acceleration

Ollama automatically detects and uses GPU if available:
- **NVIDIA**: Requires CUDA toolkit
- **Apple Silicon**: Uses Metal automatically
- **AMD**: Limited support on Linux

Check GPU usage:
```bash
# NVIDIA
nvidia-smi

# Apple Silicon
sudo powermetrics --samplers gpu_power
```

### Custom System Prompts

Modify the code to add custom behaviors:

```python
system = "You are a Python expert who writes code following PEP 8 strictly."
```

### Integration with IDEs

**VSCode with Continue.dev:**
1. Install Continue.dev extension
2. Configure to use Ollama
3. Get inline completions and chat

**Vim/Neovim:**
- Use vim-ollama or similar plugins
- Configure with local Ollama endpoint

## 🔗 Additional Resources

### Tools
- [Ollama](https://ollama.com) - Local LLM runtime
- [Continue.dev](https://continue.dev) - IDE extension
- [LM Studio](https://lmstudio.ai) - GUI alternative to Ollama

### Models
- [Ollama Model Library](https://ollama.com/library)
- [HuggingFace](https://huggingface.co/models)

### Learning
- [Ollama Documentation](https://github.com/ollama/ollama/tree/main/docs)
- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)
- [Awesome Local LLM](https://github.com/topics/local-llm)

## 📝 License

This demo code is provided for educational purposes as part of the IEEE Computer Society presentation.

## 🤝 Contributing

Feel free to extend this demo with:
- More specialized commands (SQL generation, regex help, etc.)
- Better error handling
- Streaming responses
- RAG integration for codebase context
- Web interface

## 💡 Next Steps

1. Try different models to compare quality
2. Integrate with your favorite IDE
3. Add RAG for codebase-aware assistance
4. Build custom agents for your specific needs
5. Experiment with fine-tuning for your domain
