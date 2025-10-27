#!/usr/bin/env python3
"""
Local Code Assistant - A proof-of-concept CLI tool for AI-assisted coding
Uses Ollama to run local LLMs for code generation, debugging, and more.

Requirements:
    - Ollama installed and running (https://ollama.com)
    - A code model pulled (e.g., ollama pull codellama:7b)
    - Python 3.7+
    - requests library: pip install requests

Usage:
    python local_code_assistant.py
"""

import requests
import json
import sys
from typing import Optional


class LocalCodeAssistant:
    """Interface to local LLM for code assistance tasks."""
    
    def __init__(self, model: str = "codellama:7b", base_url: str = "http://localhost:11434"):
        """
        Initialize the code assistant.
        
        Args:
            model: The Ollama model to use (default: codellama:7b)
            base_url: The Ollama API URL (default: http://localhost:11434)
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7) -> str:
        """
        Send a prompt to the local LLM and get a response.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt to set context/behavior
            temperature: Sampling temperature (0.0-1.0, higher = more creative)
            
        Returns:
            The model's response as a string
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            print(f"💭 Thinking... (using {self.model})")
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.ConnectionError:
            return "❌ Error: Cannot connect to Ollama. Is it running? Try: ollama serve"
        except requests.exceptions.Timeout:
            return "❌ Error: Request timed out. The model might be too large or slow."
        except requests.exceptions.RequestException as e:
            return f"❌ Error: {e}"
        
    @staticmethod
    def _wrap_template(template: str, **kwargs: str) -> str:
        """Simple keyword substitution; keeps the prompt templates tidy."""
        return template.format(**kwargs)
    
    def generate_code(self, description: str, language: str = "python") -> str:
        system = (
            "You are an expert software engineer. "
            "Please write a {lang} program that {desc}."
            "Handle exceptions gracefully and return meaningful error codes/messages. "
            "Return only the code files, each preceded by a clear comment with the file name. "
            "Do not wrap the code in markdown fences or add extra commentary."
        )
        prompt = f"Write a {language} program that {description}."
        system = self._wrap_template(system, lang=language, desc=description)
        return self.generate(prompt, system, temperature=0.3)

    def refactor_code(self, code: str, language: str = "python") -> str:
        system = (
            "You are a code quality expert specializing in {lang}. "
            "Refactor code for better readability, performance, and maintainability."
        )
        prompt = (
            "Refactor this {lang} code:\n\n{code}\n\n"
            "Improvements to consider:\n"
            "- Code structure and organization\n"
            "- Performance optimizations\n"
            "- Readability and naming\n"
            "- Best practices and patterns"
        )
        system = self._wrap_template(system, lang=language)
        prompt = self._wrap_template(prompt, lang=language, code=code)
        return self.generate(prompt, system, temperature=0.4)

    def code_review(self, code: str, language: str = "python") -> str:
        system = (
            "You are a senior {lang} developer conducting a code review. "
            "Provide constructive feedback on code quality, security, and best practices."
        )
        prompt = (
            "Review this {lang} code:\n\n{code}\n\n"
            "Provide feedback on:\n"
            "1. Code quality and style\n"
            "2. Potential bugs or issues\n"
            "3. Performance considerations\n"
            "4. Security concerns\n"
            "5. Suggestions for improvement"
        )
        system = self._wrap_template(system, lang=language)
        prompt = self._wrap_template(prompt, lang=language, code=code)
        return self.generate(prompt, system, temperature=0.5)

    def optimize_code(self, code: str, language: str = "python") -> str:
        system = (
            "You are a performance optimization expert for {lang}. "
            "Improve code efficiency while maintaining correctness."
        )
        prompt = (
            "Optimize this {lang} code for performance:\n\n{code}\n\n"
            "Focus on:\n"
            "- Time complexity\n"
            "- Space complexity\n"
            "- Algorithmic improvements\n"
            "Explain the optimizations made."
        )
        system = self._wrap_template(system, lang=language)
        prompt = self._wrap_template(prompt, lang=language, code=code)
        return self.generate(prompt, system, temperature=0.3)

    def test_code(self, code: str, language: str = "python") -> str:
        system = (
            "You are a test automation expert for {lang}. "
            "Write comprehensive unit tests with edge cases."
        )
        prompt = (
            "Generate unit tests for this {lang} code:\n\n{code}\n\n"
            "Include:\n"
            "- Normal cases\n"
            "- Edge cases\n"
            "- Error cases\n"
            "- Appropriate test framework (pytest for Python, JUnit for Java, etc.)"
        )
        system = self._wrap_template(system, lang=language)
        prompt = self._wrap_template(prompt, lang=language, code=code)
        return self.generate(prompt, system, temperature=0.3)

    def explain_code(self, code: str, language: str = "python") -> str:
        system = (
            "You are a senior {lang} developer explaining code. "
            "Provide a detailed analysis of the code."
        )
        prompt = (
            "Explain this {lang} code:\n\n{code}\n\n"
            "Cover:\n"
            "1. Structure and intention\n"
            "2. Key algorithms\n"
            "3. Possible improvements\n"
            "4. Common pitfalls"
        )
        system = self._wrap_template(system, lang=language)
        prompt = self._wrap_template(prompt, lang=language, code=code)
        return self.generate(prompt, system, temperature=0.3)

    def debug_code(self, code: str, language: str = "python") -> str:
        system = (
            "You are a senior software engineer for {lang}. "
            "Help debug the given code."
        )
        prompt = (
            "Debug the following {lang} code:\n\n{code}\n\n"
            "Verify:\n"
            "- The code has a logical or syntax error\n"
            "- If there is an error, provide a fix\n"
            "- Suggested test cases\n"
            "- Recommendations for robustness"
        )
        system = self._wrap_template(system, lang=language)
        prompt = self._wrap_template(prompt, lang=language, code=code)
        return self.generate(prompt, system, temperature=0.3)

    def add_documentation(self, code: str, language: str = "python") -> str:
        system = (
            "You are a senior software engineer for {lang}. "
            "Write detailed documentation for the given code."
        )
        prompt = (
            "Generate comprehensive documentation for this {lang} code:\n\n{code}\n\n"
            "Include:\n"
            "- Function documentation\n"
            "- Parameter details\n"
            "- In‑line comments for clarity\n"
            "- Module‑level description\n"
        )
        system = self._wrap_template(system, lang=language)
        prompt = self._wrap_template(prompt, lang=language, code=code)
        return self.generate(prompt, system, temperature=0.3)


def print_header():
    """Print the CLI header."""
    print("\n" + "="*70)
    print("🤖 LOCAL CODE ASSISTANT - Powered by Local LLM")
    print("="*70)
    print("Commands:")
    print("  generate   - Generate code from description")
    print("  explain    - Explain what code does")
    print("  debug      - Debug code with optional error message")
    print("  document   - Add documentation to code")
    print("  refactor   - Refactor code for better quality")
    print("  review     - Perform code review")
    print("  optimize   - Optimize code for performance")
    print("  test       - Generate unit tests")
    print("  help       - Show this help message")
    print("  quit       - Exit the program")
    print("="*70 + "\n")


def get_multiline_input(prompt: str) -> str:
    """
    Get multi-line input from user.
    
    Args:
        prompt: The prompt to display
        
    Returns:
        The complete input as a string
    """
    print(prompt)
    print("(Paste your code and press Enter twice when done)")
    lines = []
    empty_line_count = 0
    
    while True:
        try:
            line = input()
            if line == "":
                empty_line_count += 1
                if empty_line_count >= 2:
                    break
            else:
                empty_line_count = 0
                lines.append(line)
        except EOFError:
            break
    
    return "\n".join(lines)


def main():
    """Main CLI loop."""
    # Check if custom model specified
    model = sys.argv[1] if len(sys.argv) > 1 else "codellama:7b"
    
    try:
        print(f"Initializing with model: {model}")
        assistant = LocalCodeAssistant(model=model)
        print(f"✅ Initialized with model: {model}")
        print_header()
        
        while True:
            try:
                command = input("\n🔧 Command> ").strip().lower()
                
                if command == "quit" or command == "exit":
                    print("\n👋 Goodbye!")
                    break
                    
                elif command == "help":
                    print_header()
                    
                elif command == "generate":
                    description = input("  📝 Description: ")
                    language = input("  💻 Language (default: python): ").strip() or "python"
                    print("\n🔄 Generating code...\n")
                    result = assistant.generate_code(description, language)
                    print(f"\n{'='*70}\n{result}\n{'='*70}")
                    
                elif command == "explain":
                    code = get_multiline_input("\n  📋 Paste your code:")
                    language = input("  💻 Language (default: python): ").strip() or "python"
                    print("\n🔄 Analyzing code...\n")
                    result = assistant.explain_code(code, language)
                    print(f"\n{'='*70}\n{result}\n{'='*70}")
                    
                elif command == "debug":
                    code = get_multiline_input("\n  📋 Paste your code:")
                    language = input("  💻 Language (default: python): ").strip() or "python"
                    print("\n🔄 Debugging code...\n")
                    result = assistant.debug_code(code, language)
                    print(f"\n{'='*70}\n{result}\n{'='*70}")
                    
                elif command == "document":
                    code = get_multiline_input("\n  📋 Paste your code:")
                    language = input("  💻 Language (default: python): ").strip() or "python"
                    print("\n🔄 Adding documentation...\n")
                    result = assistant.add_documentation(code, language)
                    print(f"\n{'='*70}\n{result}\n{'='*70}")
                    
                elif command == "refactor":
                    code = get_multiline_input("\n  📋 Paste your code:")
                    language = input("  💻 Language (default: python): ").strip() or "python"
                    print("\n🔄 Refactoring code...\n")
                    result = assistant.refactor_code(code, language)
                    print(f"\n{'='*70}\n{result}\n{'='*70}")
                    
                elif command == "review":
                    code = get_multiline_input("\n  📋 Paste your code:")
                    language = input("  💻 Language (default: python): ").strip() or "python"
                    print("\n🔄 Reviewing code...\n")
                    result = assistant.code_review(code, language)
                    print(f"\n{'='*70}\n{result}\n{'='*70}")
                    
                elif command == "optimize":
                    code = get_multiline_input("\n  📋 Paste your code:")
                    language = input("  💻 Language (default: python): ").strip() or "python"
                    print("\n🔄 Optimizing code...\n")
                    result = assistant.optimize_code(code, language)
                    print(f"\n{'='*70}\n{result}\n{'='*70}")
                    
                elif command == "test":
                    code = get_multiline_input("\n  📋 Paste your code:")
                    language = input("  💻 Language (default: python): ").strip() or "python"
                    print("\n🔄 Generating tests...\n")
                    result = assistant.test_code(code, language)
                    print(f"\n{'='*70}\n{result}\n{'='*70}")
                    
                else:
                    print("❌ Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted. Type 'quit' to exit.")
                continue
                
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()