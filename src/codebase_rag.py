#!/usr/bin/env python3
"""
RAG-Enhanced Code Assistant - Codebase-Aware AI Agent

This implementation demonstrates Retrieval-Augmented Generation (RAG) for code.
The agent can understand and answer questions about your entire codebase by:
1. Indexing your code files with embeddings
2. Finding relevant code when you ask questions
3. Providing context to the LLM for accurate answers

Requirements:
    pip install requests chromadb sentence-transformers torch
    ollama pull codellama:7b
"""

import os
import re
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class CodebaseRAG:
    """
    RAG system for codebase understanding.
    
    This class handles:
    - Code file discovery and parsing
    - Embedding generation for code chunks
    - Semantic search for relevant code
    - Context-aware LLM queries
    """
    
    def __init__(
        self,
        codebase_path: str,
        model_name: str = "codellama:7b",
        ollama_url: str = "http://localhost:11434",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the RAG system.
        
        Args:
            codebase_path: Path to the codebase directory
            model_name: Ollama model to use
            ollama_url: Ollama API URL
            embedding_model: SentenceTransformer model for embeddings
        """
        self.codebase_path = Path(codebase_path)
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_url = f"{ollama_url}/api/generate"
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="codebase",
            metadata={"description": "Code chunks from the codebase"}
        )
        
    def parse_code_file(self, file_path: Path) -> List[Dict[str, str]]:
        """
        Parse a code file into meaningful chunks.
        
        Args:
            file_path: Path to the code file
            
        Returns:
            List of dictionaries containing code chunks and metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (UnicodeDecodeError, PermissionError):
            return []
        
        chunks = []
        
        # Strategy 1: Split by functions/classes (Python example)
        if file_path.suffix == '.py':
            chunks.extend(self._parse_python_file(content, file_path))
        
        # Strategy 2: Split by larger logical blocks for other languages
        else:
            chunks.extend(self._parse_generic_file(content, file_path))
        
        return chunks
    
    def _parse_python_file(self, content: str, file_path: Path) -> List[Dict[str, str]]:
        """Parse Python file into function/class chunks."""
        chunks = []
        lines = content.split('\n')
        
        # Simple regex patterns for Python structures
        class_pattern = re.compile(r'^class\s+(\w+)')
        func_pattern = re.compile(r'^def\s+(\w+)')
        
        current_chunk = []
        current_name = None
        current_type = None
        indent_level = 0
        
        for i, line in enumerate(lines):
            # Check for class definition
            class_match = class_pattern.match(line)
            if class_match:
                # Save previous chunk
                if current_chunk:
                    chunks.append({
                        'content': '\n'.join(current_chunk),
                        'file': str(file_path),
                        'name': current_name,
                        'type': current_type,
                        'line': i - len(current_chunk) + 1
                    })
                current_chunk = [line]
                current_name = class_match.group(1)
                current_type = 'class'
                indent_level = len(line) - len(line.lstrip())
                continue
            
            # Check for function definition
            func_match = func_pattern.match(line)
            if func_match:
                # Save previous chunk if at same or lower indent
                if current_chunk and len(line) - len(line.lstrip()) <= indent_level:
                    chunks.append({
                        'content': '\n'.join(current_chunk),
                        'file': str(file_path),
                        'name': current_name,
                        'type': current_type,
                        'line': i - len(current_chunk) + 1
                    })
                    current_chunk = []
                
                if not current_chunk or len(line) - len(line.lstrip()) <= indent_level:
                    current_chunk = [line]
                    current_name = func_match.group(1)
                    current_type = 'function'
                    indent_level = len(line) - len(line.lstrip())
                else:
                    current_chunk.append(line)
                continue
            
            # Add line to current chunk
            if current_chunk:
                current_chunk.append(line)
        
        # Save final chunk
        if current_chunk:
            chunks.append({
                'content': '\n'.join(current_chunk),
                'file': str(file_path),
                'name': current_name,
                'type': current_type,
                'line': len(lines) - len(current_chunk) + 1
            })
        
        return chunks
    
    def _parse_generic_file(self, content: str, file_path: Path) -> List[Dict[str, str]]:
        """Parse non-Python files into fixed-size chunks with overlap."""
        chunks = []
        lines = content.split('\n')
        
        chunk_size = 50  # lines per chunk
        overlap = 10     # lines of overlap
        
        for i in range(0, len(lines), chunk_size - overlap):
            chunk_lines = lines[i:i + chunk_size]
            if chunk_lines:
                chunks.append({
                    'content': '\n'.join(chunk_lines),
                    'file': str(file_path),
                    'name': f'chunk_{i}',
                    'type': 'code_block',
                    'line': i + 1
                })
        
        return chunks
    
    def index_codebase(self, file_extensions: List[str] = None) -> int:
        """
        Index all code files in the codebase.
        
        Args:
            file_extensions: List of extensions to index (e.g., ['.py', '.js'])
                           If None, indexes common programming languages
        
        Returns:
            Number of chunks indexed
        """
        if file_extensions is None:
            file_extensions = [
                '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h',
                '.cs', '.rb', '.go', '.rs', '.php', '.swift', '.kt'
            ]
        
        print(f"🔍 Scanning codebase: {self.codebase_path}")
        
        all_chunks = []
        file_count = 0
        
        # Find all code files
        for ext in file_extensions:
            for file_path in self.codebase_path.rglob(f'*{ext}'):
                # Skip common directories to ignore
                if any(skip in str(file_path) for skip in [
                    'venv', 'node_modules', '__pycache__', '.git',
                    'build', 'dist', '.pytest_cache'
                ]):
                    continue
                
                print(f"  Parsing: {file_path.relative_to(self.codebase_path)}")
                chunks = self.parse_code_file(file_path)
                all_chunks.extend(chunks)
                file_count += 1
        
        print(f"\n📊 Found {file_count} files, {len(all_chunks)} code chunks")
        
        if not all_chunks:
            print("⚠️  No code chunks found!")
            return 0
        
        # Generate embeddings and store in ChromaDB
        print("🧮 Generating embeddings...")
        
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            
            # Create enhanced content for better embedding
            enhanced_contents = []
            for chunk in batch:
                enhanced = f"File: {chunk['file']}\n"
                if chunk['name']:
                    enhanced += f"{chunk['type']}: {chunk['name']}\n"
                enhanced += f"Code:\n{chunk['content']}"
                enhanced_contents.append(enhanced)
            
            # Generate embeddings
            embeddings = self.embedder.encode(enhanced_contents).tolist()
            
            # Prepare data for ChromaDB
            ids = [f"chunk_{i+j}" for j in range(len(batch))]
            metadatas = [{
                'file': chunk['file'],
                'name': chunk.get('name', ''),
                'type': chunk.get('type', ''),
                'line': chunk.get('line', 0)
            } for chunk in batch]
            documents = [chunk['content'] for chunk in batch]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            print(f"  Indexed {i + len(batch)}/{len(all_chunks)} chunks")
        
        print(f"✅ Indexing complete! {len(all_chunks)} chunks stored.")
        return len(all_chunks)
    
    def search_codebase(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for relevant code using semantic similarity.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_dict: Optional metadata filters (e.g., {'file': 'main.py'})
        
        Returns:
            List of relevant code chunks with metadata
        """
        # Generate query embedding
        query_embedding = self.embedder.encode([query]).tolist()
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=filter_dict
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results
    
    def ask_codebase(self, question: str, n_context: int = 3) -> str:
        """
        Ask a question about the codebase using RAG.
        
        Args:
            question: Question about the codebase
            n_context: Number of code chunks to use as context
        
        Returns:
            Answer from the LLM with codebase context
        """
        # Search for relevant code
        print(f"🔍 Finding relevant code...")
        results = self.search_codebase(question, n_results=n_context)
        
        if not results:
            return "❌ No relevant code found in the codebase."
        
        # Build context from results
        context_parts = []
        for i, result in enumerate(results, 1):
            meta = result['metadata']
            context_parts.append(
                f"--- Code Snippet {i} ---\n"
                f"File: {meta['file']}\n"
                f"Type: {meta.get('type', 'N/A')} | "
                f"Name: {meta.get('name', 'N/A')} | "
                f"Line: {meta.get('line', 'N/A')}\n\n"
                f"{result['content']}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Create prompt with context
        system_prompt = (
            "You are a helpful code assistant with access to the user's codebase. "
            "Answer questions accurately based on the provided code context. "
            "Reference specific files, functions, or line numbers when relevant. "
            "If the context doesn't contain enough information, say so."
        )
        
        user_prompt = (
            f"CODEBASE CONTEXT:\n\n{context}\n\n"
            f"---\n\n"
            f"QUESTION: {question}\n\n"
            f"Provide a detailed answer based on the code context above."
        )
        
        # Query LLM
        print(f"💭 Thinking with {len(results)} code snippets as context...")
        payload = {
            "model": self.model_name,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_ctx": 4096  # Larger context window for code
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            answer = response.json()["response"]
            
            # Add source references
            sources = "\n\n📚 Sources:\n"
            for result in results:
                meta = result['metadata']
                sources += f"  • {meta['file']} (line {meta.get('line', '?')})\n"
            
            return answer + sources
            
        except requests.exceptions.RequestException as e:
            return f"❌ Error querying LLM: {e}"
    
    def explain_code_structure(self) -> str:
        """Generate an overview of the codebase structure."""
        count = self.collection.count()
        
        if count == 0:
            return "No code indexed yet. Run index_codebase() first."
        
        # Get some sample entries
        results = self.collection.get(limit=min(count, 100))
        
        # Analyze structure
        files = set()
        types = {}
        
        for meta in results['metadatas']:
            files.add(meta['file'])
            code_type = meta.get('type', 'unknown')
            types[code_type] = types.get(code_type, 0) + 1
        
        summary = f"📊 Codebase Overview:\n\n"
        summary += f"Total indexed chunks: {count}\n"
        summary += f"Files: {len(files)}\n\n"
        summary += f"Code element types:\n"
        for code_type, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
            summary += f"  • {code_type}: {count}\n"
        
        return summary


def main():
    """Interactive demo of RAG-enhanced code assistant."""
    print("="*70)
    print("🧠 RAG-ENHANCED CODE ASSISTANT")
    print("="*70)
    print("\nThis assistant can understand and answer questions about your codebase!")
    print()
    
    # Get codebase path
    codebase_path = input("Enter path to your codebase (or '.' for current): ").strip()
    if not codebase_path or codebase_path == '.':
        codebase_path = os.getcwd()
    
    # Initialize RAG system
    print("\n🚀 Initializing RAG system...")
    rag = CodebaseRAG(codebase_path)
    
    # Check if already indexed
    if rag.collection.count() == 0:
        print("\n📚 Indexing codebase (this may take a minute)...")
        num_chunks = rag.index_codebase()
        
        if num_chunks == 0:
            print("❌ No code files found. Exiting.")
            return
    else:
        print(f"✅ Using existing index ({rag.collection.count()} chunks)")
    
    # Show structure
    print("\n" + rag.explain_code_structure())
    
    # Interactive loop
    print("\n" + "="*70)
    print("Commands:")
    print("  ask <question>     - Ask about the codebase")
    print("  search <query>     - Search for code")
    print("  reindex            - Rebuild the code index")
    print("  structure          - Show codebase overview")
    print("  quit               - Exit")
    print("="*70 + "\n")
    
    while True:
        try:
            user_input = input("\n🤖 > ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'structure':
                print(rag.explain_code_structure())
                continue
            
            if user_input.lower() == 'reindex':
                print("🔄 Reindexing codebase...")
                rag.collection = rag.chroma_client.create_collection(
                    name=f"codebase_{os.getpid()}",
                    metadata={"description": "Code chunks from the codebase"}
                )
                rag.index_codebase()
                continue
            
            if user_input.lower().startswith('search '):
                query = user_input[7:].strip()
                results = rag.search_codebase(query, n_results=5)
                
                print(f"\n🔍 Found {len(results)} relevant snippets:\n")
                for i, result in enumerate(results, 1):
                    meta = result['metadata']
                    print(f"\n{'='*70}")
                    print(f"Result {i}: {meta['file']} (line {meta.get('line', '?')})")
                    print(f"Type: {meta.get('type', 'N/A')} | Name: {meta.get('name', 'N/A')}")
                    print(f"{'='*70}")
                    print(result['content'][:300] + "..." if len(result['content']) > 300 else result['content'])
                continue
            
            if user_input.lower().startswith('ask '):
                question = user_input[4:].strip()
            else:
                question = user_input
            
            # Ask the codebase
            answer = rag.ask_codebase(question)
            print(f"\n📖 Answer:\n{answer}")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Type 'quit' to exit.")
            continue
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()