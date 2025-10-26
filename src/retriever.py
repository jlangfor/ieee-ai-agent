from pathlib import Path
from typing import List
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class CodeRetriever:
    def __init__(self, project_root: Path, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.project_root = project_root
        self.embedding_model = HuggingFaceEmbedding(model_name=embed_model)
        self.index_path = project_root / ".local_index"

        if self.index_path.exists():
            storage_ctx = StorageContext.from_defaults(persist_dir=self.index_path)
            self.index = load_index_from_storage(storage_ctx)
        else:
            documents = SimpleDirectoryReader(input_files=self._collect_source_files()).load_data()
            self.index = VectorStoreIndex.from_documents(
                documents, embed_model=self.embedding_model
            )
            self.index.storage_context.persist(persist_dir=self.index_path)

    def _collect_source_files(self) -> List[Path]:
        # Include only typical source extensions, ignore venv/.git etc.
        exts = {".py", ".js", ".ts", ".cpp", ".c", ".java", ".go", ".rs", ".tsx", ".jsx"}
        ignore_dirs = {".git", "__pycache__", "node_modules", ".venv", ".local_index"}
        files = []
        for path in self.project_root.rglob("*"):
            if path.is_dir() and path.name in ignore_dirs:
                continue
            if path.suffix in exts:
                files.append(path)
        return files

    def retrieve(self, query: str, top_k: int = 5) -> str:
        """Return concatenated relevant chunks."""
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(query)
        # Simple concatenation with source file markers
        context = ""
        for node in results:
            context += f"--- {node.metadata.get('source', 'unknown')} ---\\n"
            context += node.text + "\\n\\n"
        return context