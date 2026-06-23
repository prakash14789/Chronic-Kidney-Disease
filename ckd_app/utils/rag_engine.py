# ckd_app/utils/rag_engine.py
import os
import json
import numpy as np
from typing import List, Dict, Any

# Lazy load sentence-transformers and huggingface_hub to avoid loading errors if packages aren't fully compiled yet
_ST_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    pass

_HF_AVAILABLE = False
try:
    from huggingface_hub import InferenceClient
    _HF_AVAILABLE = True
except ImportError:
    pass

class CKDRAGEngine:
    def __init__(self, db_path: str = "data/clinical_knowledge_base.json", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "HuggingFaceH4/zephyr-7b-beta"):
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        
        self.documents = []
        self.embeddings = []
        
        self._model = None  # Lazy-loaded SentenceTransformer
        self.load_vector_db()

    def get_embedding_model(self):
        """Lazy load the sentence transformer model."""
        if not _ST_AVAILABLE:
            raise ImportError("sentence-transformers is not installed. Please install it to compute embeddings.")
        if self._model is None:
            # Load the model locally (will download on first run)
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def load_vector_db(self):
        """Load document chunks and pre-computed embeddings from local JSON database."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.documents = data.get("documents", [])
                    # Convert list back to list of floats/lists
                    self.embeddings = data.get("embeddings", [])
            except Exception as e:
                print(f"Error loading vector DB: {e}")
                self.documents = []
                self.embeddings = []
        else:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self.save_vector_db()

    def save_vector_db(self):
        """Save the documents and embeddings to the local JSON database."""
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump({
                "documents": self.documents,
                "embeddings": self.embeddings
            }, f, indent=2)

    def get_embedding(self, text: str) -> List[float]:
        """Compute the embedding for a given text locally."""
        if not _ST_AVAILABLE:
            # Fallback mock embedding if library not loaded
            return [0.0] * 384
        
        model = self.get_embedding_model()
        embedding = model.encode(text)
        return embedding.tolist()

    def add_document(self, title: str, text: str, category: str = "guidelines"):
        """Chunk a document, generate embeddings, and append it to our local vector store."""
        # Simple character-based chunking with overlap (approx. 600 chars ~ 120 words)
        chunk_size = 600
        overlap = 100
        
        chunks = []
        start = 0
        if not text.strip():
            return
            
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap

        for i, chunk in enumerate(chunks):
            embedding = self.get_embedding(chunk)
            doc_id = f"{title.replace(' ', '_').lower()}_chunk_{i}"
            
            self.documents.append({
                "id": doc_id,
                "title": title,
                "content": chunk,
                "category": category,
                "chunk_index": i
            })
            self.embeddings.append(embedding)
            
        self.save_vector_db()

    def query(self, user_query: str, top_k: int = 3, token: str = None) -> Dict[str, Any]:
        """Perform semantic search and generate answer using Hugging Face Serverless Inference."""
        if not self.documents:
            return {
                "answer": "The clinical knowledge base is currently empty. Please run the seeding script or upload clinical guidelines through the API/Dashboard.",
                "sources": []
            }

        # 1. Generate query embedding
        query_emb = np.array(self.get_embedding(user_query))
        
        # 2. Calculate Cosine Similarity with all documents
        doc_embs = np.array(self.embeddings)
        
        # Avoid zero-division if embeddings are empty/mocked
        norms_doc = np.linalg.norm(doc_embs, axis=1)
        norm_query = np.linalg.norm(query_emb)
        
        if norm_query == 0 or np.any(norms_doc == 0):
            # Fallback to simple keyword search if embeddings aren't loaded or are zero-filled
            scores = []
            for doc in self.documents:
                # Count keyword matches
                match_count = sum(1 for word in user_query.lower().split() if word in doc["content"].lower())
                scores.append(match_count / (len(user_query.split()) + 1))
            top_indices = np.argsort(scores)[::-1][:top_k]
            similarities = np.array(scores)
        else:
            similarities = np.dot(doc_embs, query_emb) / (norms_doc * norm_query)
            top_indices = np.argsort(similarities)[::-1][:top_k]

        # 3. Assemble Context
        context_chunks = []
        sources = []
        
        for idx in top_indices:
            similarity = float(similarities[idx])
            # If doing keyword matching or valid cosine similarity
            if similarity > 0.05:  # Soft threshold
                doc = self.documents[idx]
                context_chunks.append(f"Source: {doc['title']} (Chunk {doc['chunk_index']})\nContent: {doc['content']}")
                sources.append({
                    "title": doc["title"],
                    "chunk": doc["chunk_index"],
                    "score": round(similarity, 4)
                })

        if not context_chunks:
            return {
                "answer": "I searched the database but could not find clinical guidelines relevant to your question. Please verify the knowledge base contains relevant topics.",
                "sources": []
            }

        context_str = "\n\n---\n\n".join(context_chunks)

        # 4. Invoke LLM via Hugging Face Inference API
        system_prompt = (
            "You are a clinical decision-support assistant specializing in Chronic Kidney Disease (CKD). "
            "You provide recommendations to medical practitioners based on the KDIGO Clinical Practice Guidelines. "
            "Using ONLY the provided medical context below, answer the clinician's query. "
            "Provide a concise, medical-grade answer with inline citations matching the Source titles. "
            "If the text does not contain the answer, state that you cannot find this in the current guidelines. "
            "Do not make up facts or ignore the provided context.\n\n"
            f"--- CLINICAL CONTEXT ---\n{context_str}\n------------------------"
        )
        
        user_prompt = f"Clinician Query: {user_query}"

        # Resolve Hugging Face Hub token
        hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

        if not hf_token:
            return {
                "answer": (
                    "⚠️ **Hugging Face API Token Missing**\n\n"
                    "I successfully performed semantic search and found relevant clinical guidelines, "
                    "but I cannot generate a synthesized answer because no Hugging Face Hub token was provided.\n\n"
                    "Please configure your token in the sidebar or set the `HF_TOKEN` environment variable on the server.\n\n"
                    f"**Matched Guidelines Excerpts (Direct Retrieval):**\n\n" + "\n\n".join([f"* **{s['title']} (Chunk {s['chunk']})**:\n{self.documents[top_indices[i]]['content']}" for i, s in enumerate(sources)])
                ),
                "sources": sources
            }

        if not _HF_AVAILABLE:
            return {
                "answer": (
                    "⚠️ **huggingface-hub Library Not Loaded**\n\n"
                    "The client library is still installing or failed to load. Here is the retrieved guideline content:\n\n" +
                    "\n\n".join([f"* **{s['title']} (Chunk {s['chunk']})**:\n{self.documents[top_indices[i]]['content']}" for i, s in enumerate(sources)])
                ),
                "sources": sources
            }

        try:
            client = InferenceClient(model=self.llm_model_name, token=hf_token)
            
            # Simple format that works universally with HF text generation models
            prompt_format = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>\n"
            
            response = client.text_generation(
                prompt=prompt_format,
                max_new_tokens=400,
                temperature=0.2,
                repetition_penalty=1.1,
                stop_sequences=["</s>", "<|im_end|>"]
            )
            
            # Post-process response to remove system artifacts if any
            clean_answer = response.strip()
            if clean_answer.endswith("</s>"):
                clean_answer = clean_answer[:-4].strip()
                
            return {
                "answer": clean_answer,
                "sources": sources
            }
            
        except Exception as e:
            return {
                "answer": f"❌ **Error generating response from Hugging Face:**\n{str(e)}\n\n"
                          f"**Retrieved Excerpts:**\n\n" + 
                          "\n\n".join([f"* **{s['title']} (Chunk {s['chunk']})**:\n{self.documents[top_indices[i]]['content']}" for i, s in enumerate(sources)]),
                "sources": sources
            }
