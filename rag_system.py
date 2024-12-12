import pickle
import numpy as np
import faiss
from openai import OpenAI
import os
from typing import List, Optional
import json
from dotenv import load_dotenv

load_dotenv()

class RAGSystem:
    def __init__(self, faiss_index_path: str, vectors_path: str, api_key: Optional[str] = None):
        """
        Initialize the RAG system.
        
        Args:
            faiss_index_path: Path to the FAISS index file
            vectors_path: Path to the pickled vectors file
            api_key: OpenAI API key (optional, will use environment variable if not provided)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found.")
        
        # Cache the OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Cache the FAISS index and documents
        self.faiss_index = self._load_faiss_index(faiss_index_path)
        self.documents = self._load_documents(vectors_path)
        
        # Set a shorter timeout for API calls
        self.timeout = 30

    def _load_faiss_index(self, index_path: str) -> faiss.Index:
        """Load the FAISS index from file."""
        try:
            index = faiss.read_index(index_path)
            print(f"FAISS index loaded from {index_path}")
            return index
        except Exception as e:
            raise Exception(f"Error loading FAISS index: {str(e)}")

    def _load_documents(self, vectors_path: str) -> List:
        """Load the document vectors from pickle file."""
        try:
            with open(vectors_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise Exception(f"The file '{vectors_path}' was not found.")

    def embed_query(self, query: str) -> np.ndarray:
        """Create embeddings for the query."""
        response = self.client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = response.data[0].embedding
        return np.array(query_embedding).astype('float32').reshape(1, -1)

    def search_similar_documents(self, query_embedding: np.ndarray, top_k: int = 5) -> List[str]:
        """Search for similar documents using the query embedding."""
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices[0]]

    def generate_response(self, context: List[str], query: str, 
                         model: str = "gpt-4o-mini", max_completion_tokens: int = 2048) -> str:
        """Generate a response using OpenAI's gpt-4o-mini model with optimized parameters."""
        context_text = "\n".join([str(doc) for doc in context])
        
        # Calculate available tokens
        context_window = 128000  # gpt-4o-mini's context window
        buffer_tokens = 256  # Buffer to account for reasoning tokens
        query_tokens = len(query.split())
        available_tokens = context_window - query_tokens - buffer_tokens
        
        # Truncate context_text if it exceeds available tokens
        context_tokens = context_text.split()
        if len(context_tokens) > available_tokens:
            context_text = " ".join(context_tokens[:available_tokens])
        
        prompt = f"""You are an idioms teacher. Please explain idioms in a clear way.
        Return your response in this JSON format:
        {{
            "idioms": [
                {{
                    "number": 1,
                    "idiom": "the idiom phrase",
                    "definition": "clear explanation of the idiom"
                }}
            ]
        }}
        Include 3 idioms at a time if there is no relevant idioms don't include any idioms.
        
        Context: {context_text}
        Query: {query}
        """
        
        # Create chat completion
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            max_tokens=max_completion_tokens,
            temperature=0.7,
            timeout=self.timeout
        )
        
        # Check if the response is valid JSON
        try:
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            return '{"error": "Failed to parse response"}'

    def query(self, query: str, top_k: int = 5) -> str:
        """
        Process a query and return a response.
        
        Args:
            query: The user's question
            top_k: Number of similar documents to retrieve
        
        Returns:
            str: Generated response as a formatted JSON string
        """
        # Create query embedding
        query_embedding = self.embed_query(query)
        
        # Retrieve similar documents
        relevant_docs = self.search_similar_documents(query_embedding, top_k)
        
        # Generate response and parse it as JSON
        response = self.generate_response(relevant_docs, query)
        return json.dumps(json.loads(response), indent=2)

def save_response_to_json(response_data: str, filename: str = "idioms_response.json") -> None:
    """Save the response to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json.loads(response_data), f, indent=2)

def main():
    # Example usage
    try:
        # Initialize the RAG system
        rag = RAGSystem(
            faiss_index_path="faiss_index.idx",
            vectors_path="vectors.pkl"
        )
        
        # Example query
        query = "I am feeling sad i want to learn idioms about sadness"
        response = rag.query(query)
        
        # Save response to JSON file
        save_response_to_json(response)
        
        print("Query:", query)
        print("\nResponse saved to idioms_response.json")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()