import pickle
import numpy as np
import faiss
from openai import OpenAI
import os
from typing import List, Optional
import json

class RAGSystem:
    def __init__(self, faiss_index_path: str, vectors_path: str, api_key: Optional[str] = None):
        """
        Initialize the RAG system.
        
        Args:
            faiss_index_path: Path to the FAISS index file
            vectors_path: Path to the pickled vectors file
            api_key: OpenAI API key (optional, will use environment variable if not provided)
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
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
        
        # Debug prints
        print("\n=== Debug: FAISS Search Results ===")
        print(f"Indices found: {indices}")
        print(f"Distances: {distances}")
        
        return [self.documents[i] for i in indices[0] if i < len(self.documents)]

    def generate_response(self, context: List[str], query: str, 
                         model: str = "gpt-4o-mini", max_completion_tokens: int = 2048) -> str:
        """Generate a response using GPT-4o-mini model with optimized parameters."""
        try:
            # Limit each document to 500 characters and take only top 5 documents
            truncated_docs = [str(doc)[:500] for doc in context[:5]]
            context_text = "\n".join(truncated_docs)
            
            # Further truncate if still too long
            if len(context_text) > 4000:  # Arbitrary limit to ensure we stay within token bounds
                context_text = context_text[:4000] + "..."
            
            prompt = f"""You are an idioms expert. Based on the query and context, provide exactly 3 idioms.
            If the query mentions a specific tone (like sarcastic, happy, sad), provide idioms that match that tone.
            
            Return your response in this exact JSON format:
            {{
                "idioms": [
                    {{
                        "phrase": "the idiom itself",
                        "meaning": "clear explanation of what it means",
                        "example": "example sentence using the idiom"
                    }}
                ]
            }}
            
            Query: {query}
            Context: {context_text}
            """
            
            # Create chat completion with adjusted max_tokens
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert at explaining idioms clearly and concisely."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7,
                timeout=self.timeout
            )
            
            # Check if the response is valid JSON
            content = response.choices[0].message.content
            print(f"Raw GPT Response: {content}")  # Debug print
            
            # Ensure we have a valid JSON response
            try:
                json_response = json.loads(content)
                if not json_response.get("idioms"):
                    raise ValueError("No idioms in response")
                return content
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing response: {str(e)}")
                # Create a fallback response
                fallback_response = {
                    "idioms": [
                        {
                            "phrase": "Tongue in cheek",
                            "meaning": "To speak in an ironic or insincere way",
                            "example": "His sarcastic remarks were clearly tongue in cheek."
                        }
                    ]
                }
                return json.dumps(fallback_response)
                
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            return '{"error": "Failed to generate response"}'

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
        
        if not relevant_docs:
            return json.dumps({
                "idioms": [],
                "message": "No relevant idioms found for your query."
            }, indent=2)
        
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
        query = "I am feeling sad I want to learn idioms about sadness"
        response = rag.query(query)
        
        # Save response to JSON file
        save_response_to_json(response)
        
        print("Query:", query)
        print("\nResponse saved to idioms_response.json")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()