from typing import Dict, List, Optional
from openai import OpenAI
import json
import faiss
import numpy as np
from rag_system import RAGSystem

class AgentOrchestrator:
    def __init__(self, rag_system: RAGSystem):
        """Initialize the orchestrator with the RAG system."""
        self.client = OpenAI()
        self.rag = rag_system
        
        # Define system prompts for different operations
        self.system_prompts = {
            "search": """You are an expert idioms teacher specializing in English idioms.
            Your role is to:
            1. Understand the user's query and intent
            2. Provide relevant idioms with clear explanations
            3. Give practical, context-appropriate examples
            4. Ensure explanations are easy to understand
            
            Format your responses consistently and focus on accuracy and clarity.
            Each idiom should include:
            - The idiom phrase
            - A clear meaning/explanation
            - A relevant example showing proper usage""",
            
            "learn": """You are an adaptive language teacher focusing on idioms.
            Your role is to:
            1. Assess the user's level and needs
            2. Create personalized learning experiences
            3. Provide constructive feedback
            4. Adapt difficulty based on user responses"""
        }
        
    def retrieve_idioms(self, query: str, top_k: int = 3) -> Dict:
        """
        Retrieve relevant idioms using the RAG system with system prompt guidance.
        
        Args:
            query: User's search query
            top_k: Number of idioms to retrieve
            
        Returns:
            Dictionary with search results
        """
        try:
            # Get embedding for the query
            query_embedding = self.rag.embed_query(query)
            
            # Search for similar documents
            similar_docs = self.rag.search_similar_documents(query_embedding, top_k)
            
            # Generate response using the retrieved documents and system prompt
            response = self.rag.generate_response(
                context=similar_docs,
                query=query,
                system_prompt=self.system_prompts["search"]
            )
            
            # Parse and return the idioms
            result = json.loads(response)
            return {
                "type": "search_result",
                "message": "Here are some relevant idioms:",
                "idioms": result.get("idioms", [])
            }
            
        except Exception as e:
            print(f"Error retrieving idioms: {e}")
            return {
                "type": "error",
                "message": "Failed to retrieve idioms",
                "error": str(e)
            }
            
    def process_query(self, query: str, mode: str = "search") -> Dict:
        """
        Process a user query based on the specified mode.
        
        Args:
            query: User's input query
            mode: Operation mode ("search" or "learn")
            
        Returns:
            Response dictionary containing idioms and/or learning content
        """
        try:
            if mode == "search":
                return self.retrieve_idioms(query)
            else:
                return {
                    "type": "error",
                    "message": "Learning mode not implemented yet"
                }
                
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "type": "error",
                "message": "An error occurred while processing your request",
                "error": str(e)
            }
            
    def get_conversation_context(self, n_messages: int = 5) -> str:
        """Get the recent conversation context."""
        recent_messages = self.conversation_history[-n_messages:] if self.conversation_history else []
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])

    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []