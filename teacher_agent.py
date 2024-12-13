from typing import Dict, List, Optional
from openai import OpenAI
import json
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TeacherAgent:
    def __init__(self, orchestrator):
        """Initialize the teacher agent with the orchestrator."""
        self.orchestrator = orchestrator
        self.client = OpenAI()
        self.current_state = "greeting"
        self.student_profile = {
            "level": None,
            "interests": [],
            "learned_idioms": set(),
            "current_lesson": None
        }
        
        self.conversation_history = []
        
    def process_message(self, message: str) -> Dict:
        """Process student message and return chatbot-style response."""
        try:
            print(f"TeacherAgent processing message: {message}")  # Debug print
            
            # Add user message to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": message
            })
            
            # Generate context from recent conversation
            context = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in self.conversation_history[-3:]
            ])
            print(f"Context: {context}")  # Debug print
            
            # Create prompt based on current state
            print(f"Current state: {self.current_state}")  # Debug print
            
            if self.current_state == "greeting":
                prompt = self._create_greeting_prompt(message, context)
            elif self.current_state == "assess_level":
                prompt = self._create_assessment_prompt(message, context)
            elif self.current_state == "teach":
                prompt = self._create_teaching_prompt(message, context)
            elif self.current_state == "practice":
                prompt = self._create_practice_prompt(message, context)
            else:
                prompt = self._create_feedback_prompt(message, context)
            
            print(f"Created prompt: {prompt}")  # Debug print
            
            # Get response from GPT
            print("Calling OpenAI API...")  # Debug print
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.orchestrator.system_prompts["learn"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            print(f"API Response: {response}")  # Debug print
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            print(f"Parsed result: {result}")  # Debug print
            
            # Update state and profile
            self._update_state_and_profile(result)
            
            # Format chatbot response
            chat_response = self._format_chat_response(result)
            print(f"Final chat response: {chat_response}")  # Debug print
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": chat_response["message"]
            })
            
            return chat_response
            
        except Exception as e:
            print(f"Error in teacher agent: {e}")
            print(f"Full error details: {str(e)}")
            return {
                "type": "chat",
                "message": "I'm having trouble understanding. Could you rephrase that?",
                "idioms": [],
                "suggestions": []
            }
    
    def _create_greeting_prompt(self, message: str, context: str) -> str:
        return f"""Context: {context}
        Student message: "{message}"
        Current state: {self.current_state}
        
        Give a brief, friendly greeting and ask about level (beginner/intermediate/advanced).
        Keep response under 2 sentences.
        
        Return in this exact JSON format:
        {{
            "message": "brief greeting and question",
            "next_state": "assess_level",
            "suggestions": ["beginner", "intermediate", "advanced"]
        }}
        """
    
    def _create_assessment_prompt(self, message: str, context: str) -> str:
        return f"""Context: {context}
        Student message: "{message}"
        Current state: {self.current_state}
        
        Acknowledge level and ask about ONE specific interest (business/casual/academic).
        Keep response under 2 sentences.
        
        Return in this exact JSON format:
        {{
            "message": "brief acknowledgment and question",
            "next_state": "teach",
            "assessment": {{
                "level": "{self.student_profile.get('level', 'beginner')}"
            }},
            "suggestions": ["business idioms", "casual idioms", "academic idioms"]
        }}
        """
    
    def _create_teaching_prompt(self, message: str, context: str) -> str:
        idioms = self.orchestrator.retrieve_idioms(
            query=f"idioms about {' '.join(self.student_profile['interests'])} for {self.student_profile['level']} level"
        )
        
        return f"""Context: {context}
        Student message: "{message}"
        Student level: {self.student_profile['level']}
        
        Teach ONE idiom clearly and concisely.
        Include: meaning and one short example.
        Keep total response under 3 sentences.
        
        Return in this exact JSON format:
        {{
            "message": "brief idiom explanation",
            "next_state": "practice",
            "taught_idioms": [{{
                "phrase": "the idiom",
                "meaning": "brief meaning",
                "example": "short example"
            }}],
            "practice_question": "simple practice question",
            "suggestions": ["2-3 word answers"]
        }}
        """
    
    def _create_practice_prompt(self, message: str, context: str) -> str:
        return f"""Context: {context}
        Student message: "{message}"
        
        Give brief feedback and ONE new practice opportunity.
        Keep response under 2 sentences.
        
        Return in this exact JSON format:
        {{
            "message": "brief feedback + question",
            "next_state": "practice",
            "suggestions": ["2-3 possible answers"]
        }}
        """
    
    def _create_feedback_prompt(self, message: str, context: str) -> str:
        return f"""Context: {context}
        Student profile: {json.dumps(self.student_profile)}
        
        Give brief encouragement and suggest next topic.
        Keep response under 2 sentences.
        
        Return in this exact JSON format:
        {{
            "message": "brief encouragement + suggestion",
            "next_state": "teach",
            "suggestions": ["2-3 topic options"]
        }}
        """
    
    def _update_state_and_profile(self, result: Dict) -> None:
        """Update agent state and student profile based on response."""
        # Update state
        self.current_state = result.get("next_state", self.current_state)
        
        # Update profile
        if result.get("detected_level"):
            self.student_profile["level"] = result["detected_level"]
        if result.get("detected_interests"):
            self.student_profile["interests"].extend(result["detected_interests"])
        if result.get("taught_idioms"):
            self.student_profile["learned_idioms"].update(result["taught_idioms"])
        if result.get("assessment"):
            self.student_profile.update(result["assessment"])
    
    def _format_chat_response(self, result: Dict) -> Dict:
        """Format the response in a chatbot-friendly structure."""
        return {
            "type": "chat",
            "message": result["message"],
            "idioms": result.get("taught_idioms", []),
            "suggestions": result.get("suggestions", []),
            "examples": result.get("examples", []),
            "practice": result.get("practice_question", ""),
            "corrections": result.get("corrections", []),
            "summary": result.get("summary", {}),
            "state": self.current_state
        } 