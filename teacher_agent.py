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
        # Store user sessions in a dictionary
        self.user_sessions = {}
        
    def get_initial_greeting(self, session_id: str) -> Dict:
        """Get the initial greeting for a new session."""
        session = self._get_or_create_session(session_id)
        
        try:
            # Get personalized greeting from LLM
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are Adam, a friendly AI English idioms teacher. Keep your response warm but professional."},
                    {"role": "user", "content": """Generate a brief, welcoming introduction that:
                        1. Introduces you as Adam
                        2. Explains you're an AI idioms teacher
                        3. Mentions the interactive learning approach
                        Keep it under 3 sentences."""}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            greeting_message = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating initial greeting: {str(e)}")
            # Fallback greeting if LLM fails
            greeting_message = "Hello! I'm Adam, your AI English idioms teacher. I'm here to help you learn and practice idioms in a fun, interactive way!"

        initial_greeting = {
            "type": "chat",
            "message": greeting_message,
            "idioms": [],
            "suggestions": ["Let's begin!"],
            "state": "greeting"
        }
        
        # Add to conversation history
        session["conversation_history"].append({
            "role": "assistant",
            "content": initial_greeting["message"]
        })
        return initial_greeting

    def get_level_question(self, session_id: str) -> Dict:
        """Get the level assessment question after initial greeting."""
        session = self._get_or_create_session(session_id)
        
        level_question = {
            "type": "chat",
            "message": "To personalize your learning experience, please tell me your level: beginner, intermediate, or advanced?",
            "idioms": [],
            "suggestions": ["beginner", "intermediate", "advanced"],
            "state": "greeting"
        }
        
        # Add to conversation history
        session["conversation_history"].append({
            "role": "assistant",
            "content": level_question["message"]
        })
        return level_question

    def _get_or_create_session(self, session_id):
        """Get or create a new user session."""
        if session_id not in self.user_sessions:
            self.user_sessions[session_id] = {
                "current_state": "greeting",
                "student_profile": {
                    "level": None,
                    "interests": [],
                    "learned_idioms": set(),
                    "current_lesson": None
                },
                "conversation_history": []
            }
        return self.user_sessions[session_id]

    def process_message(self, message: str, session_id: str) -> Dict:
        """Process student message and return chatbot-style response."""
        try:
            if not message or not session_id:
                print(f"Invalid input - message: {message}, session_id: {session_id}")
                return self._create_error_response("Missing required input")

            # Get or create user session
            session = self._get_or_create_session(session_id)
            
            print(f"Processing message for session {session_id}, current state: {session['current_state']}")
            
            # Add user message to conversation history
            session["conversation_history"].append({
                "role": "user",
                "content": message
            })
            
            # Generate context from recent conversation
            context = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in session["conversation_history"][-3:]
            ])
            
            # Create prompt based on current state
            try:
                if session["current_state"] == "greeting":
                    prompt = self._create_greeting_prompt(message, context)
                elif session["current_state"] == "assess_level":
                    prompt = self._create_assessment_prompt(message, context)
                elif session["current_state"] == "teach":
                    prompt = self._create_teaching_prompt(message, context, session["student_profile"])
                elif session["current_state"] == "practice":
                    prompt = self._create_practice_prompt(message, context)
                else:
                    prompt = self._create_feedback_prompt(message, context, session["student_profile"])
            except Exception as e:
                print(f"Error creating prompt: {str(e)}")
                return self._create_error_response("Error creating response")
            
            # Get response from GPT
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": self.orchestrator.system_prompts["learn"]},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                # Parse response
                result = json.loads(response.choices[0].message.content)
            except Exception as e:
                print(f"Error with OpenAI API or parsing response: {str(e)}")
                return self._create_error_response("Error generating response")
            
            # Update session state and profile
            self._update_session(session, result)
            
            # Format chatbot response
            chat_response = self._format_chat_response(result, session["current_state"])
            
            # Add assistant response to history
            session["conversation_history"].append({
                "role": "assistant",
                "content": chat_response["message"]
            })
            
            return chat_response
            
        except Exception as e:
            print(f"Unexpected error in process_message: {str(e)}")
            return self._create_error_response("An unexpected error occurred")

    def _update_session(self, session: Dict, result: Dict) -> None:
        """Update session state and student profile based on response."""
        # Update state
        session["current_state"] = result.get("next_state", session["current_state"])
        
        # Update profile
        if result.get("detected_level"):
            session["student_profile"]["level"] = result["detected_level"]
        if result.get("detected_interests"):
            session["student_profile"]["interests"].extend(result["detected_interests"])
        if result.get("taught_idioms"):
            session["student_profile"]["learned_idioms"].update(
                [idiom["phrase"] for idiom in result["taught_idioms"]]
            )
        if result.get("assessment"):
            session["student_profile"].update(result["assessment"])

    def _create_greeting_prompt(self, message: str, context: str) -> str:
        return f"""Context: {context}
        Student message: "{message}"
        Current state: greeting
        
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
        Current state: assess_level
        
        Acknowledge level and ask about ONE specific interest (business/casual/academic).
        Keep response under 2 sentences.
        
        Return in this exact JSON format:
        {{
            "message": "brief acknowledgment and question",
            "next_state": "teach",
            "assessment": {{
                "level": "beginner"
            }},
            "suggestions": ["business idioms", "casual idioms", "academic idioms"]
        }}
        """
    
    def _create_teaching_prompt(self, message: str, context: str, student_profile: Dict) -> str:
        idioms = self.orchestrator.retrieve_idioms(
            query=f"idioms about {' '.join(student_profile['interests'])} for {student_profile['level']} level"
        )
        
        return f"""Context: {context}
        Student message: "{message}"
        Student level: {student_profile['level']}
        
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
    
    def _create_feedback_prompt(self, message: str, context: str, student_profile: Dict) -> str:
        return f"""Context: {context}
        Student profile: {json.dumps(student_profile)}
        
        Give brief encouragement and suggest next topic.
        Keep response under 2 sentences.
        
        Return in this exact JSON format:
        {{
            "message": "brief encouragement + suggestion",
            "next_state": "teach",
            "suggestions": ["2-3 topic options"]
        }}
        """
    
    def _format_chat_response(self, result: Dict, current_state: str) -> Dict:
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
            "state": current_state
        } 

    def _create_error_response(self, message: str) -> Dict:
        """Create a standardized error response."""
        return {
            "type": "chat",
            "message": message,
            "idioms": [],
            "suggestions": ["Try saying hello", "Could you rephrase that?"],
            "state": "greeting"  # Reset to greeting state on error
        } 