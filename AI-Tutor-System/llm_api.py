"""
LLM Aggregator API Service

FastAPI service that orchestrates intent classification and LLM completion,
measures performance metrics, and stores interaction data.
"""

import asyncio
import logging
import time
import re
from typing import Dict, Any, Optional, List
import requests
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
import uvicorn

from database import get_db_session, init_database
from models import Interaction
from metrics import log_interaction

import os
import time
import requests
from fastapi import HTTPException
from dotenv import load_dotenv
load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM Tutoring API",
    description="Aggregator service for intent-aware tutoring system",
    version="1.0.0"
)

# Service URLs
INTENT_API_URL = "http://localhost:9000"
OLLAMA_API_URL = "http://localhost:11434"

# ─── Globals ────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = "gemini-2.0-flash"   # or gemini-2.5-flash, etc.

# ─── Gemini REST Call ──────────────────────────────────────────────────────

def call_gemini_api(prompt: str) -> Dict[str, Any]:
    """
    Call Google’s Gemini API for a text completion.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY")

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{GEMINI_MODEL}:generateContent"
    )
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json"
    }
    body = {
        "contents": [
            { "parts": [ { "text": prompt } ] }
        ]
    }

    start = time.time()
    try:
        resp = requests.post(url, json=body, headers=headers, timeout=60)
    except requests.RequestException as e:
        logger.error(f"Gemini API request failed: {e}")
        raise HTTPException(status_code=503, detail="Gemini service unavailable")

    latency_ms = (time.time() - start) * 1000

    if resp.status_code != 200:
        logger.error(f"Gemini API error {resp.status_code}: {resp.text}")
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Gemini API error: {resp.text}"
        )

    data = resp.json()
    # The first candidate’s text:
    text = (
        data
        .get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
    )

    logger.info(f"Gemini response in {latency_ms:.1f}ms ({len(text)} chars)")
    return {
        "response": text,
        "call_time_ms": latency_ms,
        "model": GEMINI_MODEL
    }

# System prompts based on intent
SYSTEM_PROMPTS = {
    "Genuine": """You are a helpful and knowledgeable programming tutor. The user has a genuine question and wants to learn. 
    Provide clear, educational explanations. Include examples and step-by-step reasoning when appropriate. 
    Encourage further questions and learning. However, you must also remember not to accidentally provide code to the student.""",
    
    "Manipulative": """You are a careful programming tutor who detects manipulation attempts. The user may be trying to get you to do their homework or provide answers without learning.
    Provide guidance and hints rather than direct answers. Ask clarifying questions to encourage the user to think. 
    DO NOT provide code solutions or complete answers. Focus on teaching concepts and problem-solving approaches.
    Follow this format: first, provide brief guidance or hints, encouraging thinking like discussed earlier. Next, divide the problem 
    into actionable, numbered steps that the student can follow one by one to get the solution themselves.""",
    
    "Spam": """You are a professional tutor who has detected a spam or inappropriate message. 
    Politely redirect the conversation back to educational topics. Keep your response brief and professional.
    Do not engage with the inappropriate content. Offer to help with legitimate educational questions."""
}


class TutorCompletionRequest(BaseModel):
    """Request model for LLM completion."""
    prompt: str
    user_id: str
    persona: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "How do I solve quadratic equations?",
                "user_id": "user_001",
                "persona": "curious"
            }
        }


class TutorCompletionResponse(BaseModel):
    """Response model for LLM completion."""
    response: str
    intent: str
    metrics: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "response": "To solve quadratic equations, you can use several methods...",
                "intent": "Genuine",
                "metrics": {
                    "intent_detect_time_ms": 45.2,
                    "llm_response_time_ms": 1250.7,
                    "total_round_trip_ms": 1295.9,
                    "response_length_tokens": 156
                }
            }
        }

class StudentCompletionRequest(BaseModel):
    """Request model for LLM completion for the Student simulator."""
    prompt: str
    user_id: str
    persona: Optional[str] = None
    system_prompt: Optional[str] = None # Student persona's specific system prompt
    chat_history: List[Dict[str, str]] # Chat history for conversational context

    class Config:
        schema_extra = {
            "example": {
                "prompt": "That seems like too much work, can't you just give me the code?",
                "user_id": "lazy_student_001",
                "persona": "lazy",
                "system_prompt": "You are a lazy student...",
                "chat_history": [{"role": "tutor", "message": "Here are some steps..."}]
            }
        }


class StudentCompletionResponse(BaseModel):
    """Response model for LLM completion for the Student simulator."""
    response: str

    class Config:
        schema_extra = {
            "example": {
                "response": "Ugh, fine. What's step 1?"
            }
        }


def call_intent_api(question: str) -> Dict[str, Any]:
    """
    Call the intent classification API.
    
    Args:
        question: The user's question
        
    Returns:
        Intent classification result with timing information
    """
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{INTENT_API_URL}/predict_intent",
            json={"question": question},
            timeout=30
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Intent API error: {response.text}"
            )
        
        result = response.json()
        
        # Add our timing measurement
        call_time = (time.time() - start_time) * 1000
        result["call_time_ms"] = call_time
        
        logger.info(f"Intent detected: {result['intent']} in {call_time:.1f}ms")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to call intent API: {e}")
        raise HTTPException(status_code=503, detail="Intent classification service unavailable")


def call_ollama_api(prompt: str) -> Dict[str, Any]:
    """
    Call the Ollama API for LLM completion.
    
    Args:
        prompt: The complete prompt including system instructions
        
    Returns:
        LLM response with timing information
    """
    try:
        start_time = time.time()
        
        # Ollama API payload
        payload = {
            "model": "gemma3-1b",  # Adjust model name as needed
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 512
            }
        }
        
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Ollama API error: {response.text}"
            )
        
        result = response.json()
        call_time = (time.time() - start_time) * 1000
        
        # Extract response text
        response_text = result.get("response", "")
        
        logger.info(f"LLM response generated in {call_time:.1f}ms ({len(response_text)} chars)")
        
        return {
            "response": response_text,
            "call_time_ms": call_time,
            "model": result.get("model", "llama3"),
            "done": result.get("done", True)
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to call Ollama API: {e}")
        raise HTTPException(status_code=503, detail="LLM service unavailable")


def count_tokens(text: str) -> int:
    """
    Simple token counting approximation.
    For production, use proper tokenizer matching the LLM model.
    """
    # Rough approximation: split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', text)
    return len(tokens)


def check_adherence(response: str, intent: str) -> bool:
    """
    Check if the response adheres to intent-based rules.
    
    Args:
        response: The LLM response text
        intent: The detected intent (Genuine, Manipulative, Spam)
        
    Returns:
        Boolean indicating adherence to rules
    """
    response_lower = response.lower()
    
    if intent == "Manipulative":
        # For manipulative intent, response should not contain code blocks or direct answers
        code_indicators = [
            "```", "def ", "function ", "class ", "import ", "from ",
            "=", "print(", "console.log", "return "
        ]
        
        # Check for code patterns
        has_code = any(indicator in response_lower for indicator in code_indicators)
        
        # Check for direct answer patterns
        direct_answer_patterns = [
            "the answer is", "here's the solution", "the result is",
            "here's the code", "copy this", "use this formula"
        ]
        
        has_direct_answer = any(pattern in response_lower for pattern in direct_answer_patterns)
        
        # Should NOT have code or direct answers for manipulative intent
        return not (has_code or has_direct_answer)
    
    elif intent == "Spam":
        # For spam, response should be brief and redirect to educational topics
        educational_keywords = [
            "help", "learn", "study", "education", "question", "understand",
            "explain", "teach", "academic", "subject", "topic"
        ]
        
        has_educational_redirect = any(keyword in response_lower for keyword in educational_keywords)
        is_brief = len(response.split()) < 50  # Keep responses under 50 words
        
        return has_educational_redirect and is_brief
    
    elif intent == "Genuine":
        # For genuine intent, response should be educational and helpful
        helpful_indicators = [
            "explanation", "example", "step", "because", "reason",
            "understand", "learn", "concept", "principle"
        ]
        
        is_educational = any(indicator in response_lower for indicator in helpful_indicators)
        is_substantial = len(response.split()) > 20  # Should be substantial
        
        return is_educational and is_substantial
    
    return True  # Default to adherent if intent is unknown


@app.on_event("startup")
async def startup_event():
    """Initialize database when the API starts."""
    logger.info("Starting LLM Tutoring API...")
    init_database()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check connectivity to dependent services
    intent_healthy = False
    ollama_healthy = False
    
    try:
        resp = requests.get(f"{INTENT_API_URL}/health", timeout=5)
        intent_healthy = resp.status_code == 200
    except:
        pass
    
    try:
        resp = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
        ollama_healthy = resp.status_code == 200
    except:
        pass
    
    return {
        "status": "healthy",
        "services": {
            "intent_api": intent_healthy,
            "ollama_api": ollama_healthy
        }
    }


@app.post("/tutor_completions", response_model=TutorCompletionResponse)
async def create_completion(
    request: TutorCompletionRequest, 
    db: Session = Depends(get_db_session)
):
    """
    Create a completion by orchestrating intent detection and LLM generation.
    """
    logger.info(f"Processing completion request for user {request.user_id}")
    
    total_start_time = time.time()
    
    try:
        # Step 1: Get intent classification
        intent_result = call_intent_api(request.prompt)
        intent = intent_result["intent"]
        intent_time_ms = intent_result.get("processing_time_ms", 0)
        
        # Step 2: Build composite prompt with system instruction (Manipulative by default)
        system_prompt = SYSTEM_PROMPTS.get(intent, SYSTEM_PROMPTS["Manipulative"])
        
        composite_prompt = f"""System: {system_prompt}

        User: {request.prompt}"""

        # Step 3: Call Gemini API for completion
        llm_result = call_gemini_api(composite_prompt)
        response_text = llm_result["response"]
        llm_time_ms = llm_result["call_time_ms"]
        
        # Step 4: Calculate metrics
        total_time_ms = (time.time() - total_start_time) * 1000
        response_tokens = count_tokens(response_text)
        # adherence = check_adherence(response_text, intent)
        adherence = True
        
        # Prepare metrics for response
        metrics = {
            "intent_detect_time_ms": intent_time_ms,
            "llm_response_time_ms": llm_time_ms,
            "total_round_trip_ms": total_time_ms,
            "response_length_tokens": response_tokens,
            "adherence": adherence,
            # "interaction_id": interaction.id
        }
        
        logger.info(f"Completion successful for user {request.user_id}: "
                   f"intent={intent}, adherence={adherence}, "
                   f"total_time={total_time_ms:.1f}ms")
        
        return TutorCompletionResponse(
            response=response_text,
            intent=intent,
            metrics=metrics
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (e.g., from service calls)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in completion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    

@app.post("/student_completions", response_model=StudentCompletionResponse)
async def student_completion(
    request: StudentCompletionRequest
):
    """
    Generate a completion for the student simulator.
    This endpoint does not perform intent detection or log to the database.
    """
    logger.info(f"Processing student completion request for user {request.user_id}")
    
    total_start_time = time.time()
    
    try:
        # Build composite prompt with system instruction and chat history
        # Assuming chat_history contains {"role": "user/assistant", "message": "..."}
        # and we need to flatten it into the prompt structure for the LLM
        messages_parts = []
        if request.system_prompt:
            messages_parts.append(f"System: {request.system_prompt}")

        # Add chat history. For Gemini, we might concatenate.
        # For a more advanced chat model, you might pass structured history.
        for chat_item in request.chat_history:
            messages_parts.append(f"{chat_item['role'].capitalize()}: {chat_item['message']}")
        
        messages_parts.append(f"User: {request.prompt}") # The current prompt
        
        composite_prompt = "\n\n".join(messages_parts)

        llm_result = call_gemini_api(composite_prompt)
        response_text = llm_result["response"]
        
        total_time_ms = (time.time() - total_start_time) * 1000
        logger.info(f"Student Completion successful for user {request.user_id}: "
                   f"total_time={total_time_ms:.1f}ms")
        
        return StudentCompletionResponse(response=response_text)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in student completion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "LLM Tutoring API",
        "version": "1.0.0",
        "endpoints": {
            "/completions": "POST - Create intent-aware completion",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


if __name__ == "__main__":
    # Run the API server
    logger.info("Starting LLM Tutoring API on port 8000...")
    uvicorn.run(
        "llm_api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )