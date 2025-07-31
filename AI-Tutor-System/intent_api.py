"""
Intent Classification API Service

FastAPI service that uses a HuggingFace BERT model to classify user intents
as Genuine, Manipulative, or Spam.
"""

import logging
import time
from typing import Dict, Any
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

MODEL_NAME = "prajjwal1/bert-small"
WEIGHTS_PATH = "best_model_acc.bin"

def load_model():
    global tokenizer, model, device

    # 1) set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2) tokenizer comes from the same base you trained on
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3) build a config with the right number of labels & id2label mapping
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(INTENT_LABELS),
        id2label=INTENT_LABELS,
        label2id={label: idx for idx, label in INTENT_LABELS.items()}
    )

    # 4) initialize a fresh model
    model = AutoModelForSequenceClassification.from_config(config)

    # 5) load your trained weights
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)

    # 6) move to device & eval
    model.to(device)
    model.eval()

    logger.info("✅ Loaded custom intent‐classifier from %s", WEIGHTS_PATH)


# Configure your application's logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Uvicorn log config
# This ensures Uvicorn's logs (including access logs) go to stdout
log_config = {
    "version": 1,
    "disable_existing_loggers": False, # Keep existing loggers intact
    "formatters": {
        "standard": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelname)s - %(name)s - %(message)s",
            "use_colors": True,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelname)s - %(name)s - %(message)s', # Uvicorn's default access log format
            "use_colors": True,
        },
    },
    "handlers": {
        "default": {
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout", # Send to stdout
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout", # Send to stdout
        },
    },
    "loggers": {
        # Your application's logger (configured above by basicConfig)
        "": {  # Root logger
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        # Uvicorn's error logger
        "uvicorn.error": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        # Uvicorn's access logger
        "uvicorn.access": {
            "handlers": ["access"],
            "level": "INFO", # Set to INFO to see 200 OK messages
            "propagate": False,
        },
    },
}

# Initialize FastAPI app
app = FastAPI(
    title="Intent Classification API",
    description="BERT-based intent classifier for tutoring system",
    version="1.0.0"
)

# Intent mapping
INTENT_LABELS = {
    0: "Genuine",
    1: "Manipulative"
}

# Global variables for model and tokenizer
tokenizer = None
model = None
device = None


class IntentRequest(BaseModel):
    """Request model for intent prediction."""
    question: str
    
    class Config:
        schema_extra = {
            "example": {
                "question": "Can you help me understand calculus derivatives?"
            }
        }


class IntentResponse(BaseModel):
    """Response model for intent prediction."""
    intent: str
    confidence: float
    processing_time_ms: float
    
    class Config:
        schema_extra = {
            "example": {
                "intent": "Genuine",
                "confidence": 0.95,
                "processing_time_ms": 45.2
            }
        }


def load_bert_model(model_path: str = "path/to/intent-classifier") -> None:
    """
    Load the BERT model and tokenizer for intent classification.
    
    Args:
        model_path: Path to the HuggingFace model directory or model name
    """
    global tokenizer, model, device
    
    try:
        logger.info(f"Loading BERT model from: {model_path}")
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load tokenizer and model
        # Note: In real deployment, replace with actual model path
        # For demo purposes, we'll use a general BERT model for text classification
        model_name = "distilbert-base-uncased"  # Fallback model
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=3  # Genuine, Manipulative, Spam
            )
            logger.info("Custom intent classifier loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load custom model from {model_path}: {e}")
            logger.info(f"Falling back to {model_name} for demonstration")
            
            # Fallback to a pre-trained model (for demo purposes)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=3
            )
        
        model.to(device)
        model.eval()
        
        logger.info("Model loaded and ready for inference")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def predict_intent(question: str) -> Dict[str, Any]:
    """
    Predict intent for a given question.
    
    Args:
        question: The user's question/prompt
        
    Returns:
        Dictionary containing intent, confidence, and processing time
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Tokenize input
        inputs = tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get predicted class and confidence
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Map prediction to intent label
        intent = INTENT_LABELS.get(predicted_class, "Unknown")
        
        return {
            "intent": intent,
            "confidence": confidence,
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def warm_up_model():
    test_input = "Hello!"
    _ = predict_intent(test_input)  # throw away result, just for loading
    logger.info("Model warmed up before first request")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Intent Classification API…")
    load_model()
    warm_up_model()
    # load_bert_model()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None
    }


@app.post("/predict_intent", response_model=IntentResponse)
async def predict_intent_endpoint(request: IntentRequest):
    """
    Predict the intent of a user question.
    
    Returns one of: Genuine, Manipulative, or Spam
    """
    print(f"DEBUG INTENT_API: Predicting intent for question: {request.question[:100]}...") # <-- ADD THIS LINE
    logger.info(f"Predicting intent for question: {request.question[:100]}...")
    
    try:
        result = predict_intent(request.question)
        
        logger.info(f"Intent prediction: {result['intent']} "
                   f"(confidence: {result['confidence']:.3f}, "
                   f"time: {result['processing_time_ms']:.1f}ms)")
        
        return IntentResponse(**result)
        
    except Exception as e:
        logger.error(f"Intent prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Intent prediction failed")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Intent Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/predict_intent": "POST - Classify intent of user question",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    # Run the API server
    logger.info("Starting Intent Classification API on port 9000...")
    uvicorn.run(
        "intent_api:app",
        host="127.0.0.1",
        port=9000,
        reload=False,
        log_config=log_config, # Pass the custom log config here
        log_level="info", # This sets the overall uvicorn log level
    ) 