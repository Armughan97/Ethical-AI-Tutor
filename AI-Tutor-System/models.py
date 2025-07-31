"""
SQLAlchemy models for the tutoring system metrics database.
"""

from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from typing import Optional

Base = declarative_base()


class Interaction(Base):
    """
    Model representing a single interaction between a user and the tutoring system.
    Stores metrics for intent detection, LLM response times, and adherence tracking.
    """
    __tablename__ = "interactions"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Conversation tracking
    conversation_id = Column(String(64), nullable=False, index=True)  # UUID or unique string for each conversation
    message_id = Column(Integer, nullable=False)  # Sequence/order of message in conversation
    # role = Column(String(16), nullable=False)  # 'tutor' or 'student'

    # User and persona information
    user_id = Column(String(50), nullable=False, index=True)
    persona = Column(String(50), nullable=False, index=True)
    predicted_persona = Column(String(50), nullable=False, index=True)
    
    # Intent classification
    intent = Column(String(20), nullable=False)  # Genuine, Manipulative, Spam
    
    # Conversation data
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    
    # Performance metrics (all times in milliseconds)
    intent_time_ms = Column(Float, nullable=False)
    llm_time_ms = Column(Float, nullable=False)
    total_time_ms = Column(Float, nullable=False)
    
    # Response metrics
    response_tokens = Column(Integer, nullable=False)
    
    # Quality metrics
    adherence = Column(Boolean, nullable=False)  # Whether response follows intent rules
    turn_number = Column(Integer, nullable=False)  # Turn number in conversation
    
    # Timestamp
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Scoring metrics
    persona_accuracy = Column(Boolean, nullable=False)
    pedagogical_score = Column(Float, nullable=False, default=0.5)
    persona_score = Column(Float, nullable=False, default=0.5)
    
    def __repr__(self) -> str:
        return (f"<Interaction(id={self.id}, conversation_id='{self.conversation_id}', message_id={self.message_id}, "
                f"user_id='{self.user_id}', persona='{self.persona}', predicted_persona='{self.predicted_persona}', intent='{self.intent}', "
                f"turn={self.turn_number}, adherence={self.adherence})>")

    def to_dict(self) -> dict:
        """Convert interaction to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'conversation_id': self.conversation_id,
            'message_id': self.message_id,
            'user_id': self.user_id,
            'persona': self.persona,
            'predicted_persona': self.predicted_persona,
            'intent': self.intent,
            'prompt': self.prompt,
            'response': self.response,
            'intent_time_ms': self.intent_time_ms,
            'llm_time_ms': self.llm_time_ms,
            'total_time_ms': self.total_time_ms,
            'response_tokens': self.response_tokens,
            'adherence': self.adherence,
            'turn_number': self.turn_number,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'persona_accuracy': self.persona_accuracy,
            'pedagogical_score': self.pedagogical_score,
            'persona_score': self.persona_score
        }