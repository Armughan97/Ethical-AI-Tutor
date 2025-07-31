"""
Metrics module for computing and logging tutoring system performance metrics.

This module provides functions to log interactions to the database and compute
various metrics like adherence percentage, response times, and failure rates.
"""

import csv
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from statistics import mean, median
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from models import Interaction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_interaction(
    db: Session,
    user_id: str,
    persona: str,
    intent: str,
    prompt: str,
    response: str,
    intent_time_ms: float,
    llm_time_ms: float,
    total_time_ms: float,
    response_tokens: int,
    adherence: bool,
    turn_number: int = 1
) -> Interaction:
    """
    Log an interaction to the database.
    
    Args:
        db: Database session
        user_id: User identifier
        persona: User persona (lazy, curious, persistent, strategic)
        intent: Detected intent (Genuine, Manipulative, Spam)
        prompt: User's input prompt
        response: LLM response
        intent_time_ms: Time to detect intent in milliseconds
        llm_time_ms: Time for LLM response in milliseconds
        total_time_ms: Total round-trip time in milliseconds
        response_tokens: Number of tokens in response
        adherence: Whether response adheres to intent rules
        turn_number: Turn number in the conversation
        
    Returns:
        The created Interaction object
    """
    try:
        interaction = Interaction(
            user_id=user_id,
            persona=persona,
            intent=intent,
            prompt=prompt,
            response=response,
            intent_time_ms=intent_time_ms,
            llm_time_ms=llm_time_ms,
            total_time_ms=total_time_ms,
            response_tokens=response_tokens,
            adherence=adherence,
            turn_number=turn_number,
            timestamp=datetime.utcnow()
        )
        
        db.add(interaction)
        db.commit()
        db.refresh(interaction)
        
        logger.info(f"Logged interaction {interaction.id} for user {user_id}")
        return interaction
        
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}")
        db.rollback()
        raise

def get_persona_accuracy_percentage(
    db: Session,
    user_id: Optional[str] = None,
    persona: Optional[str] = None
) -> float:
    """
    Calculate persona accuracy percentage (persona == predicted_persona).
    """
    query = db.query(Interaction)
    if user_id:
        query = query.filter(Interaction.user_id == user_id)
    if persona:
        query = query.filter(Interaction.persona == persona)
    total = query.count()
    if total == 0:
        return 0.0
    correct = query.filter(Interaction.persona == Interaction.predicted_persona).count()
    percentage = (correct / total) * 100
    logger.info(f"Persona accuracy: {correct}/{total} = {percentage:.1f}%")
    return percentage

def get_adherence_percentage(
    db: Session,
    user_id: Optional[str] = None,
    persona: Optional[str] = None
) -> float:
    """
    Calculate adherence percentage for interactions.
    
    Args:
        db: Database session
        user_id: Filter by specific user (optional)
        persona: Filter by specific persona (optional)
        
    Returns:
        Adherence percentage (0-100)
    """
    query = db.query(Interaction)
    
    if user_id:
        query = query.filter(Interaction.user_id == user_id)
    if persona:
        query = query.filter(Interaction.persona == persona)
    
    total_interactions = query.count()
    if total_interactions == 0:
        return 0.0
    
    adherent_interactions = query.filter(Interaction.adherence == True).count()
    
    percentage = (adherent_interactions / total_interactions) * 100
    logger.info(f"Adherence: {adherent_interactions}/{total_interactions} = {percentage:.1f}%")
    
    return percentage

def get_score_stats(
    db: Session,
    score_type: str = "pedagogical",
    user_id: Optional[str] = None,
    persona: Optional[str] = None
) -> dict:
    """
    Get min, max, average, and median for pedagogical_score or persona_score.
    """
    if score_type == "pedagogical":
        column = Interaction.pedagogical_score
    elif score_type == "persona":
        column = Interaction.persona_score
    else:
        raise ValueError("score_type must be 'pedagogical' or 'persona'")
    query = db.query(column)
    if user_id:
        query = query.filter(Interaction.user_id == user_id)
    if persona:
        query = query.filter(Interaction.persona == persona)
    scores = [row[0] for row in query.all()]
    if not scores:
        return {"min": 0.0, "max": 0.0, "average": 0.0, "median": 0.0, "total": 0}
    return {
        "min": min(scores),
        "max": max(scores),
        "average": mean(scores),
        "median": median(scores),
        "total": len(scores)
    }

def get_avg_response_time(
    db: Session,
    user_id: Optional[str] = None,
    persona: Optional[str] = None,
    metric_type: str = "llm"
) -> float:
    """
    Calculate average response time.
    
    Args:
        db: Database session
        user_id: Filter by specific user (optional)
        persona: Filter by specific persona (optional)
        metric_type: Type of time metric ("llm", "intent", "total")
        
    Returns:
        Average time in milliseconds
    """
    query = db.query(Interaction)
    
    if user_id:
        query = query.filter(Interaction.user_id == user_id)
    if persona:
        query = query.filter(Interaction.persona == persona)
    
    # Select the appropriate time column
    if metric_type == "llm":
        time_column = Interaction.llm_time_ms
    elif metric_type == "intent":
        time_column = Interaction.intent_time_ms
    elif metric_type == "total":
        time_column = Interaction.total_time_ms
    else:
        raise ValueError(f"Invalid metric_type: {metric_type}")
    
    result = query.with_entities(func.avg(time_column)).scalar()
    return result if result is not None else 0.0


def get_interactions_before_failure(
    db: Session,
    user_id: Optional[str] = None,
    persona: Optional[str] = None
) -> Dict[str, float]:
    """
    Calculate average interactions before failure (non-adherent response).
    
    Args:
        db: Database session
        user_id: Filter by specific user (optional)
        persona: Filter by specific persona (optional)
        
    Returns:
        Dictionary with statistics about interactions before failure
    """
    query = db.query(Interaction)
    
    if user_id:
        query = query.filter(Interaction.user_id == user_id)
    if persona:
        query = query.filter(Interaction.persona == persona)
    
    # Get interactions ordered by user and timestamp
    interactions = query.order_by(Interaction.user_id, Interaction.timestamp).all()
    
    # Group by user and calculate interactions before first failure
    user_failures = {}
    
    for interaction in interactions:
        uid = interaction.user_id
        if uid not in user_failures:
            user_failures[uid] = {"turns_before_failure": 0, "had_failure": False}
        
        if not interaction.adherence and not user_failures[uid]["had_failure"]:
            # First failure for this user
            user_failures[uid]["had_failure"] = True
        elif not user_failures[uid]["had_failure"]:
            # Still counting turns before failure
            user_failures[uid]["turns_before_failure"] += 1
    
    # Calculate statistics
    failure_counts = [data["turns_before_failure"] for data in user_failures.values()]
    
    if not failure_counts:
        return {"average": 0.0, "median": 0.0, "total_users": 0}
    
    return {
        "average": mean(failure_counts),
        "median": median(failure_counts),
        "total_users": len(failure_counts),
        "users_with_failures": sum(1 for data in user_failures.values() if data["had_failure"])
    }


def get_token_stats(
    db: Session,
    user_id: Optional[str] = None,
    persona: Optional[str] = None
) -> Dict[str, float]:
    """
    Calculate token statistics for responses.
    
    Args:
        db: Database session
        user_id: Filter by specific user (optional)
        persona: Filter by specific persona (optional)
        
    Returns:
        Dictionary with min, max, average, and median token counts
    """
    query = db.query(Interaction.response_tokens)
    
    if user_id:
        query = query.filter(Interaction.user_id == user_id)
    if persona:
        query = query.filter(Interaction.persona == persona)
    
    token_counts = [count[0] for count in query.all()]
    
    if not token_counts:
        return {"min": 0, "max": 0, "average": 0.0, "median": 0.0, "total_responses": 0}
    
    return {
        "min": min(token_counts),
        "max": max(token_counts),
        "average": mean(token_counts),
        "median": median(token_counts),
        "total_responses": len(token_counts)
    }


def get_persona_summary(db: Session, persona: str) -> Dict[str, Any]:
    """
    Get comprehensive summary statistics for a specific persona.
    
    Args:
        db: Database session
        persona: Persona to analyze
        
    Returns:
        Dictionary with all relevant metrics for the persona
    """
    summary = {
        "persona": persona,
        "adherence_percentage": get_adherence_percentage(db, persona=persona),
        "avg_response_time_ms": get_avg_response_time(db, persona=persona, metric_type="llm"),
        "avg_intent_time_ms": get_avg_response_time(db, persona=persona, metric_type="intent"),
        "avg_total_time_ms": get_avg_response_time(db, persona=persona, metric_type="total"),
        "interactions_before_failure": get_interactions_before_failure(db, persona=persona),
        "token_stats": get_token_stats(db, persona=persona),
        "persona_accuracy_percentage": get_persona_accuracy_percentage(db, persona=persona),
        "pedagogical_score_stats": get_score_stats(db, score_type="pedagogical", persona=persona),
        "persona_score_stats": get_score_stats(db, score_type="persona", persona=persona),
    }
    
    # Add total interactions count
    total_interactions = db.query(Interaction).filter(Interaction.persona == persona).count()
    summary["total_interactions"] = total_interactions
    
    # Add intent distribution
    intent_counts = db.query(
        Interaction.intent, 
        func.count(Interaction.id)
    ).filter(
        Interaction.persona == persona
    ).group_by(Interaction.intent).all()
    
    summary["intent_distribution"] = {intent: count for intent, count in intent_counts}
    
    return summary


def export_metrics_csv(db: Session, output_path: str) -> str:
    """
    Export all interaction metrics to a CSV file.
    
    Args:
        db: Database session
        output_path: Path where CSV should be saved
        
    Returns:
        Path to the exported CSV file
    """
    try:
        interactions = db.query(Interaction).order_by(Interaction.timestamp).all()
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'id', 'user_id', 'persona', 'intent', 'prompt', 'response',
                'intent_time_ms', 'llm_time_ms', 'total_time_ms', 'response_tokens',
                'adherence', 'turn_number', 'timestamp'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for interaction in interactions:
                writer.writerow({
                    'id': interaction.id,
                    'user_id': interaction.user_id,
                    'persona': interaction.persona,
                    'intent': interaction.intent,
                    'prompt': interaction.prompt,
                    'response': interaction.response,
                    'intent_time_ms': interaction.intent_time_ms,
                    'llm_time_ms': interaction.llm_time_ms,
                    'total_time_ms': interaction.total_time_ms,
                    'response_tokens': interaction.response_tokens,
                    'adherence': interaction.adherence,
                    'turn_number': interaction.turn_number,
                    'timestamp': interaction.timestamp.isoformat() if interaction.timestamp else None
                })
        
        logger.info(f"Exported {len(interactions)} interactions to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to export metrics CSV: {e}")
        raise


def get_system_overview(db: Session) -> Dict[str, Any]:
    """
    Get a high-level overview of system performance metrics.
    
    Args:
        db: Database session
        
    Returns:
        Dictionary with system-wide metrics
    """
    total_interactions = db.query(Interaction).count()
    unique_users = db.query(Interaction.user_id).distinct().count()
    
    if total_interactions == 0:
        return {
            "total_interactions": 0,
            "unique_users": 0,
            "overall_adherence_percentage": 0.0,
            "avg_response_time_ms": 0.0,
            "persona_performance": {},
            "overall_persona_accuracy_percentage": 0.0,
            "overall_pedagogical_score_stats": {},
            "overall_persona_score_stats": {},
        }
    
    # Get personas
    personas = db.query(Interaction.persona).distinct().all()
    persona_list = [p[0] for p in personas]
    
    # Get persona summaries
    persona_performance = {}
    for persona in persona_list:
        persona_performance[persona] = get_persona_summary(db, persona)
    
    return {
        "total_interactions": total_interactions,
        "unique_users": unique_users,
        "overall_adherence_percentage": get_adherence_percentage(db),
        "avg_response_time_ms": get_avg_response_time(db, metric_type="llm"),
        "avg_intent_time_ms": get_avg_response_time(db, metric_type="intent"),
        "avg_total_time_ms": get_avg_response_time(db, metric_type="total"),
        "overall_token_stats": get_token_stats(db),
        "persona_performance": persona_performance,
        "intent_distribution": dict(db.query(
            Interaction.intent, 
            func.count(Interaction.id)
        ).group_by(Interaction.intent).all()),
        "overall_persona_accuracy_percentage": get_persona_accuracy_percentage(db),
        "overall_pedagogical_score_stats": get_score_stats(db, score_type="pedagogical"),
        "overall_persona_score_stats": get_score_stats(db, score_type="persona")
    } 