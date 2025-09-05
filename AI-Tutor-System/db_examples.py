import csv
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from statistics import mean, median
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, cast, Integer
from database import get_db_context

from models import Interaction
from metrics import get_batch_score_stats
from graph_stats import get_score_stats_for_all_personas_in_batch, compare_batch_stats, plot_persona_accuracy_for_batch


def fetch_all_persona_examples(persona: str, id_start=0, id_end=40) -> List[dict]:
    """
    Fetch all interactions for a given persona.
    """
    with get_db_context() as db:
        interactions = (db.query(Interaction)
                .filter(
                    Interaction.id >= id_start,
                    Interaction.id <= id_end,
                    Interaction.persona == persona,
                    Interaction.persona_accuracy == True
                )
                .all())
        # Convert to dicts before session closes
        return [i.to_dict() for i in interactions]
    
def calculate_overall_score(interaction: dict) -> float:
    """
    Calculate an overall score for an interaction using weigted average of metrics.
    """
    adherence_weight = 0.4
    pedagogical_weight = 0.4
    persona_weight = 0.2

    adherence = interaction.get("adherence")
    pedagogical_score = interaction.get("pedagogical_score")
    persona_score = interaction.get("persona_score")

    if not adherence:
        adherence_score = 0.0  # Heavy penalty
    else:
        adherence_score = 1.0
    
    overall_score = (
        adherence_score * adherence_weight +
        pedagogical_score * pedagogical_weight +
        persona_score * persona_weight
    )
    
    return round(overall_score,2)

def good_bad_examples(interactions: List[dict], limit_good=2, limit_bad=1):
    # Good examples: high across all metrics
    sorted_list = sorted(interactions, key=lambda x: x.get('overall_score', 0), reverse=True)
    good_examples = sorted_list[:limit_good]

    # Bad examples: clear failures but still accurately predicted persona
    bad_examples = sorted_list[-limit_bad:]

    return good_examples, bad_examples

def reinforcement_learning_examples(persona: str) -> dict[str: Any]:
    """
    Create a composite dictionary of examples to be passed to the Tutor for zero shot prompting.
    """

    # step 1: Fetch all persona interactions
    db = get_db_context()
    interactions = fetch_all_persona_examples(persona)

    # step 2: calculate overall scores for each interaction
    for interaction in interactions:
        interaction["overall_score"] = calculate_overall_score(interaction)

    # step 3: get them into good and bad examples
    good_examples, bad_examples = good_bad_examples(interactions)

    # step 4: create a composite dictionary of both examples, only retain query and response and score
    composite_examples = {
        "good_examples": [
            {
                "prompt": example["prompt"],
                "response": example["response"],
                "overall_score": example["overall_score"]
            } for example in good_examples
        ],
        "bad_examples": [
            {
                "prompt": example["prompt"],
                "response": example["response"],
                "overall_score": example["overall_score"]
            } for example in bad_examples
        ]
    }

    return composite_examples


def get_persona_accuracy_percentages(db, personas, id_start=None, id_end=None):
    """
    Returns a dict of persona: accuracy_percentage (0-100) for each persona.
    Optionally restricts to a batch by id_start/id_end.
    """
    accuracy_dict = {}
    for persona in personas:
        query = db.query(
            func.sum(cast(Interaction.persona_accuracy, Integer)).label('sum_acc'),
            func.count(Interaction.persona_accuracy).label('count_acc')
        ).filter(Interaction.persona == persona)
        if id_start is not None and id_end is not None:
            query = query.filter(Interaction.id >= id_start, Interaction.id <= id_end)
        result = query.one()
        sum_acc = result.sum_acc or 0
        count_acc = result.count_acc or 0
        accuracy = (sum_acc * 100 / count_acc) if count_acc > 0 else 0.0
        accuracy_dict[persona] = round(accuracy, 2)
    return accuracy_dict

def print_batch_stats(personas: List[str]):

    with get_db_context() as db:
        batch1_stats = get_batch_score_stats(db, personas, 1, 40)
        batch2_stats = get_batch_score_stats(db, personas, 41, 80)
        batch3_stats = get_batch_score_stats(db, personas, 81, 120)
        batch4_stats = get_batch_score_stats(db, personas, 121, 160)

    print("Batch 1 Stats:\n"+"="*100)
    print(batch1_stats)

    print("="*100)

    print("Batch 2 Stats:\n"+"="*100)
    print(batch2_stats)

    print("="*100)

    print("Batch 3 Stats:\n"+"="*100)
    print(batch3_stats)

    print("="*100)

    print("Batch 4 Stats:\n"+"="*100)
    print(batch4_stats)

    print("="*100)

    # get_score_stats_for_all_personas_in_batch(batch1_stats, 'pedagogical_score', batch_number=1)
    # get_score_stats_for_all_personas_in_batch(batch1_stats, 'persona_score', batch_number=1)

    # get_score_stats_for_all_personas_in_batch(batch2_stats, 'pedagogical_score', batch_number=2)
    # get_score_stats_for_all_personas_in_batch(batch2_stats, 'persona_score', batch_number=2)

    # get_score_stats_for_all_personas_in_batch(batch3_stats, 'pedagogical_score', batch_number=3)
    # get_score_stats_for_all_personas_in_batch(batch3_stats, 'persona_score', batch_number=3)

    # get_score_stats_for_all_personas_in_batch(batch4_stats, 'pedagogical_score', batch_number=4)
    # get_score_stats_for_all_personas_in_batch(batch4_stats, 'persona_score', batch_number=4)

    # Plotting to compare multiple batches
    all_batches = {
        'Batch 1': batch1_stats,
        'Batch 2': batch2_stats,
        'Batch 3': batch3_stats,
        'Batch 4': batch4_stats,
    }

    compare_batch_stats(all_batches, 'pedagogical_score')
    compare_batch_stats(all_batches, 'persona_score')

if __name__ == "__main__":
    
    personas = ["lazy", "curious", "persistent", "strategic"]

    # print all batch stats
    print_batch_stats(personas=personas)

    # graph persona accuracies
    with get_db_context() as db:
        persona_accuracies = get_persona_accuracy_percentages(db, personas)

    # plot_persona_accuracy_for_batch(persona_accuracies)