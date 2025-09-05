import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, List

# Assuming the following functions and classes are available from the user's code:
# from sqlalchemy.orm import Session
# from models import Interaction
# from metrics import get_batch_score_stats

def get_score_stats_for_all_personas_in_batch(batch_stats: Dict[str, Any], score_type: str, batch_number: int = 1):
    """
    Creates a grouped bar chart to compare score stats across personas and overall for a single batch.

    Args:
        batch_stats: The dictionary of stats returned by get_batch_score_stats.
        score_type: The type of score to plot ('pedagogical_score' or 'persona_score').
    """
    # Prepare data for plotting
    data = []
    for persona, stats in batch_stats.items():
        if persona != "overall":
            scores = stats[score_type]
            data.append({
                'Persona': persona,
                'Statistic': 'Mean',
                'Score': scores['average']
            })
            data.append({
                'Persona': persona,
                'Statistic': 'Median',
                'Score': scores['median']
            })
            data.append({
                'Persona': persona,
                'Statistic': 'Min',
                'Score': scores['min']
            })
            data.append({
                'Persona': persona,
                'Statistic': 'Max',
                'Score': scores['max']
            })

    df = pd.DataFrame(data)

    # Plot the grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    width = 0.2  # Width of the bars
    x = range(len(batch_stats) - 1)  # -1 to exclude 'overall'
    
    mean_scores = [batch_stats[p][score_type]['average'] for p in batch_stats if p != 'overall']
    median_scores = [batch_stats[p][score_type]['median'] for p in batch_stats if p != 'overall']
    min_scores = [batch_stats[p][score_type]['min'] for p in batch_stats if p != 'overall']
    max_scores = [batch_stats[p][score_type]['max'] for p in batch_stats if p != 'overall']
    
    bars1 = ax.bar(x, mean_scores, width, label='Mean')
    bars2 = ax.bar([i + width for i in x], median_scores, width, label='Median')
    bars3 = ax.bar([i + 2 * width for i in x], min_scores, width, label='Min')
    bars4 = ax.bar([i + 3 * width for i in x], max_scores, width, label='Max')

    # Add labels, title, and legend
    ax.set_ylabel('Score')
    ax.set_xlabel('Persona')
    ax.set_title(f'Score Stats by Persona for {score_type.replace("_", " ").title()} in Batch {batch_number}')
    ax.set_xticks([i + 1.5 * width for i in x])
    ax.set_xticklabels([p for p in batch_stats if p != 'overall'])
    ax.legend()
    
    # Add an overall average line for reference
    overall_mean = batch_stats['overall'][score_type]['average']
    ax.axhline(overall_mean, color='r', linestyle='--', label=f'Overall Avg: {overall_mean:.2f}')
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_persona_accuracy_for_batch(persona_accuracy_stats: Dict[str, float], batch_number: int = 1):
    """
    Plots persona accuracy for each persona in a batch and draws an average line.
    Args:
        persona_accuracy_stats: Dict mapping persona name to accuracy percentage (0-100).
        batch_number: The batch number for labeling.
    """
    personas = list(persona_accuracy_stats.keys())
    accuracies = [persona_accuracy_stats[p] for p in personas]
    overall_avg = sum(accuracies) / len(accuracies) if accuracies else 0

    plt.figure(figsize=(10, 6))
    bars = plt.bar(personas, accuracies, color='skyblue', label='Persona Accuracy')
    plt.axhline(overall_avg, color='red', linestyle='--', label=f'Overall Avg: {overall_avg:.2f}%')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Persona')
    plt.title(f'Persona Accuracy by Persona')
    plt.ylim(0, 100)
    plt.legend()

    # Annotate bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{acc:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def compare_batch_stats(batches: Dict[str, Dict[str, Any]], score_type: str):
    """
    Creates charts to compare score stats across multiple batches.

    Args:
        batches: A dictionary of batch stats, e.g., {'Batch 1': stats_1, 'Batch 2': stats_2}.
        score_type: The type of score to plot ('pedagogical_score' or 'persona_score').
    """
    

    # --- Grouped Bar Chart for Persona Mean Scores Across Batches ---
    fig, ax = plt.subplots(figsize=(14, 8))
    persona_list = list(batches[list(batches.keys())[0]].keys())
    if 'overall' in persona_list:
        persona_list.remove('overall')

    bar_width = 0.2
    batch_labels = list(batches.keys())
    x_pos = range(len(persona_list))

    for i, batch_name in enumerate(batch_labels):
        mean_scores = [batches[batch_name][persona][score_type]['average'] for persona in persona_list]
        ax.bar([p + i * bar_width for p in x_pos], mean_scores, bar_width, label=f'{batch_name} Mean')

    ax.set_ylabel('Mean Score')
    ax.set_xlabel('Persona')
    ax.set_title(f'Mean Score Comparison for {score_type.replace("_", " ").title()} Across Batches')
    ax.set_xticks([p + bar_width for p in x_pos])
    ax.set_xticklabels(persona_list)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # --- Line Chart for Overall Mean Scores Across Batches ---
    plt.figure(figsize=(10, 6))
    overall_means = [batches[batch_name]['overall'][score_type]['average'] for batch_name in batch_labels]
    plt.plot(batch_labels, overall_means, marker='o', linestyle='-', color='b')
    plt.xlabel('Batch')
    plt.ylabel('Overall Mean Score')
    plt.title(f'Overall Mean Score Trend for {score_type.replace("_", " ").title()}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Save persona stats for each batch to CSV ---
    import csv
    for batch_name, stats in batches.items():
        csv_filename = f"{batch_name.replace(' ', '_').lower()}_{score_type}.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['persona', 'min', 'max', 'average', 'median'])
            for persona in persona_list:
                persona_stats = stats[persona][score_type]
                writer.writerow([
                    persona,
                    persona_stats['min'],
                    persona_stats['max'],
                    persona_stats['average'],
                    persona_stats['median']
                ])

# Example usage of the plotting functions with hypothetical data
if __name__ == '__main__':
    # You would typically get this data from your database as shown in the prompt
    # batch1_stats = get_batch_score_stats(db, ['lazy', 'curious', 'persistent', 'strategic'], 1, 40)
    # batch2_stats = get_batch_score_stats(db, ['lazy', 'curious', 'persistent', 'strategic'], 41, 80)
    
    # Using mock data for demonstration
    batch1_stats_mock = {
        'lazy': {'pedagogical_score': {'average': 0.7, 'median': 0.75, 'min': 0.5, 'max': 0.9},
                 'persona_score': {'average': 0.8, 'median': 0.82, 'min': 0.6, 'max': 0.95}},
        'curious': {'pedagogical_score': {'average': 0.85, 'median': 0.88, 'min': 0.7, 'max': 1.0},
                    'persona_score': {'average': 0.9, 'median': 0.92, 'min': 0.75, 'max': 1.0}},
        'persistent': {'pedagogical_score': {'average': 0.78, 'median': 0.8, 'min': 0.65, 'max': 0.92},
                       'persona_score': {'average': 0.85, 'median': 0.87, 'min': 0.7, 'max': 0.98}},
        'strategic': {'pedagogical_score': {'average': 0.92, 'median': 0.93, 'min': 0.8, 'max': 1.0},
                      'persona_score': {'average': 0.95, 'median': 0.96, 'min': 0.85, 'max': 1.0}},
        'overall': {'pedagogical_score': {'average': 0.81, 'median': 0.83, 'min': 0.5, 'max': 1.0},
                    'persona_score': {'average': 0.88, 'median': 0.9, 'min': 0.6, 'max': 1.0}}
    }

    batch2_stats_mock = {
        'lazy': {'pedagogical_score': {'average': 0.72, 'median': 0.78, 'min': 0.55, 'max': 0.9},
                 'persona_score': {'average': 0.83, 'median': 0.85, 'min': 0.65, 'max': 0.98}},
        'curious': {'pedagogical_score': {'average': 0.87, 'median': 0.9, 'min': 0.75, 'max': 1.0},
                    'persona_score': {'average': 0.91, 'median': 0.93, 'min': 0.8, 'max': 1.0}},
        'persistent': {'pedagogical_score': {'average': 0.75, 'median': 0.78, 'min': 0.6, 'max': 0.9},
                       'persona_score': {'average': 0.82, 'median': 0.85, 'min': 0.65, 'max': 0.95}},
        'strategic': {'pedagogical_score': {'average': 0.9, 'median': 0.92, 'min': 0.82, 'max': 1.0},
                      'persona_score': {'average': 0.94, 'median': 0.95, 'min': 0.88, 'max': 1.0}},
        'overall': {'pedagogical_score': {'average': 0.8, 'median': 0.82, 'min': 0.55, 'max': 1.0},
                    'persona_score': {'average': 0.87, 'median': 0.89, 'min': 0.65, 'max': 1.0}}
    }
    
    # Plotting for a single batch
    get_score_stats_for_all_personas_in_batch(batch1_stats_mock, 'pedagogical_score')
    get_score_stats_for_all_personas_in_batch(batch1_stats_mock, 'persona_score')
    
    # Plotting to compare multiple batches
    all_batches = {
        'Batch 1': batch1_stats_mock,
        'Batch 2': batch2_stats_mock
    }
    # compare_batch_stats(all_batches, 'pedagogical_score')
    # compare_batch_stats(all_batches, 'persona_score')
