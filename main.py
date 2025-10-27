"""
Main script för jämförelse av embedding modeller.
"""

import os
import json
from datetime import datetime
from embedding_comparison import EmbeddingComparison
from visualizations import EmbeddingVisualizer
from test_data import print_dataset_info


# Här definerar vi vilka embedding modeller vi vill jämföra

MODELS = {
    # Small: Snabb och kompakt modell (384 dimensioner, 22M parametrar)
    'small': 'all-MiniLM-L6-v2',
    
    # Multilingual: Stödjer många språk inkl. svenska (384 dimensioner)
    'multilingual': 'paraphrase-multilingual-MiniLM-L12-v2',
    
    # Large: Högsta kvalite men långsammare (768 dimensioner, 110M parametrar)
    'large': 'all-mpnet-base-v2',
}


def main():
    """
    Huvudfunktion som kör hela jämförelseprocessen.
    """
    
    # Visa header
    print("\n" + "=" * 70)
    print(" " * 15 + "Embedding modell Jämförelse")
    print("=" * 70)
    
    print_dataset_info()
    

    print("\n" + "=" * 70)
    print("Steg 1/4: Generera embeddings")
    print("=" * 70)
    print(f"Antal modeller att jämföra: {len(MODELS)}")
    print(f"Modeller: {', '.join(MODELS.keys())}\n")
    
    # Skapa comparison objekt
    comparison = EmbeddingComparison(MODELS)
    
    # Generera embeddings för alla modeller
    results = comparison.run_comparison()
    
    print("\n" + "=" * 70)
    print("STEG 2/4: Beräkna metrics")
    print("=" * 70)
    
    metrics = comparison.calculate_metrics()
    
    print("\n" + "=" * 70)
    print("STEG 3/4: Resultat")
    print("=" * 70)
    
    # Skriv ut metrics för varje modell
    for model_name, model_metrics in metrics.items():
        print(f"\n{'─' * 70}")
        print(f"🔹 {model_name.upper()}")
        print(f"{'─' * 70}")
        print(f"  Intra-Category Similarity:  {model_metrics['intra_similarity']:.4f}")
        print(f"  Inter-Category Separation:  {model_metrics['inter_separation']:.4f}")
        print(f"  Silhouette Score:           {model_metrics['silhouette_score']:.4f}")
        print(f"  Inference Time per Query:   {model_metrics['inference_time_per_query']*1000:.2f} ms")
        print(f"  Embedding Dimension:        {model_metrics['embedding_dimension']}")
    
    print(f"\n{'─' * 70}")
    
    print("\nSparar metrics till JSON...")
    
    # Säkerställ att results mappen finns
    os.makedirs('results', exist_ok=True)
    
    # Spara metrics
    with open('results/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print("Metrics sparade till results/metrics.json")
    
    # Spara även en sammanfattning med timestamp
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models_tested': list(MODELS.keys()),
        'num_queries': 20,
        'num_categories': 4,
        'metrics': metrics
    }
    
    with open('results/summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("Sammanfattning sparad till results/summary.json")


    print("\n" + "=" * 70)
    print("STEG 4/4: Skapa visualiseringar")
    print("=" * 70)
    
    visualizer = EmbeddingVisualizer(results, metrics)    
    # Skapa alla visualiseringar
    visualizer.create_all_visualizations()
    
    print("\n" + "=" * 70)
    print("Klart!!")
    print("=" * 70)
    print("\nResultat finns i 'results' mappen:")
    print("   - metrics.json          (Alla metrics i JSON-format)")
    print("   - summary.json          (Sammanfattning med timestamp)")
    print("   - performance.png       (Jämförelse av alla metrics)")
    print("   - embedding_space_*.png (t-SNE visualiseringar)")
    print("   - heatmap_*.png         (Similarity heatmap)")    
    print("\n" + "=" * 70)
    print(f"Sluttid: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAvbrutet av användaren")
    except Exception as e:
        print(f"\n\nEtt fel uppstod: {e}")
        print("\nFelsökningsinformation:")
        import traceback
        traceback.print_exc()
