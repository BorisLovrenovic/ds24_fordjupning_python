import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from test_data import ALL_QUERIES, LABELS

# Sätt stil för alla plottar
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class EmbeddingVisualizer:
    """
    Klass för att visualisera embedding jämförelser.
    """
    
    def __init__(self, results, metrics):
        self.results = results
        self.metrics = metrics
    
    
    def plot_performance_comparison(self, save_path='results/performance.png'):
        """
        Skapa stapeldiagram som jämför alla metrics för alla modeller.
        
        Skapar 4 subplots:
        - Intra-category similarity
        - Inter-category separation
        - Silhouette score
        - Inference time
        """
        print("\nSkapar jämförelse...")
        
        # Hämta modellnamn
        models = list(self.metrics.keys())
        
        # Skapa figur med 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Embedding Model Comparison', 
                     fontsize=16, fontweight='bold')
        
        # Färger för staplarna
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        # Intra-Category Similarity
        ax1 = axes[0, 0]
        values = [self.metrics[m]['intra_similarity'] for m in models]
        ax1.bar(models, values, color=colors[:len(models)])
        ax1.set_ylabel('Score', fontsize=11)
        ax1.set_title('Intra-Category Similarity\n(Högre = Bättre)', 
                      fontsize=12, fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.grid(axis='y', alpha=0.3)
        
        # Lägg till värden ovanpå staplarna
        for i, v in enumerate(values):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
        
        # Inter-Category Separation
        ax2 = axes[0, 1]
        values = [self.metrics[m]['inter_separation'] for m in models]
        ax2.bar(models, values, color=colors[:len(models)])
        ax2.set_ylabel('Score', fontsize=11)
        ax2.set_title('Inter-Category Separation\n(Högre = Bättre)', 
                      fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(axis='y', alpha=0.3)
        
        # Lägg till värden
        for i, v in enumerate(values):
            ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
        
        # Silhouette Score
        ax3 = axes[1, 0]
        values = [self.metrics[m]['silhouette_score'] for m in models]
        ax3.bar(models, values, color=colors[:len(models)])
        ax3.set_ylabel('Silhouette Score', fontsize=11)
        ax3.set_title('Clustering Quality\n(Högre = Bättre)', 
                      fontsize=12, fontweight='bold')
        ax3.set_ylim([0, 1])
        ax3.grid(axis='y', alpha=0.3)
        
        # Lägg till värden
        for i, v in enumerate(values):
            ax3.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
        
        # Inference Time
        ax4 = axes[1, 1]
        # Konvertera till ms
        values = [self.metrics[m]['inference_time_per_query'] * 1000 for m in models]
        ax4.bar(models, values, color=colors[:len(models)])
        ax4.set_ylabel('Tid (ms)', fontsize=11)
        ax4.set_title('Inference Time per Query\n(Lägre = Bättre)', 
                      fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        # Lägg till värden
        for i, v in enumerate(values):
            ax4.text(i, v + (max(values) * 0.02), f'{v:.2f}', 
                    ha='center', fontsize=10)
        
        # Spara figur
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sparad: {save_path}")
        plt.close()
    
    
    def plot_embedding_space_2d(self, model_name, save_path=None):
        """
        Visualisera embeddings i 2D med t-SNE.
        
        """
        print(f"\nSkapar embedding visualisering för {model_name}...")
        
        # Hämta embeddings för denna modell
        embeddings = self.results[model_name]['embeddings']
        
        # Konvertera till 2D med t-SNE
        print(f"Kör t-SNE 2D konvertering...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=5)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Skapa plot
        plt.figure(figsize=(12, 8))
        
        # Definera kategorier och färger
        categories = list(set(LABELS))
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        # Plotta varje kategori separat
        for i, category in enumerate(categories):
            # Hitta alla queries i denna kategori
            indices = [idx for idx, label in enumerate(LABELS) if label == category]
            
            # Plotta punkterna
            plt.scatter(
                embeddings_2d[indices, 0],  # x-koordinater
                embeddings_2d[indices, 1],  # y-koordinater
                label=category.capitalize(),
                color=colors[i],
                s=100,                       
                alpha=0.7,                   
                edgecolors='black',          
                linewidth=1.5
            )
        
        # Formatera plot
        plt.title(f'Embedding Space Visualization - {model_name}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = f'results/embedding_space_{model_name}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sparad {save_path}")
        plt.close()
    
    
    def plot_similarity_heatmap(self, model_name, save_path=None):
        """
        Skapa heatmap.
        
        Grön = Hög similarity (queries är lika)
        Röd = Låg similarity (queries är olika)

        """
        print(f"\nSkapar heatmap för {model_name}...")
        
        # Hämta embeddings
        embeddings = self.results[model_name]['embeddings']
        
        # Beräkna cosine similarity mellan alla query par
        sim_matrix = cosine_similarity(embeddings)
        
        # Skapa plot
        plt.figure(figsize=(14, 12))
        
        # Skapa heatmap
        sns.heatmap(
            sim_matrix,
            cmap='RdYlGn',              
            vmin=0,                     
            vmax=1,                     
            square=True,                
            cbar_kws={'label': 'Cosine Similarity'},
            xticklabels=False,          
            yticklabels=False
        )
        
        # Formatera
        plt.title(f'Query Similarity Heatmap - {model_name}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Queries', fontsize=12)
        plt.ylabel('Queries', fontsize=12)
        
        # Text förklaring
        plt.text(0.5, -0.05, 
                'Grön = Hög similarity | Röd = Låg similarity',
                ha='center', transform=plt.gca().transAxes,
                fontsize=10, style='italic')
        
        if save_path is None:
            save_path = f'results/heatmap_{model_name}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sparad: {save_path}")
        plt.close()
    
    
    def create_all_visualizations(self):
        """
        Skapa alla visualiseringar och spara till results mappen.
        """
        print("\n" + "=" * 60)
        print("Skapar visualiseringar...")
        print("=" * 60)
        
        # Säkerställ att resultsmappen finns
        os.makedirs('results', exist_ok=True)
        
        # 1. Performance comparison (jämför alla modeller)
        self.plot_performance_comparison()
        
        # 2. Embedding space visualiseringar (en per modell)
        for model_name in self.results.keys():
            self.plot_embedding_space_2d(model_name)
        
        # 3. Similarity heatmap (välj multilingual som exempel)
        if 'multilingual' in self.results:
            self.plot_similarity_heatmap('multilingual')
        elif len(self.results) > 0:
            # Om multilingual inte finns, ta första modellen
            first_model = list(self.results.keys())[0]
            self.plot_similarity_heatmap(first_model)
        
        print("\n" + "=" * 60)
        print("Alla visualeringar skapade och klara")
        print("=" * 60)

