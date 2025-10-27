import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from test_data import ALL_QUERIES, LABELS


class EmbeddingComparison:
    """
    Klass för att jämföra olika embedding modeller.
    """
    
    def __init__(self, models_dict):
        """
        Initialisera comparison objektet.
        """
        self.models_dict = models_dict
        self.results = {}
        
    
    def load_and_encode(self, model_name, model_id):
        """
        Ladda en modell och generera embeddings för alla queries.
        """
        print(f"\nLaddar {model_name} ({model_id})...")
        
        # Ladda modellen och mät tid
        start_load = time.time()
        model = SentenceTransformer(model_id)
        load_time = time.time() - start_load
        
        # Generera embeddings och mät tid
        print(f"Genererar embeddings för {len(ALL_QUERIES)} queries...")
        start_encode = time.time()
        embeddings = model.encode(ALL_QUERIES, show_progress_bar=True)
        encode_time = time.time() - start_encode
        
        # Spara resultat
        self.results[model_name] = {
            'embeddings': embeddings,           
            'load_time': load_time,             
            'encode_time': encode_time,         
            'embedding_dim': embeddings.shape[1],  
            'model_id': model_id                
        }
        
        # Skriv ut info
        print(f"{model_name} klar!")
        print(f"   Tid: {load_time:.2f}s")
        print(f"   Encoding-tid: {encode_time:.2f}s")
        print(f"   Dimensioner: {embeddings.shape}")
        
        return embeddings
    
    def run_comparison(self):
        """
        Kör jämförelse för alla modeller.
        """
        print("\n" + "=" * 60)
        print("Startar embedding jämförelse...")
        print("=" * 60)
        
        # Loopa igenom alla modeller och generera embeddings
        for name, model_id in self.models_dict.items():
            self.load_and_encode(name, model_id)
        
        print("\n" + "=" * 60)
        print("Alla embeddings genererade")
        print("=" * 60)
        
        return self.results
    
    
    def calculate_metrics(self):
        """
        Beräkna alla metrics för att jämföra modellerna.
        """
        metrics = {}
        
        print("\n" + "=" * 60)
        print("Beräkna metrics")
        print("=" * 60)
        
        # Beräkna metrics för varje modell
        for model_name, data in self.results.items():
            print(f"\nBeräknar för {model_name}...")
            
            embeddings = data['embeddings']
            
            # 1. Intra-category similarity
            intra_sim = self._calculate_intra_category_similarity(embeddings)
            
            # 2. Inter-category separation 
            inter_sep = self._calculate_inter_category_separation(embeddings)
            
            # 3. Silhouette score
            silhouette = self._calculate_clustering_quality(embeddings)
            
            # Spara metrics
            metrics[model_name] = {
                'intra_similarity': float(intra_sim),
                'inter_separation': float(inter_sep),
                'silhouette_score': float(silhouette),
                'inference_time_per_query': float(data['encode_time'] / len(ALL_QUERIES)),
                'embedding_dimension': int(data['embedding_dim'])
            }
            
            print(f"Intra-similarity: {intra_sim:.4f}")
            print(f"Inter-separation: {inter_sep:.4f}")
            print(f"Silhouette: {silhouette:.4f}")
        
        print("\n" + "=" * 60)
        print("Alla metrics beräknade")
        print("=" * 60)
        
        return metrics
    
    
    def _calculate_intra_category_similarity(self, embeddings):
        """
        Beräkna genomsnittlig similarity inom samma kategori.        
        Högre värde = queries i samma kategori ligger närmare varandra
        """
        similarities = []
        unique_labels = set(LABELS)
        
        # Beräkna similarity för varje kategori
        for label in unique_labels:
            # Hitta alla queries i denna kategori
            indices = [i for i, l in enumerate(LABELS) if l == label]
            category_embeds = embeddings[indices]
            
            # Beräkna pairwise cosine similarity
            sim_matrix = cosine_similarity(category_embeds)
            
            # Ta genomsnitt
            mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
            avg_sim = sim_matrix[mask].mean()
            similarities.append(avg_sim)
        
        # Returnera genomsnitt
        return np.mean(similarities)
    
    def _calculate_inter_category_separation(self, embeddings):
        """
        Beräkna genomsnittlig separation mellan olika kategorier.        
        Högre värde = kategorier är mer separerade från varandra
        """
        separations = []
        unique_labels = list(set(LABELS))
        
        # Jämför varje kategoripar
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                # Hitta queries för båda kategorierna
                indices1 = [idx for idx, l in enumerate(LABELS) if l == label1]
                indices2 = [idx for idx, l in enumerate(LABELS) if l == label2]
                
                embeds1 = embeddings[indices1]
                embeds2 = embeddings[indices2]
                
                # Beräkna cross-similarity mellan kategorierna
                cross_sim = cosine_similarity(embeds1, embeds2).mean()
                
                # Omvandla similarity till separation
                separation = 1 - cross_sim
                separations.append(separation)
        
        # Returnera genomsnitt
        return np.mean(separations)
    
    def _calculate_clustering_quality(self, embeddings):
        """
        Beräkna silhouette score för att mäta clustering kvalite.        
        Silhouette score mäter hur väl queries grupperas.
        Värde mellan -1 och 1.
        """
        # Använd K-means clustering med 4 clusters (vi har 4 kategorier)
        n_clusters = len(set(LABELS))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Beräkna silhouette score
        score = silhouette_score(embeddings, cluster_labels)
        
        return score
