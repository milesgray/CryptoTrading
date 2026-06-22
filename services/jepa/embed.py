import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
from typing import Optional
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JEPAEmbedder:
    """
    Embedder for JEPA models.
    """
    def __init__(self, model, device: str = None, save_dir: Optional[Path] = None):
        """
        Initialize the embedder.
        
        Args:
            model: JEPA model to embed
            device: Device to use for embedding
            save_dir: Directory to save embeddings
        """
        self.model = model
        
        self.embeddings = np.empty((0, model.encoder.latent_dim))
        self.embeddings_2d = np.empty((0, 2))

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.save_dir = Path(save_dir) if save_dir else None

    def load(self, name: str = 'best_model', folder_path: Optional[Path] = None):
        folder_path = folder_path if folder_path else self.save_dir        
        path = folder_path / f'{name}.pt'
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.embeddings = checkpoint.get('embeddings', None)
        self.embeddings_2d = checkpoint.get('embeddings_2d', None)

    def save(self, name: str = 'best_model', folder_path: Optional[Path] = None, **kwargs):
        folder_path = folder_path if folder_path else self.save_dir
        if folder_path is not None:
            folder_path = Path(folder_path)
            folder_path.mkdir(exist_ok=True, parents=True)
        else:
            logger.warning("No save directory specified, embeddings will not be saved")
            return
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'embeddings': self.embeddings,
            'embeddings_2d': self.embeddings_2d
        }, folder_path / f'{name}.pt')

    def make_embeddings(self, datasets: dict, save: bool = True, make_labels: bool = False) -> np.ndarray:
        """
        Create embeddings for the given datasets.
        
        Args:
            datasets: Dictionary of datasets to create embeddings for
            save: Whether to save the embeddings
            make_labels: Whether to create labels for the embeddings
            
        Returns:
            Embeddings array
        """
        def embed(model, dataset, label, embeddings):
            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=2
            )
            labels = []
            for batch in tqdm(loader, desc=f'Creating {label.title()} Embeddings'):
                x_context = batch['x_context'].to(self.device)
                
                embeddings_batch = model.encoder(x_context)
                embeddings.append(embeddings_batch.detach().cpu())        
                labels.append(label)
            return embeddings, labels

        embeddings = [] 
        labels = []

        for key, dataset in datasets.items():
            embeddings_batch, labels_batch = embed(self.model, dataset, key, embeddings)
            embeddings.extend(embeddings_batch)
            labels.extend(labels_batch)
            logger.info(f"Processed {key} dataset: {len(embeddings_batch)} embeddings")
            
        self.embeddings = np.vstack(embeddings)
        logger.info(f"Total embeddings: {len(self.embeddings)}")

        if save:
            self.save()
        
        if make_labels:
            return self.embeddings, labels
        else:
            return self.embeddings

    def transform_embeddings(self, embeddings: np.ndarray, method: str = 'pacmap') -> np.ndarray:
        """
        Transform embeddings to 2D using t-SNE or PaCMAP.
        
        Args:
            embeddings: Input embeddings array
            method: Transformation method ('tsne' or 'pacmap')
            
        Returns:
            2D embeddings array
        """
        from sklearn.manifold import TSNE
        from pacmap import PaCMAP
        
        # Apply t-SNE to reduce dimensionality to 2D
        if method == 'tsne':
            manifold = TSNE(n_components=2, random_state=42, perplexity=30)
        elif method == 'pacmap':
            manifold = PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0) 
        else:
            raise ValueError(f"Unknown method: {method}")
            
        embeddings_2d = manifold.fit_transform(embeddings)
        return embeddings_2d

    def make_knn_labels(self, embeddings_2d: np.ndarray = None, n_neighbors: int = 10, n_clusters: int = 10) -> tuple:
        """
        Use KNN + Spectral Clustering to determine labels and centroids.
        
        Args:
            embeddings: Input embeddings array
            n_neighbors: Number of neighbors for KNN graph
            n_clusters: Number of clusters for spectral clustering
            
        Returns:
            tuple: (labels, centroids)
        """
        from sklearn.neighbors import NearestNeighbors
        from sklearn.cluster import SpectralClustering
        
        embeddings_2d = embeddings_2d if embeddings_2d is not None else self.embeddings_2d
        assert embeddings_2d is not None, "embeddings_2d must be provided or self.embeddings_2d must be set"
        # Fit KNN to find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(embeddings_2d)
        distances, indices = nbrs.kneighbors(embeddings_2d)
        
        # Create adjacency matrix based on KNN connectivity
        n_samples = len(embeddings_2d)
        adjacency = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            adjacency[i, indices[i]] = 1
            adjacency[indices[i], i] = 1  # Make symmetric
        
        # Use spectral clustering on the adjacency matrix
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        
        # Convert adjacency to similarity matrix for spectral clustering
        similarity = adjacency
        labels = clustering.fit_predict(similarity)
        
        # Compute cluster centroids
        unique_labels = np.unique(labels)
        centroids = []
        for label in unique_labels:
            cluster_embeddings = embeddings_2d[labels == label]
            centroid = cluster_embeddings.mean(axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)
        
        logger.info(f"KNN+Spectral clustering complete: {len(unique_labels)} clusters found")
        logger.info(f"Cluster sizes: {[np.sum(labels == label) for label in unique_labels]}")
        
        return labels, centroids

    def make_dbscan_labels(self, embeddings_2d: np.ndarray = None, n_neighbors: int = 10, eps: float = 0.5, min_samples: int = 5) -> tuple:
        """
        Use KNN + DBSCAN to determine labels and centroids.
        
        Args:
            embeddings: Input embeddings array
            n_neighbors: Number of neighbors for KNN graph
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN min_samples parameter
            
        Returns:
            tuple: (labels, centroids)
        """
        from sklearn.neighbors import NearestNeighbors
        from sklearn.cluster import DBSCAN

        embeddings_2d = embeddings_2d if embeddings_2d is not None else self.embeddings_2d
        assert embeddings_2d is not None, "embeddings_2d must be provided or self.embeddings_2d must be set"
        
        # Fit KNN to find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(embeddings_2d)
        distances, indices = nbrs.kneighbors(embeddings_2d)
        
        # Create adjacency matrix based on KNN connectivity
        n_samples = len(embeddings_2d)
        adjacency = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            adjacency[i, indices[i]] = 1
            adjacency[indices[i], i] = 1  # Make symmetric
        
        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        
        # Convert adjacency to distance matrix for DBSCAN
        # Use 1 - similarity as distance (with small epsilon to avoid zero distance)
        distance_matrix = 1 - adjacency + 0.001
        labels = clustering.fit_predict(distance_matrix)
        
        # Compute cluster centroids (excluding noise points)
        unique_labels = np.unique(labels)
        # Filter out noise points (label = -1) for centroid computation
        cluster_labels = unique_labels[unique_labels != -1]
        centroids = []
        for label in cluster_labels:
            cluster_embeddings = embeddings_2d[labels == label]
            centroid = cluster_embeddings.mean(axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)
        
        n_clusters_found = len(cluster_labels)
        n_noise_points = np.sum(labels == -1)
        
        logger.info(f"DBSCAN clustering complete: {n_clusters_found} clusters found")
        logger.info(f"Noise points: {n_noise_points} ({n_noise_points/n_samples*100:.1f}%)")
        logger.info(f"Cluster sizes: {[np.sum(labels == label) for label in cluster_labels]}")
        
        return labels, centroids
    
    def plot_embeddings(self, embeddings_2d=None, labels=None):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        embeddings_2d = embeddings_2d if embeddings_2d is not None else self.embeddings_2d
        # Prepare labels for coloring
        if labels is None:
            labels = np.zeros(len(embeddings_2d))
        unique_labels = np.unique(labels)
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        numerical_labels = np.array([label_to_int[label] for label in labels])
        
        # Define a colormap for the labels
        colors = ['#D74288', '#88D742', '#4288D7', '#D78842', '#42D788', '#8842D7'] # Assign colors to different labels
        cmap = mcolors.ListedColormap(colors[:len(unique_labels)])

        # Plot the 2D embeddings
        plt.figure(figsize=(12, 10))
        _ = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=numerical_labels,
            cmap=cmap,
            alpha=0.7,
            s=0.6,
        )
        
        # Create a legend for the colors (labels)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                    markerfacecolor=cmap(label_to_int[label]), markersize=10)
                        for label in unique_labels]
        plt.legend(handles=legend_elements, title="Split")
        
        plt.title('t-SNE Visualization of Embeddings (Colored by Split)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True)


