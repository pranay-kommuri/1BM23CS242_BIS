import numpy as np
import matplotlib
matplotlib.use('Agg') # <-- ADD THIS LINE AT THE TOP
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# ... (The rest of your Particle and PSOClustering classes remain unchanged) ...

# --- 1. The Particle Class ---
# Represents a single solution (a set of K centroids) in the swarm.

class Particle:
    def __init__(self, n_clusters, data_dim, min_bounds, max_bounds):
        """
        Initializes a particle.
        """
        low_bound = np.tile(min_bounds, n_clusters)
        high_bound = np.tile(max_bounds, n_clusters)
        self.position = np.random.uniform(low_bound, high_bound, n_clusters * data_dim)
        self.velocity = np.random.rand(n_clusters * data_dim) * 0.1
        self.pbest_position = self.position.copy()
        self.pbest_fitness = float('inf')
        self.current_fitness = float('inf')

# --- 2. The Main PSO Clustering Class ---
# Encapsulates the entire algorithm logic.

class PSOClustering:
    def __init__(self, n_clusters, n_particles, data, max_iter=100, w=0.5, c1=1.5, c2=1.5):
        self.n_clusters = n_clusters
        self.n_particles = n_particles
        self.data = data
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.n_samples, self.data_dim = data.shape
        self.min_bounds = np.min(data, axis=0)
        self.max_bounds = np.max(data, axis=0)
        self.swarm = [Particle(n_clusters, self.data_dim, self.min_bounds, self.max_bounds) for _ in range(n_particles)]
        self.gbest_position = None
        self.gbest_fitness = float('inf')

    def _calculate_fitness(self, centroids):
        total_sse = 0
        for point in self.data:
            distances = np.linalg.norm(centroids - point, axis=1)
            closest_centroid_dist = np.min(distances)
            total_sse += closest_centroid_dist**2
        return total_sse / self.n_samples

    def run(self):
        print("🚀 Starting PSO for Clustering...")
        for i in range(self.max_iter):
            for particle in self.swarm:
                centroids = particle.position.reshape(self.n_clusters, self.data_dim)
                fitness = self._calculate_fitness(centroids)
                particle.current_fitness = fitness
                if fitness < particle.pbest_fitness:
                    particle.pbest_fitness = fitness
                    particle.pbest_position = particle.position.copy()
                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = particle.position.copy()
            for particle in self.swarm:
                r1 = np.random.rand(len(particle.position))
                r2 = np.random.rand(len(particle.position))
                cognitive_velocity = self.c1 * r1 * (particle.pbest_position - particle.position)
                social_velocity = self.c2 * r2 * (self.gbest_position - particle.position)
                particle.velocity = (self.w * particle.velocity) + cognitive_velocity + social_velocity
                particle.position += particle.velocity
                particle.position = np.clip(particle.position,
                                            np.tile(self.min_bounds, self.n_clusters),
                                            np.tile(self.max_bounds, self.n_clusters))
            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1}/{self.max_iter}, Best Fitness (Avg SSE): {self.gbest_fitness:.4f}")
        print("✅ PSO Clustering finished!")
        best_centroids = self.gbest_position.reshape(self.n_clusters, self.data_dim)
        return best_centroids

    def get_cluster_labels(self, centroids):
        labels = np.zeros(self.n_samples, dtype=int)
        for i, point in enumerate(self.data):
            distances = np.linalg.norm(centroids - point, axis=1)
            labels[i] = np.argmin(distances)
        return labels

# --- 3. Main Execution Block ---

if __name__ == "__main__":
    # --- A. Generate Sample Data ---
    N_SAMPLES = 300
    N_CLUSTERS = 4
    X, y_true = make_blobs(n_samples=N_SAMPLES, centers=N_CLUSTERS, cluster_std=0.8, random_state=42)

    # --- B. Run PSO Clustering ---
    pso_cluster = PSOClustering(n_clusters=N_CLUSTERS, n_particles=30, data=X, max_iter=100)
    optimal_centroids = pso_cluster.run()
    final_labels = pso_cluster.get_cluster_labels(optimal_centroids)

    # --- C. Visualize the Results ---
    plt.figure(figsize=(10, 7))
    plt.title('PSO Clustering Results')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=final_labels, cmap='viridis', alpha=0.7, edgecolors='k')
    plt.scatter(optimal_centroids[:, 0], optimal_centroids[:, 1], c='red', marker='X', s=200, label='Optimal Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # --- REPLACE plt.show() WITH THIS ---
    output_filename = 'pso_clusters.png'
    plt.savefig(output_filename)
    print(f"✅ Plot saved successfully to {output_filename}")