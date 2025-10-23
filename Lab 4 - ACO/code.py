import numpy as np
import random
import cv2
import math

class NetworkVisualizer:
    """
    A class to handle all OpenCV visualizations for the ACO network.
    """
    def __init__(self, graph, n_nodes, title="ACO Network Visualization"):
        self.graph = graph
        self.n_nodes = n_nodes
        self.window_title = title
        
        # UI settings
        self.canvas_size = (800, 800)
        self.node_radius = 25
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Pre-calculate node positions for a circular layout
        self.node_positions = self._get_node_positions()

    def _get_node_positions(self):
        """Calculates positions for nodes in a circle to draw them consistently."""
        positions = {}
        center = (self.canvas_size[0] // 2, self.canvas_size[1] // 2)
        radius = min(center) - 80 # Circle radius for layout
        
        for i in range(self.n_nodes):
            angle = 2 * math.pi * i / self.n_nodes
            x = int(center[0] + radius * math.cos(angle))
            y = int(center[1] + radius * math.sin(angle))
            positions[i] = (x, y)
        return positions

    def _draw_base_graph(self, canvas, show_weights=False):
        """Draws the static parts of the graph: nodes and edges."""
        # Draw edges
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if self.graph[i, j] != float('inf'):
                    p1 = self.node_positions[i]
                    p2 = self.node_positions[j]
                    cv2.line(canvas, p1, p2, (200, 200, 200), 1)
                    if show_weights:
                        mid_point = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
                        cv2.putText(canvas, str(int(self.graph[i,j])), (mid_point[0]+5, mid_point[1]-5), self.font, 0.6, (150, 0, 0), 2)
        
        # Draw nodes
        for i in range(self.n_nodes):
            pos = self.node_positions[i]
            cv2.circle(canvas, pos, self.node_radius, (255, 150, 0), -1)
            cv2.circle(canvas, pos, self.node_radius, (0, 0, 0), 2)
            text_size = cv2.getTextSize(str(i), self.font, 0.8, 2)[0]
            text_pos = (pos[0] - text_size[0] // 2, pos[1] + text_size[1] // 2)
            cv2.putText(canvas, str(i), text_pos, self.font, 0.8, (255, 255, 255), 2)

    def draw_initial_graph(self, filename="initial_network_graph.png"):
        """Draws and saves the initial state of the network with latency values."""
        canvas = np.ones((*self.canvas_size, 3), dtype=np.uint8) * 255
        self._draw_base_graph(canvas, show_weights=True)
        
        # Add a title
        cv2.putText(canvas, "Initial Network (with Base Latency)", (20, 40), self.font, 1, (0,0,0), 2)
        
        cv2.imwrite(filename, canvas)
        print(f"✅ Initial graph saved to '{filename}'")

    def update_live_view(self, pheromones, best_path_so_far, iteration, best_cost):
        """Creates and displays a frame for the live visualization."""
        canvas = np.ones((*self.canvas_size, 3), dtype=np.uint8) * 255
        self._draw_base_graph(canvas)
        
        # Normalize pheromones for visualization
        min_p, max_p = pheromones.min(), pheromones.max()
        if max_p == min_p: max_p += 1e-9 # Avoid division by zero

        # Draw pheromone trails
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if self.graph[i, j] != float('inf'):
                    p_level = (pheromones[i, j] - min_p) / (max_p - min_p)
                    thickness = int(1 + p_level * 10)
                    color_intensity = int(100 + p_level * 155)
                    cv2.line(canvas, self.node_positions[i], self.node_positions[j], (255, color_intensity, 0), thickness)
        
        # Draw the best path found so far
        if best_path_so_far:
            for i in range(len(best_path_so_far) - 1):
                p1 = self.node_positions[best_path_so_far[i]]
                p2 = self.node_positions[best_path_so_far[i+1]]
                cv2.line(canvas, p1, p2, (0, 200, 0), 4) # Bright green line
        
        # Display iteration info
        cv2.putText(canvas, f"Iteration: {iteration}", (20, 40), self.font, 1, (0,0,0), 2)
        cv2.putText(canvas, f"Best Cost: {best_cost:.2f}", (20, 80), self.font, 1, (0,0,0), 2)
        
        cv2.imshow(self.window_title, canvas)
        # Wait for a short duration, and allow 'q' to quit
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            return False
        return True
    
    def save_final_graph(self, best_path, final_pheromones, filename="final_optimal_path.png"):
        """Saves the final graph state highlighting the optimal path."""
        canvas = np.ones((*self.canvas_size, 3), dtype=np.uint8) * 255
        self.update_live_view(final_pheromones, best_path, "Final", 0) # Reuse drawing logic
        final_frame = canvas # A bit of a hack, but update_live_view creates the frame we need
        self.update_live_view(final_pheromones, best_path, "Final", 0)
        final_frame = cv2.imread('temp_frame.png') if cv2.getWindowProperty(self.window_title, 0) < 0 else cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        
        # Since update_live_view already draws what we need, we can just grab that frame.
        # Let's redraw it cleanly to be sure.
        canvas = np.ones((*self.canvas_size, 3), dtype=np.uint8) * 255
        self._draw_base_graph(canvas)
        min_p, max_p = final_pheromones.min(), final_pheromones.max()
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if self.graph[i, j] != float('inf'):
                    p_level = (final_pheromones[i, j] - min_p) / (max_p - min_p)
                    thickness = int(1 + p_level * 10)
                    cv2.line(canvas, self.node_positions[i], self.node_positions[j], (200, 200, 200), thickness)
        
        if best_path:
            for i in range(len(best_path) - 1):
                p1 = self.node_positions[best_path[i]]
                p2 = self.node_positions[best_path[i+1]]
                cv2.line(canvas, p1, p2, (0, 220, 0), 6)
        
        cv2.putText(canvas, "Final Optimal Path", (20, 40), self.font, 1, (0,0,0), 2)
        cv2.imwrite(filename, canvas)
        print(f"✅ Final graph with optimal path saved to '{filename}'")
        cv2.destroyAllWindows()


# --- Main ACO Class (slightly modified to use the visualizer) ---

class AntColonyOptimizer:
    def __init__(self, graph, n_ants, n_iterations, alpha, beta, rho, q, visualizer=None):
        self.graph = graph
        self.n_nodes = graph.shape[0]
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.pheromones = np.ones(self.graph.shape) / self.n_nodes
        self.congestion = np.zeros(self.graph.shape)
        self.visualizer = visualizer

    def run(self, start_node, end_node):
        best_path = None
        best_path_cost = float('inf')

        for i in range(self.n_iterations):
            all_paths = self._construct_solutions(start_node, end_node)
            self._update_pheromones(all_paths)

            current_best_path, current_best_cost = self._get_iteration_best(all_paths)
            if current_best_cost < best_path_cost:
                best_path = current_best_path
                best_path_cost = current_best_cost
            
            print(f"Iteration {i+1}/{self.n_iterations}: Best path cost = {current_best_cost:.2f}, Overall best = {best_path_cost:.2f}")

            self._simulate_dynamic_congestion()
            
            # --- VISUALIZATION HOOK ---
            if self.visualizer:
                keep_running = self.visualizer.update_live_view(self.pheromones, best_path, i + 1, best_path_cost)
                if not keep_running:
                    print("Visualization window closed by user. Stopping simulation.")
                    break
        
        # --- SAVE FINAL VISUALIZATION ---
        if self.visualizer:
            self.visualizer.save_final_graph(best_path, self.pheromones)
            
        return best_path, best_path_cost
    
    # ... (All other methods like _construct_solutions, _select_next_node, etc. are identical to the previous code) ...
    def _construct_solutions(self, start_node, end_node):
        all_paths = []
        for _ in range(self.n_ants):
            path = self._construct_path(start_node, end_node)
            path_cost = self._calculate_path_cost(path)
            all_paths.append((path, path_cost))
        return all_paths

    def _construct_path(self, start_node, end_node):
        path = [start_node]
        current_node = start_node
        while current_node != end_node:
            next_node = self._select_next_node(current_node, path)
            if next_node is None: return None
            path.append(next_node)
            current_node = next_node
        return path

    def _select_next_node(self, current_node, visited):
        neighbors = np.where(self.graph[current_node] < float('inf'))[0]
        unvisited_neighbors = [n for n in neighbors if n not in visited]
        if not unvisited_neighbors: return None
        probabilities = []
        total_prob = 0.0
        for neighbor in unvisited_neighbors:
            cost = self.graph[current_node, neighbor] + self.congestion[current_node, neighbor]
            heuristic = 1.0 / (cost + 1e-10)
            pheromone_level = self.pheromones[current_node, neighbor]
            prob_factor = (pheromone_level ** self.alpha) * (heuristic ** self.beta)
            probabilities.append(prob_factor)
            total_prob += prob_factor
        if total_prob == 0: return random.choice(unvisited_neighbors)
        probabilities = [p / total_prob for p in probabilities]
        return random.choices(unvisited_neighbors, weights=probabilities, k=1)[0]

    def _update_pheromones(self, all_paths):
        self.pheromones *= (1 - self.rho)
        for path, cost in all_paths:
            if path is None: continue
            pheromone_deposit = self.q / cost
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                self.pheromones[u, v] += pheromone_deposit
                self.pheromones[v, u] += pheromone_deposit

    def _calculate_path_cost(self, path):
        if path is None: return float('inf')
        total_cost = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            total_cost += self.graph[u, v] + self.congestion[u, v]
        return total_cost
        
    def _get_iteration_best(self, all_paths):
        best_path, best_cost = min(all_paths, key=lambda x: x[1])
        return best_path, best_cost

    def _simulate_dynamic_congestion(self):
        num_congested_links = random.randint(1, self.n_nodes // 2)
        for _ in range(num_congested_links):
            u, v = random.randint(0, self.n_nodes-1), random.randint(0, self.n_nodes-1)
            if self.graph[u, v] < float('inf'):
                self.congestion[u, v] += random.uniform(2, 10)
                self.congestion[v, u] += random.uniform(2, 10)


# --- Main Execution ---
if __name__ == "__main__":
    inf = float('inf')
    network_graph = np.array([
        [inf, 10,  inf, 25,  inf, 30],
        [10,  inf, 15,  inf, inf, inf],
        [inf, 15,  inf, 12,  20,  inf],
        [25,  inf, 12,  inf, 18,  inf],
        [inf, inf, 20,  18,  inf, 10],
        [30,  inf, inf, inf, 10,  inf]
    ])
    
    params = {
        'n_ants': 20, 'n_iterations': 100, 'alpha': 1.0, 
        'beta': 3.0, 'rho': 0.3, 'q': 100
    }
    
    # 1. Initialize the visualizer
    visualizer = NetworkVisualizer(graph=network_graph, n_nodes=network_graph.shape[0])

    # 2. Draw and save the initial network state
    visualizer.draw_initial_graph()
    
    # 3. Create optimizer and pass the visualizer to it
    aco = AntColonyOptimizer(graph=network_graph, visualizer=visualizer, **params)
    
    start_node = 0
    end_node = 4
    
    print(f"\nFinding optimal path from Node {start_node} to Node {end_node}...")
    print("Press 'q' in the visualization window to stop early.")
    
    best_path, best_cost = aco.run(start_node, end_node)
    
    print("\n" + "="*40)
    print("      ACO Traffic Management Results")
    print("="*40)
    if best_path:
        print(f"Optimal Path Found: {' -> '.join(map(str, best_path))}")
        print(f"Total Cost (Latency + Final Congestion): {best_cost:.2f}")
    else:
        print("No path was found between the start and end nodes.")