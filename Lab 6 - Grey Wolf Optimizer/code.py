import cv2
import numpy as np
from skimage import measure
import os
import time

def calculate_fitness(grayscale_image):
    """
    Calculates the fitness of a grayscale image.
    We use Shannon Entropy: higher entropy = more detail/information.
    """
    # Ensure image is in 8-bit unsigned integer format
    if grayscale_image.dtype != np.uint8:
        grayscale_image = grayscale_image.astype(np.uint8)
        
    entropy = measure.shannon_entropy(grayscale_image)
    return entropy

def apply_weights(color_image, weights):
    """
    Applies the [Wr, Wg, Wb] weights to a color image to create a grayscale image.
    Note: OpenCV loads images as BGR, so weights are applied as [Wb, Wg, Wr]
    """
    Wb, Wg, Wr = weights
    # Split the B, G, R channels
    b, g, r = cv2.split(color_image)
    
    # Apply the weighted sum.
    # We use .astype(np.float64) to prevent overflow during multiplication
    grayscale_image = (Wb * b.astype(np.float64) +
                       Wg * g.astype(np.float64) +
                       Wr * r.astype(np.float64))
                       
    # Normalize to 0-255 range and convert to 8-bit integer
    grayscale_image = np.clip(grayscale_image, 0, 255).astype(np.uint8)
    return grayscale_image

def run_gwo(color_image, num_wolves=20, max_iterations=50):
    """
    Runs the Grey Wolf Optimizer to find the best grayscale weights.
    """
    print("Starting Grey Wolf Optimizer...")
    
    # The search space is 3-dimensional [Wb, Wg, Wr]
    dim = 3
    
    # Initialize the positions (weights) of the wolves
    # We initialize randomly and then normalize so Wb + Wg + Wr = 1
    positions = np.random.rand(num_wolves, dim)
    positions /= np.sum(positions, axis=1, keepdims=True)
    
    # Initialize fitness scores
    fitness = np.zeros(num_wolves)
    
    # Initialize Alpha, Beta, and Delta wolves (position and fitness)
    # We start with negative infinity because we want to MAXIMIZE entropy
    alpha_pos = np.zeros(dim)
    alpha_score = -np.inf
    
    beta_pos = np.zeros(dim)
    beta_score = -np.inf
    
    delta_pos = np.zeros(dim)
    delta_score = -np.inf
    
    start_time = time.time()
    
    # Main optimization loop
    for iteration in range(max_iterations):
        # 1. Calculate fitness for each wolf
        for i in range(num_wolves):
            # Apply the wolf's weights to the image
            weights = positions[i]
            grayscale_img = apply_weights(color_image, weights)
            
            # Calculate the fitness (entropy)
            fitness[i] = calculate_fitness(grayscale_img)
            
            # Update Alpha, Beta, Delta
            if fitness[i] > alpha_score:
                # This wolf is the new best
                alpha_score = fitness[i]
                alpha_pos = positions[i].copy()
            elif fitness[i] > beta_score:
                # This wolf is the new second best
                beta_score = fitness[i]
                beta_pos = positions[i].copy()
            elif fitness[i] > delta_score:
                # This wolf is the new third best
                delta_score = fitness[i]
                delta_pos = positions[i].copy()
        
        # Linearly decrease 'a' from 2 to 0
        # This parameter controls the exploration/exploitation balance
        a = 2 - iteration * (2 / max_iterations)
        
        # 2. Update the position of all (Omega) wolves
        for i in range(num_wolves):
            # GWO hunting equations
            r1, r2 = np.random.rand(2)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * alpha_pos - positions[i])
            X1 = alpha_pos - A1 * D_alpha
            
            r1, r2 = np.random.rand(2)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * beta_pos - positions[i])
            X2 = beta_pos - A2 * D_beta
            
            r1, r2 = np.random.rand(2)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * delta_pos - positions[i])
            X3 = delta_pos - A3 * D_delta
            
            # Update the wolf's position based on the average of the top 3
            positions[i] = (X1 + X2 + X3) / 3
        
        # 3. Apply constraints:
        # Weights must be between 0 and 1
        positions = np.clip(positions, 0, 1)
        # Weights must sum to 1
        positions /= np.sum(positions, axis=1, keepdims=True)

        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{max_iterations}, Best Fitness (Entropy): {alpha_score:.4f}")

    end_time = time.time()
    print(f"Optimization finished in {end_time - start_time:.2f} seconds.")
    
    # Return the best weights found
    return alpha_pos

# --- Main execution ---
if __name__ == "__main__":
    
    # --- 1. Load Input Image ---
    input_filepath = "input_image.png"
    print(f"Loading '{input_filepath}'...")
    
    color_img = cv2.imread(input_filepath)
    if color_img is None:
        print(f"Error: Could not read '{input_filepath}'.")
        print("Please make sure the image is in the same directory as the script.")
        exit()

    # --- 2. Run GWO Optimized Conversion ---
    print("\n--- Running GWO to find optimal weights ---")
    # Run the optimizer
    optimal_weights = run_gwo(color_img, num_wolves=20, max_iterations=50)
    
    # Apply the best weights found
    optimized_gray = apply_weights(color_img, optimal_weights)
    
    output_filepath = "output_image.png"
    cv2.imwrite(output_filepath, optimized_gray)
    
    optimized_fitness = calculate_fitness(optimized_gray)
    print(f"\nSuccessfully saved '{output_filepath}'")
    # Format weights for printing (OpenCV is BGR, so we list them as Wb, Wg, Wr)
    print(f"Optimized Weights (B,G,R): [{optimal_weights[0]:.4f}, {optimal_weights[1]:.4f}, {optimal_weights[2]:.4f}]")
    print(f"Optimized Fitness (Entropy): {optimized_fitness:.4f}")

