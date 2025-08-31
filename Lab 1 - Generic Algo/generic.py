# Python3 program to recreate a target image using a Genetic Algorithm
# Features: Live spectating, 40x40 binary (black & white) images.

import random
import os
import numpy as np
from PIL import Image
import cv2 # Used for live image display

# --- Constants ---
POPULATION_SIZE = 100
IMG_WIDTH = 10
IMG_HEIGHT = 10
TARGET_IMAGE_PATH = 'target_image10.png'
OUTPUT_DIR = 'output'

# --- Load and Prepare Target Image ---
try:
    # Load the target image
    target_img_pil = Image.open(TARGET_IMAGE_PATH)
    # Resize to 40x40 and convert to grayscale
    target_img_pil = target_img_pil.resize((IMG_WIDTH, IMG_HEIGHT)).convert('L')
    
    # Convert to NumPy array
    TARGET_IMAGE = np.array(target_img_pil)
    
    # Threshold the image to make it strictly black (0) and white (255)
    TARGET_IMAGE = np.where(TARGET_IMAGE > 127, 255, 0).astype(np.uint8)

except FileNotFoundError:
    print(f"Error: Target image not found at '{TARGET_IMAGE_PATH}'")
    exit()

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class Individual(object):
    '''
    An individual is a 40x40 binary image (chromosome).
    '''
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.cal_fitness()

    @classmethod
    def mutated_pixel(cls):
        '''Returns a random binary pixel value (0 or 255).'''
        return random.choice([0, 255])

    @classmethod
    def create_gnome(cls):
        '''Creates a random 40x40 chromosome with black and white pixels.'''
        return np.random.choice([0, 255], size=(IMG_HEIGHT, IMG_WIDTH), p=[0.5, 0.5]).astype(np.uint8)

    def mate(self, par2):
        '''Performs crossover and mutation.'''
        child_chromosome = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

        for i in range(IMG_HEIGHT):
            for j in range(IMG_WIDTH):
                prob = random.random()
                
                if prob < 0.45: # 45% chance from parent 1
                    child_chromosome[i, j] = self.chromosome[i, j]
                elif prob < 0.90: # 45% chance from parent 2
                    child_chromosome[i, j] = par2.chromosome[i, j]
                else: # 10% chance for mutation
                    child_chromosome[i, j] = self.mutated_pixel()
        
        return Individual(child_chromosome)

    def cal_fitness(self):
        '''
        Calculates fitness as the sum of absolute differences.
        Lower is better.
        '''
        fitness = np.sum(np.abs(self.chromosome.astype(np.int64) - TARGET_IMAGE.astype(np.int64)))
        return fitness

# --- Main Program Logic ---
def main():
    generation = 1
    population = []

    print("Creating initial population...")
    for _ in range(POPULATION_SIZE):
        gnome = Individual.create_gnome()
        population.append(Individual(gnome))
    print("Initial population created. Starting evolution...")

    while True:
        # Sort the population by fitness score
        population = sorted(population, key=lambda x: x.fitness)
        best_individual = population[0]

        # Calculate number of incorrect pixels for display
        incorrect_pixels = best_individual.fitness // 255
        print(f"Generation: {generation}\t Mismatched Pixels: {incorrect_pixels}")

        # --- Live Spectating Logic ---
        cv2.imshow('Evolution in Progress (Press Q to quit)', best_individual.chromosome)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Check for solution
        if best_individual.fitness <= 0:
            print("\nSolution Found!")
            break

        # Generate the next generation
        new_generation = []
        s = int(0.10 * POPULATION_SIZE) # Elitism
        new_generation.extend(population[:s])
        
        s = int(0.90 * POPULATION_SIZE) # Mating
        for _ in range(s):
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            child = parent1.mate(parent2)
            new_generation.append(child)

        population = new_generation
        generation += 1

    # --- Cleanup ---
    print(f"Evolution stopped at Generation: {generation}")
    
    # Save the final image
    final_image = Image.fromarray(population[0].chromosome)
    final_path = os.path.join(OUTPUT_DIR, "final_solution.png")
    final_image.save(final_path)
    print(f"Final image saved to '{final_path}'")

    # Keep the final window open until a key is pressed
    cv2.imshow('Evolution in Progress (Press any key to exit)', population[0].chromosome)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()