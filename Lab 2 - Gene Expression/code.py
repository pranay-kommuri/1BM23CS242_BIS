# Python Black & White Pixel Image Recreator with Live View
#
# This script is designed to recreate a 200x200 pixel image
# where each pixel is either pure black or pure white.
# It uses a hill-climbing algorithm and displays the progress live using OpenCV.
#
# Required libraries:
# pip install Pillow numpy opencv-python

from PIL import Image
import numpy as np
import random
import os
import cv2

# --- Configuration ---
IMAGE_SIZE = (200, 200)
TARGET_FILENAME = "target_200x200.png"
OUTPUT_FILENAME = "recreated_200x200.png"
# Note: For a 200x200 image (40,000 pixels), this will take some time to perfect.
MAX_ITERATIONS = 2000000 
# Update the live view window every N improvements to avoid slowing down the process.
UPDATE_VIEW_EVERY_N_IMPROVEMENTS = 100

# Define black and white colors for clarity
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

def create_target_image():
    """
    Generates a sample 200x200 black and white target image if it doesn't exist.
    This creates a checkerboard pattern suitable for the canvas.
    """
    if os.path.exists(TARGET_FILENAME):
        return

    print(f"Sample target file '{TARGET_FILENAME}' not found. Creating one.")
    img = Image.new('RGB', IMAGE_SIZE, BLACK)
    pixels = img.load()
    
    # Create a checkerboard pattern
    square_size = 20
    for y in range(IMAGE_SIZE[1]):
        for x in range(IMAGE_SIZE[0]):
            if (x // square_size) % 2 == (y // square_size) % 2:
                pixels[x, y] = WHITE
            else:
                pixels[x, y] = BLACK

    img.save(TARGET_FILENAME)
    print("Sample image created successfully.")


def calculate_difference(image1, image2):
    """
    Calculates the number of mismatched pixels between two images.
    A score of 0 means the images are identical.
    """
    arr1 = np.array(image1)
    arr2 = np.array(image2)
    # The sum of non-equal pixels, divided by 3 (for RGB channels) gives the count
    return np.sum(arr1 != arr2) // 3

def recreate_image():
    """Main function to run the generative process with live viewing."""
    # --- 1. SETUP ---
    try:
        target_image = Image.open(TARGET_FILENAME).convert("RGB")
        if target_image.size != IMAGE_SIZE:
            print(f"Error: Target image must be {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} pixels.")
            return
        print(f"Successfully loaded target image '{TARGET_FILENAME}'.")
    except FileNotFoundError:
        print(f"ERROR: Target file not found. Please create '{TARGET_FILENAME}'.")
        return

    width, height = target_image.size

    # --- 2. INITIAL STATE ---
    # Start with a random black and white canvas
    current_image = Image.new("RGB", IMAGE_SIZE)
    current_pixels = current_image.load()
    for x in range(width):
        for y in range(height):
            current_pixels[x, y] = random.choice([BLACK, WHITE])
    
    best_score = calculate_difference(target_image, current_image)
    print(f"Starting with a random canvas. Mismatched pixels: {best_score}")
    print("-" * 30)
    print("Starting optimization process... Press 'q' on the image window to quit early.")
    
    # Setup OpenCV window for live viewing
    cv2.namedWindow('Live Reconstruction', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Live Reconstruction', width, height)
    
    improvement_counter = 0

    # --- 3. OPTIMIZATION LOOP (GENERATIONS) ---
    for i in range(1, MAX_ITERATIONS + 1):
        if best_score == 0:
            print(f"\nPerfect match found after {i - 1} generations!")
            break

        x, y = random.randint(0, width - 1), random.randint(0, height - 1)
        
        test_image = current_image.copy()
        test_pixels = test_image.load()

        original_color = test_pixels[x, y]
        flipped_color = WHITE if original_color == BLACK else BLACK
        test_pixels[x, y] = flipped_color

        # --- 4. EVALUATE ---
        current_score = calculate_difference(target_image, test_image)

        if current_score < best_score:
            current_image = test_image
            best_score = current_score
            improvement_counter += 1
            
            if improvement_counter % 100 == 0:
                 print(f"Generation {i}: Improvement found! Mismatched pixels remaining: {best_score}")

            # Update live view periodically
            if improvement_counter % UPDATE_VIEW_EVERY_N_IMPROVEMENTS == 0:
                # Convert PIL image to an OpenCV image (RGB -> BGR)
                cv_image = np.array(current_image)
                cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                cv2.imshow('Live Reconstruction', cv_image_bgr)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nUser quit the process early.")
                    break

    # --- 5. FINISH ---
    if best_score != 0:
        print(f"\nMax generations ({MAX_ITERATIONS}) reached or process quit.")

    # Show final image before closing
    final_cv_image = np.array(current_image)
    final_cv_image_bgr = cv2.cvtColor(final_cv_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Live Reconstruction', final_cv_image_bgr)
    print("Displaying final image. Press any key in the window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n" + "="*30)
    print("Recreation process finished.")
    print(f"Final mismatched pixels: {best_score}")

    current_image.save(OUTPUT_FILENAME)
    print(f"Final {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} image saved to '{OUTPUT_FILENAME}'")
    
if __name__ == "__main__":
    create_target_image()
    recreate_image()

