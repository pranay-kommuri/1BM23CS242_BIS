import cv2
import numpy as np
import os # For checking file existence

# --- Configuration ---
input_filename = "input_image.png"
output_filename = "output_image_denoised_cellular.png"
NOISE_AMOUNT = 0.05       # Percentage of pixels to add salt-and-pepper noise
FILTER_KERNEL_SIZE = 3  # Size of the median filter kernel (e.g., 3 for 3x3, 5 for 5x5)
                        # Must be an odd number (3, 5, 7, etc.)
NUM_ITERATIONS = 2      # How many times to apply the cellular median filter

# --- Helper Function: Add Salt-and-Pepper Noise (FIXED) ---
def add_salt_and_pepper_noise(image, amount=0.05):
    """Adds salt-and-pepper noise to a grayscale or color image."""
    noisy_image = np.copy(image)
    
    height, width = image.shape[:2]
    total_pixels = height * width

    num_salt = int(np.ceil(amount * total_pixels * 0.5))
    num_pepper = int(np.ceil(amount * total_pixels * 0.5))

    # For Salt (white pixels)
    # Generate random coordinates for salt noise
    salt_coords_y = np.random.randint(0, height - 1, num_salt)
    salt_coords_x = np.random.randint(0, width - 1, num_salt)

    if len(image.shape) == 2: # Grayscale
        noisy_image[salt_coords_y, salt_coords_x] = 255
    else: # Color
        # Need to iterate for color as direct tuple indexing with multi-element assignment causes shape mismatch
        for i in range(num_salt):
            noisy_image[salt_coords_y[i], salt_coords_x[i]] = [255, 255, 255] # White

    # For Pepper (black pixels)
    # Generate random coordinates for pepper noise
    pepper_coords_y = np.random.randint(0, height - 1, num_pepper)
    pepper_coords_x = np.random.randint(0, width - 1, num_pepper)

    if len(image.shape) == 2: # Grayscale
        noisy_image[pepper_coords_y, pepper_coords_x] = 0
    else: # Color
        # Need to iterate for color as direct tuple indexing with multi-element assignment causes shape mismatch
        for i in range(num_pepper):
            noisy_image[pepper_coords_y[i], pepper_coords_x[i]] = [0, 0, 0] # Black
        
    return noisy_image

# --- Parallel Cellular Algorithm Implementation: Median Filter ---
def cellular_median_filter(image_data, kernel_size, iterations):
    """
    Applies a median filter as a parallel cellular algorithm.
    image_data: Input image as a NumPy array.
    kernel_size: Size of the neighborhood (e.g., 3 for 3x3). Must be odd.
    iterations: Number of times to apply the filter.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    height, width = image_data.shape[:2]
    padding = kernel_size // 2 # How many pixels to extend on each side for the neighborhood

    current_data = np.copy(image_data)
    
    # Determine if it's a grayscale or color image
    is_grayscale = (len(image_data.shape) == 2)

    print(f"Applying cellular median filter ({kernel_size}x{kernel_size}) for {iterations} iterations...")

    for iteration in range(iterations):
        next_data = np.copy(current_data) # This will be the new state for the next generation

        # Iterate over each pixel (cell) in the image
        # Using a loop here in Python, but conceptually, these updates are parallel.
        # In a true parallel system (GPU, multi-core), each (y, x) calculation
        # would run on a separate thread/core.
        for y in range(height):
            for x in range(width):
                # Extract the neighborhood
                # Handle image boundaries by padding (reflection is common)
                
                # Define slice for the neighborhood
                y_start = max(0, y - padding)
                y_end = min(height, y + padding + 1)
                x_start = max(0, x - padding)
                x_end = min(width, x + padding + 1)
                
                # Get the actual neighborhood values from current_data
                neighborhood = current_data[y_start:y_end, x_start:x_end]
                
                # Flatten the neighborhood to find the median
                if is_grayscale:
                    # For grayscale, just flatten all values
                    pixel_values = neighborhood.flatten()
                    new_pixel_value = np.median(pixel_values)
                else:
                    # For color, flatten each channel separately and find median for R, G, B
                    new_pixel_value = [
                        np.median(neighborhood[:, :, 0].flatten()), # Red channel
                        np.median(neighborhood[:, :, 1].flatten()), # Green channel
                        np.median(neighborhood[:, :, 2].flatten())  # Blue channel
                    ]
                
                # Update the cell in the next_data grid
                next_data[y, x] = new_pixel_value
        
        current_data = next_data.astype(image_data.dtype) # Swap grids for the next iteration

        print(f"  Iteration {iteration + 1}/{iterations} complete.")

    return current_data

# --- Main Program Flow ---
if __name__ == "__main__":
    if not os.path.exists(input_filename):
        print(f"Error: Input image '{input_filename}' not found.")
        print("Please place an image named 'input_image.png' in the same directory.")
    else:
        print(f"Loading '{input_filename}'...")
        original_img = cv2.imread(input_filename, cv2.IMREAD_UNCHANGED)

        if original_img is None:
            print(f"Error: Could not read the image '{input_filename}'. Is it a valid image file?")
        else:
            print(f"Original image shape: {original_img.shape}")

            # Convert to appropriate type for processing (usually float or 8-bit int)
            # Ensure it's in a modifiable format if needed, though uint8 is fine for median.
            if original_img.dtype != np.uint8:
                original_img = cv2.convertScaleAbs(original_img) # Convert to 8-bit unsigned integer

            # Add noise to demonstrate denoising
            print(f"Adding {NOISE_AMOUNT*100}% salt-and-pepper noise...")
            noisy_img = add_salt_and_pepper_noise(original_img, amount=NOISE_AMOUNT)

            # Apply our cellular median filter
            denoised_img = cellular_median_filter(noisy_img, FILTER_KERNEL_SIZE, NUM_ITERATIONS)

            # Save the results
            cv2.imwrite("noisy_input_for_demonstration.png", noisy_img) # Save noisy version too
            cv2.imwrite(output_filename, denoised_img)

            print(f"\nOriginal noisy image saved as 'noisy_input_for_demonstration.png'")
            print(f"Denoised image (using custom cellular algorithm) saved as '{output_filename}'.")

            # Optionally, display images (requires matplotlib)
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(15, 5))

                plt.subplot(1, 3, 1)
                plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB) if len(original_img.shape)==3 else original_img, cmap='gray')
                plt.title('Original Image')
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB) if len(noisy_img.shape)==3 else noisy_img, cmap='gray')
                plt.title(f'Noisy Image ({NOISE_AMOUNT*100}% S&P)')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB) if len(denoised_img.shape)==3 else denoised_img, cmap='gray')
                plt.title(f'Denoised (Cellular Median {FILTER_KERNEL_SIZE}x{FILTER_KERNEL_SIZE}, {NUM_ITERATIONS} iter)')
                plt.axis('off')

                plt.suptitle('Image Denoising with Custom Parallel Cellular Median Filter')
                plt.show()
            except ImportError:
                print("\nInstall 'matplotlib' (pip install matplotlib) to view images directly.")