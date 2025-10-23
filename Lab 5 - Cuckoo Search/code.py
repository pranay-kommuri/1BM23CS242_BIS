import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import math  # Import math for gamma and sin functions

# Objective function: measure image quality (entropy-based fitness)
def fitness_function(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    return entropy  # higher entropy → better image detail

# Enhancement transformation (contrast & brightness)
def enhance_image(image, alpha, beta):
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced

# Cuckoo Search parameters
n = 10              # number of nests
pa = 0.25           # discovery rate
iterations = 30
alpha_range = (1.0, 3.0)
beta_range = (0, 80)

# Load image
image = cv2.imread('input.jpg')

# Initialize nests (alpha, beta) randomly
nests = np.array([[random.uniform(*alpha_range), random.uniform(*beta_range)] for _ in range(n)])
fitness = np.zeros(n)

# Evaluate initial fitness
for i in range(n):
    enhanced = enhance_image(image, nests[i][0], nests[i][1])
    fitness[i] = fitness_function(enhanced)

def levy_flight(Lambda):
    sigma1 = (math.gamma(1 + Lambda) * math.sin(math.pi * Lambda / 2)) / \
             (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))
    sigma = sigma1 ** (1 / Lambda)
    u = np.random.randn() * sigma
    v = np.random.randn()
    step = u / abs(v) ** (1 / Lambda)
    return step

# Main Cuckoo Search loop
for t in range(iterations):
    for i in range(n):
        step_size = levy_flight(1.5)
        new_nest = nests[i] + step_size * np.random.randn(2)
        new_nest[0] = np.clip(new_nest[0], *alpha_range)
        new_nest[1] = np.clip(new_nest[1], *beta_range)

        new_image = enhance_image(image, new_nest[0], new_nest[1])
        new_fit = fitness_function(new_image)

        if new_fit > fitness[i]:
            nests[i] = new_nest
            fitness[i] = new_fit

    # Discovery and randomization
    discover = np.random.rand(n) < pa
    for i in range(n):
        if discover[i]:
            nests[i] = [random.uniform(*alpha_range), random.uniform(*beta_range)]

# Get best enhanced image
best_nest = nests[np.argmax(fitness)]
best_img = enhance_image(image, best_nest[0], best_nest[1])

# Apply sharpening to the optimized output
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
best_img = cv2.filter2D(best_img, -1, kernel)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title('Cuckoo Search Enhanced')
plt.imshow(cv2.cvtColor(best_img, cv2.COLOR_BGR2RGB))
plt.show()

# Save enhanced output
cv2.imwrite('enhanced_output.jpg', best_img)
print("Enhanced image saved as enhanced_output.jpg")
