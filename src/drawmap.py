import cv2
import matplotlib.pyplot as plt

# Set font to support English labels
plt.rcParams['font.family'] = 'Arial'  # Use Arial font for English text

# Read image
image = cv2.imread("example_image.jpg", cv2.IMREAD_GRAYSCALE)

# Original model feature point detection
orb = cv2.ORB_create()
kp1 = orb.detect(image, None)
img1 = cv2.drawKeypoints(image, kp1, None, color=(0, 255, 0), flags=0)

# Compressed model feature point detection (simulate reduced feature points)
kp2 = kp1[:len(kp1)//2]  # Keep only half of the feature points
img2 = cv2.drawKeypoints(image, kp2, None, color=(0, 255, 0), flags=0)

# Plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img1, cmap="gray")
plt.title("Original Model Feature Points")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img2, cmap="gray")
plt.title("Compressed Model Feature Points")
plt.axis("off")

plt.savefig("figure2.png")  # Save the figure
plt.show()