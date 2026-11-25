from src.image_filtering import (
    load_image,
    convert_to_grayscale,
    process_image
)

# Load the image
image = load_image("data/cameraman.tif")

# Convert to grayscale
gray_image = convert_to_grayscale(image)

# Apply Fourier Transform and filters
results = process_image(gray_image, cutoff=30)

print("Image loaded successfully!")
print("Grayscale shape:", gray_image.shape)

print("Fourier spectrum:", results["magnitude_spectrum"].shape)
print("Low-pass result:", results["low_pass_image"].shape)
print("High-pass result:", results["high_pass_image"].shape)

print("Student 5 test completed successfully!")

import matplotlib.pyplot as plt

# Display the results
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(gray_image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(results["magnitude_spectrum"], cmap="gray")
plt.title("Fourier Spectrum")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(results["low_pass_image"], cmap="gray")
plt.title("Low-Pass Filter")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(results["high_pass_display"], cmap="gray")
plt.title("High-Pass Filter")
plt.axis("off")

plt.tight_layout()
plt.show()