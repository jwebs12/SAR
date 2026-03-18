import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from scipy.ndimage import median_filter

#-----------------------------------------------------------------------------------------
# FUNCTIONS
#-----------------------------------------------------------------------------------------
# Read a SAR GeoTIFF file
def load_sar_image(filepath):
    print(f"Loading image: {filepath}...")
    with rasterio.open(filepath) as src:
        # Read the first band
        image = src.read(1)
        profile = src.profile
    print(f"Image loaded. Shape: {image.shape}")
    return image, profile


#-----------------------------------------------------------------------------------------
# Converts raw amplitude/intensity to Decibels (dB)
def convert_to_db(image):
    # Formula: 10 * log10(pixel_value)
    print("Converting to Decibels (dB)...")
    # Avoid log of zero by adding a tiny epsilon
    image_db = 10 * np.log10(image + 1e-7)
    return image_db


#-----------------------------------------------------------------------------------------
# Applies a Lee Filter to remove SAR speckle (granular noise)
def lee_speckle_filter(image, window_size=5):
    print("Applying Lee Speckle Filter...")

    # Calculate local mean and variance
    img_mean = uniform_filter(image, (window_size, window_size))
    img_sqr_mean = uniform_filter(image ** 2, (window_size, window_size))
    img_var = img_sqr_mean - img_mean ** 2

    # Estimate overall noise variance (sigma_v^2)
    # In SAR, variance is often proportional to the mean squared.
    # For a simple implementation, we estimate noise from a homogeneous area or use a constant.
    # Here we calculate the weighting function 'k' directly.
    overall_var = np.var(image)

    # Lee filter weights
    weights = img_var / (img_var + overall_var)

    # Final filtered image
    filtered_image = img_mean + weights * (image - img_mean)
    return filtered_image


#-----------------------------------------------------------------------------------------
# Plots the steps side-by-side.
def visualize_process(original, db_converted, filtered):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Raw Image (NOTE: often looks black due to high dynamic range)
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=np.percentile(original, 99))
    axes[0].set_title("1. Raw SAR Data (Amplitude)")
    axes[0].axis('off')

    # 2. dB Converted (Log scale reveals details)
    axes[1].imshow(db_converted, cmap='gray')
    axes[1].set_title("2. Log-Scaled (dB)")
    axes[1].axis('off')

    # 3. Speckle Filtered (NOTE: Smoother visualization)
    # Use a 'inferno' or 'magma' colormap for Radar heatmaps
    im = axes[2].imshow(filtered, cmap='inferno')
    axes[2].set_title("3. Speckle Filtered (Final)")
    axes[2].axis('off')

    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label='Backscatter (dB)')
    plt.tight_layout()
    plt.show()


#-----------------------------------------------------------------------------------------
# Save processed image as new .tif file
def save_result(image, profile, output_name="processed_sar.tif"):
    # Saves the processed numpy array back to a GeoTIFF.
    profile.update(dtype=rasterio.float32, count=1, nodata=None)
    with rasterio.open(output_name, 'w', **profile) as dst:
        dst.write(image.astype(rasterio.float32), 1)
    print(f"Saved processed image to {output_name}")

#-----------------------------------------------------------------------------------------
# --- MAIN EXECUTION ---
#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Path to downloaded SAR .tiff file
    input_file = "/Users/jweber/PycharmProjects/Umbra/data/2023-11-07-09-15-04_UMBRA-05_GEC.tif"

    try:
        # Step 1: Load
        raw_img, meta_profile = load_sar_image(input_file)

        # Step 2: Convert to dB (Visualization Scaling)
        # Note: Real scientific calibration requires metadata (calibration vectors),
        # but this log-transform is standard for visual inspection.
        img_db = convert_to_db(raw_img)

        # Step 3: Denoise (Speckle Filtering)
        # We run the filter on the dB image for better visual results
        clean_img = lee_speckle_filter(img_db, window_size=5)

        # Step 4: Visualize
        visualize_process(raw_img, img_db, clean_img)

        # Step 5: Save
        save_result(clean_img, meta_profile, "final_sar_output.tif")

    except FileNotFoundError:
        print("Error: Please update the 'input_file' variable with a valid path to a .tif file.")



