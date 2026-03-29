import cv2
import matplotlib.pyplot as plt
import os

def main():
    # 1. Image Loading
    image_path = './data/image.jpg'
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("Error: Failed to load the image.")
        return

    # 2. Color Space Conversion (BGR to RGB for plot, BGR to GRAY for Sobel)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 3. Sobel Edge Detection
    # Calculate horizontal gradient (X-axis)
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    # Calculate vertical gradient (Y-axis)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

    # Convert to 8-bit unsigned integer (absolute value)
    abs_x = cv2.convertScaleAbs(sobel_x)
    abs_y = cv2.convertScaleAbs(sobel_y)

    # 4. Combine X and Y gradients
    sobel_combined = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

    # 5. Visualization and Saving
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    # Sobel Edge Detection Image
    plt.subplot(1, 2, 2)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Sobel Edge Detection')
    plt.axis('off')

    os.makedirs('./output', exist_ok=True)
    save_path = './output/sobel_result.png'
    plt.savefig(save_path)
    print(f"Sobel edge detection result saved to {save_path}")
    
    # plt.show() # Uncomment to display interactively

if __name__ == "__main__":
    main()
