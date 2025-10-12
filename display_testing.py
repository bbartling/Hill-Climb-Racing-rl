# test_display.py

import cv2
import os

def test_image_display(image_path):
    """
    Loads and displays an image using OpenCV.

    Args:
        image_path (str): The relative or absolute path to the image file.
    """
    # Check if the file exists before trying to read it
    if not os.path.exists(image_path):
        print(f"Error: The file was not found at the path: {image_path}")
        print("Please make sure the script is in the correct root directory and the path is correct.")
        return

    # Read the image from the specified file
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Failed to load the image. The file might be corrupted or in an unsupported format.")
        return

    # --- This is the core test for imshow ---
    try:
        print("Displaying image. Press any key to close the window.")
        cv2.imshow("Image Display Test", image) # Creates a window and displays the image
        cv2.waitKey(0)                         # Waits indefinitely for a key press
        cv2.destroyAllWindows()                # Closes the window
        print("Window closed successfully.")
    except cv2.error as e:
        print(f"\nAn OpenCV GUI error occurred: {e}")
        print("This often happens when running the script in an environment without a display,")
        print("such as over an SSH connection or in a Docker container.")


if __name__ == "__main__":
    # Define the relative path to your image
    # This path is relative to the location of the script
    path_to_image = os.path.join("config_screenshots", "images_for_cv", "90_deg_up.PNG")
    
    test_image_display(path_to_image)