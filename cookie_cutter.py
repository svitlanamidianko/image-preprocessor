import cv2
import numpy as np
import os
from pathlib import Path
import shutil

class BatchCardCutter:
    def __init__(self, input_folder="input", output_folder="output", temp_folder="temp"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.temp_folder = temp_folder
        self.template_coords = None
        self.standard_dims = None
        
        # Create necessary folders
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        Path(temp_folder).mkdir(parents=True, exist_ok=True)

    def learn_template(self, template_path):
        """Learn card coordinates from template image"""
        image = cv2.imread(template_path)
        if image is None:
            raise Exception(f"Could not read template image: {template_path}")

        coords = []
        window_name = 'Template Image'
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                coords.append([x, y])
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow(window_name, image)
                print(f"Point {len(coords)} clicked at ({x}, {y})")
                
                if len(coords) == 4:
                    print("All points collected. Press any key to continue...")
                    cv2.waitKey(1000)
                    cv2.destroyWindow(window_name)

        cv2.imshow(window_name, image)
        print("\nClick the four corners of the card in this order:")
        print("1. Top-left")
        print("2. Top-right")
        print("3. Bottom-right")
        print("4. Bottom-left")
        
        cv2.setMouseCallback(window_name, mouse_callback)
        
        while len(coords) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                cv2.destroyAllWindows()
                raise Exception("Template learning cancelled by user")
        
        self.template_coords = np.array(coords, dtype=np.float32)
        
        # Use fixed dimensions
        self.standard_dims = (1187, 1802)
        print(f"Using fixed dimensions: 1187x1802")
        
        print("Template learning completed successfully!")

    def cut_card(self, image):
        if self.template_coords is None:
            raise Exception("Template coordinates not learned. Call learn_template() first.")
            
        width, height = self.standard_dims
        dst_pts = np.array([
            [0, 0],
            [width-1, 0],
            [width-1, height-1],
            [0, height-1]
        ], dtype="float32")
        
        M = cv2.getPerspectiveTransform(self.template_coords, dst_pts)
        warped = cv2.warpPerspective(image, M, (width, height))
        
        return warped

    def process_batch(self):
        # Get all images from input folder
        all_images = [f for f in os.listdir(self.input_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Get list of already processed images (removing 'cut_' prefix if it exists)
        processed_images = [f.replace('cut_', '') for f in os.listdir(self.output_folder)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Filter out already processed images
        unprocessed_images = [img for img in all_images if img not in processed_images]
        
        if not unprocessed_images:
            print("No images left to process!")
            return False

        print(f"\nFound {len(unprocessed_images)} unprocessed images")
        
        # Select template from unprocessed images
        print("\nSelect template image from the following:")
        for i, img in enumerate(unprocessed_images):
            print(f"{i+1}. {img}")
        
        template_idx = int(input("\nEnter the number of the template image: ")) - 1
        template_path = os.path.join(self.input_folder, unprocessed_images[template_idx])
        
        # Learn template from selected image
        self.learn_template(template_path)
        
        # Process all images
        for filename in unprocessed_images:
            image_path = os.path.join(self.input_folder, filename)
            print(f"\nProcessing: {filename}")
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                continue
            
            card = self.cut_card(image)
            output_path = os.path.join(self.temp_folder, f"cut_{filename}")
            cv2.imwrite(output_path, card)
            print(f"Saved to: {output_path}")
        
        # Ask user to review results
        input("\nPlease review the results in the temp folder."
              "\nKeep the good ones by moving them to the output folder."
              "\nDelete the bad ones."
              "\nPress Enter when done...")
        
        # Clean up and prepare for next batch
        for filename in os.listdir(self.temp_folder):
            os.remove(os.path.join(self.temp_folder, filename))
        
        return True

def main():
    try:
        cutter = BatchCardCutter()
        
        while True:
            if not cutter.process_batch():
                break
            
            continue_processing = input("\nProcess another batch? (y/n): ").lower()
            if continue_processing != 'y':
                break
        
        print("\nProcessing completed!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        cv2.destroyAllWindows()
        # Clean up temp folder
        if os.path.exists("temp"):
            shutil.rmtree("temp")

if __name__ == "__main__":
    main()