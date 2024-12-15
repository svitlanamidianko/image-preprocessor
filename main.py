from card_processor import process_card

# Process a single image
input_path = "path/to/your/input/image.jpg"
output_path = "path/to/your/output/image.jpg"

try:
    process_card(input_path, output_path)
    print(f"Successfully processed {input_path}")
except Exception as e:
    print(f"Error processing image: {str(e)}")

# To process multiple images in a directory:
import os

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                process_card(input_path, output_path)
                print(f"Successfully processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# Example usage:
process_directory("input/", "output/") 