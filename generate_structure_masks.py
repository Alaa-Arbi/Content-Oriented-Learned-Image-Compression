
import cv2, os, argparse


def compute_structure_mask(image, min, max):
    binary_structure = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), min, max, L2gradient=True)
    return binary_structure

def generate_masks(args):
    # Traverse through all files and subdirectories
    for root, _, files in os.walk(args.input_dir):
        for file_name in files:
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):

                input_path = os.path.join(root, file_name)
                relative_path = os.path.relpath(input_path, args.input_dir)
                output_path = os.path.join(args.output_dir, os.path.splitext(relative_path)[0]+".png")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                try:
                    image = cv2.imread(input_path)
                    mask = compute_structure_mask(image, args.min, args.max)
                    cv2.imwrite(output_path, mask)
                    print(f"Processed: {input_path} -> {output_path}")
                except Exception as e:
                    print(f"Error processing {input_path}: {str(e)}")
    print("Processing completed.")


if __name__ == '__main__':
    description = "Generate structure masks."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir", required=True, default=None, help="Path to input directory.")
    parser.add_argument("--output_dir", required=True, default=None, help="Path to output directory.")
    parser.add_argument("--min", default=70, help="Min threshold for canny detector")
    parser.add_argument("--max", default=200, help="Max threshold for canny detector")
    generate_masks(parser.parse_args())

    
    