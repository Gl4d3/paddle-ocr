# Neccessary libraries
import os
import json

# Define paths
base_path = 'Meter-dial-3'
datasets = ['Train', 'Valid', 'Test']
image_folder = 'images'
label_folder = 'labels'

# Define a dictionary for class labels (can be expanded as needed)
class_labels = {
    0: '0',  # Replace with actual class names if more exist
}

# Function to convert YOLOv8 labels to PaddleOCR format
def create_detection_txt(dataset_name):
    dataset_path = os.path.join(base_path, dataset_name.lower())
    images_path = os.path.join(dataset_path, image_folder)
    labels_path = os.path.join(dataset_path, label_folder)
    output_file = os.path.join(base_path, f'{dataset_name.lower()}_detection.txt')

    print(f"Processing {dataset_name} dataset...")

    with open(output_file, 'w') as out:
        for label_file in os.listdir(labels_path):
            if label_file.endswith('.txt'):
                label_file_path = os.path.join(labels_path, label_file)
                image_file = label_file.replace('.txt', '.jpg')  # Adjust if different extension
                image_path = os.path.join(images_path, image_file)
                objects = []

                with open(label_file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])
                        
                        # Convert to x_min, y_min, x_max, y_max
                        x_min = x_center - width / 2
                        y_min = y_center - height / 2
                        x_max = x_center + width / 2
                        y_max = y_center + height / 2

                        # Convert to points for PaddleOCR
                        points = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                        transcription = class_labels[class_id]

                        obj_data = {
                            "points": points,
                            "transcription": transcription
                        }
                        objects.append(obj_data)

                if objects:
                    line = f"{image_path}\t{json.dumps(objects)}\n"
                    out.write(line)
                else:
                    print(f"No objects found for image {image_path}")

# Run the conversion for all datasets
for dataset in datasets:
    create_detection_txt(dataset)
