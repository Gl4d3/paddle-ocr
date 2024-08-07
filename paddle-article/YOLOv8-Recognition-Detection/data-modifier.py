import os

def create_recognition_txt(dataset_type):
    base_path = "Meter-digits"
    images_dir = os.path.join(base_path, dataset_type, "images")
    labels_dir = os.path.join(base_path, dataset_type, "labels")
    output_file = f"{dataset_type}_recognition.txt"

    with open(output_file, 'w', encoding='utf-8') as out_file:
        for label_file in os.listdir(labels_dir):
            if label_file.endswith('.txt'):
                image_file = label_file.replace('.txt', '.jpg')
                image_path = os.path.join(images_dir, image_file)
                label_path = os.path.join(labels_dir, label_file)
                
                if os.path.exists(image_path):
                    with open(label_path, 'r', encoding='utf-8') as label_f:
                        # Read all lines and join them as a single string (for multi-line labels)
                        label = ' '.join([line.strip() for line in label_f])
                        out_file.write(f"{image_path}\t{label}\n")
                else:
                    print(f"Image file not found: {image_path}")

datasets = ["Train", "Valid", "Test"]
for dataset in datasets:
    print(f"Processing {dataset} dataset...")
    create_recognition_txt(dataset)
