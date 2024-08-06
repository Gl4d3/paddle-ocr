# # VERSION 1.0

# import os
# import json

# # Define paths
# base_path = 'Meter-dial-3-COCO'
# datasets = ['Train', 'Valid', 'Test']

# # Function to convert COCO format to PaddleOCR format for detection
# def create_detection_txt(dataset_name):
#     dataset_path = os.path.join(base_path, dataset_name)
#     annotation_file = os.path.join(dataset_path, '_annotations.coco.json')
#     output_file = os.path.join(base_path, f'{dataset_name.lower()}_detection.txt')

#     if not os.path.isfile(annotation_file):
#         print(f"Annotation file not found: {annotation_file}")
#         return

#     with open(annotation_file, 'r') as f:
#         coco_data = json.load(f)

#     with open(output_file, 'w') as out:
#         for image_info in coco_data['images']:
#             image_id = image_info['id']
#             file_name = image_info['file_name']
#             image_path = os.path.join(dataset_path, file_name)
#             objects = []

#             for ann in coco_data['annotations']:
#                 if ann['image_id'] == image_id and 'bbox' in ann:
#                     bbox = ann['bbox']
#                     x_min, y_min, width, height = bbox
#                     x_max = x_min + width
#                     y_max = y_min + height
#                     label_id = ann['category_id']

#                     # Assuming we want to use the category name
#                     label = coco_data['categories'][label_id]['name']

#                     # Convert bbox to points format
#                     points = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
#                     obj_data = {
#                         "points": points,
#                         "transcription": label
#                     }
#                     objects.append(obj_data)

#             line = f"{image_path}\t{json.dumps(objects)}\n"
#             out.write(line)

# # Run the conversion for all datasets
# for dataset in datasets:
#     create_detection_txt(dataset)


# VERSION 2.0
# import os
# import json

# # Define paths
# base_path = 'Meter-dial-3-COCO'
# datasets = ['Train', 'valid', 'Test']

# # Function to convert COCO format to PaddleOCR format for detection
# def create_detection_txt(dataset_name):
#     dataset_path = os.path.join(base_path, dataset_name)
#     annotation_file = os.path.join(dataset_path, '_annotations.coco.json')
#     output_file = os.path.join(base_path, f'{dataset_name.lower()}_detection.txt')

#     if not os.path.isfile(annotation_file):
#         print(f"Annotation file not found: {annotation_file}")
#         return

#     with open(annotation_file, 'r') as f:
#         coco_data = json.load(f)

#     if not coco_data['images']:
#         print(f"No images found in {annotation_file}")
#         return

#     if not coco_data['annotations']:
#         print(f"No annotations found in {annotation_file}")
#         return

#     with open(output_file, 'w') as out:
#         for image_info in coco_data['images']:
#             image_id = image_info['id']
#             file_name = image_info['file_name']
#             image_path = os.path.join(dataset_path, file_name)
#             objects = []

#             for ann in coco_data['annotations']:
#                 if ann['image_id'] == image_id and 'bbox' in ann:
#                     bbox = ann['bbox']
#                     x_min, y_min, width, height = bbox
#                     x_max = x_min + width
#                     y_max = y_min + height
#                     label_id = ann['category_id']

#                     # Assuming we want to use the category name
#                     label = coco_data['categories'][label_id]['name']

#                     # Convert bbox to points format
#                     points = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
#                     obj_data = {
#                         "points": points,
#                         "transcription": label
#                     }
#                     objects.append(obj_data)

#             if objects:
#                 line = f"{image_path}\t{json.dumps(objects)}\n"
#                 out.write(line)
#             else:
#                 print(f"No objects found for image {image_path}")

# # Run the conversion for all datasets
# for dataset in datasets:
#     create_detection_txt(dataset)

# VERSION 3.0

import os
import json

# Define paths
base_path = 'Meter-dial-3-COCO'
datasets = ['Train', 'Valid', 'Test']

# Function to convert COCO format to PaddleOCR format for detection
def create_detection_txt(dataset_name):
    dataset_path = os.path.join(base_path, dataset_name)
    annotation_file = os.path.join(dataset_path, '_annotations.coco.json')
    output_file = os.path.join(base_path, f'{dataset_name.lower()}_detection.txt')

    print(f"Processing {dataset_name} dataset...")
    
    if not os.path.isfile(annotation_file):
        print(f"Annotation file not found: {annotation_file}")
        return

    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    print(f"Loaded annotations from {annotation_file}")

    if not coco_data['images']:
        print(f"No images found in {annotation_file}")
        return

    if not coco_data['annotations']:
        print(f"No annotations found in {annotation_file}")
        return

    with open(output_file, 'w') as out:
        has_written = False
        for image_info in coco_data['images']:
            image_id = image_info['id']
            file_name = image_info['file_name']
            image_path = os.path.join(dataset_path, file_name)
            objects = []

            for ann in coco_data['annotations']:
                if ann['image_id'] == image_id and 'bbox' in ann:
                    bbox = ann['bbox']
                    x_min, y_min, width, height = bbox
                    x_max = x_min + width
                    y_max = y_min + height
                    label_id = ann['category_id']

                    # Assuming we want to use the category name
                    label = coco_data['categories'][label_id]['name']

                    # Convert bbox to points format
                    points = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                    obj_data = {
                        "points": points,
                        "transcription": label
                    }
                    objects.append(obj_data)

            if objects:
                line = f"{image_path}\t{json.dumps(objects)}\n"
                out.write(line)
                has_written = True
            else:
                print(f"No objects found for image {image_path}")

        if not has_written:
            print(f"No data written for {dataset_name} dataset")

# Run the conversion for all datasets
for dataset in datasets:
    create_detection_txt(dataset)
