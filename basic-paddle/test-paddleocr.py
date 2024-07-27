from paddleocr import PaddleOCR

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True)

print("PaddleOCR initialized successfully!")

# Run PaddleOCR on an image
img_path = 'test.jpg'
result = ocr.ocr(img_path, cls=True)

# Print the result
# print(result)       
table_data = []
table_data.append([line[1][0], f"{line[1][1][0]:.4f}, {line[1][1][1]:.4f}", str(line[0])])
for line in result:
    table_data.append([line[1][0], f"{line[1][1]:.4f}", str(line[0])])

headers = ["Text", "Confidence", "Position"]
print(tabulate(table_data, headers=headers, tablefmt= "grid"))