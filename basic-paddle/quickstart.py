from paddleocr import PaddleOCR, draw_ocr
from PIL import ImageFont # for custom font

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch') #use_angle_cls=True for text detection model
print("PaddleOCR initialized successfully!")

# Path to the image
img_path = 'table-ocr-1.jpg'

# Run the model on an image
result = ocr.ocr(img_path, cls=True)

# Print the result
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)

# draw result
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]

font_path = 'C:\\Windows\\Fonts\\Arial.ttf'
im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save('./images/results/result.jpg')


# # Print the result
# # print(result)       
# table_data = []
# table_data.append([line[1][0], f"{line[1][1][0]:.4f}, {line[1][1][1]:.4f}", str(line[0])])
# for line in result:
#     table_data.append([line[1][0], f"{line[1][1]:.4f}", str(line[0])])

# headers = ["Text", "Confidence", "Position"]
# print(tabulate(table_data, headers=headers, tablefmt= "grid"))