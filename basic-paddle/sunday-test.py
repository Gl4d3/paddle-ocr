# Import Library
from paddleocr import PaddleOCR

# Initialize "client"
myOcr = PaddleOCR(use_angle_cls=True, lang='ch')
print("PaddleOCR initialized successfully")

# Performing OCR
img_path = "./images/meter-1.jfif"
result = myOcr.ocr(img_path, cls=True)
print('model run successfully')

# Here's the output:
    # PaddleOCR initialized successfully
    # [2024/08/11 09:35:47] ppocr DEBUG: dt_boxes num : 11, elapsed : 0.4686739444732666
    # [2024/08/11 09:35:47] ppocr DEBUG: cls num  : 11, elapsed : 0.16307377815246582
    # [2024/08/11 09:35:48] ppocr DEBUG: rec_res num  : 11, elapsed : 1.4785797595977783
    # model run successfully

# Print the text
for line in result:
    for word_info in line:
        print(word_info[-1])

# Here's the output:
    # ('Made in Slovenia', 0.8718852996826172)
    # ('2007', 0.9976920485496521)
    # ('回', 0.980918824672699)
    # ('KWh', 0.7692621350288391)
    # ('016932.45', 0.9916929602622986)
    # ('T1T2', 0.9237879514694214)
    # ('DROPDF', 0.9340569376945496)
    # ('Soroli', 0.8708110451698303)
    # ('1000mp/kWh', 0.9368378520011902)
    # ('AC-1Phase2Wire', 0.972956120967865)
    # ('Rate1=Low00:00-07:00GMT', 0.9406930804252625)