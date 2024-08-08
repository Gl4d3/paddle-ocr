# Good performing general purpose dataset for OCR
# 4000+ images of digital meters (Notices decimal points & labels-kwh)
from roboflow import Roboflow

rf = Roboflow(api_key="MM2rTKK4OnFjzDPdpQBe")
project = rf.workspace("hstech").project("ocrscale")

version = project.version(6)
dataset = version.download("yolov8")

# Performs poorly on analog meters.