from roboflow import Roboflow

rf = Roboflow(api_key="MM2rTKK4OnFjzDPdpQBe")
project = rf.workspace("meter-i5vud").project("meter-digits-frvqn")
version = project.version(2)
dataset = version.download("yolov8")