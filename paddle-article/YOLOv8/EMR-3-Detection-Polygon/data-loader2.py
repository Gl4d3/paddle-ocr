from roboflow import Roboflow

rf = Roboflow(api_key="MM2rTKK4OnFjzDPdpQBe")
project = rf.workspace("yoloem").project("emr-dpn12")
version = project.version(3)
dataset = version.download("yolov8")
