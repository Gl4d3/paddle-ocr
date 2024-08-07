from roboflow import Roboflow

rf = Roboflow(api_key="MM2rTKK4OnFjzDPdpQBe")
project = rf.workspace("work-n51jy").project("ssc-4wymd")
version = project.version(2)
dataset = version.download("yolov8")
