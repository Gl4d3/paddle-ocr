from roboflow import Roboflow

rf = Roboflow(api_key="MM2rTKK4OnFjzDPdpQBe")
project = rf.workspace("hust-9u80b").project("analog-classification")
version = project.version(1)
dataset = version.download("folder")
