from roboflow import Roboflow

rf = Roboflow(api_key="MM2rTKK4OnFjzDPdpQBe")
project = rf.workspace("generateddisplay2").project("meter-dial")
version = project.version(3)
dataset = version.download("coco")

