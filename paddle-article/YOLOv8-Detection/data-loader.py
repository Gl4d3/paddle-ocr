from roboflow import Roboflow

# Initialize Roboflow API with your API key
rf = Roboflow(api_key="MM2rTKK4OnFjzDPdpQBe")

# Specify the workspace, project, and version you want to download the dataset from
workspace_name = "generateddisplay2"
project_name = "meter-dial"
version_number = 3

# Get the specified version of the project
project = rf.workspace(workspace_name).project(project_name)
version = project.version(version_number)

# Download the dataset associated with the specified version
dataset = version.download("yolov8")
