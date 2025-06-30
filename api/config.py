import os
from pathlib import Path
import cv2
from openvino.inference_engine import IECore
from openvino.runtime import Core


# OpenVino Global Variables
# Model Configurations
ie = Core()

model_ac = ie.read_model(model="/ml/model/v3-small_224_1.0_float.xml")
compiled_model_ic = ie.compile_model(model=model_ac, device_name="CPU")

output_layer_ic = compiled_model_ic.output(0)

model_rs = ie.read_model(model="/ml/model/road-segmentation-adas-0001.xml")
compiled_model_rs = ie.compile_model(model=model_rs, device_name="CPU")

input_layer_rs = compiled_model_rs.input(0)
output_layer_rs = compiled_model_rs.output(0)

# Setting
DEVICE = "CPU"
# 1032: 4x superresolution, 1033: 3x superresolution
MODEL_FILE = "/ml/model/single-image-super-resolution-1032.xml"
model_name = os.path.basename(MODEL_FILE)
model_xml_path = Path(MODEL_FILE).with_suffix(".xml")

# Load the Superresolution Model
ie = IECore()
net = ie.read_network(model=str(model_xml_path))
exec_net = ie.load_network(network=net, device_name=DEVICE)

# Load Caffe Model 
net = cv2.dnn.readNetFromCaffe('/ml/model/deploy.prototxt', '/ml/model/mobilenet_iter_73000.caffemodel')
