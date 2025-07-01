from flask import Flask, request, jsonify
import serverless_wsgi
from model import image_classify, superresolution, road_segmentation, object_detection
from helper import decode_base64_image, encode_image_base64
from config import exec_net

app = Flask(__name__)

# Root API endpoint 
@app.route("/", methods=['GET'])
def index():
    return jsonify({
    "html":"""<h1>Welcome to Computer Vison As Service API!</h1>
    <p><b><i>Developed by Mohamad Oghli<i></b></p>"""
    })

# Image Classifier API endpoint
@app.route('/classify_image', methods=['POST'])
def cls_img_inference():
    data = request.get_json()
    if 'image' in data:
        # Decode base64 string back to binary
        req_image = decode_base64_image(data['image'])
        image_class = image_classify(req_image)
        return jsonify({"img_class": image_class})
    return jsonify({"message": "Sorry, Invalid Image Parameter!"})

# Road Segmentation API endpoint
@app.route('/road_segmentation', methods=['POST'])
def rseg_inference():
    data = request.get_json()
    if 'image' in data:
        # Decode base64 string back to binary
        req_image = decode_base64_image(data['image'])
        seg_image = road_segmentation(req_image)

        # Encode segmentation image as base64
        seg_image_base64 = encode_image_base64(seg_image)

        return jsonify({"seg_image": seg_image_base64})
    return jsonify({"message": "Sorry, Invalid Image Parameter!"})


# Super Resolution API endpoint
@app.route('/super_resolution', methods=['POST'])
def sr_inference():
    data = request.get_json()
    if 'image' in data:
        # Decode base64 string back to binary
        req_image = decode_base64_image(data['image'])
        sr_image = superresolution(req_image, exec_net)

        # Encode super image as base64
        sr_image_base64 = encode_image_base64(sr_image)

        return jsonify({"super": sr_image_base64})
    return jsonify({"message": "Sorry, Invalid Image Parameter!"})


# Object Detection API endpoint
@app.route('/object_detection', methods=['POST'])
def obj_detect_inference():
    # Flags for optional response fields
    return_image = request.args.get('return_image', 'false').lower() == 'true'
    data = request.get_json()
    if 'image' in data:
        # Decode base64 string back to binary
        req_image = decode_base64_image(data['image'])
        if 'confidence' in data:
            detection_results, detection_image = object_detection(req_image, confidence_threshold=data['confidence'])
        else:
            detection_results, detection_image = object_detection(req_image)

        # Build final JSON response
        response = {
            'detections': detection_results
        }
        
        if return_image:
            # Encode detection image as base64
            detection_image_base64 = encode_image_base64(detection_image)
            response['detection_image'] = detection_image_base64

        return jsonify(response)
    return jsonify({"message": "Sorry, Invalid Image Parameter!"})


def lambda_handler(event, context):
    return serverless_wsgi.handle_request(app, event, context)
