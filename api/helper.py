import numpy as np
import cv2
import base64 

def to_rgb(image_data) -> np.ndarray:
    """
    Convert image_data from BGR to RGB
    """
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)


def encode_image_base64(image_data):
    _, buffer = cv2.imencode('.png', image_data)  # You can change to .jpg if you want
    return base64.b64encode(buffer).decode('utf-8')



def decode_base64_image(image_data):
    # Decode base64 string back to binary
    image_bytes = base64.b64decode(image_data)
    # Convert binary to numpy image
    np_array = np.frombuffer(image_bytes, np.uint8)
    origin_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return origin_image



def convert_result_to_image(result) -> np.ndarray:
    """
    Convert network result of floating point numbers to image with integer
    values from 0-255. Values outside this range are clipped to 0 and 255.

    :param result: a single superresolution network result in N,C,H,W shape
    """
    result = result.squeeze(0).transpose(1, 2, 0)
    result *= 255
    result[result < 0] = 0
    result[result > 255] = 255
    result = result.astype(np.uint8)
    return result