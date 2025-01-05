from flask import Flask, render_template, request
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import os
import torch
import uuid
import pytesseract
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from environs import Env
import mimetypes

env = Env()
env.read_env()

# API Key configuration
api_key = env('API_KEY', None)
if api_key is None:
    raise ValueError("Please enter API Key")

# Configure the Gemini API
genai.configure(api_key=api_key)

# Initialize Flask app
app = Flask(__name__)

# Set Tesseract command path for OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load Object Detection Model with caching
def load_object_detection_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

object_detection_model = load_object_detection_model()

def detect_objects(image, threshold=0.3, iou_threshold=0.5):
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image)
    predictions = object_detection_model([img_tensor])[0]
    keep = torch.ops.torchvision.nms(predictions['boxes'], predictions['scores'], iou_threshold)
    
    filtered_predictions = {
        'boxes': predictions['boxes'][keep],
        'labels': predictions['labels'][keep],
        'scores': predictions['scores'][keep],
    }
    return filtered_predictions

def draw_boxes(image, predictions, threshold=0.5):
    draw = ImageDraw.Draw(image)
    for label, box, score in zip(predictions['labels'], predictions['boxes'], predictions['scores']):
        if score > threshold:
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=5)
    return image

# text from image can be done through tesseract(offline) or google gemini (online)
def extract_text_from_image(uploaded_file):
    img = Image.open(uploaded_file)
    extracted_text = pytesseract.image_to_string(img)
    return extracted_text.strip() or "No text found in the image."

def image_to_bytes(file):
    file.seek(0)  # Reset file pointer to the beginning of the BytesIO object
    bytes_data = file.read()  # Read the actual bytes from the BytesIO object
    mime_type, _ = mimetypes.guess_type(file.filename)  # Guess MIME type based on filename
    return [{"mime_type": mime_type or 'application/octet-stream', "data": bytes_data}]

def get_assistance_response(input_prompt, image_data):
    system_prompt = """You are a specialized AI that provides accessibility assistance to visually impaired individuals. Visually impaired user will ask you queries and your goal is to provide clear answer with step by step process(where applicable)."""
    
    full_prompt = f"{system_prompt}\n{input_prompt}"
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    response = model.generate_content([full_prompt, image_data[0]])
    return response.text


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file:
            image = Image.open(uploaded_file)
            predictions = detect_objects(image)
            if predictions:
                # Image with Boxes returned (Object Detection)
                image_with_boxes = draw_boxes(image.copy(), predictions)
                # What objects are present in the image returned
                image_data = image_to_bytes(uploaded_file)
                user_prompt = "Generate descriptive textual output that interprets the content of the uploaded image for eg. Objects and to understand the scene effectively."
                response = get_assistance_response(user_prompt, image_data)  # Assuming this function is defined elsewhere
                
                try:
                    output_filename = os.path.join(app.root_path, 'static', f"{uuid.uuid4().hex}.jpg")
                    image_with_boxes.save(output_filename)
                except Exception as e:
                    print(f"Error saving image: {e}")
                    output_filename = None
            else:
                output_filename = None

            # Text extraction from image
            extracted_text = extract_text_from_image(uploaded_file)

            return render_template('index.html', 
                                   image_with_boxes=f"static/{os.path.basename(output_filename)}" if output_filename else None,
                                   extracted_text=extracted_text,
                                   scene_analysis=response)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
