# image_processor/views.py

import os
import uuid
import mimetypes
import google.generativeai as genai
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch
import pytesseract

# Configure the Gemini API
genai.configure(api_key=settings.API_KEY)

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

def extract_text_from_image(image_path):
    img = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(img)
    return extracted_text.strip() or "No text found in the image."

def image_to_bytes(file_path):
    with open(file_path, 'rb') as file:
        bytes_data = file.read()
    mime_type, _ = mimetypes.guess_type(file_path)
    return [{"mime_type": mime_type or 'application/octet-stream', "data": bytes_data}]

def get_assistance_response(input_prompt, image_data):
    system_prompt = """You are a specialized AI that provides accessibility assistance to visually impaired individuals. Visually impaired user will ask you queries and your goal is to provide clear answer with step by step process(where applicable)."""
    
    full_prompt = f"{system_prompt}\n{input_prompt}"
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    response = model.generate_content([full_prompt, image_data[0]])
    return response.text

def index(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')
        if uploaded_file:
            # Save the uploaded file
            file_name = default_storage.save(uploaded_file.name, ContentFile(uploaded_file.read()))
            file_path = os.path.join(settings.MEDIA_ROOT, file_name)

            # Process the image
            image = Image.open(file_path)
            predictions = detect_objects(image)
            
            if predictions:
                # Image with Boxes returned (Object Detection)
                image_with_boxes = draw_boxes(image.copy(), predictions)
                
                # What objects are present in the image returned
                image_data = image_to_bytes(file_path)
                user_prompt = "Generate descriptive textual output that interprets the content of the uploaded image for eg. Objects and to understand the scene effectively."
                response = get_assistance_response(user_prompt, image_data)
                
                try:
                    output_filename = f"{uuid.uuid4().hex}.jpg"
                    output_path = os.path.join(settings.MEDIA_ROOT, output_filename)
                    image_with_boxes.save(output_path)
                except Exception as e:
                    print(f"Error saving image: {e}")
                    output_filename = None
            else:
                output_filename = None
                response = "No objects detected in the image."

            # Text extraction from image
            extracted_text = extract_text_from_image(file_path)

            context = {
                'image_with_boxes': f"{settings.MEDIA_URL}{output_filename}" if output_filename else None,
                'extracted_text': extracted_text,
                'scene_analysis': response,
            }
            return render(request, 'image_processor/index.html', context)

    return render(request, 'image_processor/index.html')