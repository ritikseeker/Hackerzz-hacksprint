<<<<<<< HEAD
import os
import uuid
import mimetypes
from google import generativeai as genai
=======
# image_processor/views.py

import os
import uuid
import mimetypes
import google.generativeai as genai
>>>>>>> 04dd203642660d3a027ca02d7d892977e6b89616
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch
<<<<<<< HEAD
=======
import pytesseract
>>>>>>> 04dd203642660d3a027ca02d7d892977e6b89616

# Configure the Gemini API
genai.configure(api_key=settings.API_KEY)

<<<<<<< HEAD
=======
# Set Tesseract command path for OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

>>>>>>> 04dd203642660d3a027ca02d7d892977e6b89616
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
<<<<<<< HEAD
    image_data = image_to_bytes(image_path)
    user_prompt = "Extract and return only the text present in this image. If no text is found, return 'No text found in the image.'"
    response = get_assistance_response(user_prompt, image_data)
    return response.strip() or "No text found in the image."
=======
    img = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(img)
    return extracted_text.strip() or "No text found in the image."
>>>>>>> 04dd203642660d3a027ca02d7d892977e6b89616

def image_to_bytes(file_path):
    with open(file_path, 'rb') as file:
        bytes_data = file.read()
    mime_type, _ = mimetypes.guess_type(file_path)
    return [{"mime_type": mime_type or 'application/octet-stream', "data": bytes_data}]

def get_assistance_response(input_prompt, image_data):
<<<<<<< HEAD
    system_prompt = """You are a specialized AI that provides accessibility assistance to visually impaired individuals. You will be given you an image and query related to that image. Give appropriate response to the query and provide step by step instructions wherever applicable."""    
    full_prompt = f"{system_prompt}\n{input_prompt}"
    model = genai.GenerativeModel("gemini-1.5-flash")
    # print("Full Prompt Sent to Model:", full_prompt)
    response = model.generate_content([full_prompt, image_data[0]])
    return response.text

def process_image(request):
    if request.method == 'POST':
        print('PORCESS IMAGE WOKRING')
        uploaded_file = request.FILES.get('file')
        print(uploaded_file)
        
        user_query = request.POST.get('query', '')  # Using 'get' to avoid KeyError if 'query' is not present
        print(user_query)

        # print(uploaded_file)
=======
    system_prompt = """You are a specialized AI that provides accessibility assistance to visually impaired individuals. Visually impaired user will ask you queries and your goal is to provide clear answer with step by step process(where applicable)."""
    
    full_prompt = f"{system_prompt}\n{input_prompt}"
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    response = model.generate_content([full_prompt, image_data[0]])
    return response.text

def index(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')
>>>>>>> 04dd203642660d3a027ca02d7d892977e6b89616
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
<<<<<<< HEAD
                # user_prompt = "Generate descriptive textual output that interprets the content of the uploaded image for eg. Objects and to understand the scene effectively."
                response = get_assistance_response(user_query, image_data)
=======
                user_prompt = "Generate descriptive textual output that interprets the content of the uploaded image for eg. Objects and to understand the scene effectively."
                response = get_assistance_response(user_prompt, image_data)
>>>>>>> 04dd203642660d3a027ca02d7d892977e6b89616
                
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
<<<<<<< HEAD
            return render(request, 'process_image.html', context)

    return render(request, 'process_image.html')

=======
            return render(request, 'image_processor/index.html', context)

    return render(request, 'image_processor/index.html')
>>>>>>> 04dd203642660d3a027ca02d7d892977e6b89616
