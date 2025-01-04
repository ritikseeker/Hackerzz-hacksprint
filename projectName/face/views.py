import cv2
import face_recognition
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Person
import base64
import json



print(dir(face_recognition))
def index1(request):
    return render(request, 'face.html')

@csrf_exempt
def recognize_face(request):
    if request.method == 'POST':
        image_data = json.loads(request.body)['image']
        image_data = image_data.split(',')[1]
        image_data = base64.b64decode(image_data)
        
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)
        print(img)
        
        if len(face_encodings) == 0:
            return JsonResponse({'message': 'No face detected'})
        
        known_faces = Person.objects.all()
        
        for face_encoding in face_encodings:
            for known_face in known_faces:
                known_encoding = np.frombuffer(known_face.face_encoding, dtype=np.float64)
                match = face_recognition.compare_faces([known_encoding], face_encoding)[0]
                if match:
                    return JsonResponse({'name': known_face.name, 'message': 'Face recognized'})
        
        return JsonResponse({'message': 'Face not recognized'})

@csrf_exempt
def add_person(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        image_data = request.POST.get('image')
        image_data = image_data.split(',')[1]
        image_data = base64.b64decode(image_data)
        
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)
        
        if len(face_encodings) == 0:
            return JsonResponse({'message': 'No face detected'})
        
        face_encoding = face_encodings[0]
        
        person = Person(name=name, face_encoding=face_encoding.tobytes())
        person.save()
        
        return JsonResponse({'message': 'Person added successfully'})

