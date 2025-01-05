import face_recognition
import cv2

# Load an image file
img = face_recognition.load_image_file("your_image.jpg")

# Find all face locations in the image
face_locations = face_recognition.face_locations(img)

# Print the face locations
print(f"Found {len(face_locations)} face(s) in the image.")
