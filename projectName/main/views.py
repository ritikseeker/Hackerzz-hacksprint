# appName/views.py
from django.shortcuts import render

def home(request):
    return render(request, 'index.html')  # Adjust the path as necessary
