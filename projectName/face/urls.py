from django.urls import path
from . import views

urlpatterns = [
    path('recognize/', views.recognize_face, name='recognize'),
    path('add_person/', views.add_person, name='add_person'),
]

