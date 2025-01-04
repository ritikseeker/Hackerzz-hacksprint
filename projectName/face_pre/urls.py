from django.urls import path
from . import views

urlpatterns = [
    path('', views.index1, name='index'),
    path('recognize/', views.recognize_face, name='recognize_face'),
    path('add_person/', views.add_person, name='add_person'),
]

