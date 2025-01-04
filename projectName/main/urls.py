from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Example view
    path('face/', include('face.urls')),  # This ensures all URLs starting with 'face/' are routed to face app

]
