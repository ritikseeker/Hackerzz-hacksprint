from django.urls import path, include
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('', views.home, name='home'),  # Example view
    path('face/', include('face.urls')),  # This ensures all URLs starting with 'face/' are routed to face app
<<<<<<< HEAD
    path('', include('object.urls')),  # This ensures all URLs starting with 'face/' are routed to face app

    

=======
    
    
>>>>>>> 04dd203642660d3a027ca02d7d892977e6b89616
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)