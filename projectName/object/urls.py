<<<<<<< HEAD

# image_processor/urls.py
from django.conf.urls.static import static
from django.conf import settings
from django.urls import path
from . import views

urlpatterns = [
    path('process-image/', views.process_image, name='process_image'),
]
=======
# image_processing_project/urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('image_processor.urls')),
]

>>>>>>> 04dd203642660d3a027ca02d7d892977e6b89616
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)