"""
URL configuration for projectName project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('main.urls')),  # Include app UR
    path('face/', include('main.urls')),  # Include app UR
<<<<<<< HEAD
     path('object/', include('object.urls')),  # Include app UR
=======
    path('object/', include('object.urls')),  # Include app UR
>>>>>>> 04dd203642660d3a027ca02d7d892977e6b89616

 
]



