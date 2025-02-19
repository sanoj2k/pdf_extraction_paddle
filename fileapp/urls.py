from django.urls import path
from .views import home_page, upload_and_classify_pdf

urlpatterns = [
    path('', home_page, name='home_page'),  # Renders the upload form (upload.html)
    path('upload-and-classify-pdf/', upload_and_classify_pdf, name='upload_and_classify_pdf'), 
]
