from django.urls import path
from .views import upload_page, upload_and_classify_pdf

urlpatterns = [
    path('llama/', upload_page, name='upload_page'),  # Renders the upload form (upload.html)
    path('upload-and-classify-pdf/', upload_and_classify_pdf, name='upload_and_classify_pdf'), 
]
