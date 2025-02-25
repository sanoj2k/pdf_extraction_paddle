from django.urls import path
from .views import document_upload

urlpatterns = [
    # path('', model_page, name='model_page'),  # Renders the upload form (upload.html)
    path('document_upload/', document_upload, name='document_upload'), 
]
