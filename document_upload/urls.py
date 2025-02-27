from django.urls import path
from .views import document_upload, model_page

urlpatterns = [
    path('', model_page, name='model_page'),  
    path('document_upload/', document_upload, name='document_upload'), 
]
