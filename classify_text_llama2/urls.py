from django.urls import path
from .views import process_pdfs_and_generate_report

urlpatterns = [
    path('process_pdfs_and_generate_report/', process_pdfs_and_generate_report, name='process_pdfs_and_generate_report'), 
]
