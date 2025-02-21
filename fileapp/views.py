import os
import re
import logging
import numpy as np
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from dotenv import load_dotenv
from .utils import classify_text_with_openai, classify_text_with_mistral_latest

# Load environment variables
load_dotenv()

# Initialize PaddleOCR with German language
ocr = PaddleOCR(lang='german', use_angle_cls=True)  # Added angle classifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to remove invalid XML characters
def sanitize_text(text):
    pattern = r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x84\x86-\x9F]'
    return re.sub(pattern, '', text)

# Function to process uploaded PDF file
@csrf_exempt
def upload_and_classify_pdf(request):
    if request.method != 'POST' or 'file' not in request.FILES:
        return JsonResponse({"error": "Invalid request or missing file"}, status=400)

    selected_method = request.POST.get('selected_method', '').strip()
    print('selected method is', selected_method)
    uploaded_file = request.FILES['file']
    selected_category = request.POST.get('selected_category', '').strip()
    save_path = os.path.join(settings.MEDIA_ROOT, 'uploads', uploaded_file.name)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure folder exists
    with open(save_path, 'wb') as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)

    logger.info(f"File saved at: {save_path}")

    try:
        images = convert_from_path(save_path, poppler_path="/usr/bin/")[:3]
        extracted_text = ""

        for i, image in enumerate(images):
            try:
                image_np = np.array(image)
                result = ocr.ocr(image_np, cls=True)
                text = "\n".join(
                    [" ".join([word_info[1][0] for word_info in line]) for line in result if line]
                )
                extracted_text += sanitize_text(text) + "\n"    
            
            except Exception as ocr_error:
                logger.error(f"OCR error on page {i+1}: {ocr_error}")

        # Classify the extracted text based on the selected method
        if selected_method == 'Openai':
            extracted_category = classify_text_with_openai(extracted_text)
        elif selected_method == 'Mistral:latest':
            extracted_category = classify_text_with_mistral_latest(extracted_text, selected_method)
        elif selected_method == 'Llama2':
            extracted_category = classify_text_with_mistral_latest(extracted_text, selected_method)
        elif selected_method == 'Llama3:8b':
            extracted_category = classify_text_with_mistral_latest(extracted_text, selected_method)
        else:
            extracted_category = "NA"  # Default value if no method is selected
            logger.warning(f"Unknown method selected: {selected_method}")

        logger.info(f"Extracted category: {extracted_category}")

        # Compare extracted category with user-selected category
        category_match = extracted_category == selected_category
        print('extracted category is:', extracted_category)

        # Delete file after processing
        os.remove(save_path)
        logger.info(f"Deleted file: {save_path}")

        return JsonResponse({
            "category": extracted_category,
            "match": category_match
        }, status=200)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JsonResponse({"error": f"Unexpected error: {str(e)}"}, status=500)

# Function to render the upload page
def home_page(request):
    return render(request, 'home.html')
