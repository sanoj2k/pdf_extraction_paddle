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
from fileapp.utils import classify_text_with_openai, classify_text_with_mistral_latest

# Load environment variables
load_dotenv()

# Initialize PaddleOCR with German language
ocr = PaddleOCR(lang='german', use_angle_cls=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Remove invalid XML characters
def sanitize_text(text):
    pattern = r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x84\x86-\x9F]'
    return re.sub(pattern, '', text)

# Classification method dispatcher
CLASSIFICATION_METHODS = {
    'Openai': classify_text_with_openai,
    'Mistral:latest': classify_text_with_mistral_latest,
    'Llama2': classify_text_with_mistral_latest,
    'llama3:8b': classify_text_with_mistral_latest,
    'deepseek-r1:14b': classify_text_with_mistral_latest
}

# Process uploaded PDF file
@csrf_exempt
def document_upload(request):
    if request.method != 'POST' or not request.FILES.get('file'):
        return JsonResponse({"error": "Invalid request or missing file"}, status=400)

    selected_method = request.POST.get('selected_method', '').strip()
    uploaded_file = request.FILES['file']
    selected_category = request.POST.get('in_category', '').strip()
    selected_file_id = request.POST.get('file_id', '').strip()
    # user_details = request.POST.get('user_details', {})

    if len(selected_file_id) > 15:
        return JsonResponse("file_id length exceeds the maximum limit of 15 characters.")
    
    if not re.match(r'^[a-z0-9]+$', selected_file_id):
        file_id = "Not valid"
    else:
        file_id = "Valid"
    

    # Check if the uploaded file is a PDF
    if not uploaded_file.name.lower().endswith('.pdf'):
        return JsonResponse({"error": "Only PDF files are allowed."}, status=400)

    save_path = os.path.join(settings.MEDIA_ROOT, 'uploads', uploaded_file.name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        # Save the uploaded file
        with open(save_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        logger.info(f"File saved at: {save_path}")

        # Convert PDF to images (first 3 pages only)
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

        # Select the classification method dynamically
        print('extracted text is:', extracted_text)
        classify_function = CLASSIFICATION_METHODS.get(selected_method)
        if classify_function:
            extracted_category = classify_function(extracted_text, selected_method)
        else:
            print("not using any method")
            extracted_category = "NA"
            logger.warning(f"Unknown method selected: {selected_method}")

        logger.info(f"Extracted category: {extracted_category}")

        # Compare extracted category with user-selected category
        category_match = extracted_category == selected_category
        # print('selected category is:', selected_category)
        print('extracted category is:', extracted_category)
        category_score = 100


        return JsonResponse({
            "out_category": extracted_category,
            "match": category_match,
            "category_score": category_score,
            "file_id": file_id
            # "user_details": user_details
        }, status=200)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JsonResponse({"error": "Unexpected error occurred", "details": str(e)}, status=500)

    finally:
        # Ensure the uploaded file is deleted after processing
        if os.path.exists(save_path):
            os.remove(save_path)
            logger.info(f"Deleted file: {save_path}")


def model_page(request):
    return render(request, 'model.html')


