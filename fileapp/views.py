import os
import re
import logging
import numpy as np
import pandas as pd
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import openai
from dotenv import load_dotenv
import os

load_dotenv()
# from .models import UploadedFile  


# Initialize PaddleOCR with German language
ocr = PaddleOCR(lang='german')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=OPENAI_API_KEY)


# Define categories
CATEGORIES = [
    "Aufenthaltstitel", "Aufteilungsplan", "Baubeschreibung", "Energieausweis",
    "Exposé", "Flurkarte", "Grundbuchauszug", "Grundriss", "Kaufvertragsentwurf",
    "Lohnsteuerbescheinigung", "Passport", "payslip", "Personalausweis",
    "Teilungserklarung", "Wohnflachenberechnung"
]

# Function to remove invalid XML characters
def sanitize_text(text):
    pattern = r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x84\x86-\x9F]'
    return re.sub(pattern, '', text)

# Function to classify text using Mistral
def classify_text_with_openai(text):
    print("Prompt:", text)
    try:
        prompt = f"""
        The following text is extracted from a document. Identify the category from the predefined categories: {', '.join(CATEGORIES)}.
        If none of the categories apply, return 'NA'.
        
        Text: {text}
        
        Response format: Only return the category name, nothing else. If there is no exact match, return 'NA'.
        """

        chat_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a strict classifier and do not infer categories."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10
        )

        logger.info(f"Full chat response: {chat_response}")

        if hasattr(chat_response, 'choices') and len(chat_response.choices) > 0:
            assistant_message = chat_response.choices[0].message
            return assistant_message.content.strip() if assistant_message.content.strip() in CATEGORIES else "NA"
        
        return "NA"
    
    except Exception as e:
        logger.error(f"Error in classification: {e}")
        return "NA"

# Function to process uploaded PDF file

@csrf_exempt
def upload_and_classify_pdf(request):
    if request.method != 'POST' or 'file' not in request.FILES:
        return JsonResponse({"error": "Invalid request or missing file"}, status=400)

    uploaded_file = request.FILES['file']
    selected_category = request.POST.get('selected_category', '').strip()  # Get selected category
    save_path = os.path.join(settings.MEDIA_ROOT, 'uploads', uploaded_file.name)
    

    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure folder exists
    with open(save_path, 'wb') as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)

    logger.info(f"File saved at: {save_path}")

    try:
        # poppler_path = r'C:\\MY SOTF\\poppler-24.08.0\\Library\\bin'
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

        extracted_category = classify_text_with_openai(extracted_text)

        # Store the file info in the PostgreSQL database
        # UploadedFile.objects.create(
        #     file_name=uploaded_file.name,
        #     category=extracted_category
        # )

        # Compare extracted category with user-selected category
        category_match = extracted_category == selected_category


        # Delete file after processing
        os.remove(save_path)
        logger.info(f"Deleted file: {save_path}")

        # Return the category as response
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
