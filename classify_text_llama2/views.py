# import os
# import re
# import logging
# import numpy as np
# import pandas as pd
# from django.conf import settings
# from django.shortcuts import render
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from django.core.files.storage import default_storage
# from django.core.files.base import ContentFile
# from pdf2image import convert_from_path
# from paddleocr import PaddleOCR
# import subprocess

# # Initialize PaddleOCR with German language
# ocr = PaddleOCR(lang='german')

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Define categories
# CATEGORIES = [
#     "Aufenthaltstitel", "Aufteilungsplan", "Baubeschreibung", "Energieausweis",
#     "Exposé", "Flurkarte", "Grundbuchauszug", "Grundriss", "Kaufvertragsentwurf",
#     "Lohnsteuerbescheinigung", "Passport", "payslip", "Personalausweis",
#     "Teilungserklarung", "Wohnflachenberechnung"
# ]

# # Function to remove invalid XML characters
# def sanitize_text(text):
#     pattern = r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x84\x86-\x9F]'
#     return re.sub(pattern, '', text)

# # Function to classify text using Llama2
# def classify_text_with_llama2(text):
#     print("ext test is:", text)
#     try:
#         # Prepare the prompt for Llama2 
#         prompt = f"""
#         The following text is extracted from a document. Identify the category from the predefined categories: {', '.join(CATEGORIES)}.
#         If none of the categories apply, return 'NA'.
#         When responding, always standardize the category name to the exact form in the predefined categories list, e.g., return 'Teilungserklarung' instead of 'Teilungserklärung'.

#         Text: {text[:1000]}  # Limiting to first 1000 characters

#         Response format: Only return the category name, nothing else. If there is no exact match, return 'NA'.
#         """
        

#         # Run the Llama2 model using ollama
#         result = subprocess.run(
#             ["ollama", "run", "mistral:latest", prompt],
#             capture_output=True, text=True
#         )

#         # Capture and process the output from Llama2
#         if result.returncode == 0:
#             output = result.stdout.strip()
#             return output if output in CATEGORIES else "NA"
#         else:
#             logger.error(f"Error from Llama3: {result.stderr}")
#             return "NA"

#     except Exception as e:
#         logger.error(f"Error in Llama3 classification: {e}")
#         return "NA"

# # Function to process uploaded PDF file
# @csrf_exempt
# def upload_and_classify_pdf(request):
#     # print('test')
#     if request.method != 'POST' or 'file' not in request.FILES:
#         return JsonResponse({"error": "Invalid request or missing file"}, status=400)

#     uploaded_file = request.FILES['file']
#     selected_category = request.POST.get('selected_category', '').strip()  # Get selected category
#     save_path = os.path.join(settings.MEDIA_ROOT, 'uploads', uploaded_file.name)
    
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure folder exists
#     with open(save_path, 'wb') as f:
#         for chunk in uploaded_file.chunks():
#             f.write(chunk)

#     logger.info(f"File saved at: {save_path}")

#     try:
#         images = convert_from_path(save_path)[:3]
#         extracted_text = ""

#         for i, image in enumerate(images):
#             try:
#                 image_np = np.array(image)
#                 result = ocr.ocr(image_np, cls=True)
#                 text = "\n".join(
#                     [" ".join([word_info[1][0] for word_info in line]) for line in result if line]
#                 )
#                 extracted_text += sanitize_text(text) + "\n"
            
#             except Exception as ocr_error:
#                 logger.error(f"OCR error on page {i+1}: {ocr_error}")



#         extracted_category = classify_text_with_llama2(extracted_text)  # Use Llama2 here
#         print('extracted category is:', extracted_category)
#         # print('test')
#         # Store the file info in the PostgreSQL database
#         # UploadedFile.objects.create(
#         #     file_name=uploaded_file.name,
#         #     category=extracted_category
#         # )

#         # Compare extracted category with user-selected category
#         category_match = extracted_category == selected_category

#         # Delete file after processing
#         os.remove(save_path)
#         logger.info(f"Deleted file: {save_path}")

#         # Return the category as response
#         return JsonResponse({
#             "category": extracted_category,
#             "match": category_match
#         }, status=200)

#     except Exception as e:
#         logger.error(f"Unexpected error: {e}")
#         return JsonResponse({"error": f"Unexpected error: {str(e)}"}, status=500)

# # Function to render the upload page
# def upload_page(request):
#     return render(request, 'upload.html')
