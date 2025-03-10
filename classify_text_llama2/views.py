import os
import json
import re
import numpy as np
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import pandas as pd
import subprocess
from django.conf import settings
from datetime import datetime
from fileapp.utils import classify_text_with_mistral_latest

# Define categories
CATEGORIES = [
    "Aufenthaltstitel", "Aufteilungsplan", "Baubeschreibung", "Energieausweis",
    "Exposé", "Flurkarte", "Grundbuchauszug", "Grundriss", "Kaufvertragsentwurf",
    "Lohnsteuerbescheinigung", "Passport", "payslip", "Personalausweis",
    "Teilungserklarung", "Wohnflachenberechnung"
]

# Initialize PaddleOCR with German language
ocr = PaddleOCR(lang='german')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Function to remove invalid XML characters
def sanitize_text(text):
    pattern = r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x84\x86-\x9F]'
    return re.sub(pattern, '', text)

@csrf_exempt
def process_pdfs_and_generate_report(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        body = json.loads(request.body)
        folder_path = body.get('folder_path')
        # Generate dynamic filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_excel = body.get("output_excel", f"classification_report_{timestamp}.xlsx")
        
        if not os.path.isdir(folder_path):
            return JsonResponse({"error": "Invalid folder path"}, status=400)

        if not output_excel.endswith(".xlsx"):
            output_excel += ".xlsx"
        
        poppler_path = "/usr/bin/"
        results = []
        errors = []

        for root, _, files in os.walk(folder_path):
            for file_name in files:
                if not file_name.endswith('.pdf'):
                    continue

                pdf_path = os.path.join(root, file_name)
                relative_folder = os.path.relpath(root, folder_path)
                logger.info(f"Processing PDF: {pdf_path}")

                try:
                    images = convert_from_path(pdf_path, poppler_path=poppler_path)[:3]  # Convert first 3 pages
                    text = ""

                    for i, image in enumerate(images):
                        try:
                            image_np = np.array(image)
                            result = ocr.ocr(image_np, cls=True)
                            extracted_text = "\n".join(
                                [" ".join([word_info[1][0] for word_info in line]) for line in result if line]
                            )
                            text += sanitize_text(extracted_text) + "\n"
                        except Exception as ocr_error:
                            logger.error(f"Error in OCR processing on page {i+1} of {file_name}: {ocr_error}")
                            continue  # Continue processing other pages
                    print('text is:',text)
                    if not text.strip():
                        logger.warning(f"OCR extracted empty or low-quality text for {file_name}")
                        category = "NA"
                    else:
                        #selected_method = 'llama3.8b'
                        selected_method = 'qwen2.5'
                        category = classify_text_with_mistral_latest(text, selected_method)
                    
                    results.append({
                        "Folder Name": relative_folder,
                        "File Name": file_name,
                        "Category Name": category
                    })
                
                except Exception as pdf_error:
                    error_msg = f"Error processing {file_name}: {str(pdf_error)}"
                    logger.error(error_msg, exc_info=True)
                    errors.append(error_msg)
                    continue  # Continue processing other PDFs

        if results:
            save_to_excel(results, output_excel)
            response = {"message": "Processing and classification completed", "output_file": output_excel}
        else:
            response = {"message": "Processing completed, but no valid classifications found."}

        if errors:
            response["errors"] = errors
        return JsonResponse(response, status=200)

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON format"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Unexpected error: {str(e)}"}, status=500)

# # Function to classify text using Llama3
# def classify_text_with_llama_latest(text):
#     try:
#         prompt = f"""
#         The following text is extracted from a document. Identify the category from the predefined categories:
#         Aufenthaltstitel, Aufteilungsplan, Baubeschreibung, Energieausweis, Exposé, Flurkarte, Grundbuchauszug, Grundriss, Kaufvertragsentwurf, Lohnsteuerbescheinigung, Passport, payslip, Personalausweis, Teilungserklarung, Wohnflachenberechnung.

#         If none of the categories apply, return "NA".

#         ### Category Selection Rules:

#         1. **Exact Match (With OCR Error Handling):**  
#         If the document explicitly contains a term that matches or closely resembles one of the predefined categories, return that category immediately. Handle common OCR variations:
#         - "AUFENTHANTSTITEI", "AUFENTHANTSTITEL" → "Aufenthaltstitel"
#         - "Baubeschreihbung" → "Baubeschreibung"
#         - "Grundrizz", "Grundrizz", "Grunriss" → "Grundriss"
#         - "ENERGIEAUSWEI5", "ENERGIE AUSWEIS" → "Energieausweis"

#         2. **Contextual Classification:**
#         - **Flurkarte:**  
#         Classify as **Flurkarte** if the text contains terms like:
#         - "Flurstück", "Liegenschaftskarte", "Geobasisdaten", "Digitale Flurkarte", "Liegenschaftskataster", or "Maßstab".  
#         - Mentions of **parcels**, **land plots**, **boundary lines**, or any cadastral mapping data.  

#         - **Exposé:** Prioritize this if the text primarily contains **property descriptions** (price, amenities, agent contact, nearby locations).  

#         - **Baubeschreibung:**  
#         Classify as Baubeschreibung if the text contains construction details like:
#         - "Bauweise", "Dacheindeckung", "Fenster", "Heizung", or "Fußböden".
#         - Or explicitly mentions **"Baubeschreibung"**, **"Rohbau"**, **"Fundamente"**, **"Decke"**, **"Dach"**, **"Wärmedämmung"**, or **"Estrich"**.

#         - **Grundriss:**  
#         Classify as Grundriss if the document focuses on:
#         - **Layout plans**, **room distribution**, or **floor dimensions** without mentioning construction details or cadastral mapping.

#         3. **Identity Documents:**  
#         Only classify as Passport, Personalausweis, or Lohnsteuerbescheinigung if the exact term (or its OCR variant) explicitly appears.

#         4. **Standardization:**  
#         Return the category in the exact predefined form (e.g., "Teilungserklarung" instead of "Teilungserklärung").

#         {text[:1500]}  
#         Response format: Return only the category name, nothing else. If no match, return "NA".
#         """
        
#         result = subprocess.run(
#             ["ollama", "run", 'llama3:8b', prompt],
#             capture_output=True, text=True, timeout=60
#         )
        
#         if result.returncode == 0:
#             output = result.stdout.strip()
#             return output if output in CATEGORIES else "NA"
#         else:
#             logger.error(f"Error from Llama3: {result.stderr}")
#             return "NA"
#     except subprocess.TimeoutExpired:
#         logger.error("Llama3 classification timed out")
#         return "NA"
#     except FileNotFoundError:
#         logger.error("Ollama is not installed or not running")
#         return "NA"
#     except Exception as e:
#         logger.error(f"Unexpected error in Llama3 classification: {e}")
#         return "NA"


# Function to save results to an Excel file inside the media folder
def save_to_excel(data, filename="output.xlsx"):
    if data:
        # Ensure the media folder exists
        media_folder = settings.MEDIA_ROOT
        if not os.path.exists(media_folder):
            os.makedirs(media_folder)
        
        # Define the full path inside media folder
        output_excel = os.path.join(media_folder, filename)

        # Save the DataFrame as an Excel file
        df = pd.DataFrame(data)
        df.to_excel(output_excel, index=False, engine='openpyxl')

        logger.info(f"Excel file saved at: {output_excel}")
        return output_excel  # Return the file path if needed
    else:
        logger.warning("No valid classifications found, skipping Excel report.")
