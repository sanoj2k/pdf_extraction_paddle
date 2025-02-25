
import openai
from dotenv import load_dotenv
import os
import subprocess

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
import subprocess

load_dotenv()
# from .models import UploadedFile  

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


# Function to classify text using Openai
def classify_text_with_openai(text, selected_method):
    print('selected method is', selected_method)
    print("Text is:", text)
    try:
        prompt = f"""
        The following text is extracted from a document. Identify the category from the predefined categories: {', '.join(CATEGORIES)}.
        If none of the categories apply, return 'NA'.
        When responding, always standardize the category name to the exact form in the predefined categories list, e.g., return 'Teilungserklarung' instead of 'Teilungserklärung'.

        Text: {text[:1000]}  # Limiting to first 1000 characters

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


# Initialize PaddleOCR with German language
ocr = PaddleOCR(lang='german')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Function to classify text using Llama2
def classify_text_with_mistral_latest(text, selected_method):
    print('selected method is', selected_method)
    print("ext test is:", text)
    try:
        # Prepare the prompt for Llama2 
        prompt = f"""
        The following text is extracted from a document. Identify the category from the predefined categories:
        Aufenthaltstitel, Aufteilungsplan, Baubeschreibung, Energieausweis, Exposé, Flurkarte, Grundbuchauszug, Grundriss, Kaufvertragsentwurf, Lohnsteuerbescheinigung, Passport, payslip, Personalausweis, Teilungserklarung, Wohnflachenberechnung.

        If none of the categories apply, return "NA".

        ### Category Selection Rules:

        1. **Exact Match (With OCR Error Handling):**  
        If the document explicitly contains a term that matches or closely resembles one of the predefined categories, return that category immediately. Handle common OCR variations:
        - "AUFENTHANTSTITEI", "AUFENTHANTSTITEL" → "Aufenthaltstitel"
        - "Baubeschreihbung" → "Baubeschreibung"
        - "Grundrizz", "Grundrizz", "Grunriss" → "Grundriss"
        - "ENERGIEAUSWEI5", "ENERGIE AUSWEIS" → "Energieausweis"

        2. **Contextual Classification:**
        - **Flurkarte:**  
        Classify as **Flurkarte** if the text contains terms like:
        - "Flurstück", "Liegenschaftskarte", "Geobasisdaten", "Digitale Flurkarte", "Liegenschaftskataster", or "Maßstab".  
        - Mentions of **parcels**, **land plots**, **boundary lines**, or any cadastral mapping data.  

        - **Exposé:** Prioritize this if the text primarily contains **property descriptions** (price, amenities, agent contact, nearby locations).  

        - **Baubeschreibung:**  
        Classify as Baubeschreibung if the text contains construction details like:
        - "Bauweise", "Dacheindeckung", "Fenster", "Heizung", or "Fußböden".
        - Or explicitly mentions **"Baubeschreibung"**, **"Rohbau"**, **"Fundamente"**, **"Decke"**, **"Dach"**, **"Wärmedämmung"**, or **"Estrich"**.

        - **Grundriss:**  
        Classify as Grundriss if the document focuses on:
        - **Layout plans**, **room distribution**, or **floor dimensions** without mentioning construction details or cadastral mapping.

        3. **Identity Documents:**  
        Only classify as Passport, Personalausweis, or Lohnsteuerbescheinigung if the exact term (or its OCR variant) explicitly appears.

        4. **Standardization:**  
        Return the category in the exact predefined form (e.g., "Teilungserklarung" instead of "Teilungserklärung").

        {text[:1500]}  
        Response format: Return only the category name, nothing else. If no match, return "NA".
        """

        


        
        if selected_method == 'Mistral:latest':
            model = 'mistral:latest'
        elif selected_method == 'Llama2':
            model = 'llama2:latest'
        elif selected_method == 'llama3:8b':
            model = 'llama3:8b'
        elif selected_method == 'deepseek-r1:14b':
            model = 'deepseek-r1:14b'
        else:
            model = 'Mistral:latest'


        # Run the Llama2 model using ollama
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True, text=True
        )

        # Capture and process the output from Llama2
        if result.returncode == 0:
            output = result.stdout.strip()
            return output if output in CATEGORIES else "NA"
        else:
            logger.error(f"Error from Llama3: {result.stderr}")
            return "NA"

    except Exception as e:
        logger.error(f"Error in Llama3 classification: {e}")
        return "NA"
