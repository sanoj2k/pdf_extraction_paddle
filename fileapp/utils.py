
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
def classify_text_with_openai(text):
    print("Prompt:", text)
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

        2. **Flurkarte (Cadastral Map) Prioritization:**  
        - If the text includes **land registry or mapping references**, classify it as **Flurkarte** immediately.
        - **Key terms to trigger Flurkarte classification:**  
            - `"Flurkarte"`, `"Liegenschaftskataster"`, `"Flurstück"`, `"Geobasisdaten"`, `"Digitale Flurkarte"`, `"Katasteramt"`, `"ALKIS"`
        - Ignore property descriptions or construction details if the dominant focus is land parcel information.

        3. **Baubeschreibung (Construction Description) Priority:**  
        - Classify as **Baubeschreibung** if the text contains detailed construction terms like:
            - `"Bauweise"`, `"Dacheindeckung"`, `"Fenster"`, `"Heizung"`, `"Fußböden"`, `"Rohbau"`, `"Fundamente"`, `"Decke"`, `"Dach"`, `"Wärmedämmung"`, `"Estrich"`

        4. **Grundriss (Floor Plan) Handling:**  
        - If the text describes **layout, room distribution, or measurements** but lacks construction details, classify it as **Grundriss**.

        5. **Other Documents:**  
        - Identity documents (Passport, Personalausweis, Lohnsteuerbescheinigung) should only be selected if explicitly mentioned.
        - Exposé should be chosen if the text focuses on property descriptions, amenities, or real estate marketing.

        6. **Final Decision:**  
        - Select the **most relevant** category based on the dominant focus.
        - If no match is found, return `"NA"`.

        {text[:1500]}  
        Response format: Return only the category name, nothing else. If no match, return "NA".
        """

        


        
        if selected_method == 'Mistral:latest':
            model = 'mistral:latest'
        elif selected_method == 'Llama2':
            model = 'llama2:latest'
        elif selected_method == 'Llama3:8b':
            model = 'llama3:8b'
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
