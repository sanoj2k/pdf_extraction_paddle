
import openai
from dotenv import load_dotenv
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
    "Lohnsteuerbescheinigung", "Passport", "Payslip", "Personalausweis",
    "Teilungserklarung", "Wohnflachenberechnung"
]

import unicodedata

def normalize_text(text):
    """Normalize text by converting umlauts and stripping spaces."""
    replacements = {
        "ä": "a", "ö": "o", "ü": "u", "ß": "ss",
        "Ä": "A", "Ö": "O", "Ü": "U"
    }
    for umlaut, replacement in replacements.items():
        text = text.replace(umlaut, replacement)
    return text

def clean_category(category):
    category = category.strip()  # Remove extra spaces and newlines
    category = normalize_text(category.lower())  # Normalize case and umlauts

    # Standardize variations of "Payslip"
    if "payslip" in category or "payroll statement" in category:
        return "Payslip"

    # Normalize categories for exact match
    normalized_categories = {normalize_text(c.lower()): c for c in CATEGORIES}

    return normalized_categories.get(category, "NA")


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
def classify_text_with_llm(text, selected_method):
    print('selected method is', selected_method)
    # print("ext test is:", text)
    try:
        # Prepare the prompt for Llama2 
        prompt = f"""
        The following text is extracted from a document. Identify the category from the predefined categories:  
        Aufenthaltstitel, Aufteilungsplan, Baubeschreibung, Energieausweis, Exposé, Flurkarte, Grundbuchauszug, Grundriss, Kaufvertragsentwurf, Lohnsteuerbescheinigung, Passport, payslip, Personalausweis, Teilungserklarung, Wohnflachenberechnung.  

        If none of the categories apply, return "NA".  

        ### Category Selection Rules:

        1. **Exact Match (With OCR Error Handling):**  
        - "AUFENTHANTSTITEI", "AUFENTHANTSTITEL" → "Aufenthaltstitel"  
        - "Baubeschreihbung" → "Baubeschreibung"  
        - "Grundrizz", "Grunriss" → "Grundriss"  
        - "ENERGIEAUSWEI5", "ENERGIE AUSWEIS" → "Energieausweis"  

        2. **Contextual Classification:**

        - **Kaufvertragsentwurf (Draft Sales Contract)**  
            Classify as **Kaufvertragsentwurf** if the document contains:  
            - Legal terms related to **property sales contracts** such as:  
            - "Notar", "beurkunde", "Kaufvertrag", "Verkäufer", "Erwerber", "Kaufgegenstand", "Eigentumsübertragung", "Miteigentumsanteil", "Übergang von Nutzen und Lasten", "Kaufpreis", "Verwalterzustimmung erforderlich", "Grundbuchinhalt", "Belastungen", "Wohnungs-/Teileigentumsgrundbuch".  
            - Mentions of **buyers (Käufer) and sellers (Verkäufer)** in a contractual context.  
            - Sections indicating **contractual obligations or notarization**.  
            - If "Kaufvertrag" or "beurkunde" appear, classify as **Kaufvertragsentwurf** immediately.  

        -**Baubeschreibung**
            -Classify as Baubeschreibung if the text contains:
                -"BAUBESCHREIBUNG" explicitly.
                -Technical construction details, such as:
                    -"Rohbau," "Fundamente," "Dacheindeckung," "Fenster," "Heizung," "Isolierung," "Estrich," "Wärmedämmung," "Stahlbeton," "Mauerwerk," "Tiefgarage," "Dachstuhl," "Schallschutzziegel," "Bodenbeläge," "Kalksandstein"
            -Mentions of building materials, structural components, insulation techniques, or engineering details.
            -If "Baubeschreibung" appears anywhere in the text, classify it as Baubeschreibung immediately.
        -**Exposé**
            -Classify as Exposé if the text primarily describes:

                -Property listings or advertisements for real estate.
                -Key real estate details such as:
                    -"Wohnfläche," "Kaufpreis," "Provision," "Makler," "Lage," "Ausstattung," "Tiefgarage," "Balkon," "Energieeffizienz," "Miete," "Angebotspreis," "Grundstück"
                    -Lifestyle descriptions, e.g., "zentral eingebundene Lage," "moderne Architektur," "hochwertige Ausstattung," "komfortables Wohnen"
                    -Descriptions of nearby schools, transport, parks, or amenities.
        -**Do NOT classify as Exposé if**:
            -"BAUBESCHREIBUNG" appears in the text.
            -The document contains technical construction details (e.g., foundation, insulation, materials).
            -The focus is on engineering or structural descriptions rather than property marketing.

        - **Flurkarte:**  
            Classify as **Flurkarte** if the text contains terms like:
            - "Flurstück", "Liegenschaftskarte", "Geobasisdaten", "Digitale Flurkarte", "Liegenschaftskataster", or "Maßstab".
            - Mentions of **parcels, land plots, boundary lines**, or **cadastral mapping data**.
            
        - **Personalausweis**:
            - If "PERSONALAUSWEIS" appears anywhere in the text, classify it as "Personalausweis" immediately.
            - Do NOT classify as "Aufenthaltstitel" if "Personalausweis" is present.
            - If the text contains **"PERSONALAUSWEIS"** explicitly, classify it as **"Personalausweis"**.
            - Handle common OCR variations:
                - "PFRSONALAUSWEIS", "PERSONALAUSWELS", "PER5ONALAUSWEI5" → "Personalausweis".
            - If "BUNDESREPUBLIK DEUTSCHLAND" appears together with **"IDENTITY CARD"** or **"CARTED'IDENTITÉ"**, classify as **"Personalausweis"**.


        - **Do NOT classify as Exposé if**:  
            - The document contains **contractual legal language** or mentions **notarization**.  
            - The text describes **ownership transfer, legal obligations, or official registry details**.  

        3. **Identity Documents:**  
        - Only classify as **Passport, or Lohnsteuerbescheinigung** if the exact term (or its OCR variant) explicitly appears.  

        4. **Standardization:**  
        - Return the category in the exact predefined form (e.g., "Teilungserklarung" instead of "Teilungserklärung").  

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
        elif selected_method == 'qwen2.5':
            model = 'qwen2.5:14b'
        else:
            model = 'Mistral:latest'


        # Run the Llama2 model using ollama
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True, text=True
        )
        # print('result is:', result)
        # Capture and process the output from Llama2
        if result.returncode == 0:
            output = result.stdout.strip()
            print('output is:', output)
            cleaned_category = clean_category(output) 
            return cleaned_category if cleaned_category in CATEGORIES else "NA"
        else:
            logger.error(f"Error from Llama3: {result.stderr}")
            return "NA"

    except Exception as e:
        logger.error(f"Error in Llama3 classification: {e}")
        return "NA"