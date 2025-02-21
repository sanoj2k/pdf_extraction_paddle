
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
        1. **Exact Match Priority**:  
        - If the document explicitly contains a term that directly matches one of the predefined categories (e.g., "Grundbuchauszug" or "Energieausweis"), return that category **immediately**, without considering the document’s context.  

        2. **Contextual Classification (Only if No Exact Match Found):**  
        - **Exposé** → A **real estate listing**.  
            - If the document contains property descriptions (e.g., price, amenities, nearby locations, and a general property summary) along with energy efficiency details, classify it as **"Exposé"**.  
            - If the document **only contains energy efficiency details and consumption ratings**, classify it as **"Energieausweis"**, not "Exposé".  

        - **Grundriss** (Floor Plan) vs. **Baubeschreibung** (Construction Description):  
            - If the text contains **"floor plan", "site plan", "overview", "room layout"**, classify it as **"Grundriss"**.  
            - If the text contains **"construction description", "structural work", "foundations", "roof", "construction", "materials"**, classify it as **"Baubeschreibung"**.  
            - If both terms exist, prioritize **"Baubeschreibung"** if it contains detailed construction elements.  

        - **Teilungserklarung** → A **property division agreement** in legal/real estate contexts.  
        - **Kaufvertragsentwurf** → A **real estate contract** (mentions "Kaufvertrag", notary, buyer/seller).  

        3. **Identity Documents Rule:**  
        - **Only classify as Passport, Personalausweis, or Lohnsteuerbescheinigung if there is a direct, explicit mention of these terms in the document.**  
        - **DO NOT infer these categories based on names, dates, or legal mentions alone.**  

        4. **Standardization:**  
        - Always return the category name in the exact predefined form (e.g., **"Teilungserklarung" instead of "Teilungserklärung"**).  

        **Text (in German):** {text[:1500]}  

        **Response format: Return only the category name, nothing else. If there is no exact match, return "NA".**  
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

# # Function to classify text using Llama2
# def classify_text_with_llama2_latest(text):
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
#             ["ollama", "run", "llama2:latest", prompt],
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
