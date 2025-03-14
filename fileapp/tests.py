import os
import requests
from docx import Document
import textract
import openai
from dotenv import load_dotenv

load_dotenv()

# Load OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Read and extract text from various file formats
def read_file(file_path):
    try:
        if file_path.lower().endswith(".txt"):
            return read_txt(file_path)
        elif file_path.lower().endswith(".docx"):
            return read_docx(file_path)
        elif file_path.lower().endswith(".doc"):
            return read_doc(file_path)
        else:
            return "Error: Unsupported file format."
    except Exception as e:
        return f"Error: Unable to read file {file_path}. Exception: {str(e)}"

# Read text from DOCX file
def read_docx(file_path):
    try:
        doc = Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading DOCX file: {str(e)}"

# Read text from DOC file (older Word format)
def read_doc(file_path):
    try:
        text = textract.process(file_path).decode("utf-8")
        return text.strip()
    except Exception as e:
        return f"Error reading DOC file: {str(e)}"

# Read text from TXT file
def read_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        return f"Error reading TXT file: {str(e)}"

# Verify extracted MOM using OpenAI API
def verify_mom(transcript, agenda, extracted_mom):
    # Input validation
    if not transcript.strip() or not agenda.strip() or not extracted_mom.strip():
        return "Error: One or more input files are empty. Please provide valid data."

    prompt = f"Given the following Meeting Transcript:\n{transcript}\nAnd the following Agenda:\n{agenda}\nVerify whether the following Extracted MOM is correct:\n{extracted_mom}\nHighlight any missing points or hallucinations."

    try:
        # Using OpenAI API with your preferred client approach
        chat_response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use your model here
            messages=[
                {"role": "system", "content": "You are a strict classifier and do not infer categories."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200  # Adjust max_tokens as needed
        )

        # Debugging to understand the structure of the response
        # print("Chat Response:", chat_response)

        # Adjusting the way to access the response
        if hasattr(chat_response, 'choices') and len(chat_response.choices) > 0:
            assistant_message = chat_response.choices[0].message
            return assistant_message.content.strip() if assistant_message.content.strip() else "NA"
        
        return "NA"
    except Exception as e:
        return f"Error while validating MOM: {str(e)}"


if __name__ == "__main__":
    # File paths (Replace with actual file paths)
    transcript_file = "/home/blooms/Desktop/transcript_file.docx"  # Provide actual file path
    agenda_file = "/home/blooms/Desktop/agenda_file.docx"
    extracted_mom_file = "/home/blooms/Desktop/extracted_mom_file.docx"

    # Check if files exist
    # print(f"Checking transcript file: {transcript_file}")
    # print(f"Checking agenda file: {agenda_file}")
    # print(f"Checking extracted MOM file: {extracted_mom_file}")

    # print(f"Does transcript file exist? {os.path.exists(transcript_file)}")
    # print(f"Does agenda file exist? {os.path.exists(agenda_file)}")
    # print(f"Does extracted MOM file exist? {os.path.exists(extracted_mom_file)}")

    # Read input files
    transcript = read_file(transcript_file)
    # print("Transcript content:", transcript)
    agenda = read_file(agenda_file)
    # print("Agenda content:", agenda)
    extracted_mom = read_file(extracted_mom_file)
    # print("Extracted MOM content:", extracted_mom)

    # Run verification
    verification_result = verify_mom(transcript, agenda, extracted_mom)
    print("Verification Result:\n", verification_result)

prompt = f"Given the following Meeting Transcript:\n{transcript}\nAnd the following Agenda:\n{agenda}\nVerify whether the following Extracted MOM is correct:\n{extracted_mom}\nHighlight any missing points or hallucinations. check any invalid content in extracted for {agenda} if any invalid content in {extracted_mom} then display in the result."
