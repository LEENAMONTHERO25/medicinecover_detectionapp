import os
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from io import BytesIO
from PIL import Image
from google.cloud import vision, translate_v2 as translate
from google.oauth2 import service_account
import openai
import re
import pytesseract
import time
from dotenv import load_dotenv  # Import dotenv

# Load environment variables
load_dotenv()
app = Flask(__name__)

# OpenAI and Google Cloud setup
openai.api_key = os.getenv("OPENAI_API_KEY")  # Get API key from .env
key_path = r"C:/Program Files/credentials/medicineapp-441202-4b011d3a2b75.json"
credentials = service_account.Credentials.from_service_account_file(key_path)
vision_client = vision.ImageAnnotatorClient(credentials=credentials)
translate_client = translate.Client(credentials=credentials)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to translate text
def translate_text(text, target_language):
    if not text:
        return ""
    result = translate_client.translate(text, target_language=target_language)
    return result['translatedText']

# Extract text with Google Vision
def extract_text_with_google_vision(image_data):
    image = vision.Image(content=image_data)
    response = vision_client.text_detection(image=image)
    if response.text_annotations:
        return response.text_annotations[0].description
    return ""

# Extract text with Tesseract OCR
def extract_text_with_tesseract(image):
    text = pytesseract.image_to_string(image, config="--psm 6")
    return text

# Process image and extract information
def extract_information(text):
    # Enhanced regex for wider language support and special characters
    name_match = re.search(r'([A-Za-z0-9\s\-一-龯ぁ-んァ-ヶء-ي\u0600-\u06ff\u0400-\u04FF]+)', text)
    return {"Name": name_match.group() if name_match else "Not found"}

# Function to get medicine details from ChatGPT
def get_medicine_details_with_chatgpt(medicine_name):
    prompt = f"Provide a structured response about the medicine '{medicine_name}' including usage, dosage, and side effects. Format the response as follows:\nUsage:\n[details]\n\nDosage:\n[details]\n\nSide effects:\n[details]"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant about medicine."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    except openai.error.RateLimitError:
        time.sleep(10)
        return get_medicine_details_with_chatgpt(medicine_name)
    except Exception as e:
        print(f"Error: {e}")
        return None

# Extract structured details from the ChatGPT response
def extract_details_from_chatgpt_response(response):
    if not response:
        return {"usage": "Not found", "dosage": "Not found", "side_effects": "Not found"}

    usage = re.search(r'Usage:\n(.*?)(?=\n\n|$)', response, re.DOTALL)
    dosage = re.search(r'Dosage:\n(.*?)(?=\n\n|$)', response, re.DOTALL)
    side_effects = re.search(r'Side effects:\n(.*?)(?=\n\n|$)', response, re.DOTALL)

    return {
        "usage": usage.group(1).strip() if usage else "Not found",
        "dosage": dosage.group(1).strip() if dosage else "Not found",
        "side_effects": side_effects.group(1).strip() if side_effects else "Not found"
    }

# Main image processing function
def process_image(image_data, target_language):
    # Decode the image
    image_data = base64.b64decode(image_data.split(',')[1])
    image = Image.open(BytesIO(image_data))
    
    # Extract text using Google Vision API (or fallback to Tesseract)
    text = extract_text_with_google_vision(image_data)
    if not text:
        text = extract_text_with_tesseract(image)
    
    # Extract the medicine name
    info = extract_information(text)
    medicine_name = info.get("Name", "Not found")

    # If the medicine name is found, get details from ChatGPT
    if medicine_name != "Not found":
        details = get_medicine_details_with_chatgpt(medicine_name)
        extracted_details = extract_details_from_chatgpt_response(details)
        
        # Translate each extracted detail
        translated_details = {
            "name": translate_text(medicine_name, target_language),
            "usage": translate_text(extracted_details["usage"], target_language),
            "dosage": translate_text(extracted_details["dosage"], target_language),
            "side_effects": translate_text(extracted_details["side_effects"], target_language)
        }
        return translated_details
    
    return {"name": "Not found", "usage": "N/A", "dosage": "N/A", "side_effects": "N/A"}

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    data = request.get_json()
    image_data = data['image']
    language = data['language']
    result = process_image(image_data, language)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
