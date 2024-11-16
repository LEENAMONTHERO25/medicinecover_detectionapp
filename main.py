import os
from google.cloud import vision, translate_v2 as translate  # Added translate_v2 for translation
from google.oauth2 import service_account
import cv2
import pytesseract
import re
from PIL import Image
import openai
import time

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update with your correct path

# Explicitly set up the credentials for Google Vision and Translate API
key_path =  r"C:/Program Files/credentials/medicineapp-441202-4b011d3a2b75.json" # Update with your correct key path
credentials = service_account.Credentials.from_service_account_file(key_path)

# Create a Vision API client and Translate API client with the credentials
client = vision.ImageAnnotatorClient(credentials=credentials)
translate_client = translate.Client(credentials=credentials)

# OpenAI API setup
openai.api_key = os.getenv("OPENAI_API_KEY")  # Get OpenAI API key from .env

# Function to preprocess the image for OCR
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    processed_image = cv2.medianBlur(thresh, 3)
    cv2.imwrite("processed_image_debug.png", processed_image)
    return processed_image

# Extract text from the image using pytesseract
def extract_text(image_path):
    processed_image = preprocess_image(image_path)
    temp_image_path = "processed_image.png"
    cv2.imwrite(temp_image_path, processed_image)
    text = pytesseract.image_to_string(
        Image.open(temp_image_path),
        config="--psm 6"
    )
    cleaned_text = re.sub(r'[^\x20-\x7E]+', '', text)  # Clean non-ASCII characters
    return cleaned_text

# Translate extracted text to English
def translate_text(text, target_language='en'):
    if not text:
        return ""
    result = translate_client.translate(text, target_language=target_language)
    return result['translatedText']

# Function to extract information from OCR text
def extract_information(text):
    name_match = re.search(r'\b([A-Za-z0-9\s\-]+)\b', text)
    name = name_match.group() if name_match else "Not found"
    return {"Name": name}

# Extract text using Google Vision API
def extract_text_with_google_vision(image_path):
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    if response.text_annotations:
        return response.text_annotations[0].description
    else:
        return ""

# Retrieve medicine details using OpenAI's ChatGPT
def get_medicine_details_with_chatgpt(medicine_name):
    prompt = f"Provide details about the medicine {medicine_name}. Include information on its usage, dosage, and side effects."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant knowledgeable about medicine details."},
                {"role": "user", "content": prompt}
            ]
        )
        details = response['choices'][0]['message']['content']
        return details
    except openai.error.RateLimitError:
        time.sleep(10)
        return get_medicine_details_with_chatgpt(medicine_name)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Main function to run the program
def main(target_language='en'):
    cap = cv2.VideoCapture(0)
    print("Press 's' to capture an image.")
    while True:
        ret, frame = cap.read()
        cv2.imshow("Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            image_path = "captured_image.jpg"
            cv2.imwrite(image_path, frame)
            print("Image saved as", image_path)
            break
    cap.release()
    cv2.destroyAllWindows()

    # Step 1: Extract text using Google Vision API
    text = extract_text_with_google_vision(image_path)
    
    # Step 2: Translate text to English if necessary
    if text:
        print("Original Text Detected:\n", text)
        text = translate_text(text, target_language)  # Translate to English
        print("Translated Text (English):\n", text)
    else:
        print("Google Vision API did not detect any text. Using Tesseract for OCR.")
        text = extract_text(image_path)
    
    # Extract the name of the medicine from the OCR text
    info = extract_information(text)
    medicine_name = info.get("Name", "Not found")
    print("\nExtracted Information:")
    print(f"Name: {medicine_name}")
    
    # If a medicine name is found, use OpenAI API to fetch additional information
    if medicine_name != "Not found":
        details = get_medicine_details_with_chatgpt(medicine_name)
        print("\nAdditional Information (from ChatGPT):")
        print(details)
    else:
        print("Medicine name not found in the image.")

if __name__ == "__main__":
    main(target_language='en')