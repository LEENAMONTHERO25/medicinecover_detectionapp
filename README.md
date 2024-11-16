# Medicine Cover Detection App

This application helps users to capture an image of a medicine cover (e.g., tablets, creams, etc.), extract the text from it, and provide detailed information about the medicine such as its name, usage, dosage, and side effects in a selected language. The app uses Google Cloud Vision API for Optical Character Recognition (OCR), OpenAI’s GPT for medicine-related information, and Google Translate API for translating the results to the desired language.

## Features

- Capture images of medicine covers and extract the text using Google Cloud Vision API and Tesseract OCR.
- Extract and display the medicine name, usage, dosage, and side effects.
- Translate the extracted information into the selected language.
- Uses OpenAI's GPT model to provide structured details about the medicine (usage, dosage, and side effects).
- Web-based interface using Flask.

## Requirements

To run this project, you need the following:

- Python 
- Google Cloud Vision API
- OpenAI API Key
- Tesseract-OCR
- Flask
- Google Cloud Translation API
- dotenv for managing environment variables

