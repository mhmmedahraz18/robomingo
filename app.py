from flask import Flask, render_template, request, jsonify, send_file

from flask_cors import CORS

import speech_recognition as sr

from rapidfuzz import process, fuzz  # Improved fuzzy matching

from googletrans import Translator  # Google Translate API

from gtts import gTTS

import os

import tempfile

from flask import after_this_request

from flask_socketio import SocketIO, emit  # Import for WebSockets

import cv2  # OpenCV for person detection

import numpy as np  # NumPy for array manipulation

import threading  # For running person detection in background

import time    # For delays in person detection

import requests # For Gemini API calls

import json    # For handling JSON data

import eventlet


app = Flask(__name__)

CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet') # Initialize SocketIO


# Initialize Google Translator
translator = Translator()

# Gemini API Key (Remember to set yours as an environment variable for production)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBVRHqvCCHcj5QC77A_511-B_JMsxzxD_U")

# Predefined responses (No need for external file)
responses = {
    "tell me about your college": "P.A. College of Engineering, established in 1999 by Dr. P.A. Ibrahim Haji, is located in Mangalore, Karnataka. It is affiliated with VTU and approved by AICTE.",
    "where is the college located": "P.A. College of Engineering is located at Nadupadav, Montepadav Post, Kairangala, Mangalore - 574153, Karnataka, India.",
    "what courses does pace offer": "PACE offers undergraduate programs in AI, Biotechnology, Civil, Computer Science, Electronics, Mechanical, and Cyber Security.",
    "does the college provide mba programs": "Yes, PACE offers an MBA program with specializations in Finance, Marketing, and Human Resources.",
    "is the college affiliated with vtu": "Yes, P.A. College of Engineering is affiliated with Visvesvaraya Technological University, Belgaum.",
    "is pace approved by aicte": "Yes, P.A. College of Engineering is approved by the All India Council for Technical Education (AICTE), New Delhi.",
    "is pace nba accredited": "Yes, P.A. College of Engineering was the first institution in Mangalore to receive NBA accreditation in 2009.",
    "does pace have research programs": "Yes, seven departments of the college are recognized as VTU research centers, offering Ph.D. and M.Sc. programs.",
    "what facilities are available in the college": "PACE provides a central library, auditorium, hostels, transportation, a gymnasium, Wi-Fi, and a sports & recreation area.",
    "is hostel accommodation available": "Yes, PACE provides hostel facilities for both male and female students with essential amenities.",
    "how can i reach pace from mangalore": "PACE is accessible via local transportation from Mangalore city, and the college provides bus services.",
    "what are the placement opportunities at pace": "PACE has a dedicated placement cell that assists students in securing jobs in top companies across various industries.",
    "does pace have industry collaborations": "Yes, the college collaborates with industry leaders for training, internships, and placement opportunities.",
    "how do i apply for admission": "You can apply online through the official PACE website or contact the admissions office for guidance.",
    "how can i contact the college": "You can reach PACE via phone at +91 824 2284701 or email at admission@pace.edu.in.",
    "who is the chairman of pace": "Dr. P.A. Ibrahim Haji is the chairman of the PACE Group.",
    "who is the principal of pace": "Dr. Ramis M. K. is the principal of P.A. College of Engineering.",
    "does pace conduct extracurricular activities": "Yes, PACE organizes cultural events, technical fests, sports activities, and student clubs.",
    "does the college have an alumni association": "Yes, PACE has an active alumni association that connects former students and supports networking opportunities.",
    "how can i join the alumni association": "You can register on the official college website or contact the alumni coordinator for details.",
    "does the college provide scholarships": "Yes, PACE offers scholarships based on merit and need. You can check the college website for details.",
    "what are the recent achievements of pace students": "PACE students have won VTU ranks, participated in the ACCS Design Challenge, and a student was selected as a lead for Google Developer Students Club.",
    "who is eligible for the phd program": "Candidates with a postgraduate degree in a relevant field and meeting VTU guidelines are eligible for the Ph.D. program at PACE.",
    "how can i contact the admission officer": "You can contact the Admission Officer at +91 9980022000 or email admission@pace.edu.in.",
    "who is the vice principal of pace": "The Vice Principal of P.A. College of Engineering is Dr. Sharmila Kumari M. She is also the Head of the Department of Computer Science & Engineering.",
    "who is the HOD of the computer science department": "Dr. Sharmila Kumari M is the Head of the Computer Science & Engineering Department at P.A. College of Engineering.",
    "who are the faculties of the computer science department": "The faculty members of the Computer Science & Engineering Department at P.A. College of Engineering are:\n\n"
                                                                "1. Dr. Sharmila Kumari M\n"
                                                                "2. Dr. Sayed Abdulhayan\n"
                                                                "3. Dr. Mohammed Hafeez M K\n"
                                                                "4. Dr. Saleem Malik\n"
                                                                "5. Mrs. Sakeena\n"
                                                                "6. Mr. Mohammed Saifudeen\n"
                                                                "7. Mr. Habeeb Ur Rehman P.B\n"
                                                                "8. Ms. Avvanhi\n"
                                                                "9. Mrs. Divya K K\n"
                                                                "10. Mrs. Ankitha Bekal\n"
                                                                "11. Mrs. Fathimath Raihan\n"
                                                                "12. Mr. Jalaluddeen B M",
}


# Language Mapping for gTTS (Keep your existing one)
LANGUAGE_CODES = {"en": "en", "hi": "hi", "ml": "ml", "kn": "kn", "ta": "ta", "te": "te"}

def translate_text(text, dest_lang):
    """Safely translates text using Google Translate API with error handling."""
    try:
        translated_text = translator.translate(text, dest=dest_lang).text
        return translated_text if translated_text else text  # Fallback to original text
    except Exception as e:
        print(f"Translation Error: {e}")
        return text  # Fallback in case of API failure

def generate_content(api_key, text_input):
    """Generates content using the Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": text_input}]}]}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        generated_content = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No content generated.")
        return generated_content
    except requests.exceptions.RequestException as e:
        print(f"Gemini API Error: {e}")
        return "Error in generating content."

def get_custom_chatbot_response(user_input, language="en"):
    """Checks for college info and then uses Gemini if not found."""
    if language != "en":
        english_input = translate_text(user_input, "en")
    else:
        english_input = user_input

    # Check if the English input is in our college info dictionary (or a close match)
    best_match, score, _ = process.extractOne(english_input, responses.keys(), scorer=fuzz.ratio)
    threshold = 80  # Adjust threshold as needed for matching accuracy

    if score >= threshold:
        response = responses.get(best_match, "Sorry, I don't have that specific information.")
        if language != "en":
            response = translate_text(response, language)
        return response
    else:
        # If not found in college info, use Gemini
        if language != "en":
            gemini_response_en = generate_content(GEMINI_API_KEY, english_input)
            final_response = translate_text(gemini_response_en, language)
        else:
            final_response = generate_content(GEMINI_API_KEY, english_input)
        return final_response

def get_response(user_input, language):
    """Uses the custom chatbot (checks college info then Gemini) to generate a response."""
    return get_custom_chatbot_response(user_input, language)



# --- Person Detection Variables ---

person_detected = False

conversation_active = False

chat_history = []  # To store the chat history (optional, for more advanced reset)

net = None

layer_names = None

classes = None

cap = None


def load_yolo():

    global net, layer_names, classes

    try:

        script_dir = os.path.dirname(os.path.abspath(__file__))

        weights_path = os.path.join(script_dir, "yolov2.weights")

        config_path = os.path.join(script_dir, "yolov2.cfg")

        names_path = os.path.join(script_dir, "coco.names")

        net = cv2.dnn.readNet(weights_path, config_path)

        layer_names = net.getUnconnectedOutLayersNames()

        with open(names_path, "r") as f:

            classes = [line.strip() for line in f.readlines()]

        print("YOLO model loaded successfully.")

    except Exception as e:

        print(f"Error loading YOLO model: {e}")


def start_person_detection():

    global person_detected, conversation_active, chat_history, cap, net, layer_names, classes

    camera_index = 0

    cap = cv2.VideoCapture(camera_index)


    if not cap.isOpened():

        print(f"Error: Could not open camera at index {camera_index}.")

        return

    else:

        print(f"Successfully opened camera at index {camera_index}.")


    print("Person detection started.")

    previous_person_detected = False  # Track the previous state

    frame_read_errors = 0

    max_frame_read_errors = 5  # Limit consecutive errors before giving up


    while True:

        ret, frame = cap.read()

        if not ret:

            print(f"Error: Could not read frame ({frame_read_errors + 1}). Trying to reconnect...")

            frame_read_errors += 1

            cap.release()

            time.sleep(1)  # Wait a bit before reconnecting

            cap = cv2.VideoCapture(camera_index)

            if not cap.isOpened():

                print(f"Error: Could not reopen camera after error.")

                break

            if frame_read_errors >= max_frame_read_errors:

                print("Too many consecutive frame read errors. Exiting person detection.")

                break

            continue  # Skip the rest of the loop and try to read again

        else:

            frame_read_errors = 0 # Reset error counter


        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(layer_names)


        detected_person = False

        for out in outs:

            for detection in out:

                scores = detection[5:]

                class_id = np.argmax(scores)

                confidence = scores[class_id]

                if confidence > 0.5 and classes[class_id] == "person":

                    detected_person = True

                    break

            if detected_person:

                break


        if detected_person and not previous_person_detected:

            person_detected = True

            conversation_active = False

            socketio.emit('person_detected', {'message': 'New person detected! Ready to chat.'})

            socketio.emit('reset_chat')  # Emit event to reset chat on frontend

            print("New person detected (backend) - Resetting chat.")

        elif not detected_person and previous_person_detected:

            person_detected = False

            conversation_active = False

            socketio.emit('person_left', {'message': 'Person left. Ending chat.'})

            print("Person left (backend).")


        previous_person_detected = detected_person

        time.sleep(0.1)


    if cap.isOpened():

        cap.release()

    print("Person detection stopped.")


@socketio.on('connect')

def handle_connect():

    print('Client connected')

    global person_detected, conversation_active

    if person_detected:

        emit('person_detected', {'message': 'Ready to chat.'})

        if conversation_active:

            emit('start_conversation', {'message': 'Conversation active.'})


@socketio.on('start_chat_request')

def handle_start_chat():

    global conversation_active

    conversation_active = True

    emit('start_conversation', {'message': 'Conversation started.'})

    print("Chat started (backend)")


@app.route("/")

def index():

    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    language = data.get("language", "en")
    response_text = get_response(user_input, language)
    return jsonify({"response": response_text})

@app.route("/voice_input", methods=["POST"])
def voice_input():
    """Captures voice input and returns the recognized text."""
    data = request.json
    language = data.get("language", "en")

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)  # Reduces background noise
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=7)  # Prevents infinite listening

            text = recognizer.recognize_google(audio, language=language)
            return jsonify({"message": text})

        except sr.UnknownValueError:
            return jsonify({"message": "Sorry, I didn't catch that. Please try again."})
        except sr.RequestError:
            return jsonify({"message": "Speech recognition service is unavailable."})

@app.route("/voice_output", methods=["POST"])
def voice_output():
    """Converts text to speech and returns the audio file in the selected language."""
    data = request.json
    text = data.get("message", "")
    language = data.get("language", "en")

    try:
        # Generate temporary audio file
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_filename = temp_audio.name
        temp_audio.close()  # Close file so gTTS can write to it

        # Convert text to speech using the selected language
        tts = gTTS(text, lang=LANGUAGE_CODES.get(language, "en"))
        tts.save(temp_filename)

        @after_this_request
        def remove_file(response):
            try:
                os.remove(temp_filename)  # Delete file AFTER response is sent
            except Exception as e:
                print(f"Error deleting file: {e}")
            return response

        return send_file(temp_filename, as_attachment=True)

    except Exception as e:
        print(f"Voice Output Error: {e}")
        return jsonify({"error": "Failed to generate voice output."})

# ... (rest of your Flask and SocketIO routes remain the same)


if __name__ == "__main__":

    load_yolo()

    detection_thread = threading.Thread(target=start_person_detection)

    detection_thread.daemon = True

    detection_thread.start()

    socketio.run(app, debug=True, host='0.0.0.0')