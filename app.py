from flask import Flask, render_template, request, jsonify, send_file, after_this_request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import speech_recognition as sr
from rapidfuzz import process, fuzz
from googletrans import Translator
from gtts import gTTS
import os
import tempfile
import cv2
import numpy as np
import threading
import time
import requests
import json
import eventlet

eventlet.monkey_patch()

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Initialize Google Translator
translator = Translator()

# Gemini API Key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Predefined responses
responses = {
    "tell me about your college": "P.A. College of Engineering, established in 1999 by Dr. P.A. Ibrahim Haji, is located in Mangalore, Karnataka. It is affiliated with VTU and approved by AICTE.",
    "where is the college located": "P.A. College of Engineering is located at Nadupadav, Montepadav Post, Kairangala, Mangalore - 574153, Karnataka, India.",
    "what courses does pace offer": "PACE offers undergraduate programs in AI, Biotechnology, Civil, Computer Science, Electronics, Mechanical, and Cyber Security.",
    # ... (other predefined responses)
}

LANGUAGE_CODES = {"en": "en", "hi": "hi", "ml": "ml", "kn": "kn", "ta": "ta", "te": "te"}

def translate_text(text, dest_lang):
    try:
        translated_text = translator.translate(text, dest=dest_lang).text
        return translated_text if translated_text else text
    except Exception as e:
        print(f"Translation Error: {e}")
        return text

def generate_content(api_key, text_input):
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
    if language != "en":
        english_input = translate_text(user_input, "en")
    else:
        english_input = user_input

    best_match, score, _ = process.extractOne(english_input, responses.keys(), scorer=fuzz.ratio)
    threshold = 80

    if score >= threshold:
        response = responses.get(best_match, "Sorry, I don't have that specific information.")
        if language != "en":
            response = translate_text(response, language)
        return response
    else:
        if language != "en":
            gemini_response_en = generate_content(GEMINI_API_KEY, english_input)
            final_response = translate_text(gemini_response_en, language)
        else:
            final_response = generate_content(GEMINI_API_KEY, english_input)
        return final_response

def get_response(user_input, language):
    return get_custom_chatbot_response(user_input, language)

# Person Detection Variables
person_detected = False
conversation_active = False
chat_history = []
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

    print(f"Successfully opened camera at index {camera_index}.")
    print("Person detection started.")
    previous_person_detected = False
    frame_read_errors = 0
    max_frame_read_errors = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame ({frame_read_errors + 1}). Trying to reconnect...")
            frame_read_errors += 1
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                print(f"Error: Could not reopen camera after error.")
                break
            if frame_read_errors >= max_frame_read_errors:
                print("Too many consecutive frame read errors. Exiting person detection.")
                break
            continue
        else:
            frame_read_errors = 0

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
            socketio.emit('reset_chat')
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
    data = request.json
    language = data.get("language", "en")
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=7)
            text = recognizer.recognize_google(audio, language=language)
            return jsonify({"message": text})
        except sr.UnknownValueError:
            return jsonify({"message": "Sorry, I didn't catch that. Please try again."})
        except sr.RequestError:
            return jsonify({"message": "Speech recognition service is unavailable."})

@app.route("/voice_output", methods=["POST"])
def voice_output():
    data = request.json
    text = data.get("message", "")
    language = data.get("language", "en")

    try:
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_filename = temp_audio.name
        temp_audio.close()

        tts = gTTS(text, lang=LANGUAGE_CODES.get(language, "en"))
        tts.save(temp_filename)

        @after_this_request
        def remove_file(response):
            try:
                os.remove(temp_filename)
            except Exception as e:
                print(f"Error deleting file: {e}")
            return response

        return send_file(temp_filename, as_attachment=True)
    except Exception as e:
        print(f"Voice Output Error: {e}")
        return jsonify({"error": "Failed to generate voice output."})

if __name__ == "__main__":
    load_yolo()
    detection_thread = threading.Thread(target=start_person_detection)
    detection_thread.daemon = True
    detection_thread.start()
    socketio.run(app, host="0.0.0.0", port=5000)
