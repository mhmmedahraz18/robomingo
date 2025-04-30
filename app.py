# eventlet.monkey_patch() MUST be the first import that uses patched modules like time or threading.
import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, jsonify, make_response # Import make_response
from flask_cors import CORS
from flask_socketio import SocketIO, emit # start_background_task is a method in older versions (assuming your version)
import speech_recognition as sr
from rapidfuzz import process, fuzz  # Improved fuzzy matching
from googletrans import Translator  # Google Translate API
from gtts import gTTS
import os
import cv2  # OpenCV for person detection
import numpy as np  # NumPy for array manipulation
# import threading # Use eventlet.monkey_patch() and start_background_task instead
# import time # Use eventlet.sleep() instead
import io # Import BytesIO for in-memory audio handling
# Import the Response class from werkzeug to fix the TypeError
from werkzeug.wrappers import Response


import requests # For Gemini API calls
import json     # For handling JSON data
import logging # Use logging instead of print

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app) # Enable CORS for all origins and routes
# Initialize SocketIO - use eventlet's async mode
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Initialize Google Translator
translator = Translator()

# --- Gemini API Key Configuration ---
# IMPORTANT: Directly setting your API key here is less secure than using environment variables.
# This is done for simplicity based on your request, but use environment variables for production.
# Replace "YOUR_ACTUAL_GEMINI_KEY_HERE_PUT_IT_HERE" with your key if the hardcoded one below doesn't work.
# Your provided key: AIzaSyBVRHqvCCHcj5QC77A_511-B_JMsxzxD_U
GEMINI_API_KEY = "" # <-- Your key is now hardcoded here directly

# We still define a placeholder key, but only for the *check* to see if the key was replaced/set
# This is NO LONGER the fallback value from os.environ.get
PLACEHOLDER_KEY_FOR_CHECK = "MY_NEW_UNIQUE_ROBOMIGO_PLACEHOLDER" # <-- New placeholder string

# The warning now checks if the hardcoded/environment key is the placeholder
if not GEMINI_API_KEY or GEMINI_API_KEY == PLACEHOLDER_KEY_FOR_CHECK:
    logging.warning("GEMINI_API_KEY seems unset or is the placeholder. Gemini functionality may be limited or fail.")
    # Note: With the key hardcoded above, this warning should *not* appear if the key is present.


# Predefined responses (Keep your existing dictionary - full version assumed)
# Add any additional relevant questions and answers about the college here
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
    "who are the faculties of the computer science department": (
        "The faculty members of the Computer Science & Engineering Department at P.A. College of Engineering are:\n\n"
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
        "12. Mr. Jalaluddeen B M"
    ),
    # Add other navigation points here if they are tied to responses, otherwise they are only frontend video triggers
    "where is the principal's office": "You can find directions to the Principal's office using the navigation panel.",
    "show me directions to the canteen": "Please select the Canteen from the navigation panel to see the video directions.",
    # Removed "hi", "hello", "hey" - they will now be handled by Gemini
}


LANGUAGE_CODES = {"en": "en", "hi": "hi", "ml": "ml", "kn": "kn", "ta": "ta", "te": "te"}

def translate_text(text, dest_lang):
    """Safely translates text using Google Translate API with error handling."""
    if not text or not text.strip():
        return ""
    if dest_lang == "en": # No need to translate if destination is English
        return text
    try:
        translated_text = translator.translate(text, dest=dest_lang).text
        return translated_text if translated_text else text  # Fallback to original text
    except Exception as e:
        logging.error(f"Translation Error for text '{text[:50]}...': {e}") # Log first 50 chars
        # The gTTS timeout errors seem to be related to translation calls sometimes.
        # This suggests the network issue might affect translate.google.com generally,
        # which gTTS also uses.
        return text  # Fallback in case of API failure

def generate_content(api_key, text_input):
    """Generates content using the Gemini API."""
    # Now check against the new placeholder value
    if not api_key or api_key == PLACEHOLDER_KEY_FOR_CHECK:
         logging.warning("Gemini API key is not set or is placeholder. Cannot generate external content.")
         return "Gemini API key is not set. Cannot generate external content." # This message should now only appear if you modify the hardcoded key to the placeholder

    if not text_input or not text_input.strip():
        return "Please provide some input for the AI."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    # It's a good idea to add a system instruction or context to Gemini
    # to keep responses relevant to a campus chatbot.
    # Example: "As a helpful chatbot for P.A. College of Engineering, answer questions about the college or general topics concisely. If asked about sensitive topics, decline appropriately. Keep answers under 100 words if possible."
    data = {"contents": [{"parts": [{"text": text_input}]}]} # Simple input without specific context for now
    try:
        logging.info(f"Calling Gemini API with input: '{text_input[:100]}...'") # Log first 100 chars
        response = requests.post(url, headers=headers, json=data, timeout=15) # Added timeout
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        response_data = response.json()
        logging.info(f"Gemini API response received.") # Log response data structure might be too verbose
        # logging.debug(f"Gemini Response Data: {response_data}") # Use debug for full response

        # Safely access the generated text, handling potential missing keys
        candidates = response_data.get("candidates")
        if candidates and len(candidates) > 0:
            content = candidates[0].get("content")
            if content and content.get("parts") and len(content["parts"]) > 0:
                generated_content = content["parts"][0].get("text", "No content generated.")
                # Basic cleanup for potential markdown if not intended
                generated_content = generated_content.replace('**', '') # Remove bold markdown
                return generated_content.strip() # Return stripped content
            else:
                 logging.warning("Gemini API did not return expected content structure (parts). Response data keys: %s", response_data.keys())
                 return "Received unexpected response structure from AI."
        else:
            # Check for 'promptFeedback' if candidates are empty - indicates safety issues etc.
            feedback = response_data.get("promptFeedback")
            if feedback:
                 logging.warning("Gemini API prompt feedback: %s", feedback)
                 # Try to provide a user-friendly message based on feedback
                 if feedback.get("blockReason"):
                      return "I'm sorry, I cannot respond to that query due to safety concerns."
                 elif feedback.get("safetyRatings"):
                      # You could parse safetyRatings to be more specific
                       return "I'm sorry, I cannot provide a response that violates safety policies."

            logging.warning("Gemini API did not return any candidates. Response data keys: %s", response_data.keys())
            return "AI did not provide a response."

    except requests.exceptions.Timeout:
        logging.error("Gemini API request timed out.")
        return "The AI took too long to respond. Please try again."
    except requests.exceptions.RequestException as e:
        logging.error(f"Gemini API Request Error: {e}", exc_info=True)
        return f"Error communicating with AI: {e}" # Return the error for debugging
    except Exception as e:
        logging.error(f"Error processing Gemini response: {e}", exc_info=True)
        return "Error processing AI response."


def get_custom_chatbot_response(user_input, language="en"):
    """Checks for college info and then uses Gemini if not found."""
    if not user_input or not user_input.strip(): # Handle empty input
        return translate_text("Please say something.", language)

    # Translate input to English for matching against internal responses
    english_input = translate_text(user_input, "en")
    logging.info(f"Processing user input (English) for response: '{english_input}'")

    # Check if the English input is in our college info dictionary (or a close match)
    # Use score_cutoff correctly: it returns None if *no* item meets the cutoff
    # Lower threshold slightly for flexibility, require at least a basic match (e.g., 60 or 70)
    # The 'hi' match should now fall to Gemini.
    best_match = process.extractOne(english_input, responses.keys(), scorer=fuzz.ratio, score_cutoff=65) # Adjusted threshold

    logging.info(f"Result of process.extractOne: {best_match}") # Log the result

    if best_match: # Check if a match was found (i.e., best_match is not None)
        best_match_key, score, _ = best_match # Unpack the tuple (this is safe now)
        logging.info(f"Found internal match: '{best_match_key}' with score {score:.2f}")
        response = responses.get(best_match_key, "Sorry, I don't have that specific information.")
        # Translate the response back to the target language if needed
        if language != "en":
            response = translate_text(response, language)
        return response
    else:
        # If not found in college info, use Gemini
        logging.info(f"No high-confidence internal match (score < 65) found for '{english_input}'. Using Gemini.")
        # Add a prompt to Gemini to keep answers relevant to college/campus?
        # Example: prompt = f"As a chatbot for P.A. College of Engineering, {english_input}"
        # For now, let's just send the translated query
        gemini_response_en = generate_content(GEMINI_API_KEY, english_input)

        # Translate the Gemini response to the target language
        if language != "en":
            final_response = translate_text(gemini_response_en, language)
        else:
            final_response = gemini_response_en

        return final_response

def get_response(user_input, language):
    """Uses the custom chatbot (checks college info then Gemini) to generate a response."""
    # Check if conversation is active before processing chat messages
    global conversation_active
    if not conversation_active:
        logging.warning("Received message but conversation not active.")
        # Send a message back indicating the chat isn't started
        # Return a response with 400 status code to indicate to frontend that chat is not active
        response_text = translate_text("Please click 'Start Chat' to begin the conversation.", language)
        # Need app context for make_response when called from outside a request context (e.g., socket event)
        # But this function is only called from the /get_response route handler, which IS in a request context.
        # So no need for app context here.
        return make_response(jsonify({"response": response_text}), 400)

    return get_custom_chatbot_response(user_input, language)


# --- Person Detection Variables ---
person_detected = False
conversation_active = False
# chat_history = []  # Optional - keep if you want to use it for context (not currently used)
net = None
layer_names = None
classes = None
cap = None
detection_running = False # Flag to indicate if detection thread is active

def load_yolo():
    """Loads the YOLO model files."""
    global net, layer_names, classes
    try:
        # Use os.path.join for robust path creation
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(script_dir, "yolov2.weights")
        config_path = os.path.join(script_dir, "yolov2.cfg")
        names_path = os.path.join(script_dir, "coco.names")

        # Ensure files exist before trying to read
        if not os.path.exists(weights_path):
            logging.error(f"YOLO weights file not found at {weights_path}")
            return False
        if not os.path.exists(config_path):
             logging.error(f"YOLO config file not found at {config_path}")
             return False
        if not os.path.exists(names_path):
             logging.error(f"COCO names file not found at {names_path}")
             return False

        net = cv2.dnn.readNet(weights_path, config_path)
        layer_names = net.getUnconnectedOutLayersNames()
        with open(names_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        logging.info("YOLO model loaded successfully.")
        return True
    except Exception as e:
        logging.error(f"Error loading YOLO model: {e}", exc_info=True)
        net = None # Ensure net is None if loading fails
        return False

def start_person_detection():
    """Continuously runs person detection and emits SocketIO events."""
    global person_detected, conversation_active, cap, net, layer_names, classes, detection_running

    if net is None or layer_names is None or classes is None:
        logging.error("YOLO model not loaded. Cannot start person detection.")
        detection_running = False
        return

    detection_running = True
    logging.info("Person detection loop started.")
    previous_person_detected = False
    camera_index = -1 # Start with no camera found
    cap = None # Ensure cap is initially None

    # Attempt to open camera (try multiple indices)
    logging.info("Attempting to open camera...")
    # Increased the number of indices to try for better compatibility
    for i in range(10): # Try indices 0, 1, 2, ... 9
        cap = cv2.VideoCapture(i)
        if cap and cap.isOpened():
            camera_index = i
            logging.info(f"Camera opened successfully using index {camera_index}.")
            break
        # Release the camera if it was opened but not valid
        if cap:
            cap.release()
            cap = None
        eventlet.sleep(0.1) # Short delay before trying next index

    if camera_index == -1 or not cap or not cap.isOpened():
        logging.error("Error: Could not open any camera for person detection.")
        detection_running = False
        return # Exit the thread if no camera opens


    frame_read_errors = 0
    max_frame_read_errors = 30 # Increased attempts before stopping detection (allows for brief camera glitches)

    logging.info("Entering main person detection loop.")
    # Need app context for socketio.emit from a background task
    with app.app_context():
        while detection_running:
            # Use eventlet.sleep for cooperative yielding
            eventlet.sleep(0.05) # Process frames slightly faster, adjust as needed (e.g., 0.1)

            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Error: Could not read frame ({frame_read_errors + 1}). Attempting to re-read...")
                frame_read_errors += 1
                if frame_read_errors >= max_frame_read_errors:
                    logging.error(f"Too many consecutive frame read errors ({max_frame_read_errors}). Exiting person detection loop.")
                    detection_running = False # Stop the loop
                    break # Exit the while loop
                # No need to release and re-open on single read error, camera might recover
                eventlet.sleep(0.5) # Wait a bit before trying to read again
                continue # Skip processing for this frame
            else:
                frame_read_errors = 0 # Reset error count on successful read

            # Ensure frame is not None before processing (although ret should handle this)
            if frame is None:
                logging.warning("Received None frame from camera. Skipping processing.")
                eventlet.sleep(0.1)
                continue


            height, width, channels = frame.shape
            # Adjust blob size/scale if needed depending on YOLO version, but 416 is standard for Tiny YOLO
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            # Get only output layer names
            output_layers = net.getUnconnectedOutLayersNames()
            # Run forward pass
            outs = net.forward(output_layers)

            detected_person_in_frame = False
            # Optional: Collect bounding box data if needed later (commented out)
            # class_ids = []
            # confidences = []
            # boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    # Filter detections: must be a person and confidence must be above threshold
                    # Adjusted confidence threshold to 0.6 for fewer false positives
                    if confidence > 0.6 and classes[class_id] == "person":
                        # Optional: Collect bounding box data
                        # center_x = int(detection[0] * width)
                        # center_y = int(detection[1] * height)
                        # w = int(detection[2] * width)
                        # h = int(detection[3] * height)
                        # x = int(center_x - w / 2)
                        # y = int(center_y - h / 2)
                        # boxes.append([x, y, w, h])
                        # confidences.append(float(confidence))
                        # class_ids.append(class_id)
                        detected_person_in_frame = True
                        # Found a person with high confidence, no need to check other detections in this 'out'
                        break
                if detected_person_in_frame:
                    # Found a person in this frame, no need to check other 'outs'
                    break


            # --- State Transition Logic ---
            if detected_person_in_frame and not previous_person_detected:
                # Person just appeared
                person_detected = True
                conversation_active = False # Reset conversation state for a new user interaction
                logging.info("State Change: New person detected. Emitting person_detected and reset_chat.")
                # Use socketio.emit directly as we are in an eventlet thread WITHIN app_context
                # Provide a message for the frontend to display/speak
                socketio.emit('person_detected', {'message': translate_text('Hello! Please click "Start Chat" to begin.', 'en'), 'language': 'en'}) # Send initial message in English
                socketio.emit('reset_chat')  # Emit event to reset chat on frontend
                # The frontend's person_detected listener handles enabling the button and displaying/speaking the message.

            elif not detected_person_in_frame and previous_person_detected:
                # Person just left
                person_detected = False
                conversation_active = False # End conversation
                logging.info("State Change: Person left. Emitting person_left.")
                # Provide a farewell message for the frontend
                socketio.emit('person_left', {'message': translate_text('Goodbye! Come back soon.', 'en'), 'language': 'en'})
                # The frontend's person_left listener handles disabling the button, hiding input, and displaying/speaking the message.

            # --- Keep track of the previous state ---
            previous_person_detected = detected_person_in_frame


    # Cleanup when detection_running becomes False (loop exits)
    if cap and cap.isOpened():
        cap.release()
        logging.info("Camera released.")
    cv2.destroyAllWindows() # Close any OpenCV windows if any were created (unlikely here)
    logging.info("Person detection loop stopped.")


@socketio.on('connect')
def handle_connect():
    """Handles client connections via WebSocket."""
    logging.info('Client connected')
    global person_detected, conversation_active
    # Inform the newly connected client about the current state
    if person_detected:
        logging.info("Informing new client: Person currently detected.")
        # Send the "Ready to chat" message only if a person is currently detected
        # Frontend listener for person_detected should handle displaying this and enabling button
        emit('person_detected', {'message': translate_text('Ready to chat.', 'en'), 'language': 'en'})
        # If a conversation was active when the client connected (less common scenario but possible)
        if conversation_active:
             logging.info("Informing new client: Conversation active.")
             # Frontend listener for start_conversation should handle displaying this and showing input
             emit('start_conversation', {'message': translate_text('Conversation is ongoing.', 'en'), 'language': 'en'})
    else:
         # If no person is detected, inform the new client about the idle state
         logging.info("Informing new client: No person detected.")
         emit('bot_message', {'response': translate_text('Stand in front of the robot to begin.', 'en'), 'language': 'en'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handles client disconnections."""
    logging.info('Client disconnected')
    # The person_detected and conversation_active states are physical,
    # so they don't change just because a client disconnects.


@socketio.on('start_chat_request')
def handle_start_chat():
    """Handles the 'Start Chat' button request from the frontend."""
    global person_detected, conversation_active
    logging.info("Start chat request received.")
    # Only start conversation if a person is detected AND a conversation isn't already active
    if person_detected and not conversation_active:
        conversation_active = True
        logging.info("Start chat request accepted. Emitting start_conversation.")
        # Send a welcome message to the user in the default language (or detect language from frontend if sent)
        initial_message = translate_text("Hello! How can I help you today?", 'en')
        # Need app context for emit from handler not directly triggered by request
        # with app.app_context(): # Not needed for emit in this context
        emit('start_conversation', {'message': initial_message, 'language': 'en'}) # Emit initial bot message
        # The frontend's 'start_conversation' listener handles playing audio and showing input.
    elif conversation_active:
         logging.warning("Start chat request received but conversation is already active.")
         # Optionally emit a message back to the client indicating this, though the frontend state should prevent the button click.
         emit('bot_message', {'response': translate_text('Conversation is already active.', 'en'), 'language': 'en'})
    else:
         logging.warning("Start chat request received but no person detected.")
         # Optionally emit a message back
         emit('bot_message', {'response': translate_text('Please stand in front of me to start the chat.', 'en'), 'language': 'en'})


@app.route("/")
def index():
    """Renders the main HTML page."""
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def chat():
    """Handles text chat input and returns a response."""
    # This route is now guarded by the get_response function's conversation_active check
    data = request.json
    user_input = data.get("message", "")
    language = data.get("language", "en")
    logging.info(f"Received chat message: '{user_input}' (Lang: {language})")

    # get_response now checks conversation_active and returns a response object (potentially 400) or the response text
    response = get_response(user_input, language)

    # Check if get_response returned a Flask Response object (like the 400 error)
    # We use the actual Response class from werkzeug.wrappers
    if isinstance(response, Response):
        logging.info("get_response returned a Flask Response object (e.g., 400 error).")
        return response
    else:
        # Otherwise, it returned the response text, wrap it in JSON with 200 status
        logging.info("get_response returned a text response (200 OK).")
        return jsonify({"response": response})


@app.route("/voice_input", methods=["POST"])
def voice_input():
    """Captures voice input and returns the recognized text."""
    global conversation_active
    data = request.json
    language = data.get("language", "en")
    logging.info(f"Voice input request received (Lang: {language}).")

    # Check if conversation is active before starting listening
    if not conversation_active:
        logging.warning("Voice input requested but conversation not active.")
        response_text = translate_text("Please click 'Start Chat' to begin the conversation.", language)
        # Return a response with 400 status code to indicate to frontend that chat is not active
        return make_response(jsonify({"message": response_text}), 400)


    recognizer = sr.Recognizer()
    # Adjust recognizer settings as needed (e.g., energy threshold, dynamic_energy_threshold)
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 4000 # Example: Adjust if microphone is too sensitive/insensitive

    with sr.Microphone() as source:
        try:
            logging.info("Adjusting for ambient noise...")
            # Increased duration for better ambient noise adjustment
            recognizer.adjust_for_ambient_noise(source, duration=1.5)
            logging.info("Listening for voice input...")
            # Increased timeout and phrase limit
            audio = recognizer.listen(source, timeout=4, phrase_time_limit=10)

            logging.info("Processing voice input...")
            # Use language parameter for recognition
            # Fallback to 'en-IN' or other common English variants if 'en' is too broad
            recognition_language = LANGUAGE_CODES.get(language, "en")
            # Use Indian English as a common default if just 'en' is too general
            if recognition_language == 'en': recognition_language = 'en-IN'

            text = recognizer.recognize_google(audio, language=recognition_language)
            logging.info(f"Recognized text: '{text}'")
            return jsonify({"message": text})

        except sr.WaitTimeoutError:
            logging.warning("Voice input timed out: Didn't hear anything.")
            return jsonify({"message": translate_text("Timeout: Didn't hear anything.", language)})
        except sr.UnknownValueError:
            logging.warning("Google Speech Recognition could not understand audio.")
            return jsonify({"message": translate_text("Sorry, I didn't catch that. Could you please repeat?", language)})
        except sr.RequestError as e:
            logging.error(f"Could not request results from Google Speech Recognition service; {e}", exc_info=True)
            return jsonify({"message": translate_text("Speech recognition service is currently unavailable.", language)})
        except Exception as e:
             logging.error(f"An unexpected error occurred during voice input: {e}", exc_info=True)
             return jsonify({"message": translate_text("An error occurred during voice input.", language)})


@app.route("/voice_output", methods=["POST"])
def voice_output():
    """Converts text to speech and returns the audio data as a response."""
    data = request.json
    text = data.get("message", "")
    language = data.get("language", "en")

    if not text or not text.strip():
        logging.warning("Received empty text for voice output.")
        return jsonify({"error": "No text provided for voice output."}), 400 # Bad Request

    # Use BytesIO to avoid writing to a temporary file on disk
    # This avoids potential file permission or cleanup issues and the recursion depth error
    try:
        tts_lang = LANGUAGE_CODES.get(language, "en")
        logging.info(f"Generating voice output for language: {tts_lang} for text: '{text[:50]}...'") # Log first 50 chars
        tts = gTTS(text, lang=tts_lang)

        # Save the audio to a BytesIO object in memory
        audio_stream = io.BytesIO()
        tts.write_to_fp(audio_stream)
        audio_stream.seek(0) # Rewind to the beginning of the stream

        logging.info("Generated voice output into BytesIO stream.")

        # Create a Flask response from the BytesIO stream data
        response = make_response(audio_stream.read())
        response.headers['Content-Type'] = 'audio/mpeg'
        # You might add Content-Disposition if you wanted to force download, but not for playback
        # response.headers['Content-Disposition'] = 'inline; filename="voice_output.mp3"'


        logging.info("Returning audio data response from BytesIO")
        # Close the stream after reading
        audio_stream.close()
        return response

    except Exception as e:
        logging.error(f"Voice Output Error: {e}", exc_info=True) # Log traceback
        # Ensure stream is closed if created and not already closed
        if 'audio_stream' in locals() and not audio_stream.closed:
             audio_stream.close()
             logging.warning("Closed BytesIO stream after error.")

        return jsonify({"error": "Failed to generate voice output."}), 500 # Internal Server Error


# --- Main Execution ---
if __name__ == "__main__":
    # Load YOLO model first
    if load_yolo():
        # If YOLO loaded successfully, start the person detection thread
        logging.info("Starting person detection background task.")
        # Use socketio.start_background_task with eventlet's async_mode
        # The target function should be the one that runs the loop
        # We pass the app context to the background task if needed, although not strictly required for detection loop itself
        # with app.app_context(): # App context added inside start_person_detection
        socketio.start_background_task(target=start_person_detection)

        # Run the Flask app with SocketIO using eventlet server
        logging.info("Starting SocketIO server...")
        # Use debug=False in production
        # use_reloader=False is important when using background tasks or threads
        socketio.run(app, debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    else:
        logging.error("YOLO model loading failed. Cannot start the application.")
        # Optionally exit or run the app without detection features
        # If you want to run without detection if it fails, you'd remove the 'if load_yolo():' block
        # and maybe set a flag so detection-dependent features are disabled.
        # For this example, we exit if YOLO fails.
