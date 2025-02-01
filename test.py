import streamlit as st
from ultralytics import YOLO
import torch
import cv2
from dotenv import load_dotenv
import os
import time
from datetime import datetime
from twilio.rest import Client



load_dotenv() 
# Twilio credentials
TWILIO_PHONE_NUMBER = ""
TWILIO_ACCOUNT_SID = "" #enter your API key
TWILIO_AUTH_TOKEN = "" #add token
RECEIVER_PHONE_NUMBER ="" # Replace with the receiver's phone number



# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Load your trained YOLO model
trained_model_path = r'C:\Users\Rohan\Desktop\3.01wildlife\runs\detect\train\weights\best.pt'  # Replace with the actual path to your model

@st.cache_resource
def load_trained_model():
    return YOLO(trained_model_path)

model = load_trained_model()

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    st.write(f"Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    st.write("Running on CPU")

# Create a folder to save detected frames if it doesn’t exist
save_folder = 'detected_frames'
os.makedirs(save_folder, exist_ok=True)

# SMS function
def send_sms(animal_name, location):
    message = twilio_client.messages.create(
        body=f"{animal_name} detected at location: {location}",
        from_=TWILIO_PHONE_NUMBER,
        to=RECEIVER_PHONE_NUMBER
    )
    st.write(f"SMS sent: {message.sid}")

# Define frame saving cooldown
SAVE_COOLDOWN = 10  # Time in seconds
last_save_time = 0

# Streamlit UI
st.title("Real-Time Wildlife Detection with Custom YOLO Model and Webcam")
conf_threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.6)
start_feed = st.button("Start Detection")
stop_feed = st.button("Stop Detection")
frame_placeholder = st.empty()

# Start detection
if start_feed:
    cap = cv2.VideoCapture(0)
    running = True
    
    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image.")
            break

        # Convert frame to RGB for display
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results = model.predict(img_rgb, device=device, conf=conf_threshold)
        detected = False
        detected_animal = None
        
        # Draw bounding boxes
        for result in results:
            for box in result.boxes.cpu().numpy():
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                score = box.conf[0]

                # Draw bounding box and label if above threshold
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_rgb, f"{label} ({score:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                detected = True
                detected_animal = label

        # Save frame if an object was detected and cooldown has passed
        current_time = time.time()
        if detected and (current_time - last_save_time > SAVE_COOLDOWN):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_folder, f"detected_{detected_animal}_{timestamp}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            last_save_time = current_time
            st.write(f"Frame saved: {save_path}")

            # Send SMS
            location = "Latitude: 12.9716, Longitude: 77.5946"  # Replace with actual location data if available
            send_sms(detected_animal, location)

        # Display the annotated frame in Streamlit
        frame_placeholder.image(img_rgb, channels="RGB", use_column_width=True)

        # Stop the feed if "Stop Detection" is clicked
        if stop_feed:
            running = False

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()
