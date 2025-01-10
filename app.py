import streamlit as st
import cv2
import numpy as np
import tempfile
from tensorflow import keras

# Constants
SEQUENCE_LENGTH = 20
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
CLASSES_LIST=['Basketball','Biking','Diving','WalkingWithDog','PullUps','Swing','PushUps','HighJump','PlayingGuitar'] 

def frame_extraction(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)
    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_list.append(normalized_frame)
    video_reader.release()
    return frames_list

@st.cache_resource
def load_model():
    return keras.models.load_model("D:\OFFICIAL\SUCCESS Journey of Coding Life\Deep Learning\Project\human_activity-main\human_action_recognition_1.h5")

model = load_model()

st.title("Human Action Recognition")
st.write("Upload a video to predict the human action.")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded file as a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name

    # Extract frames
    st.write("Extracting frames...")
    frames = frame_extraction(temp_video_path)

    if len(frames) == SEQUENCE_LENGTH:
        features = np.asarray([frames])
        st.write("Predicting action...")
        predicted_label_probabilities = model.predict(features)[0]
        predicted_label = np.argmax(predicted_label_probabilities)
        st.write(f"Predicted Action: **{CLASSES_LIST[predicted_label]}**")
    else:
        st.error("Unable to extract the required number of frames. Please upload a longer video.")