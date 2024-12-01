import cv2
import time
import streamlit as st
from ultralytics import YOLO
from draw_utils_live import plot_boxes_live, color_map_live
from threading import Thread

def live_detection(plot_boxes, model_path="best.pt", webcam_resolution=(640, 480)):
    frame_width, frame_height = webcam_resolution
    cap = cv2.VideoCapture(0)  # Changed to 0 for default webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO(model_path).to('cpu')
    frame_placeholder, object_description_placeholder = st.empty(), st.empty()

    # Store the state of detection in session state
    if "is_detecting" not in st.session_state:
        st.session_state.is_detecting = False

    # Start the live detection thread
    def detection_thread():
        while st.session_state.is_detecting:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break

            results = model(frame)
            frame, labels, descriptions = plot_boxes_live(results, frame, model, color_map_live)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            description_display = "<div class='title-box'>Detected Objects:</div>"
            for label, desc in zip(labels, descriptions):
                description_display += f"<div class='description-box'><b>{label}</b>: {desc}</div>"
            object_description_placeholder.markdown(description_display, unsafe_allow_html=True)

            time.sleep(0.1)

        cap.release()
        st.success("Live detection stopped.")

    # Control Panel
    st.sidebar.title("Control Panel")
    if st.sidebar.button("Start Live Detection"):
        st.session_state.is_detecting = True
        Thread(target=detection_thread).start()  # Start detection in a new thread

    if st.sidebar.button("Stop Live Detection"):
        st.session_state.is_detecting = False

# Call the function within Streamlit
if __name__ == "__main__":
    st.title("YOLO Live Webcam Detection")
    live_detection(plot_boxes_live)
