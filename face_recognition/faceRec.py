import cv2
import streamlit as st
import numpy as np
import os
import pickle
from datetime import datetime
import pandas as pd
from PIL import Image

class OpenCVFaceRecognizer:
    def __init__(self):
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Initialize face recognizer
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()

        self.known_faces = []
        self.known_names = []
        self.known_ids = []
        self.next_id = 0

        self.database_file = "opencv_face_database.pkl"
        self.model_file = "face_recognition_model.yml"
        self.attendance_file = "attendance_log.csv"

        self.load_database()

    def load_database(self):
        """Load face database and trained model"""
        if os.path.exists(self.database_file):
            try:
                with open(self.database_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data.get('faces', [])
                    self.known_names = data.get('names', [])
                    self.known_ids = data.get('ids', [])
                    self.next_id = data.get('next_id', 0)

                # Load trained model if it exists
                if os.path.exists(self.model_file) and len(self.known_faces) > 0:
                    self.face_recognizer.read(self.model_file)
                    st.success(f"✅ Loaded {len(self.known_names)} faces from database")
                else:
                    st.info("📝 Database found but no trained model. Add faces to train.")
            except Exception as e:
                st.warning(f"⚠️ Could not load database: {str(e)}")
                self.reset_database()

    def save_database(self):
        """Save face database"""
        data = {
            'faces': self.known_faces,
            'names': self.known_names,
            'ids': self.known_ids,
            'next_id': self.next_id
        }
        with open(self.database_file, 'wb') as f:
            pickle.dump(data, f)

    def reset_database(self):
        """Reset all data"""
        self.known_faces = []
        self.known_names = []
        self.known_ids = []
        self.next_id = 0

    def extract_face_from_image(self, image):
        """Extract face from image"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) == 0:
            return None, "No face detected"

        if len(faces) > 1:
            return None, "Multiple faces detected. Please use image with single face"

        # Extract the face region
        (x, y, w, h) = faces[0]
        face_region = gray[y:y+h, x:x+w]

        # Resize to standard size
        face_region = cv2.resize(face_region, (200, 200))

        return face_region, "Face extracted successfully"

    def add_person(self, image, name):
        """Add a new person to the database"""
        try:
            # Convert PIL to numpy if needed
            if isinstance(image, Image.Image):
                image = np.array(image)

            # Extract face
            face_region, message = self.extract_face_from_image(image)

            if face_region is None:
                return False, f"❌ {message}"

            # Check if name already exists
            if name in self.known_names:
                # Add another sample for existing person
                person_id = self.known_ids[self.known_names.index(name)]
            else:
                # New person
                person_id = self.next_id
                self.next_id += 1
                self.known_names.append(name)
                self.known_ids.append(person_id)

            # Add face sample
            self.known_faces.append((face_region, person_id))

            # Retrain the model
            self.train_model()

            # Save database
            self.save_database()

            return True, f"✅ {name} added successfully! Total samples: {len([f for f in self.known_faces if f[1] == person_id])}"

        except Exception as e:
            return False, f"❌ Error: {str(e)}"

    def train_model(self):
        """Train the face recognition model"""
        if len(self.known_faces) < 2:
            return False, "Need at least 2 face samples to train"

        # Prepare training data
        faces = [face[0] for face in self.known_faces]
        labels = [face[1] for face in self.known_faces]

        # Train the recognizer
        self.face_recognizer.train(faces, np.array(labels))

        # Save the trained model
        self.face_recognizer.save(self.model_file)

        return True, "Model trained successfully"

    def recognize_face(self, face_region, confidence_threshold=50):
        """Recognize a face region"""
        if len(self.known_faces) == 0:
            return "Unknown", 0

        try:
            # Resize face to standard size
            face_resized = cv2.resize(face_region, (200, 200))

            # Predict
            label, confidence = self.face_recognizer.predict(face_resized)

            # Convert confidence to percentage (lower is better for LBPH)
            confidence_percent = max(0, 100 - confidence)

            # Check if confidence is above threshold
            if confidence <= confidence_threshold:
                # Find name by ID
                try:
                    name_index = self.known_ids.index(label)
                    name = self.known_names[name_index]
                    return name, confidence_percent
                except ValueError:
                    return "Unknown", confidence_percent
            else:
                return "Unknown", confidence_percent

        except Exception as e:
            return "Error", 0

    def detect_and_recognize_faces(self, frame, confidence_threshold=50):
        """Detect and recognize faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        results = []

        for (x, y, w, h) in faces:
            # Extract face region
            face_region = gray[y:y+h, x:x+w]

            # Recognize face
            name, confidence = self.recognize_face(face_region, confidence_threshold)

            results.append({
                'location': (x, y, w, h),
                'name': name,
                'confidence': confidence
            })

        return results

    def log_attendance(self, name):
        """Log attendance"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if os.path.exists(self.attendance_file):
            df = pd.read_csv(self.attendance_file)
        else:
            df = pd.DataFrame(columns=['Name', 'Timestamp', 'Date'])

        new_entry = pd.DataFrame({
            'Name': [name],
            'Timestamp': [timestamp],
            'Date': [datetime.now().strftime("%Y-%m-%d")]
        })

        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(self.attendance_file, index=False)

def run_opencv_recognition(recognizer, confidence_threshold, show_confidence, log_attendance, box_color):
    """Main recognition loop"""
    cap = cv2.VideoCapture(0)

    print("OpenCV Face Recognition Started!")
    print("Press 'q' to quit")
    print("Press 'a' to log attendance")

    logged_today = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Recognize faces
        results = recognizer.detect_and_recognize_faces(frame, confidence_threshold)

        # Draw results
        for result in results:
            x, y, w, h = result['location']
            name = result['name']
            confidence = result['confidence']

            # Draw rectangle
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # Prepare label
            if show_confidence:
                label = f"{name} ({confidence:.1f}%)"
            else:
                label = name

            # Draw label
            cv2.rectangle(frame, (x, y - 30), (x + w, y), box_color, -1)
            cv2.putText(frame, label, (x + 5, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display stats
        total_faces = len(results)
        recognized = sum(1 for r in results if r['name'] != "Unknown")

        cv2.putText(frame, f'Faces: {total_faces} | Recognized: {recognized}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('OpenCV Face Recognition', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a') and log_attendance:
            today = datetime.now().strftime("%Y-%m-%d")
            for result in results:
                name = result['name']
                if name != "Unknown" and f"{name}_{today}" not in logged_today:
                    recognizer.log_attendance(name)
                    logged_today.add(f"{name}_{today}")
                    print(f"✅ Logged: {name}")

    cap.release()
    cv2.destroyAllWindows()

def get_color_bgr(color_name):
    """Convert color name to BGR"""
    colors = {
        "Green": (0, 255, 0), "Red": (0, 0, 255), "Blue": (255, 0, 0),
        "Yellow": (0, 255, 255), "Purple": (255, 0, 255), "Cyan": (255, 255, 0),
        "White": (255, 255, 255), "Orange": (0, 165, 255)
    }
    return colors.get(color_name, (0, 255, 0))

def main():
    st.set_page_config(page_title="OpenCV Face Recognition", page_icon="👤", layout="wide")

    # Initialize recognizer
    if 'recognizer' not in st.session_state:
        st.session_state.recognizer = OpenCVFaceRecognizer()

    recognizer = st.session_state.recognizer

    # Sidebar
    st.sidebar.header("👥 Face Database")
    st.sidebar.write(f"**People**: {len(set(recognizer.known_names))}")
    st.sidebar.write(f"**Face samples**: {len(recognizer.known_faces)}")

    if recognizer.known_names:
        st.sidebar.write("**Registered:**")
        for name in set(recognizer.known_names):
            count = recognizer.known_names.count(name)
            st.sidebar.write(f"• {name} ({count} samples)")

    # Add person section
    st.sidebar.subheader("➕ Add Person")
    uploaded_file = st.sidebar.file_uploader("Upload face photo", type=['jpg', 'jpeg', 'png'])
    person_name = st.sidebar.text_input("Person's name")

    if st.sidebar.button("Add Person") and uploaded_file and person_name:
        image = Image.open(uploaded_file)
        success, message = recognizer.add_person(image, person_name)
        if success:
            st.sidebar.success(message)
            st.rerun()
        else:
            st.sidebar.error(message)

    # Settings
    st.sidebar.subheader("⚙️ Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 30, 80, 50, 5)
    show_confidence = st.sidebar.checkbox("Show Confidence", True)

    box_color_name = st.sidebar.selectbox("Box Color",
        ["Green", "Blue", "Purple", "Cyan", "Yellow", "Orange", "Red", "White"])
    box_color = get_color_bgr(box_color_name)

    # Attendance
    st.sidebar.subheader("📋 Attendance")
    log_attendance = st.sidebar.checkbox("Enable Attendance Logging")

    # Database management
    if st.sidebar.button("🗑️ Clear Database"):
        recognizer.reset_database()
        if os.path.exists(recognizer.database_file):
            os.remove(recognizer.database_file)
        if os.path.exists(recognizer.model_file):
            os.remove(recognizer.model_file)
        st.sidebar.success("Database cleared!")
        st.rerun()

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## 🚀 **OpenCV Face Recognition**

        This system uses **OpenCV's LBPH (Local Binary Pattern Histograms)** algorithm - no dlib required!

        ### 📋 **How to Use:**

        1. **Add People to Database:**
           - Upload clear face photos
           - Enter person's name
           - Add multiple photos per person for better accuracy

        2. **Start Recognition:**
           - Adjust confidence threshold (lower = more strict)
           - Click "Start Recognition"

        3. **During Recognition:**
           - Press **'q'** to quit
           - Press **'a'** to log attendance

        ### 💡 **Tips:**
        - Add 3-5 photos per person for best results
        - Use photos with different angles/lighting
        - Ensure good lighting during recognition
        - Face should be clearly visible and front-facing
        """)

    # Start recognition
    if st.button("🎥 Start Recognition", type="primary"):
        if len(recognizer.known_faces) == 0:
            st.warning("⚠️ Add at least one person first!")
        elif len(recognizer.known_faces) < 2:
            st.warning("⚠️ Add at least 2 face samples for better accuracy!")
        else:
            try:
                st.info("🎥 Recognition started! Check OpenCV window.")
                run_opencv_recognition(recognizer, confidence_threshold,
                                     show_confidence, log_attendance, box_color)
                st.success("✅ Recognition completed!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Show attendance log
    if log_attendance and os.path.exists(recognizer.attendance_file):
        st.markdown("## 📊 Recent Attendance")
        df = pd.read_csv(recognizer.attendance_file)
        if not df.empty:
            st.dataframe(df.tail(10))



if __name__ == "__main__":
    main()