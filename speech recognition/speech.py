import streamlit as st
import speech_recognition as sr
import os
import datetime

# Supported APIs
RECOGNITION_APIS = ["Google", "Sphinx (Only English)"]
LANGUAGES = {
    "English (US)": "en-US",
    "Français": "fr-FR",
    "العربية": "ar-DZ",
    "Español": "es-ES",
    "Deutsch": "de-DE"
}

def transcribe_speech(api_choice, language):
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        st.info("Listening... Speak now.")
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=20)
        except sr.WaitTimeoutError:
            return "No speech detected. Please try again."
    
    try:
        if api_choice == "Google":
            return recognizer.recognize_google(audio, language=language)
        elif api_choice == "Sphinx":
            return recognizer.recognize_sphinx(audio, language=language)
        else:
            return "Unsupported API selected."
    except sr.UnknownValueError:
        return "Speech was unintelligible. Please try again."
    except sr.RequestError as e:
        return f"API request error: {e}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def save_transcription(text):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transcription_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return filename

def main():
    st.title("🎙️ Speech Recognition App")
    st.markdown("Select a speech recognition API and speak to transcribe.")

    # API and language selection
    api_choice = st.selectbox("Choose Speech Recognition API", RECOGNITION_APIS)
    language_choice = st.selectbox("Choose Language", list(LANGUAGES.keys()))
    language_code = LANGUAGES[language_choice]

    # Simulated pause/resume (not real due to Streamlit's statelessness)
    if "is_paused" not in st.session_state:
        st.session_state.is_paused = False

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎤 Start Recording"):
            st.session_state.is_paused = False
            text = transcribe_speech(api_choice, language_code)
            st.session_state.transcribed_text = text
            st.success("Transcription completed.")
            st.write("**Transcription:**", text)

    with col2:
        if st.button("⏸️ Pause Listening"):
            st.session_state.is_paused = True
            st.warning("Listening paused.")

    # Save transcription to file
    if "transcribed_text" in st.session_state and st.session_state.get("transcribed_text"):
        if st.button("💾 Save Transcription to File"):
            filename = save_transcription(st.session_state.transcribed_text)
            st.success(f"Transcription saved as {filename}")
            with open(filename, "r", encoding="utf-8") as f:
                st.download_button("Download Transcription", data=f, file_name=filename)

if __name__ == "__main__":
    main()
