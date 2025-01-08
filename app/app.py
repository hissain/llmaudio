import streamlit as st
import numpy as np
from transformers import pipeline
import sounddevice as sd
import soundfile as sf
from gtts import gTTS
import tempfile
import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM
import pygame
import re

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize pygame mixer for audio
pygame.mixer.init()

def play_beep():
    freq = 440  # A4 note
    duration = 0.2
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    beep = np.sin(2 * np.pi * freq * t)
    sd.play(beep, sample_rate)
    sd.wait()

def play_audio_file(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def is_goodbye(text):
    goodbye_patterns = ['good bye', 'bye', 'quit', 'exit', 'stop']
    return any(pattern in text.lower() for pattern in goodbye_patterns)

def get_model(llm_engine, llm_model_name=None):
    try:
        if llm_engine == "Gemini":
            api_key = st.session_state.user_api_key or os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("Gemini API Key is not set.")
            return ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=api_key,
                temperature=0.1
            )
        elif llm_engine == "Local Ollama" and llm_model_name:
            return OllamaLLM(model=llm_model_name)
        else:
            raise ValueError("Invalid LLM configuration")
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        raise

class VoiceAssistant:
    def __init__(self):
        self.asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")
        
    def record_audio(self, duration=5, sample_rate=16000):
        logger.info("Recording audio...")
        play_beep()
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
        sd.wait()
        logger.info("Audio recording complete.")
        return audio_data

    def process_audio(self, audio_data, sample_rate=16000):
        logger.info("Processing audio for transcription...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            sf.write(temp_audio.name, audio_data, sample_rate)
            transcription = self.asr(temp_audio.name)["text"]
            os.unlink(temp_audio.name)
            logger.info(f"Transcription: {transcription.strip()}")
            return transcription.strip()

    def generate_response(self, text):
        logger.info("Generating response audio...")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            tts = gTTS(text=text, lang='en')
            tts.save(temp_audio.name)
            logger.info("Response audio generated.")
            return temp_audio.name

def restart_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

def main():
    st.set_page_config(page_title="Voice Assistant", layout="wide")
    st.title("Voice Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "assistant" not in st.session_state:
        st.session_state.assistant = VoiceAssistant()
    if "active" not in st.session_state:
        st.session_state.active = False
    if "user_api_key" not in st.session_state:
        st.session_state.user_api_key = ""

    status = st.empty()

    st.sidebar.header("Settings")
    llm_engine = st.sidebar.radio("Select LLM Type", options=['Gemini', 'Local Ollama'])
    llm_model_name = None if llm_engine == 'Gemini' else st.sidebar.text_input("Ollama Model Name", value="llama3.2:latest")

    if llm_engine == 'Gemini':
        user_api_key = st.sidebar.text_input("Google Gemini API Key", type="password")
        st.session_state.user_api_key = user_api_key.strip()

    recording_duration = st.sidebar.slider("Recording Duration (seconds)", min_value=1, max_value=10, value=5)

    if st.sidebar.button("Restart App"):
        restart_app()

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    if st.button("Start Conversation", disabled=st.session_state.active):
        st.session_state.active = True
        logger.info("Conversation started.")
        try:
            while st.session_state.active:
                status.info("Listening...")
                audio_data = st.session_state.assistant.record_audio(duration=recording_duration)
                user_text = st.session_state.assistant.process_audio(audio_data)

                if is_goodbye(user_text):
                    final_response = "Goodbye! Have a great day!"
                    logger.info("User said goodbye. Ending conversation.")
                    audio_file = st.session_state.assistant.generate_response(final_response)
                    play_audio_file(audio_file)
                    os.unlink(audio_file)
                    st.session_state.active = False
                    st.rerun()
                    break

                status.info("Processing...")
                llm = get_model(llm_engine, llm_model_name)
                response = llm.invoke(f"Answer the query within few sentences, query: {user_text}")
                assistant_response = str(response.content if hasattr(response, 'content') else response)

                audio_file = st.session_state.assistant.generate_response(assistant_response)

                st.session_state.messages.append({"role": "user", "content": user_text})
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})

                logger.info(f"User: {user_text}")
                logger.info(f"Assistant: {assistant_response}")

                play_audio_file(audio_file)
                os.unlink(audio_file)

        except Exception as e:
            logger.error(f"Error: {e}")
            status.error(f"Error occurred: {str(e)}")
            st.session_state.active = False

if __name__ == "__main__":
    main()
