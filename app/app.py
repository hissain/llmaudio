import streamlit as st
import numpy as np
from transformers import pipeline
import sounddevice as sd
import soundfile as sf
from gtts import gTTS
import tempfile
import os
import queue
import logging
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_model(llm_engine, llm_model_name=None):
    try:
        if llm_engine == "Gemini" and os.getenv('GEMINI_API_KEY'):
            return ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=os.getenv('GEMINI_API_KEY'),
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
        logger.info("Initializing Voice Assistant...")
        try:
            self.asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")
        except Exception as e:
            logger.error(f"ASR initialization failed: {e}")
            raise
        
    def record_audio(self, duration=5, sample_rate=16000):
        logger.info("Recording audio...")
        try:
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32
            )
            sd.wait()
            return audio_data
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            raise

    def process_audio(self, audio_data, sample_rate=16000):
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                sf.write(temp_audio.name, audio_data, sample_rate)
                transcription = self.asr(temp_audio.name)["text"]
                os.unlink(temp_audio.name)
                return transcription.strip()
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise

    def generate_response(self, text):
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                tts = gTTS(text=text, lang='en')
                tts.save(temp_audio.name)
                return temp_audio.name
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise

def main():
    st.set_page_config(page_title="Voice Assistant")
    st.title("Voice Assistant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "assistant" not in st.session_state:
        try:
            st.session_state.assistant = VoiceAssistant()
        except Exception as e:
            st.error(f"Failed to initialize Voice Assistant: {e}")
            return
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
    if "temp_audio" not in st.session_state:
        st.session_state.temp_audio = None

    status = st.empty()
    
    st.sidebar.header("Settings")
    llm_engine = st.sidebar.radio("Select LLM Type", options=['Gemini', 'Local Ollama'])

    if llm_engine == 'Gemini':
        llm_model_name = None
        api_key = st.sidebar.text_input("Google Gemini API Key", type="password", value=os.getenv('GEMINI_API_KEY', ''))
        if api_key:
            os.environ['GEMINI_API_KEY'] = api_key
    else:
        llm_model_name = st.sidebar.text_input("Ollama Model Name", value="llama2")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "audio" in message and os.path.exists(message["audio"]):
                st.audio(message["audio"])

    if st.button("Start Recording", disabled=st.session_state.is_recording):
        try:
            status.info("Recording started... Speak now!")
            st.session_state.is_recording = True
            
            audio_data = st.session_state.assistant.record_audio()
            user_text = st.session_state.assistant.process_audio(audio_data)
            
            status.info("Processing response...")
            llm = get_model(llm_engine, llm_model_name)
            response = llm.invoke(user_text)
            assistant_response = str(response.content if hasattr(response, 'content') else response)
            
            audio_file = st.session_state.assistant.generate_response(assistant_response)
            
            if st.session_state.temp_audio and os.path.exists(st.session_state.temp_audio):
                os.unlink(st.session_state.temp_audio)
            
            st.session_state.messages.append({"role": "user", "content": user_text})
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_response,
                "audio": audio_file
            })
            
            st.session_state.temp_audio = audio_file
            st.session_state.is_recording = False
            status.success("Done! Click 'Start Recording' to speak again.")
            st.rerun()
            
        except Exception as e:
            logger.error(f"Error: {e}")
            status.error(f"Error occurred: {str(e)}")
            st.session_state.is_recording = False

if __name__ == "__main__":
    main()