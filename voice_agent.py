"""
Sannidhi Multilingual Conversational AI Voice Pipeline
======================================================
1. Active Microphone Listening with Voice Activity Detection (VAD)
2. Regional Language Support (Hindi, Marathi, Gujarati, Tamil, Telugu, English)
3. Empathetic LLM / Zero-shot Regional Intent Engine
4. Caregiver WhatsApp Alerting with Full Conversation Context
5. Conversation Logging for Health & Psychological Analysis
"""

import os
import sys
import time
import tempfile
import threading
import subprocess
import numpy as np

import sounddevice as sd
from scipy.io import wavfile

from profile_manager import load_profile, SUPPORTED_LANGUAGES
from caregiver_alert import send_caregiver_whatsapp
from conversation_logger import log_conversation

try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

# Regional Prompts & Responses
REGIONAL_PROMPTS = {
    "hi": {
        "greeting": "नमस्ते {name} जी, मुझे कमरे में अचानक हलचल महसूस हुई। क्या आप ठीक हैं, या मैं मदद के लिए आपकी बेटी को फोन करूं?",
        "ok_reply": "यह सुनकर बहुत खुशी हुई कि आप बिल्कुल ठीक हैं, {name} जी। मैं सामान्य निगरानी जारी रख रही हूं।",
        "emergency_reply": "कृपया शांत रहें और उठने की जल्दी न करें, {name} जी। मैंने आपके परिवार को तुरंत व्हाट्सएप अलर्ट भेज दिया है।",
        "unresponsive_reply": "मुझे आपकी आवाज स्पष्ट सुनाई नहीं दी, {name} जी। आपकी सुरक्षा के लिए मैं परिवार को तुरंत सूचित कर रही हूं।"
    },
    "mr": {
        "greeting": "नमस्कार {name}, मला खोलीत अचानक पडल्यासारखी हालचाल जाणवली. आपण ठीक आहात ना, की मी मदतीसाठी फोन करू?",
        "ok_reply": "तुम्ही ठीक आहात हे ऐकून खूप आनंद झाला, {name}. मी लक्ष ठेवून आहे, काळजी घ्या!",
        "emergency_reply": "कृपया शांत राहा आणि उठायची घाई करू नका, {name}. मी लगेच तुमच्या कुटुंबाला संदेश पाठवला आहे, मदत येत आहे.",
        "unresponsive_reply": "मला आपला आवाज स्पष्ट आला नाही, {name}. तुमच्या सुरक्षेसाठी मी लगेच कुटुंबाला कळवत आहे."
    },
    "en": {
        "greeting": "Hello {name}, I noticed a sudden fall movement in the room. Are you alright or do you need assistance?",
        "ok_reply": "I am so glad to hear you are feeling okay, {name}. I will resume normal monitoring.",
        "emergency_reply": "Please lie still and do not try to move quickly, {name}. I am notifying your caregiver right now for assistance.",
        "unresponsive_reply": "I didn't hear a clear response, {name}. To be safe, I am notifying your family immediately."
    },
    "gu": {
        "greeting": "નમસ્તે {name}, મને રૂમમાં અચાનક હિલચાલ જણાઈ. તમે ઠીક છો ને, કે હું મદદ માટે ફોન કરું?",
        "ok_reply": "તમે ઠીક છો એ જાણીને આનંદ થયો, {name}. હું ધ્યાન રાખી રહી છું.",
        "emergency_reply": "મહેરબાની કરીને શાંત રહો, {name}. મેં તમારા પરિવારને મદદ માટે સંદેશ મોકલી દીધો છે.",
        "unresponsive_reply": "મને તમારો અવાજ સ્પષ્ટ ના સંભળાયો, {name}. સુરક્ષા માટે હું પરિવારને જાણ કરું છું."
    }
}


class VoiceAgent:
    def __init__(self):
        self.is_active = False
        self.current_status = "Idle"
        self.last_result = None

    def speak(self, text: str, lang_code: str = "hi"):
        """Multilingual TTS using macOS native speech engine or fallback."""
        print(f"[VoiceAgent 🔊 ({lang_code})] '{text}'")
        if sys.platform == "darwin":
            voice = SUPPORTED_LANGUAGES.get(lang_code, {}).get("tts_voice", "Samantha")
            subprocess.run(["say", "-v", voice, text], check=False)
        else:
            print(f"[VoiceAgent] (TTS): {text}")

    def active_listen(self, stt_lang_code: str = "hi-IN", sample_rate: int = 16000, max_duration_sec: float = 12.0) -> tuple[str, str]:
        """
        Actively listens to the microphone with Voice Activity Detection (VAD).
        Waits for speech and naturally stops 1.5s after speech ends.
        """
        if not HAS_SR:
            return "", "SpeechRecognition module not available"

        print(f"[VoiceAgent 🎙 ({stt_lang_code})] Active listening started...")
        self.current_status = f"🎙 Listening ({stt_lang_code})..."

        chunk_dur = 0.2
        chunk_samples = int(sample_rate * chunk_dur)
        silence_threshold = 300

        recorded_chunks = []
        speech_started = False
        silence_chunks = 0
        max_silence_chunks = int(1.5 / chunk_dur)
        max_total_chunks = int(max_duration_sec / chunk_dur)
        total_chunks = 0

        def callback(indata, frames, time_info, status):
            nonlocal speech_started, silence_chunks, total_chunks
            total_chunks += 1
            audio_int16 = (indata * 32767).astype(np.int16)
            energy = np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2))

            if energy > silence_threshold:
                speech_started = True
                silence_chunks = 0
            elif speech_started:
                silence_chunks += 1

            recorded_chunks.append(audio_int16.copy())

        try:
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32',
                                blocksize=chunk_samples, callback=callback):
                while total_chunks < max_total_chunks:
                    sd.sleep(100)
                    if speech_started and silence_chunks >= max_silence_chunks:
                        print("[VoiceAgent] Speech finished (silence detected).")
                        break
        except Exception as e:
            print(f"[VoiceAgent] Mic recording error: {e}")
            return "", f"mic_error ({e})"

        if not recorded_chunks or not speech_started:
            return "", "timeout_no_speech"

        full_audio = np.concatenate(recorded_chunks, axis=0)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            wavfile.write(tmp_path, sample_rate, full_audio)

        recognizer = sr.Recognizer()
        transcript = ""
        try:
            with sr.AudioFile(tmp_path) as source:
                audio_data = recognizer.record(source)
                self.current_status = "Analyzing spoken words..."
                transcript = recognizer.recognize_google(audio_data, language=stt_lang_code).lower()
                print(f"[VoiceAgent 🗣] Recognized ({stt_lang_code}): '{transcript}'")
        except sr.UnknownValueError:
            print("[VoiceAgent] Audio detected but speech was unclear.")
            transcript = "unclear audio"
        except Exception as e:
            print(f"[VoiceAgent] STT error: {e}")
            transcript = "stt_error"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return transcript, "success"

    def analyze_intent_agentic(self, transcript: str, status: str, senior_name: str, lang_code: str) -> dict:
        """
        Multilingual Agentic Reasoning Engine.
        Evaluates senior speech in their native language and returns emergency status & responses.
        """
        regional = REGIONAL_PROMPTS.get(lang_code, REGIONAL_PROMPTS["en"])

        # 1. LLM reasoning if Gemini Key available
        if GEMINI_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = (
                    f"You are Sannidhi, an empathetic AI caregiver assistant for an elderly resident named {senior_name}.\n"
                    f"A Wi-Fi radar detected a sudden body fall. Language: {lang_code}.\n"
                    f"Senior's Spoken Response: '{transcript}' (Status: {status})\n\n"
                    f"Return ONLY JSON:\n"
                    f"{{\n"
                    f'  "is_emergency": true/false,\n'
                    f'  "spoken_reply": "1-2 empathetic sentences in language {lang_code}",\n'
                    f'  "caregiver_summary": "concise summary of senior response"\n'
                    f"}}\n"
                )
                res = model.generate_content(prompt)
                import json
                text = res.text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                return json.loads(text)
            except Exception as e:
                print(f"[VoiceAgent] Gemini API fallback to local intent engine: {e}")

        # 2. Local Multilingual Intent Parser
        t = transcript.lower()

        fine_keywords = [
            "fine", "okay", "ok", "good", "no problem", "didn't fall",
            "ठीक", "ठीक हूं", "कोई बात नहीं", "सब ठीक", "अच्छा हूँ",
            "बरे आहे", "काही नाही", "मी ठीक आहे", "સાંજૂ છું", "ઠીક છું"
        ]
        emergency_keywords = [
            "help", "pain", "hurt", "fell", "fallen", "slip", "slipped", "cant get up", "no",
            "मदद", "दर्द", "चोट", "गिर गया", "गिर गई", "बचाओ", "उठ नहीं", "दर्द हो रहा है",
            "मदत", "दुखत आहे", "पडलो", "पडले", "मदत करा", "મદદ", "દુખાવો"
        ]

        is_fine = any(w in t for w in fine_keywords) and not any(w in t for w in emergency_keywords)
        is_help = any(w in t for w in emergency_keywords) or status in ["timeout_no_speech", "mic_error"]

        if is_fine:
            return {
                "is_emergency": False,
                "spoken_reply": regional["ok_reply"].format(name=senior_name),
                "caregiver_summary": f"Senior confirmed they are fine. Voice: '{transcript}'"
            }
        elif is_help or t:
            return {
                "is_emergency": True,
                "spoken_reply": regional["emergency_reply"].format(name=senior_name),
                "caregiver_summary": f"Senior reported distress/injury. Voice: '{transcript}'"
            }
        else:
            return {
                "is_emergency": True,
                "spoken_reply": regional["unresponsive_reply"].format(name=senior_name),
                "caregiver_summary": f"Senior was unresponsive to voice query. (Status: {status})"
            }

    def trigger_confirmation_flow(self, metrics: dict = None, on_complete_callback=None):
        if self.is_active:
            print("[VoiceAgent] Flow already running, skipping duplicate trigger.")
            return

        metrics = metrics or {}
        profile = load_profile()
        senior  = profile.get("senior", {})
        s_name  = senior.get("name", "Senior Resident")
        lang    = senior.get("language", "hi")
        stt_code = SUPPORTED_LANGUAGES.get(lang, {}).get("stt_code", "hi-IN")
        regional = REGIONAL_PROMPTS.get(lang, REGIONAL_PROMPTS["en"])

        def flow():
            self.is_active = True
            self.current_status = f"🔊 Speaking to {s_name} ({lang})..."

            # 1. Natural Regional Initial Inquiry
            greeting_msg = regional["greeting"].format(name=s_name)
            self.speak(greeting_msg, lang_code=lang)

            # 2. Active Listen with Silence Detection (VAD) in regional STT code
            transcript, listen_status = self.active_listen(stt_lang_code=stt_code, max_duration_sec=12.0)
            self.last_result = transcript or listen_status

            # 3. Agentic Intent Reasoning
            self.current_status = "🧠 Agentic AI evaluating response..."
            reasoning = self.analyze_intent_agentic(transcript, listen_status, s_name, lang)

            is_emergency = reasoning["is_emergency"]
            spoken_reply = reasoning["spoken_reply"]
            summary      = reasoning["caregiver_summary"]

            # 4. Spoken Feedback to Senior
            self.current_status = f"🔊 Responding to resident..."
            self.speak(spoken_reply, lang_code=lang)

            # 5. Log for Health & Analytics
            log_conversation(
                senior_name=s_name,
                language=lang,
                transcript=transcript,
                ai_response=spoken_reply,
                is_emergency=is_emergency,
                summary=summary
            )

            # 6. Caregiver WhatsApp Alerting
            if is_emergency:
                self.current_status = "🚨 EMERGENCY — WHATSAPP DISPATCHED"
                print(f"[VoiceAgent 🚨] Emergency confirmed! Dispatching WhatsApp alert...")
                send_caregiver_whatsapp(
                    event_type="EMERGENCY ALERT",
                    voice_outcome=f"• Senior Voice: '{transcript or 'No voice response'}'\n• Language: {lang}\n• AI Analysis: {summary}",
                    metrics=metrics
                )
            else:
                self.current_status = "✅ Senior Confirmed OK"

            self.is_active = False
            if on_complete_callback:
                on_complete_callback(self.current_status)

        t = threading.Thread(target=flow, daemon=True, name="voice_confirmation")
        t.start()


# Global instance
voice_agent = VoiceAgent()
