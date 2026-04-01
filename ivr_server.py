import os
import uuid
import requests
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from twilio.twiml.voice_response import VoiceResponse
from faster_whisper import WhisperModel
from gtts import gTTS

# Import our MPR Engine
import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from inference import MunicipalInferenceEngine

app = FastAPI(title="Municipal Voice IVR")

# Ensure static directory exists for serving MP3s
STATIC_DIR = os.path.join(ROOT, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

print("Loading local Faster-Whisper Model (tiny)...")
# tiny model is extremely fast on CPU
stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")

print("Loading Municipal Inference Engine...")
mpr_engine = MunicipalInferenceEngine()
print("IVR System Ready!")

def generate_tts_mp3(text: str) -> str:
    """Generates an MP3 using Google TTS and returns the filename."""
    filename = f"response_{uuid.uuid4().hex[:8]}.mp3"
    filepath = os.path.join(STATIC_DIR, filename)
    tts = gTTS(text=text, lang="hi") # 'hi' works well for Hindi + English mix
    tts.save(filepath)
    return filename

@app.post("/voice")
async def incoming_call(request: Request):
    """
    Twilio Webhook: Answers the call, plays a greeting, and records the user.
    """
    resp = VoiceResponse()
    resp.say(
        "Welcome to the Municipal Grievance system. Please state your complaint after the beep, and press any key when finished.", 
        voice="alice", language="en-IN"
    )
    # Record the audio and send it to /recording_complete
    resp.record(
        action="/recording_complete", 
        method="POST", 
        play_beep=True, 
        max_length=30
    )
    resp.hangup()
    
    return Response(content=str(resp), media_type="text/xml")

@app.post("/recording_complete")
async def handle_recording(
    request: Request,
    RecordingUrl: str = Form(...),
):
    """
    Twilio Webhook: Receives the recording URL after the user finishes speaking.
    """
    print(f"\n[+] Processing New Recording: {RecordingUrl}")
    
    # 1. Download the audio file from Twilio
    audio_path = os.path.join(STATIC_DIR, f"temp_{uuid.uuid4().hex[:8]}.wav")
    try:
        r = requests.get(RecordingUrl)
        with open(audio_path, 'wb') as f:
            f.write(r.content)
    except Exception as e:
        print(f"Error downloading audio: {e}")
        resp = VoiceResponse()
        resp.say("Sorry, we could not process your audio.")
        return Response(content=str(resp), media_type="text/xml")
        
    # 2. Transcribe Audio -> Text (Faster Whisper)
    print("Transcribing audio...")
    segments, info = stt_model.transcribe(audio_path, beam_size=5)
    transcript = " ".join([segment.text for segment in segments]).strip()
    
    print(f" => Transcript: {transcript}")
    
    # Cleanup temp audio
    if os.path.exists(audio_path):
        os.remove(audio_path)
        
    # 3. Pass Text -> MPR Engine
    if not transcript:
        reply_text = "Sorry, I could not hear you clearly. Please try calling again."
    else:
        print("Routing via MPR Engine...")
        result = mpr_engine.process_complaint(transcript)
        
        # If it's a conversational response (like emergency or query)
        if "response" in result:
            reply_text = result["response"]
        else:
            # Standard complaint routing
            dept = result['department'].replace('_', ' ').title()
            reply_text = f"Thank you. Your complaint has been successfully registered with the {dept} department. The severity level is {result['severity']}."
            
    print(f" => System Reply: {reply_text}")
    
    # 4. Convert AI Reply -> Speech (TTS)
    mp3_filename = generate_tts_mp3(reply_text)
    
    # Base URL for playing the MP3 back (FastAPI grabs the ngrok/localtunnel host dynamically)
    base_url = str(request.base_url).rstrip("/")
    audio_url = f"{base_url}/static/{mp3_filename}"
    
    # 5. Tell Twilio to play the MP3 over the phone
    resp = VoiceResponse()
    resp.play(audio_url)
    resp.say("Thank you for contacting the municipality. Have a good day.")
    resp.hangup()
    
    return Response(content=str(resp), media_type="text/xml")

# Run via: uvicorn ivr_server:app --port 8000 --reload
