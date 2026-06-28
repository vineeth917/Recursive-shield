import os
import requests
import wave
import math
import struct
from redteam import config

def generate_mock_wav(output_path: str, text: str = "This is a fallback audio instruction.", duration_sec: float = 3.0, sample_rate: int = 16000):
    """
    Generates a spoken WAV file using pyttsx3 offline TTS (SAPI5 on Windows).
    Falls back to a 440Hz sine wave beep if pyttsx3 fails.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Moderate speed
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        print(f"Generated spoken audio offline via pyttsx3: '{text}' -> {output_path}")
    except Exception as e:
        print(f"pyttsx3 offline synthesis failed: {e}. Generating beep fallback.")
        nchannels = 1
        sampwidth = 2  # 16-bit
        nframes = int(duration_sec * sample_rate)
        
        with wave.open(output_path, 'wb') as wav:
            wav.setparams((nchannels, sampwidth, sample_rate, nframes, 'NONE', 'not compressed'))
            for i in range(nframes):
                value = int(16000.0 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
                data = struct.pack('<h', value)
                wav.writeframesraw(data)
        print(f"Generated beep fallback audio at {output_path}")

def text_to_speech(text: str, output_path: str, voice_id: str = "male-qn-01"):
    """
    Calls MiniMax TTS API to render speech for text. Falls back to generating a mock WAV if
    API keys are missing or if the API call fails.
    """
    if not config.MINIMAX_API_KEY or config.MINIMAX_API_KEY == "your_minimax_api_key_here":
        print("MiniMax API key not configured. Falling back to offline pyttsx3 speech synthesis.")
        generate_mock_wav(output_path, text=text)
        return

    # Determine endpoint/GroupId
    group_id_query = f"?GroupId={config.MINIMAX_GROUP_ID}" if config.MINIMAX_GROUP_ID else ""
    url = f"https://api.minimax.io/v1/t2a_v2{group_id_query}"
    
    headers = {
        "Authorization": f"Bearer {config.MINIMAX_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": config.MINIMAX_MODEL_ID,
        "text": text,
        "stream": False,
        "voice_setting": {
            "voice_id": voice_id,
            "speed": 1.0,
            "vol": 1.0,
            "pitch": 0
        },
        "audio_setting": {
            "format": "wav",  # We request WAV directly to match pipeline
            "sample_rate": 16000,
            "channel": 1
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            # Check if response is actually JSON error message
            try:
                json_resp = response.json()
                if "base_resp" in json_resp or "status_code" in json_resp:
                    print(f"MiniMax API returned JSON error: {json_resp}. Falling back to offline synthesis.")
                    generate_mock_wav(output_path, text=text)
                    return
            except ValueError:
                pass
                
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"MiniMax TTS generated audio saved to {output_path}")
        else:
            print(f"MiniMax API Error {response.status_code}: {response.text}")
            print("Falling back to offline pyttsx3 speech synthesis.")
            generate_mock_wav(output_path, text=text)
    except Exception as e:
        print(f"Failed to call MiniMax TTS API: {e}. Falling back to offline pyttsx3 speech synthesis.")
        generate_mock_wav(output_path, text=text)
