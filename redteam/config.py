import os
from dotenv import load_dotenv

# Load environment variables from .env file at the root of the project
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY")
MINIMAX_GROUP_ID = os.environ.get("MINIMAX_GROUP_ID")
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")

# Speech generation defaults. Environment overrides keep local/prod runs reproducible
# without hard-coding secrets or account-specific cloned voices.
MINIMAX_MODEL_ID = os.environ.get("MINIMAX_MODEL_ID", "speech-2.8-hd")
MINIMAX_VOICE_ID = os.environ.get("MINIMAX_VOICE_ID", "English_expressive_narrator")
VOYAGE_MODEL_ID = "voyage-3"       # Standard default, can override in .env
ANTIGRAVITY_AGENT_ID = "antigravity-preview-05-2026"
GEMINI_AUDIO_MODEL = os.environ.get("GEMINI_AUDIO_MODEL", "gemini-3.5-flash")
