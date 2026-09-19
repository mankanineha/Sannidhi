"""
Conversation Logger & Senior Mental Health / Wellbeing Tracker
================================================================
Logs natural voice interactions for long-term health analytics.
Enables early detection of loneliness, anxiety, or cognitive decline
on the Raspberry Pi 5 agentic layer.
"""

import os
import json
import time
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "conversation_history.jsonl")

os.makedirs(LOG_DIR, exist_ok=True)


def log_conversation(senior_name: str,
                     language: str,
                     transcript: str,
                     ai_response: str,
                     is_emergency: bool,
                     summary: str):
    """Log a voice interaction session for health analytics."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "senior_name": senior_name,
        "language": language,
        "user_transcript": transcript,
        "ai_spoken_response": ai_response,
        "is_emergency": is_emergency,
        "summary": summary,
        "session_duration_sec": time.time()
    }

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[ConversationLogger] 📝 Saved session log to {LOG_FILE}")
    except Exception as e:
        print(f"[ConversationLogger] ❌ Failed to write log: {e}")


def get_conversation_history(limit: int = 50) -> list:
    """Retrieve recent conversation logs for analysis."""
    if not os.path.exists(LOG_FILE):
        return []
    logs = []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
        return logs[-limit:]
    except Exception as e:
        print(f"[ConversationLogger] Error reading logs: {e}")
        return []
