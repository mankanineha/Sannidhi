"""
Sannidhi Product Profile & Onboarding Setup Wizard
===================================================
Manages senior citizen medical profiles, caregiver contacts, and
regional language preferences (English, Hindi, Marathi, etc.)
"""

import os
import json

PROFILE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_profile.json")

SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "stt_code": "en-IN", "tts_voice": "Samantha"},
    "hi": {"name": "Hindi (हिंदी)", "stt_code": "hi-IN", "tts_voice": "Lekha"},
    "mr": {"name": "Marathi (मराठी)", "stt_code": "mr-IN", "tts_voice": "Lekha"},
    "gu": {"name": "Gujarati (ગુજરાતી)", "stt_code": "gu-IN", "tts_voice": "Lekha"},
    "ta": {"name": "Tamil (தமிழ்)", "stt_code": "ta-IN", "tts_voice": "Veena"},
    "te": {"name": "Telugu (తెలుగు)", "stt_code": "te-IN", "tts_voice": "Veena"},
    "bn": {"name": "Bengali (বাংলা)", "stt_code": "bn-IN", "tts_voice": "Lekha"}
}

DEFAULT_PROFILE = {
    "senior": {
        "name": "Senior Resident",
        "age": 74,
        "medical_conditions": ["Parkinson's Disease", "Hypertension", "Epilepsy"],
        "home_address": "Living Room Zone",
        "language": "hi"  # Default Hindi for regional warmth
    },
    "caregiver": {
        "name": "Primary Caregiver",
        "relationship": "Daughter",
        "phone": "+919876543210",
        "whatsapp": "+919876543210"
    },
    "device_settings": {
        "room_placement": "Living Room Corner",
        "privacy_mode": True,  # Zero camera, zero wearable
        "sensitivity": "balanced" # 'balanced' or 'low' to eliminate atmospheric false alarms
    }
}


def load_profile() -> dict:
    if not os.path.exists(PROFILE_FILE):
        save_profile(DEFAULT_PROFILE)
        return DEFAULT_PROFILE
    try:
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Ensure language key exists
            if "language" not in data.get("senior", {}):
                data["senior"]["language"] = "hi"
            return data
    except Exception as e:
        print(f"[ProfileManager] ⚠️ Error loading profile: {e}. Using default.")
        return DEFAULT_PROFILE


def save_profile(profile_data: dict):
    try:
        with open(PROFILE_FILE, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2, ensure_ascii=False)
        print(f"[ProfileManager] ✅ User profile updated successfully!")
    except Exception as e:
        print(f"[ProfileManager] ❌ Failed to save profile: {e}")


def configure_interactive():
    print("=" * 65)
    print("  🌸 Sannidhi Senior Safety & Privacy System — Product Setup")
    print("  (Zero Cameras • Zero Wearables • Privacy First)")
    print("=" * 65)

    curr = load_profile()
    senior = curr.get("senior", DEFAULT_PROFILE["senior"])
    caregiver = curr.get("caregiver", DEFAULT_PROFILE["caregiver"])

    s_name = input(f"1. Senior Resident Name [{senior['name']}]: ").strip() or senior['name']
    try:
        s_age = int(input(f"2. Senior Resident Age [{senior['age']}]: ").strip() or senior['age'])
    except ValueError:
        s_age = senior['age']

    s_cond_str = input(f"3. Medical Conditions (e.g., Parkinson's, Epilepsy) [{', '.join(senior['medical_conditions'])}]: ").strip()
    s_conds = [c.strip() for c in s_cond_str.split(",")] if s_cond_str else senior['medical_conditions']

    print("\n4. Preferred Conversation Language:")
    for code, info in SUPPORTED_LANGUAGES.items():
        print(f"   [{code}] {info['name']}")
    lang_choice = input(f"Select language code [{senior.get('language', 'hi')}]: ").strip().lower()
    if lang_choice not in SUPPORTED_LANGUAGES:
        lang_choice = senior.get('language', 'hi')

    c_name = input(f"\n5. Caregiver Name [{caregiver['name']}]: ").strip() or caregiver['name']
    c_rel  = input(f"6. Relationship (e.g. Daughter/Granddaughter) [{caregiver['relationship']}]: ").strip() or caregiver['relationship']
    c_num  = input(f"7. Caregiver WhatsApp Number (+91...) [{caregiver['whatsapp']}]: ").strip() or caregiver['whatsapp']

    new_profile = {
        "senior": {
            "name": s_name,
            "age": s_age,
            "medical_conditions": s_conds,
            "home_address": senior.get("home_address", "Living Room"),
            "language": lang_choice
        },
        "caregiver": {
            "name": c_name,
            "relationship": c_rel,
            "phone": c_num,
            "whatsapp": c_num
        },
        "device_settings": DEFAULT_PROFILE["device_settings"]
    }

    save_profile(new_profile)
    print("\n" + "=" * 65)
    print("✅ Setup complete! Device is ready for unobtrusive home protection.")
    print("=" * 65)


if __name__ == "__main__":
    configure_interactive()
