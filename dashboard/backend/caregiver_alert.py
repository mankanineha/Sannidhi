"""
Caregiver Alerting System via WhatsApp (Twilio API)
===================================================
Sends urgent WhatsApp notification to caregiver(s) when a fall is detected
and confirmed (or when senior citizen is unresponsive).
Dynamically pulls resident profile & caregiver details from profile_manager.
"""

import os
import time
from datetime import datetime
from dotenv import load_dotenv

from profile_manager import load_profile

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM        = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")


def format_whatsapp_message(event_type: str, voice_outcome: str, metrics: dict) -> str:
    """Format a clear, urgent, and detailed WhatsApp message for caregiver."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    profile = load_profile()
    senior = profile.get("senior", {})
    caregiver = profile.get("caregiver", {})

    fn_p = metrics.get("fallnet_prob", 0.0)
    sn_p = metrics.get("spectranet_prob", 0.0)
    energy = metrics.get("motion_energy", 0.0)

    if "CONFIRMED BY USER" in event_type or "EMERGENCY" in event_type.upper():
        header = "🚨 *CRITICAL EMERGENCY: FALL DETECTED & ASSISTANCE NEEDED*"
        severity = "HIGH - Immediate Assistance Required"
    else:
        header = "🚨 *URGENT ALERT: UNRESPONSIVE FALL DETECTED*"
        severity = "HIGH - Senior Citizen Unresponsive to Voice Query"

    res_name = senior.get('preferred_name') or senior.get('name', '')
    if not res_name or res_name.lower() in ["senior resident", "resident", "suman", "suman pagare"]:
        res_name = "Senior Citizen ji"

    diagnostic_lines = [
        f"  • FallGuardNet model probability: {fn_p*100:.1f}%",
    ]
    if sn_p is not None:
        diagnostic_lines.append(f"  • SpectraNet model probability: {sn_p*100:.1f}%")
    diagnostic_lines.append(f"  • CSI energy change: {energy:.3f}")

    msg = (
        f"{header}\n\n"
        f"📅 *Time*: {now_str}\n"
        f"👤 *Resident*: {res_name} (Age {senior.get('age', 70)})\n"
        f"📍 *Location*: {senior.get('home_address', 'Home Residence')}\n"
        f"⚕️ *Medical Conditions*: {', '.join(senior.get('medical_conditions', []))}\n"
        f"👥 *Contact*: {caregiver.get('name', 'Caregiver')} ({caregiver.get('relationship', 'Family')})\n\n"
        f"🎙 *Voice Interaction Log*:\n{voice_outcome}\n\n"
        f"⚠️ *Severity*: {severity}\n\n"
        f"📊 *Detection Evidence*:\n"
        + "\n".join(diagnostic_lines)
        + "\n\n"
        f"👉 *Action Required*: Please check on {res_name} immediately or visit!"
    )
    return msg


def send_caregiver_whatsapp(event_type: str, voice_outcome: str, metrics: dict) -> bool:
    """Send WhatsApp alert via Twilio REST API."""
    profile = load_profile()
    caregiver = profile.get("caregiver", {})

    target_num = os.getenv("CAREGIVER_WHATSAPP_TO") or caregiver.get("whatsapp") or caregiver.get("phone", "")

    msg_body = format_whatsapp_message(event_type, voice_outcome, metrics)

    print("\n" + "═"*60)
    print("📱 [CaregiverAlert] OUTGOING WHATSAPP ALERT:")
    print("═"*60)
    print(msg_body)
    print("═"*60 + "\n")

    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not target_num:
        print("[CaregiverAlert] 💡 Twilio credentials or caregiver phone not set.")
        print("[CaregiverAlert] Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and CAREGIVER_WHATSAPP_TO in .env file or user_profile.json")
        return False

    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        to_num = target_num if target_num.startswith("whatsapp:") else f"whatsapp:{target_num}"
        from_num = TWILIO_FROM if TWILIO_FROM.startswith("whatsapp:") else f"whatsapp:{TWILIO_FROM}"

        message = client.messages.create(
            body=msg_body,
            from_=from_num,
            to=to_num
        )
        print(f"[CaregiverAlert] ✅ WhatsApp message sent successfully to {to_num}! SID: {message.sid}")
        return True
    except Exception as e:
        print(f"[CaregiverAlert] ❌ Failed to send WhatsApp message: {e}")
        return False
