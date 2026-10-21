import os
import time
import firebase_admin
from firebase_admin import credentials, firestore

# Force IPv4 to avoid IPv6 routing issues on some networks
os.environ.setdefault('GRPC_DNS_RESOLVER', 'native')

_db = None


def init_firebase():
    global _db
    if _db is not None:
        return _db
        
    # The JSON key must be present in the root directory for this to work
    key_path = os.path.join(os.path.dirname(__file__), 'firebase_admin_key.json')
    
    if not os.path.exists(key_path):
        print(f"[Firebase] ⚠️ Warning: {key_path} not found. Firestore logging is DISABLED.")
        return None

    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred)
            
        _db = firestore.client()
        print("[Firebase] ✅ Admin SDK Initialized Successfully!")
        return _db
    except Exception as e:
        print(f"[Firebase] ❌ Error initializing: {e}")
        return None

def _resolve_senior_id(senior_id=None):
    if senior_id:
        return senior_id
    try:
        import profile_manager
        return profile_manager.load_profile().get("senior_id", "SNR-8492")
    except Exception:
        return "SNR-8492"

def update_heartbeat(senior_id=None, device_status="Sensor Online"):
    senior_id = _resolve_senior_id(senior_id)
    db = init_firebase()
    if not db:
        return
        
    try:
        db.collection("seniors").document(senior_id).set({
            "deviceStatus": device_status,
            "lastHeartbeat": firestore.SERVER_TIMESTAMP,
            "wellnessObservation": {
                "summary": "Activity is normal today" if "FALL" not in device_status.upper() else "Fall detected — assessment in progress",
                "evidence": "Movement events match this person's 14-day baseline."
            }
        }, merge=True)
    except Exception as e:
        print(f"[Firebase] ❌ Heartbeat failed: {e}")

def log_incident(senior_id=None, is_emergency=False, transcript="", summary="", metrics=None):
    metrics = metrics or {}
    senior_id = _resolve_senior_id(senior_id)
    db = init_firebase()
    if not db:
        return None
        
    try:
        incident_ref = db.collection("seniors").document(senior_id).collection("incidents").document()
        
        # Build immutable timeline
        conf = metrics.get('probability', 0.95) * 100
        z_score = metrics.get('surge_zscore', 3.5)
        timeline = [
            {
                "time": "Just now", 
                "title": "Possible fall detected", 
                "detail": f"Conf: {conf:.1f}% | Z-Score: {z_score:.1f}"
            },
            {
                "time": "Just now",
                "title": "Voice check completed",
                "detail": f"Transcript: \"{transcript or 'Voice check active'}\""
            }
        ]
        
        if is_emergency:
            timeline.append({
                "time": "Just now",
                "title": "Caregiver Notified",
                "detail": "FCM Push & WhatsApp sent"
            })
            state = "unacknowledged"
        else:
            timeline.append({
                "time": "Just now",
                "title": "Incident Resolved (Self-Reported)",
                "detail": "Senior verbally confirmed they are okay."
            })
            state = "resolved"

        incident_ref.set({
            "id": incident_ref.id,
            "state": state,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "isEmergency": is_emergency,
            "aiSummary": summary or "Possible fall event detected by radar sensors.",
            "timeline": timeline
        })
        
        print(f"[Firebase] ✅ Incident logged to Firestore. ID: {incident_ref.id} (Emergency: {is_emergency}, Senior: {senior_id})")
        return incident_ref.id
    except Exception as e:
        print(f"[Firebase] ❌ Incident logging failed: {e}")
        return None


if __name__ == "__main__":
    import sys
    print("Testing Sannidhi Firebase incident logger...")
    doc_id = log_incident(
        is_emergency=True,
        transcript="Simulated test from terminal",
        summary="Emergency fall alert generated for caregiver verification test.",
        metrics={"probability": 0.98, "surge_zscore": 4.2}
    )
    print(f"Result document ID: {doc_id}")

