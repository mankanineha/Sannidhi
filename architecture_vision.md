# Sannidhi — Senior Safety & Autonomous Fall Care System

## 🎯 Mission Statement
To bridge the critical emergency response gap for elderly individuals living independently (especially those with neurological disorders such as **Parkinson's Disease**, **Epilepsy**, or **Dementia**) by providing non-invasive, privacy-preserving Wi-Fi CSI fall detection, automated voice interaction, and instant multi-modal caregiver alerting.

---

## 🏗️ End-to-End System Architecture

```
                       ┌─────────────────────────┐
                       │  ESP32 Wi-Fi CSI Node   │
                       │ (Subcarrier Monitoring) │
                       └────────────┬────────────┘
                                    │ (CSI Tensors @ 100 Hz)
                                    ▼
                       ┌─────────────────────────┐
                       │ Dual AI Classifier      │
                       │ • CSI-FallNet (25ms)    │
                       │ • CSISpectraNet (96.4%) │
                       └────────────┬────────────┘
                                    │ (Probabilities)
                                    ▼
                       ┌─────────────────────────┐
                       │ Anti-Hallucination Gate │
                       │ • Motion Energy Check   │
                       │ • Temporal Debouncing   │
                       │ • Post-Alert Cooldown   │
                       └────────────┬────────────┘
                                    │ (Verified Fall Event)
                                    ▼
                       ┌─────────────────────────┐
                       │   Voice Agent (TTS/STT) │
                       │ "Are you okay? Say YES  │
                       │   or NO if you need..." │
                       └────────────┬────────────┘
                                    │
             ┌──────────────────────┴──────────────────────┐
             │                                             │
      ✅ [User OK]                                🚨 [Unresponsive or "HELP"]
             │                                             │
  • Clear Alert & Log                           ┌──────────▼──────────┐
  • Resume Monitoring                           │ Caregiver Alert Engine│
                                                │ (Twilio WhatsApp)   │
                                                └──────────┬──────────┘
                                                           │
                                                           ▼
                                                📱 WhatsApp Notification to
                                                   Daughter / Granddaughter
                                                • Resident Medical Context
                                                • AI Diagnostics & Energy
                                                • Voice Interaction Log
```

---

## 🤖 Next Step: Raspberry Pi 5 Agentic Layer Architecture

```
                  ┌─────────────────────────────────────┐
                  │      Raspberry Pi 5 Edge Hub        │
                  │                                     │
                  │  ┌───────────────────────────────┐  │
                  │  │ Local Fall Detection Engine   │  │
                  │  └──────────────┬────────────────┘  │
                  │                 │                   │
                  │  ┌──────────────▼────────────────┐  │
                  │  │ Conversational AI Agent       │  │
                  │  │ • Daily Senior Companion      │  │
                  │  │ • Medication Reminders        │  │
                  │  │ • Medical History Tracker     │  │
                  │  └──────────────┬────────────────┘  │
                  │                 │                   │
                  │  ┌──────────────▼────────────────┐  │
                  │  │ Emergency Dispatch Gateway    │  │
                  │  │ • WhatsApp / SMS / Call       │  │
                  │  └───────────────────────────────┘  │
                  └─────────────────────────────────────┘
```

---

## 📱 Live WhatsApp Message Format

```text
🚨 *URGENT ALERT: POTENTIAL UNRESPONSIVE FALL*

📅 *Time*: 2026-09-18 15:48:43
👤 *Resident*: Senior Resident (Age 74)
📍 *Location*: Home Residence - Living Room Zone
⚕️ *Medical Alert*: Parkinson's Disease, Hypertension

🎙 *Voice Response*: No response within 7s (timeout)
⚠️ *Severity Level*: HIGH - Senior Citizen Unresponsive to Voice Query

📊 *AI Diagnostic Confidence*:
  • FallNet Probability: 94.1%
  • SpectraNet Probability: 96.3%
  • Physics Motion Impact: 0.087 (Verified physical event)

👉 *Action Required*: Please call senior citizen immediately or check home camera / visit!
```
