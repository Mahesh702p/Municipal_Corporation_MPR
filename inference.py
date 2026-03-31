"""
inference.py
============
Main Inference Router for the Municipal Corporation MLM + RAG System.

Pipeline:
  User Input
      ↓
  [Level 1] Intent Classifier  →  complaint / query / status_check / emergency / service_request
      ↓
  Router:
    complaint       → [Level 2] Department Classifier → log complaint
    query           → RAG Retriever → return FAQ answer
    status_check    → extract IDs → redirect to portal
    emergency       → flag HIGH priority → emergency contacts
    service_request → [Level 2] Department Classifier → redirect to service portal

Usage:
  from inference import MunicipalInferenceEngine
  engine = MunicipalInferenceEngine()
  result = engine.process("garbage nahi utha ward 5 se 3 din se")
  print(result)
"""

import json
import os
import re
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "abctokz_repo" / "src"))
sys.path.insert(0, str(ROOT))

# ── Department → contact info mapping ──────────────────────
# Keys MUST match the trained model's label names exactly (from label_encoders.json)
DEPT_INFO = {
    "roads": {
        "office": "Roads & Infrastructure Department",
        "helpline": "1800-MC-ROADS",
        "portal": "https://mc.gov.in/complaints/roads",
    },
    "water_supply": {
        "office": "Water Supply Department",
        "helpline": "1916",
        "portal": "https://mc.gov.in/complaints/water",
    },
    "solid_waste": {
        "office": "Solid Waste Management Department",
        "helpline": "1800-MC-WASTE",
        "portal": "https://mc.gov.in/complaints/swm",
    },
    "health": {
        "office": "Public Health Department",
        "helpline": "1800-MC-HEALTH",
        "portal": "https://mc.gov.in/complaints/health",
    },
    "revenue": {
        "office": "Revenue & Tax Department",
        "helpline": "1800-MC-TAX",
        "portal": "https://mc.gov.in/propertytax",
    },
    "electricity": {
        "office": "Electricity Department",
        "helpline": "1912",
        "portal": "https://mc.gov.in/complaints/electricity",
    },
    "sewerage": {
        "office": "Sewerage & Drainage Department",
        "helpline": "1800-MC-DRAIN",
        "portal": "https://mc.gov.in/complaints/sewerage",
    },
    "parks": {
        "office": "Parks & Gardens Department",
        "helpline": "1800-MC-PARKS",
        "portal": "https://mc.gov.in/complaints/parks",
    },
    "disaster_management": {
        "office": "Disaster Management Cell",
        "helpline": "1077",
        "portal": "https://mc.gov.in/disaster",
    },
    "General": {
        "office": "Municipal Corporation Helpdesk",
        "helpline": "1800-MC-HELP",
        "portal": "https://mc.gov.in",
    },
}

EMERGENCY_RESPONSE = {
    "message": "🚨 EMERGENCY DETECTED — Contacting relevant services immediately.",
    "contacts": {
        "Fire": "101",
        "Ambulance": "108",
        "Police": "100",
        "MC Emergency": "1800-MC-HELP",
        "Disaster Mgmt": "1077",
    },
    "instruction": "Please call the relevant number immediately. Stay safe.",
}

STATUS_PORTAL = "https://mc.gov.in/track-complaint"


class MunicipalInferenceEngine:
    """
    Main inference engine for the Municipal Corporation helpdesk system.

    Loads:
    - Intent model (Level 1)
    - Department model (Level 2)
    - MunicipalTokenizer
    - RAG Retriever
    - Label encoders
    """

    def __init__(
        self,
        intent_model_path: str = "artifacts/intent_model",
        dept_model_path: str = "artifacts/ensemble_model",
        tok_path: str = "artifacts/municipal_bpe_tok",
        rag_index_path: str = "artifacts/rag_index",
        label_enc_path: str = "artifacts/label_encoders.json",
        max_len: int = 60,
    ):
        self.max_len = max_len
        self._loaded = False

        self.intent_model_path = str(ROOT / intent_model_path)
        self.dept_model_path = str(ROOT / dept_model_path)
        self.tok_path = str(ROOT / tok_path)
        self.rag_index_path = str(ROOT / rag_index_path)
        self.label_enc_path = str(ROOT / label_enc_path)

    def load(self):
        """Lazy-load all models (call once before first inference)."""
        import tensorflow as tf
        from tokenizer.municipal_tokenizer import MunicipalTokenizer
        from rag.retriever import MunicipalRetriever
        from models.ensemble_model import BahdanauAttention

        print("[Engine] Loading models...")

        # Tokenizer
        self.tok = MunicipalTokenizer.load(self.tok_path)

        # Intent model
        self.intent_model = tf.keras.models.load_model(
            os.path.join(self.intent_model_path, "best_model.keras"),
            compile=False,
            custom_objects={"BahdanauAttention": BahdanauAttention}
        )

        # Department model
        self.dept_model = tf.keras.models.load_model(
            os.path.join(self.dept_model_path, "best_model.keras"),
            compile=False,
            custom_objects={"BahdanauAttention": BahdanauAttention}
        )

        # Label encoders
        with open(self.label_enc_path) as f:
            enc = json.load(f)
        self.intent_idx2label = {int(k): v for k, v in enc["intent"]["idx2label"].items()}
        self.dept_idx2label = {int(k): v for k, v in enc["department"]["idx2label"].items()}

        # RAG Retriever
        try:
            self.retriever = MunicipalRetriever.load(self.rag_index_path)
            print("[Engine] RAG Database loaded.")
        except Exception as e:
            print(f"[Engine] RAG Database not found at {self.rag_index_path}. 'Query' intents will fallback to standard helpdesk responses.")
            self.retriever = None

        self._loaded = True
        print("[Engine] Ready.")

    def _encode(self, text: str) -> np.ndarray:
        """Encode single text → (1, max_len) array."""
        return self.tok.encode_batch([text], max_len=self.max_len)

    def _predict_intent(self, text: str) -> tuple[str, float]:
        """Predict intent class and confidence."""
        x = self._encode(text)
        probs = self.intent_model.predict(x, verbose=0)[0]
        idx = int(probs.argmax())
        return self.intent_idx2label[idx], float(probs[idx])

    def _predict_department(self, text: str) -> tuple[str, float]:
        """Predict department and confidence."""
        x = self._encode(text)
        probs = self.dept_model.predict(x, verbose=0)[0]
        idx = int(probs.argmax())
        return self.dept_idx2label[idx], float(probs[idx])

    def _extract_ids(self, text: str) -> dict:
        """Extract complaint IDs, application IDs from text."""
        complaint_ids = re.findall(r"CMP\d{4,}", text, re.IGNORECASE)
        app_ids = re.findall(r"MC\d{4,}", text, re.IGNORECASE)
        ward = re.findall(r"ward\s*(?:no\.?\s*)?\d+|ward\s+\w+", text, re.IGNORECASE)
        return {
            "complaint_ids": complaint_ids,
            "app_ids": app_ids,
            "ward": ward[0] if ward else None,
        }

    def _assess_severity(self, text: str, dept: str) -> str:
        """Simple keyword-based severity scoring."""
        text_lower = text.lower()
        high_keywords = [
            "aag", "fire", "blast", "emergency", "urgent", "death", "accident",
            "child", "baby", "hospital", "shock", "dead", "fallen", "collapse",
            "आग", "मृत्यु", "दुर्घटना", "गिरा",
        ]
        if any(k in text_lower for k in high_keywords):
            return "HIGH"
        # Keys match trained model labels (from label_encoders.json)
        if dept in ["disaster_management", "electricity", "water_supply"]:
            return "HIGH"
        if dept in ["roads", "health", "sewerage"]:
            return "MEDIUM"
        return "LOW"

    # ──────────────────────────────────────
    # Main process method
    # ──────────────────────────────────────
    def process(self, user_input: str) -> dict:
        """
        Process user input through the full pipeline.

        Args:
            user_input: Raw text from citizen (any language/script).

        Returns:
            Dict with:
              intent, department, severity, response, action, routing
        """
        if not self._loaded:
            self.load()

        text = user_input.strip()

        # ── Level 1: Intent Detection ───────────────────────
        intent, intent_conf = self._predict_intent(text)

        # ── Route by intent ─────────────────────────────────

        # EMERGENCY — fast path, skip Level 2
        if intent == "emergency":
            return {
                "intent": "emergency",
                "intent_confidence": intent_conf,
                "department": "Fire Services / Emergency",
                "severity": "HIGH",
                "response": EMERGENCY_RESPONSE["message"],
                "emergency_contacts": EMERGENCY_RESPONSE["contacts"],
                "instruction": EMERGENCY_RESPONSE["instruction"],
                "action": "ALERT_EMERGENCY",
            }

        # STATUS CHECK — extract IDs and redirect
        if intent == "status_check":
            ids = self._extract_ids(text)
            return {
                "intent": "status_check",
                "intent_confidence": intent_conf,
                "department": "General",
                "severity": "LOW",
                "response": (
                    f"To track your complaint/application status, please visit: {STATUS_PORTAL}\n"
                    "Enter your Complaint ID or Application ID to check status."
                ),
                "extracted_ids": ids,
                "portal": STATUS_PORTAL,
                "action": "REDIRECT_STATUS",
            }

        # QUERY — use RAG
        if intent == "query":
            if self.retriever is None:
                return {
                    "intent": "query",
                    "intent_confidence": intent_conf,
                    "department": "General",
                    "severity": "LOW",
                    "response": (
                        "The RAG Knowledge Base is not loaded on this deployment. "
                        "Please contact your municipal ward office or call 1800-MC-HELP."
                    ),
                    "action": "FAQ_FALLBACK_NO_RAG",
                }
                
            results = self.retriever.retrieve(text, top_k=2)
            if results:
                best = results[0]
                response = f"{best['answer']}"
                if len(results) > 1:
                    response += f"\n\n(Also see: {results[1]['question']})"
                return {
                    "intent": "query",
                    "intent_confidence": intent_conf,
                    "department": best["department"],
                    "severity": "LOW",
                    "response": response,
                    "rag_results": results,
                    "action": "FAQ_ANSWER",
                }
            else:
                return {
                    "intent": "query",
                    "intent_confidence": intent_conf,
                    "department": "General",
                    "severity": "LOW",
                    "response": (
                        "I don't have a specific answer for this. "
                        "Please contact your municipal ward office or call 1800-MC-HELP."
                    ),
                    "action": "FAQ_FALLBACK",
                }

        # COMPLAINT or SERVICE_REQUEST — Level 2: Department Classification
        dept, dept_conf = self._predict_department(text)
        severity = self._assess_severity(text, dept)
        dept_info = DEPT_INFO.get(dept, DEPT_INFO["General"])
        ids = self._extract_ids(text)

        if intent == "service_request":
            return {
                "intent": "service_request",
                "intent_confidence": intent_conf,
                "department": dept,
                "dept_confidence": dept_conf,
                "severity": "LOW",
                "response": (
                    f"For {dept} service requests, please visit:\n"
                    f"Portal: {dept_info['portal']}\n"
                    f"Helpline: {dept_info['helpline']}\n"
                    f"Office: {dept_info['office']}"
                ),
                "extracted": ids,
                "action": "REDIRECT_SERVICE",
            }

        # COMPLAINT (default)
        return {
            "intent": "complaint",
            "intent_confidence": intent_conf,
            "department": dept,
            "dept_confidence": dept_conf,
            "severity": severity,
            "response": (
                f"Your complaint has been registered with the {dept} department.\n"
                f"Severity: {severity}\n"
                f"Expected resolution: {'24 hours' if severity == 'HIGH' else '48–72 hours' if severity == 'MEDIUM' else '7 working days'}.\n"
                f"Track status at: {STATUS_PORTAL}\n"
                f"For urgent issues, call: {dept_info['helpline']}"
            ),
            "dept_office": dept_info["office"],
            "helpline": dept_info["helpline"],
            "extracted": ids,
            "action": "LOG_COMPLAINT",
        }


# ──────────────────────────────────────────────────────────────
# CLI Demo
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = MunicipalInferenceEngine()
    engine.load()

    test_inputs = [
        "garbage nahi utha ward 5 mein 3 din se",
        "birth certificate ke liye kya documents chahiye",
        "meri complaint CMP1234 ka kya hua",
        "ALAG fire lag gayi ward 12 mein help karo",
        "naya water connection chahiye ward 8 mein",
        "potholes on road near market ward 22 very dangerous",
        "property tax kaise calculate hota hai",
        "{ward} 15 mein drain block hai badbu aa rahi",
        "pipe phoot gayi aag lag gayi",
    ]

    print("\n" + "=" * 65)
    print(" MUNICIPAL CORPORATION HELPDESK — INFERENCE DEMO")
    print("=" * 65)

    for inp in test_inputs:
        print(f"\nInput:  {inp}")
        result = engine.process(inp)
        print(f"Intent: {result['intent']} ({result.get('intent_confidence', 0):.2f})")
        print(f"Dept:   {result.get('department', '-')} ({result.get('dept_confidence', 0):.2f})")
        print(f"Action: {result['action']}")
        print(f"Response preview: {result['response'][:100]}...")
        print("-" * 65)
