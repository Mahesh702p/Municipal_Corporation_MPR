import os
import sys
import warnings

# Suppress TF logging noise for a clean terminal UI
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from inference import MunicipalInferenceEngine

print("Booting up local Neural Networks...")
engine = MunicipalInferenceEngine()
engine.load()

print("\n" + "━"*65)
print(" 🏛️  MUNICIPAL AI HELPDESK — TERMINAL INFERENCE MODE")
print("━"*65)
print("Test the engine! Type a messy, Hinglish complaint or query.")
print("Type 'exit' to quit.\n")

while True:
    try:
        user_input = input("🗣️ You: ")
        if user_input.strip().lower() in ['exit', 'quit']:
            print("Shutting down engine...")
            break
        if not user_input.strip():
            continue
            
        print("🤖 Processing via CNN/BiLSTM Ensemble...")
        result = engine.process(user_input)
        
        print("\n┌── ARCHITECTURE ROUTING ────────────────────────")
        print(f"│ ➜ Intent Level 1: {result.get('intent', 'UNKNOWN').upper()} ({result.get('intent_confidence', 0):.2f})")
        if 'dept_confidence' in result:
            print(f"│ ➜ Dept Level 2:   {result.get('department', 'GENERAL').upper()} ({result.get('dept_confidence', 0):.2f})")
        else:
            print(f"│ ➜ Dept Level 2:   {result.get('department', 'GENERAL').upper()}")
        print(f"│ ➜ Core Action:    {result.get('action', '')}")
        print("└────────────────────────────────────────────────\n")
        
        print("📜 OFFICIAL RESPONSE:")
        print(result.get('response', 'No response generated.'))
        print("\n" + "━"*65 + "\n")
        
    except KeyboardInterrupt:
        print("\nShutting down engine...")
        break
    except Exception as e:
        print(f"Terminal Error: {e}")
