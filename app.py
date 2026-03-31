"""
app.py
======
Interactive Web Interface for the Municipal Corporation AI System.
Powered by Gradio.

Allows users to type queries in English, Hindi, or Hinglish,
and routes them through the Pipeline (Intent -> Dept/RAG -> Response).

Usage:
  python app.py
  (Then open http://localhost:7860 in your browser)
"""

import gradio as gr
from inference import MunicipalInferenceEngine

# Load the engine once at startup
print("Starting Municipal AI Web Server...")
print("Loading models (this takes a few seconds)...")
try:
    engine = MunicipalInferenceEngine()
    engine.load()
    print("Models loaded successfully!")
except Exception as e:
    print(f"Failed to load models. Have you run the training pipeline yet? Error: {e}")
    engine = None

def process_query(user_msg):
    """Gradio handler: Takes text, returns Markdown response."""
    if not user_msg.strip():
        return "Please enter a message.", ""
        
    if engine is None:
        return "**System Error**: AI Engine not loaded. Run pipeline first.", ""
        
    try:
        # Run through our inference router
        result = engine.process(user_msg)
        
        # Build a nice markdown response
        intent = result["intent"].upper()
        dept = result.get("department", "Unknown")
        intent_conf = result.get("intent_confidence", 0.0)
        
        # Format the top debug banner
        banner = f"### System Routing Data\n"
        banner += f"- **Detected Intent:** `{intent}` ({intent_conf:.0%} confidence)\n"
        
        if intent in ["complaint", "service_request"]:
            dept_conf = result.get("dept_confidence", 0.0)
            banner += f"- **Routed Department:** `{dept}` ({dept_conf:.0%} confidence)\n"
            banner += f"- **Severity Score:** `{result.get('severity', 'LOW')}`\n"
            if result.get("extracted", {}).get("ward"):
                banner += f"- **Extracted Ward:** `{result['extracted']['ward']}`\n"
                
        elif intent == "emergency":
            banner += f"- **Status:** 🚨 `HIGH ALERT`\n"
            
        elif intent == "query":
            banner += f"- **RAG Retrieved Department:** `{dept}`\n"

        # Format the main reply
        reply = f"**Avatar Response:**\n\n{result['response']}"
        
        # If there are RAG results, list keywords
        if "rag_results" in result and len(result["rag_results"]) > 0:
            top_kw = ", ".join(result["rag_results"][0].get("keywords", []))
            if top_kw:
                banner += f"- **Keyword Matches:** `{top_kw}`\n"

        return reply, banner
        
    except Exception as e:
        import traceback
        return f"**Error processing query**: {str(e)}", f"```text\n{traceback.format_exc()}\n```"

# ── Gradio Layout ──────────────────────────────────────────

# Custom CSS for municipal aesthetics (Blue/Orange theme)
custom_css = """
.gradio-container { background-color: #f7f9fa; }
h1 { color: #1a365d; text-align: center; font-family: 'Helvetica Neue', sans-serif; }
.debug-panel { background-color: #f1f5f9; border-left: 4px solid #64748b; padding: 10px; font-family: monospace; font-size: 0.9em; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Municipal Demo") as demo:
    gr.Markdown("# 🏛️ Smart Municipal Corporation Helpdesk AI")
    gr.Markdown(
        "Welcome! Try asking about documents, complaining about a pothole, checking a status, or reporting an emergency. "
        "The AI understands **English, Hindi, and Hinglish**."
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": "Namaste! How can I help you today? (नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?)"}], 
                elem_id="chat-box", height=400
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your complaint or query here... (e.g. 'ward 5 mein kachra nahi utha')", 
                    show_label=False,
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
                
            gr.Examples(
                examples=[
                    "birth certificate ke liye kya documents chahiye?",
                    "garbage nahi utha hai ward 5 se please help",
                    "mera property tax kitna calculate hota hai?",
                    "aag lag gayi hai ward 12 junction pe!!",
                    "meri complaint CMP2948 ka status kya hai?",
                    "there is a huge pothole causing accidents in Andheri",
                ],
                inputs=msg,
            )
            
        with gr.Column(scale=1):
            gr.Markdown("### AI Processing Details")
            gr.Markdown("Watch how the 2-level router and RAG engine analyze your text instantly.")
            debug_box = gr.Markdown("*(Awaiting input...)*", elem_classes="debug-panel")

    # Handle submitting messages
    def respond(user_message, chat_history):
        response, debug_info = process_query(user_message)
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": response})
        return "", chat_history, debug_info
        
    msg.submit(respond, [msg, chatbot], [msg, chatbot, debug_box])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot, debug_box])

if __name__ == "__main__":
    # Launch on port 7860
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
