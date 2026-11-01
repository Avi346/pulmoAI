import os
import streamlit as st
from PIL import Image

# Page setup - MUST be first Streamlit command
st.set_page_config(
    page_title="AI Radiologist Assistant",
    page_icon="🩺",
    layout="wide",
)

import threading

def _background_warmup():
    try:
        from src.build_rag import get_rag_chain
        get_rag_chain()
    except Exception as e:
        print("Background RAG warmup error:", e)
    try:
        from src.train_vision import get_cached_vision_model
        get_cached_vision_model()
    except Exception as e:
        print("Background vision warmup error:", e)

@st.cache_resource(show_spinner=False)
def init_background_warmup():
    t = threading.Thread(target=_background_warmup, daemon=True)
    t.start()
    return t

init_background_warmup()

# Title and header
st.title("🩺 AI Radiologist Assistant")
st.markdown("Automated **Chest X-Ray Pneumonia Classification** and **Clinical Guidelines RAG Assistant**.")
st.divider()

# Sidebar for overview & status
with st.sidebar:
    st.header("⚙️ System Status")
    st.success("🟢 Dashboard: Ready")
    st.info("💡 **Tip**: Use sample images from `data/chest_xray/test/` to test X-ray predictions.")
    st.divider()
    st.markdown("### 📖 About")
    st.caption("This assistant uses transfer learning (DenseNet-121 at 384x384 resolution) to screen X-rays and LangChain + Mistral AI to retrieve clinical guidelines.")

# Two clean tabs
tab1, tab2 = st.tabs(["🖼️ X-Ray Screening", "💬 Clinical Guideline Assistant"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("1️⃣ Upload Chest X-Ray")
        uploaded_file = st.file_uploader(
            "Select an X-ray image file",
            type=["png", "jpg", "jpeg"],
            help="Upload a standard chest radiograph"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

    with col2:
        st.subheader("2️⃣ Diagnosis & Findings")
        if uploaded_file is not None:
            if st.button("🔍 Run AI Analysis", type="primary", use_container_width=True):
                # Ensure scratch directory exists for safe PIL handling
                scratch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch")
                os.makedirs(scratch_dir, exist_ok=True)
                temp_path = os.path.join(scratch_dir, "temp_upload.png")

                # Save uploaded file temporarily for backend predict() function
                image = Image.open(uploaded_file).convert("RGB")
                image.save(temp_path)

                with st.spinner("Examining the X-ray..."):
                    try:
                        from src.train_vision import predict
                        label, confidence = predict(temp_path)
                        st.session_state.last_diagnosis = label
                        st.session_state.last_confidence = confidence
                        # Display clean results
                        if label == "NORMAL":
                            st.success("### 🌟 Diagnosis: Normal")
                            st.metric("Model Confidence", f"{confidence * 100:.2f}%")
                            st.info("No radiographic evidence of pneumonia detected.")
                        else:
                            st.error("### ⚠️ Diagnosis: Pneumonia Detected")
                            st.metric("Model Confidence", f"{confidence * 100:.2f}%")
                            st.warning("**Clinical Action Recommended**: Consult treatment guidelines or verify findings.")
                            st.markdown("👉 Switch to the **Clinical Guidelines RAG** tab to check recommended empirical antibiotics or admission criteria.")
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
        else:
            st.info("👈 Upload an X-ray image on the left to start screening.")

with tab2:
    st.subheader("💬 Clinical AI Assistant & Guidelines Consultant")
    st.write("Ask any clinical question, patient recovery scenario, diet, or treatment plan below.")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your AI Clinical Healthcare Assistant. How can I help you or your patient today?"}
        ]

    # Dynamic Sample Topics based on Screening Result
    last_dx = st.session_state.get("last_diagnosis")
    if last_dx == "PNEUMONIA":
        st.markdown("**💡 Tailored Topics for Pneumonia Diagnosis:**")
        t_label1, t_prompt1 = "🍎 Recovery Diet & Care", "I was diagnosed with pneumonia. What medical steps, diet, nutrition, and daily recovery habits should I adopt?"
        t_label2, t_prompt2 = "💊 Antibiotic Protocols", "What is the brief recommended empirical antibiotic therapy for outpatient adults with CAP?"
        t_label3, t_prompt3 = "🚨 Warning & Emergency Signs", "What are the red-flag warning symptoms of worsening pneumonia requiring immediate emergency care?"
    elif last_dx == "NORMAL":
        st.markdown("**💡 Tailored Topics for Normal Findings:**")
        t_label1, t_prompt1 = "🛡️ Infection Prevention", "My chest X-ray was normal. What daily habits and immune nutrition can help me prevent respiratory infections?"
        t_label2, t_prompt2 = "🩺 Persistent Cough Care", "My X-ray is normal but I still have a cough. What clinical guidelines recommend for managing acute bronchitis or persistent viral cough?"
        t_label3, t_prompt3 = "📅 Follow-up Guidance", "When should a patient with normal initial chest radiograph seek a follow-up imaging study?"
    else:
        st.markdown("**💡 General Sample Topics (No X-ray screened yet):**")
        t_label1, t_prompt1 = "🍎 Pneumonia Recovery Diet", "What diet, nutrition, and recovery steps should someone with pneumonia adopt?"
        t_label2, t_prompt2 = "💊 Outpatient Treatment", "What is the recommended empirical antibiotic therapy for CAP?"
        t_label3, t_prompt3 = "ℹ️ Childhood Pneumonia Signs", "What are the WHO diagnostic criteria for pneumonia in children?"

    q_col1, q_col2, q_col3 = st.columns(3)
    triggered_prompt = None
    if q_col1.button(t_label1, use_container_width=True):
        triggered_prompt = t_prompt1
    if q_col2.button(t_label2, use_container_width=True):
        triggered_prompt = t_prompt2
    if q_col3.button(t_label3, use_container_width=True):
        triggered_prompt = t_prompt3

    # Display existing chat messages
    chat_container = st.container(height=420, border=True)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Accept user input
    user_input = st.chat_input("Type your clinical question or patient scenario here...")
    prompt_to_process = triggered_prompt if triggered_prompt else user_input

    if prompt_to_process:
        # Build patient screening context
        if last_dx:
            conf_val = st.session_state.get("last_confidence", 0) * 100
            pat_ctx = f"AI X-ray Screening Diagnosis: {last_dx} ({conf_val:.1f}% confidence)."
        elif uploaded_file is None:
            pat_ctx = "No X-ray uploaded or screened yet."
        else:
            pat_ctx = "X-ray uploaded but AI screening not run yet."

        # Maintain small chat history context (last 6 messages prior to current query)
        history_lines = []
        for m in st.session_state.messages[-6:]:
            role_name = "User" if m["role"] == "user" else "Assistant"
            history_lines.append(f"{role_name}: {m['content']}")
        chat_hist_str = "\n".join(history_lines) if history_lines else "None"

        # Dynamic spinner text check
        p_lower = prompt_to_process.lower().strip()
        if len(p_lower) < 5 or any(k in p_lower for k in ["hi", "hello", "hey", "thanks", "thank", "who are", "what can", "help"]):
            dynamic_spinner = "💭 Thinking..."
        elif any(k in p_lower for k in ["diet", "food", "habit", "recover", "nutrition"]):
            dynamic_spinner = "🍎 Retrieving WHO & IDSA clinical nutrition & patient recovery guidelines..."
        elif any(k in p_lower for k in ["treat", "antibiotic", "drug", "medicine", "cap", "therapy"]):
            dynamic_spinner = "💊 Retrieving WHO & IDSA empirical antimicrobial protocols..."
        elif any(k in p_lower for k in ["admit", "hospital", "emergency", "icu", "severe", "sign", "child"]):
            dynamic_spinner = "🏥 Retrieving WHO clinical diagnostic criteria & admission guidelines..."
        else:
            dynamic_spinner = "🔍 Retrieving evidence from indexed WHO & ATS/IDSA pneumonia guidelines..."

        # 1. Add user message to state and display
        st.session_state.messages.append({"role": "user", "content": prompt_to_process})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt_to_process)

            # 2. Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner(dynamic_spinner):
                    try:
                        from src.build_rag import get_rag_chain
                        chain = get_rag_chain()
                        ai_response = chain.invoke({
                            "question": prompt_to_process,
                            "patient_context": pat_ctx,
                            "chat_history": chat_hist_str
                        })
                    except Exception as e:
                        ai_response = f"⚠️ Error querying guidelines: {e}"
                st.markdown(ai_response)

        # 3. Save assistant response to state
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
