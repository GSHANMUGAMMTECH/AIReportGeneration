import streamlit as st
from PIL import Image
import os
from agents.router_agent import MedicalPipeline
import dspy
from utils.utils import create_pdf, translate_to_german
from gtts import gTTS
import io

# Translation map for UI strings
UI = {
    "page_title": {"en": "Medical Image Report Generator using Agentic AI", "de": "Medizinischer Bericht-Generator mit Agentic AI"},
    "title": {"en": "Medical Report Generator using Agentic AI", "de": "Medizinischer Bericht-Generator mit Agentic AI"},
    "intro": {"en": "Upload a medical image to generate a report.", "de": "Laden Sie ein medizinisches Bild hoch, um einen Bericht zu erstellen."},
    "select_language": {"en": "Select Report Language", "de": "Berichtssprache auswählen"},
    "upload_prompt": {"en": "Choose a medical image...", "de": "Wählen Sie ein medizinisches Bild..."},
    "analysis_subheader": {"en": "Analysis", "de": "Analyse"},
    "generate_button": {"en": "Generate Report", "de": "Bericht erstellen"},
    "analyzing_spinner": {"en": "Analyzing image...", "de": "Bild wird analysiert..."},
    "report_generated": {"en": "Report Generated!", "de": "Bericht erstellt!"},
    "modality": {"en": "Modality:", "de": "Modalität:"},
    "body_part": {"en": "Body Part:", "de": "Körperteil:"},
    "analysis_label": {"en": "Specialist Analysis:", "de": "Fachärztliche Analyse:"},
    "report_content": {"en": "📄 Report Content", "de": "📄 Berichtsinhalte"},
    "audio_report": {"en": "🔊 Audio Report", "de": "🔊 Audio-Bericht"},
    "generating_audio": {"en": "Generating audio...", "de": "Audio wird erzeugt..."},
    "download_audio": {"en": "📥 Download Audio", "de": "📥 Audio herunterladen"},
    "pdf_report": {"en": "📑 PDF Report", "de": "📑 PDF-Bericht"},
    "download_pdf": {"en": "📥 Download PDF", "de": "📥 PDF herunterladen"},
    "hallucination_score": {"en": "Hallucination Score", "de": "Halluzinations-Score"},
    "initializing_pipeline": {"en": "Initializing MedicalPipeline... (this should only happen once)", "de": "MedicalPipeline wird initialisiert... (sollte nur einmal passieren)"},
    "uploaded_image": {"en": "Uploaded Image", "de": "Hochgeladenes Bild"},
    "error_generating_audio": {"en": "Error generating audio:", "de": "Fehler bei der Audioerzeugung:"}
}

def ui_text(key: str, lang: str) -> str:
    return UI.get(key, {}).get('de' if lang == 'German' else 'en', '')

# Determine current UI language for labels (default to English)
current_ui_lang = st.session_state.get('language_selector', 'English')

st.set_page_config(page_title=ui_text('page_title', current_ui_lang), layout="wide")

# Custom CSS for blue button
st.markdown("""
    <style>
    .stButton > button[kind=\"primary\"] {
        background-color: #0066CC;
        color: white;
        border: none;
        font-weight: bold;
    }
    .stButton > button[kind=\"primary\"]:hover {
        background-color: #0052A3;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Determine current UI language for labels (default to English)
current_ui_lang = st.session_state.get('language_selector', 'English')

st.title(ui_text('title', current_ui_lang))
st.markdown(ui_text('intro', current_ui_lang))

# Language selector (keeps internal values English/German)
language = st.selectbox(ui_text('select_language', current_ui_lang), ["English", "German"], index=0, key="language_selector")


# Translation helpers imported from utils

# Audio generation helper
def generate_audio(text: str, lang: str = "English") -> bytes | None:
    """Generate audio from text using gTTS for the selected language.

    `lang` should be the UI language string, e.g. "English" or "German".
    """
    if not text or not text.strip():
        return None
        
    try:
        if lang == "German":
            tts_lang = "de"
        else:
            tts_lang = "en"
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.getvalue()
    except Exception as e:
        st.error(f"{ui_text('error_generating_audio', lang)} {e}")
        return None

# Initialise the medical pipeline (singleton)
@st.cache_resource
def get_pipeline():
    print("Initializing MedicalPipeline... (this should only happen once)")
    return MedicalPipeline()

if 'pipeline_ready' not in st.session_state:
    st.toast(ui_text('initializing_pipeline', current_ui_lang), icon="⏳")
    st.session_state['pipeline_ready'] = True

pipeline = get_pipeline()

uploaded_file = st.file_uploader(ui_text('upload_prompt', language if 'language' in locals() else current_ui_lang), type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    with col1:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption=ui_text('uploaded_image', language), width=250)
    with col2:
        st.subheader(ui_text('analysis_subheader', language))
        if st.button(ui_text('generate_button', language), type="primary"):
            with st.spinner(ui_text('analyzing_spinner', language)):
                result = pipeline.process(image, language)
                if "error" in result:
                    st.error(result["error"])
                else:
                    modality = result["modality"]
                    body_part = result["body_part"]
                    analysis = result["analysis"]
                    report_text = result["report"]
                    # Apply translation if needed
                    if language == "German":
                        report_text = translate_to_german(report_text)
                    # Store in session state
                    st.session_state['report_text'] = report_text
                    st.session_state['modality'] = modality
                    st.session_state['body_part'] = body_part
                    st.session_state['analysis'] = analysis
                    st.session_state['image'] = image
                    st.session_state['hallucination_score'] = result.get('hallucination_score')
                    st.success(ui_text('report_generated', language))
                    st.write(f"**{ui_text('modality', language)}** {modality}")
                    st.write(f"**{ui_text('body_part', language)}** {body_part}")
                    st.write(f"**{ui_text('analysis_label', language)}** {analysis}")
    # Display report and additional outputs if available
    if 'report_text' in st.session_state:
        with col1:
            st.markdown("---")
            st.subheader(ui_text('report_content', language))
            st.text_area("", st.session_state['report_text'], height=350, label_visibility="collapsed")
            if st.session_state.get('hallucination_score') is not None:
                st.metric(label=ui_text('hallucination_score', language), value=st.session_state['hallucination_score'])
        with col2:
            # Audio
            st.markdown("---")
            st.subheader(ui_text('audio_report', language))
            with st.spinner(ui_text('generating_audio', language)):
                audio_bytes = generate_audio(st.session_state['report_text'], language)
            if audio_bytes:
                st.audio(audio_bytes, format='audio/mp3')
                st.download_button(
                    label=ui_text('download_audio', language),
                    data=audio_bytes,
                    file_name=f"medical_report_{'de' if language == 'German' else ('kn' if language == 'Kannada' else 'en')}.mp3",
                    mime="audio/mp3",
                    use_container_width=True
                )
            # PDF
            st.markdown("---")
            st.subheader(ui_text('pdf_report', language))
            pdf_bytes = create_pdf(
                st.session_state['report_text'],
                uploaded_file.name,
                st.session_state['modality'],
                st.session_state['body_part'],
                st.session_state['image'],
                language=language
            )
            st.download_button(
                label=ui_text('download_pdf', language),
                data=pdf_bytes,
                file_name=f"medical_report_{'de' if language == 'German' else ('kn' if language == 'Kannada' else 'en')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
