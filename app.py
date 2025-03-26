import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image
import faiss
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from dotenv import load_dotenv
import os
import speech_recognition as sr
from langdetect import detect

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

st.set_page_config(page_title="📄 Assistant d'Analyse de Documents", layout="wide")
print(f"Hugging Face Token: {hf_token}")


# --- Hugging Face  --- 
@st.cache_resource
def authenticate():
    try:
        login(token=os.getenv("HF_TOKEN", hf_token))
        return True
    except Exception as e:
        st.error(f"échec de l'authentification: {str(e)}")
        return False

if not authenticate():
    st.stop()


@st.cache_resource
def load_models(language='fr'):
    if language == 'en':
        embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        model_name = "distilbert-base-uncased-distilled-squad"
    else:
        embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        model_name = "etalab-ia/camembert-base-squadFR-fquad-piaf"

    qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    device = 0 if torch.cuda.is_available() else -1  # 0=GPU, -1=CPU

    qa_pipeline = pipeline(
        "question-answering",
        model=qa_model,
        tokenizer=qa_tokenizer,
        device=device
    )
    return embedding_model, qa_pipeline


# --- Language detection --- 
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'fr'  # Default to French if detection fails


# --- Voice input --- 
def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎙️ Enregistrement en cours...")
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio, language="fr-FR")
        st.write(f"Vous avez dit: {text}")
        return text
    except sr.UnknownValueError:
        st.warning("Désolé, je n'ai pas compris. Essayez à nouveau.")
        return ""
    except sr.RequestError:
        st.error("Erreur lors de la connexion au service de reconnaissance vocale.")
        return ""


# --- Extract Text from File --- 
def extract_text(file):
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            return text if text else None
    else:
        try:
            return pytesseract.image_to_string(Image.open(file))
        except Exception:
            return None


# --- Create FAISS index --- 
def create_faiss_index(text):
    if not text:
        return None, None
        
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
    if len(sentences) < 3:
        return None, None
    
    embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.cpu().numpy())
    return index, sentences


# --- Retrieve Context --- 
def retrieve_context(question, index, sentences, top_k=3):
    if not index or not sentences:
        return None
        
    query_embed = embedding_model.encode(question, convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(np.array([query_embed]), top_k)
    return ". ".join(sentences[i] for i in indices[0] if i < len(sentences))


# --- Feedback mechanism --- 
def collect_feedback(answer):
    feedback = st.radio("Avez-vous trouvé la réponse utile?", ["Oui", "Non"])
    if feedback == "Non":
        additional_feedback = st.text_area("Comment améliorer la réponse:")
        if st.button("Envoyer le retour"):
            with open("feedback.txt", "a") as f:
                f.write(f"Réponse: {answer}\nFeedback: {additional_feedback}\n\n")
            st.success("Merci pour votre retour!")


# --- Streamlit UI --- 
st.title("📄 Assistant d'Analyse de Documents")

uploaded_file = st.file_uploader("📂 Téléverser un document", 
                               type=["pdf", "png", "jpg", "jpeg"],
                               accept_multiple_files=False)

if uploaded_file:
    with st.spinner("Analyse du document en cours..."):
        text = extract_text(uploaded_file)
        if not text:
            st.error("Impossible d'extraire du texte d'un document, essayez un autre fichier")
            st.stop()
        
        # Detect language
        language = detect_language(text)
        embedding_model, qa_pipeline = load_models(language)
        
        index, sentences = create_faiss_index(text)
        if not index:
            st.warning("Texte trop court pour créer un index sémantique")
            st.stop()
        
        st.success("Document prêt pour interrogation !")
        st.text_area("Texte extrait (extrait):", 
                    value=text[:500] + ("..." if len(text) > 500 else ""), 
                    height=150)

    question = st.text_input("❓ Posez votre question ici", key="question")
    
    if question:
        with st.spinner("Recherche de la réponse..."):
            context = retrieve_context(question, index, sentences)
            if not context:
                st.warning("Aucun contexte n'a pu être trouvé")
                st.stop()
                
            try:
                answer = qa_pipeline(question=question, context=context)
                st.subheader("Réponse:")
                st.write(answer["answer"])
                
                # Collect feedback
                collect_feedback(answer["answer"])

                with st.expander("Voir le contexte utilisé"):
                    st.write(context)
            except Exception as e:
                st.error(f"Échec de la demande: {str(e)}")

    # Voice input button
    if st.button("🎤 Poser une question par voix"):
        question = record_audio()
        if question:
            with st.spinner("Recherche de la réponse..."):
                context = retrieve_context(question, index, sentences)
                if not context:
                    st.warning("Aucun contexte n'a pu être trouvé")
                    st.stop()

                try:
                    answer = qa_pipeline(question=question, context=context)
                    st.subheader("Réponse:")
                    st.write(answer["answer"])

                    # Collect feedback
                    collect_feedback(answer["answer"])

                    with st.expander("Voir le contexte utilisé"):
                        st.write(context)
                except Exception as e:
                    st.error(f"Échec de la demande: {str(e)}")


with st.sidebar:
    st.markdown(""" 
    **Instructions:**
    1. Téléversez un PDF ou une image
    2. Posez votre question en français
    3. Obtenez une réponse précise
    
    **Technologies:**
    - OCR: Tesseract
    - NLP: CamemBERT (modèle français)
    - Sémantique: Sentence-Transformers
    """)
    
    if st.checkbox("Afficher les informations techniques"):
        st.write(f"Device utilisé: {'GPU' if qa_pipeline.device != -1 else 'CPU'}")
        st.write(f"Modèle: {qa_pipeline.model.name_or_path}")
