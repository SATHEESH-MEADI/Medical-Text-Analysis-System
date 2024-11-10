import streamlit as st
import zipfile
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForQuestionAnswering
import torch
import requests
import shutil
import fitz  # PyMuPDF for PDF handling
import google.generativeai as genai

import openai

# Set your API key here
API_KEY = "AIzaSyBLUELtvdlQr3T5g5CU8UhN5JSBnDIXyQA"



# Download required NLTK data
nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words("english"))

#Languages------------------------------------------------------------------

LANGUAGES = {
    'English': 'en',
    'Telugu': 'te',
    'Hindi': 'hi',
    'Arabic': 'ar',
    'Chinese': 'zh',
    'Dutch': 'nl',
    'Korean': 'ko',
    'Russian': 'ru',
    'Spanish': 'es',
    'Portuguese': 'pt',
    'Japanese': 'ja',
    'Italian': 'it',
    'German': 'de',
    'French': 'fr',
    'Greek': 'el',
    'Thai': 'th'
}

#<---------------------------------------------------------Translator----------------------------->
class Translator:
    def __init__(self):
        self.translations_cache = {}

    def translate_text(self, text, target_language_code):
        """Translate text using Google Translation API v2 with caching to minimize API calls."""
        cache_key = (text, target_language_code)
        
        # Check cache first
        if cache_key in self.translations_cache:
            return self.translations_cache[cache_key]
        
        # API request if no cached result
        url = "https://translation.googleapis.com/language/translate/v2"
        params = {'q': text, 'target': target_language_code, 'key': API_KEY}
        
        try:
            response = requests.get(url, params=params)
            response_data = response.json()
            translation = response_data['data']['translations'][0]['translatedText']
            
            # Cache the result to minimize API calls
            self.translations_cache[cache_key] = translation
            return translation
        
        except Exception as e:
            st.error(f"Translation failed: {response_data.get('error', {}).get('message', 'Unknown error')}")
            return text

#<---------------------------------------------------------PubMed model----------------------------->
# # Move this function outside the class
class PubMedBERTSummarizer:
    def __init__(self):
        self.model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def preprocess_text(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    @st.cache_data
    def get_sentence_embeddings(_self, text):
        inputs = _self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        inputs = {k: v.to(_self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = _self.model(**inputs)
        
        return outputs.last_hidden_state.mean(dim=1)

    def get_pubmedbert_summary(self, text):
        try:
            processed_text = self.preprocess_text(text)
            doc_embedding = self.get_sentence_embeddings(processed_text)
            sentences = sent_tokenize(processed_text)
            
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                sent_embedding = self.get_sentence_embeddings(sentence)
                similarity = torch.nn.functional.cosine_similarity(doc_embedding, sent_embedding).item()
                sentence_scores.append((i, sentence, similarity))
            
            sorted_sentences = sorted(sentence_scores, key=lambda x: x[2], reverse=True)
            selected_sentences = sorted_sentences[:5]  # Select top 5 sentences
            summary_sentences = sorted(selected_sentences, key=lambda x: x[0])
            
            summary = ' '.join(sent for _, sent, _ in summary_sentences)
            return summary
            
        except Exception as e:
            st.error(f"Summarization error: {str(e)}")
            return text



#<-----------------------------------------------------Extracting the Text Data -------------------------------->




def extract_text_from_pdf(pdf_path):
    """Extract text from each page of a PDF file using PyMuPDF."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF file {pdf_path}: {e}")
    return text

def extract_files(uploaded_file):
    extract_to = "extracted_text_files"
    os.makedirs(extract_to, exist_ok=True)
    text_files = []

    # Handle zip files
    if uploaded_file.name.endswith(".zip"):
        print(f"Extracting zip file: {uploaded_file.name}")
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        text_files = [os.path.join(root, f) for root, _, files in os.walk(extract_to) for f in files if f.endswith('.txt')]
        print(f"Text files extracted from zip: {text_files}")

    # Handle single text files
    elif uploaded_file.name.endswith(".txt"):
        print(f"Processing text file: {uploaded_file.name}")
        file_path = os.path.join(extract_to, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        text_files.append(file_path)

    # Handle PDF files
    elif uploaded_file.name.endswith(".pdf"):
        print(f"Processing PDF file: {uploaded_file.name}")
        pdf_path = os.path.join(extract_to, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_text = extract_text_from_pdf(pdf_path)

        # Save extracted text to a .txt file
        text_file_path = os.path.join(extract_to, uploaded_file.name.replace(".pdf", ".txt"))
        with open(text_file_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(pdf_text)
        text_files.append(text_file_path)
        print(f"Extracted text saved to: {text_file_path}")

    # Ensure files are found
    if not text_files:
        raise FileNotFoundError("No text files found in the uploaded file.")

    return text_files




#<---------------------------------------------------------Chatbot model----------------------------->

openai.api_key = "Your Api key for chat bot "




# Medical Chatbot class using OpenAI for Q&A
class MedicalChatbot:
    def __init__(self):
        self.conversation_history = []

    def get_answer(self, question, context):
        messages = [
            {"role": "system", "content": "You are a medical expert chatbot. Answer based on the context provided."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ] + self.conversation_history[-3:]  # Keep only the last 3 interactions for context

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
            max_tokens=150
        )

        answer = response['choices'][0]['message']['content']
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})
        return answer

    def clear_history(self):
        self.conversation_history = []
#<---------------------------------------------------Initialization----------------------------->

# Initialize session state
def initialize_session_state():
    for key, val in [
        ('summarizer', PubMedBERTSummarizer()),
        ('chatbot', MedicalChatbot()),
        ('translator', Translator()),
        ('current_summary', None),
        ('translated_summary', None),
        ('selected_language', 'English'),
        ('chat_history', [])  # Store chat history in session state
    ]:
        st.session_state.setdefault(key, val)

def main():
    st.title("Medical Text Analysis System")
    st.write("Upload medical texts for summarization, translation, and interactive Q&A")
    initialize_session_state()

    target_language = st.sidebar.selectbox("Select Target Language", LANGUAGES.keys(), 
                                           index=list(LANGUAGES.keys()).index(st.session_state.selected_language))
    st.session_state.selected_language = target_language
    uploaded_file = st.file_uploader("Upload medical text file(s)", type=["txt", "zip", "pdf"])

    if uploaded_file:
        try:
            text_files = extract_files(uploaded_file)
            for text_file in text_files:
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                st.write(f"### Processing: {os.path.basename(text_file)}")
                tabs = st.tabs(["Original Text", "Summary & Translation", "Q&A"])

                with tabs[0]:
                    st.write("Original Text:")
                    st.text(content)

                with tabs[1]:
                    st.write("PubMedBERT Summary:")
                    summary = st.session_state.summarizer.get_pubmedbert_summary(content)
                    st.session_state.current_summary = summary
                    st.write(summary)
                    
                    if target_language != "English":
                        translated_text = st.session_state.translator.translate_text(summary, LANGUAGES[target_language])
                        st.session_state.translated_summary = translated_text
                        st.write(f"Translation ({target_language}):\n{translated_text}")
                        st.button("Copy Translation", key=f"copy_{text_file}", on_click=lambda: st.write(translated_text))

                with tabs[2]:
                    st.write("**Chat with the Medical Expert Bot**")
                    
                    # Display chat history
                    if st.session_state.chat_history:
                        for chat in st.session_state.chat_history:
                            st.write(f"🤵 You: {chat['question']}")
                            st.write(f"🤖 Bot: {chat['answer']}")

                    question = st.text_input("Enter your question:", key=f"question_{text_file}")
                    answer_language = st.radio("Select answer language:", ["English", target_language], horizontal=True)

                    if question:
                        answer = st.session_state.chatbot.get_answer(question, st.session_state.current_summary)
                        if answer_language != "English":
                            answer = st.session_state.translator.translate_text(answer, LANGUAGES[answer_language])
                        st.write("🤵 You:", question)
                        st.write("🤖 Bot:", answer)

                        # Append the question-answer pair to the chat history
                        st.session_state.chat_history.append({"question": question, "answer": answer})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            if os.path.exists("extracted_text_files"):
                shutil.rmtree("extracted_text_files")

if __name__ == "__main__":
    main()




