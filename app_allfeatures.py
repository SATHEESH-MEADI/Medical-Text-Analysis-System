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
import pandas as pd
import plotly.express as px

# Set your API key here for the language translation this is from the google cloud api 
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
#<---------------------------------------------------------Sentiment Analysis----------------------------->


class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = pipeline("sentiment-analysis")

    def analyze_sentiment(self, text):
        try:
            # Handle empty text
            if not text or len(text.strip()) == 0:
                return None

            # Split text into smaller chunks (to handle long texts)
            max_length = 1000
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            
            if not chunks:
                return None

            sentiments = []
            for chunk in chunks:
                try:
                    result = self.analyzer(chunk)
                    if result and len(result) > 0:
                        sentiments.append(result[0])
                except Exception as e:
                    st.warning(f"Chunk analysis failed: {str(e)}")
                    continue

            if not sentiments:
                return None

            # Count sentiments
            positive_count = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
            negative_count = sum(1 for s in sentiments if s['label'] == 'NEGATIVE')
            neutral_count = len(sentiments) - positive_count - negative_count

            total = len(sentiments)
            if total == 0:
                return None

            # Calculate percentages
            sentiment_scores = {
                'positive': (positive_count / total) * 100,
                'negative': (negative_count / total) * 100,
                'neutral': (neutral_count / total) * 100
            }

            # Determine overall sentiment
            max_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])
            
            # Map sentiment to color and label
            sentiment_mapping = {
                'positive': {'color': 'green', 'label': 'Positive'},
                'negative': {'color': 'red', 'label': 'Negative'},
                'neutral': {'color': 'blue', 'label': 'Neutral'}
            }

            return {
                'overall_sentiment': sentiment_mapping[max_sentiment[0]]['label'],
                'color': sentiment_mapping[max_sentiment[0]]['color'],
                'confidence': max_sentiment[1] / 100,
                'breakdown': sentiment_scores
            }

        except Exception as e:
            st.error(f"Sentiment analysis failed: {str(e)}")
            return None



#<--------------------------------------------------Named Entity Model----------------------------------->

class MedicalNER:
    def __init__(self):
        self.nlp = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")

    def get_named_entities(self, text):
        entities = self.nlp(text)
        return [(entity['word'], entity['entity_group']) for entity in entities]







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


# Configure OpenAI API to use Ollama's local server
openai.api_base = 'http://localhost:11434/v1'
openai.api_key = 'ollama'  # Placeholder key, not used by Ollama

# Medical Chatbot class using Ollama for Q&A
class MedicalChatbot:
    def __init__(self):
        self.conversation_history = []

    def get_answer(self, question, context):
        messages = [
            {"role": "system", "content": "You are a medical expert chatbot. Answer based on the context provided."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ] + self.conversation_history[-3:]  # Keep only the last 3 interactions for context

        # Call the Ollama model using OpenAI's compatible API structure with "llama3.2"
        response = openai.ChatCompletion.create(
            model="llama3.2",  # Use "llama3.2" as the model name for Ollama
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )

        answer = response['choices'][0]['message']['content']
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})
        return answer

    def clear_history(self):
        self.conversation_history = []


#<---------------------------------------------------Initialization----------------------------->

def initialize_session_state():
    for key, val in [
        ('summarizer', PubMedBERTSummarizer()),
        ('chatbot', MedicalChatbot()),
        ('translator', Translator()),
        ('ner', MedicalNER()),
        ('sentiment_analyzer', SentimentAnalyzer()),
        ('current_summary', None),
        ('translated_summary', None),
        ('selected_language', 'English'),
        ('chat_history', [])  # Store chat history in session state
    ]:
        st.session_state.setdefault(key, val)



def main():
    st.title("Medical Text Analysis System")
    st.write("Upload medical texts or enter raw text for summarization, translation, NER, and interactive Q&A")
    initialize_session_state()

    target_language = st.sidebar.selectbox("Select Target Language", LANGUAGES.keys(), 
                                           index=list(LANGUAGES.keys()).index(st.session_state.selected_language))
    st.session_state.selected_language = target_language



    # Option to either upload a file or input raw text
    st.write("### Input Options:")
    uploaded_files = st.file_uploader("Upload medical text file(s)", type=["txt", "zip", "pdf"], accept_multiple_files=True)
    raw_text = st.text_area("Or, paste raw text here:")

    if uploaded_files or raw_text:
        try:
            text_data = ""

            # Process uploaded files
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    text_files = extract_files(uploaded_file)
                    for text_file in text_files:
                        with open(text_file, 'r', encoding='utf-8') as f:
                            text_data += f.read() + "\n"

            # Use raw text if provided
            if raw_text:
                text_data += raw_text

            st.write(f"### Processing Text")
            tabs = st.tabs(["Original Text", "Summary & Translation", "NER", "Q&A", "Sentiment Analysis"])

            with tabs[0]:
                st.write("Original Text:")
                st.text(text_data)

            with tabs[1]:
                st.write("PubMedBERT Summary:")
                summary = st.session_state.summarizer.get_pubmedbert_summary(text_data)
                st.session_state.current_summary = summary
                st.write(summary)

                if target_language != "English":
                    translated_text = st.session_state.translator.translate_text(summary, LANGUAGES[target_language])
                    st.session_state.translated_summary = translated_text
                    st.write(f"Translation ({target_language}):\n{translated_text}")
            # Format and display entities in "Entity - Type" format

            with tabs[2]:
                st.write("Named Entity Recognition (NER):")
                entities = st.session_state.ner.get_named_entities(text_data)
                if entities:
                    for entity, entity_type in entities:
                        clean_entity = entity.replace("#", "")
                        st.write(f"{clean_entity} - {entity_type}")
                else:
                    st.write("No Named Entities found in the text...")

            with tabs[3]:
                st.write("**Chat with the Medical Expert Bot**")
                if st.session_state.chat_history:
                    for chat in st.session_state.chat_history:
                        st.write(f"🤵 You: {chat['question']}")
                        st.write(f"🤖 Bot: {chat['answer']}")

                question = st.text_input("Enter your question:")
                answer_language = st.radio("Select answer language:", ["English", target_language], horizontal=True)

                if question:
                    answer = st.session_state.chatbot.get_answer(question, st.session_state.current_summary)
                    if answer_language != "English":
                        answer = st.session_state.translator.translate_text(answer, LANGUAGES[answer_language])
                    st.write("🤵 You:", question)
                    st.write("🤖 Bot:", answer)
                    st.session_state.chat_history.append({"question": question, "answer": answer})       

            with tabs[4]:
                st.write("### Sentiment Analysis")
                if text_data:  # Only analyze if there's text
                    sentiment_result = st.session_state.sentiment_analyzer.analyze_sentiment(text_data)
                    
                    if sentiment_result:
                        # Display overall sentiment
                        st.markdown(
                            f"<h3 style='color: {sentiment_result['color']}'>"
                            f"Overall Sentiment: {sentiment_result['overall_sentiment']}</h3>",
                            unsafe_allow_html=True
                        )
                        
                        # Display confidence
                        st.write(f"Confidence: {sentiment_result['confidence']*100:.1f}%")
                        
                        # Create columns for sentiment breakdown
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(
                                f"<p style='color: green'>Positive: "
                                f"{sentiment_result['breakdown']['positive']:.1f}%</p>",
                                unsafe_allow_html=True
                            )
                        
                        with col2:
                            st.markdown(
                                f"<p style='color: red'>Negative: "
                                f"{sentiment_result['breakdown']['negative']:.1f}%</p>",
                                unsafe_allow_html=True
                            )
                        
                        with col3:
                            st.markdown(
                                f"<p style='color: blue'>Neutral: "
                                f"{sentiment_result['breakdown']['neutral']:.1f}%</p>",
                                unsafe_allow_html=True
                            )
                        
                        # Create visualization
                        try:
                            chart_data = pd.DataFrame({
                                'Sentiment': ['Positive', 'Negative', 'Neutral'],
                                'Percentage': [
                                    sentiment_result['breakdown']['positive'],
                                    sentiment_result['breakdown']['negative'],
                                    sentiment_result['breakdown']['neutral']
                                ]
                            })
                            
                            fig = px.bar(
                                chart_data,
                                x='Sentiment',
                                y='Percentage',
                                color='Sentiment',
                                color_discrete_map={
                                    'Positive': 'green',
                                    'Negative': 'red',
                                    'Neutral': 'blue'
                                }
                            )
                            st.plotly_chart(fig)
                            
                            # Add contextual message
                            if sentiment_result['overall_sentiment'] == 'Positive':
                                st.success("✓ The text contains predominantly positive indicators")
                            elif sentiment_result['overall_sentiment'] == 'Negative':
                                st.error("⚠️ The text contains significant negative elements")
                            else:
                                st.info("ℹ️ The text maintains a neutral tone")
                                
                        except Exception as e:
                            st.warning(f"Could not create visualization: {str(e)}")
                    else:
                        st.warning("Could not determine sentiment for this text")
                else:
                    st.info("Please enter or upload text to analyze sentiment")            

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            if os.path.exists("extracted_text_files"):
                shutil.rmtree("extracted_text_files")




if __name__ == "__main__":
    main()

