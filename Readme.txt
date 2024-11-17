# 📚 Medi Text Summarizer

## 📌 Table of Contents

1. [Introduction](#introduction)  
2. [Features](#features)  
3. [Web Application Showcase](#web-application-showcase)  
4. [Key Functionalities](#key-functionalities)  
5. [📸 Image Showcase](#image-showcase)  
6. [Packages and Tools Used](#packages-and-tools-used)  
7. [Setup Instructions](#setup-instructions)  
8. [Versioning](#versioning)  
9. [Contributing](#contributing)  
10. [License](#license)  
11. [Author](#author)  

---

## 📝 Introduction  

**Medi Text Summarizer** is an advanced text processing and analysis tool designed to make understanding medical literature effortless. Whether it's summarizing large documents, translating text into multiple languages, extracting named entities, or analyzing sentiment, this application leverages cutting-edge AI models to provide precise results.  

With a sleek and interactive user interface built using **Streamlit**, Medi Text Summarizer transforms the way users interact with medical text, making it more accessible, insightful, and actionable.  

---

## 🌟 Features  

- **PubMedBERT Summarization**: Summarizes medical text with precision using state-of-the-art transformer models.  
- **Language Translation**: Translate summaries into 15+ languages via Google Cloud API.  
- **Named Entity Recognition (NER)**: Identifies and categorizes entities in medical documents (e.g., drugs, diseases, treatments).  
- **Interactive Q&A**: Chat with a medical expert bot powered by **Ollama Llama** for contextual question-answering.  
- **Sentiment Analysis**: Detects the emotional tone of medical text using fine-tuned models like DistilBERT.  

---

## 🌐 Web Application Showcase  

### 📂 **Upload Options**  
- Upload `.txt`, `.zip`, or `.pdf` files for automated processing.  
- Enter raw text directly in the sidebar for instant analysis.  

### 🌍 **Language Selector**  
- Translate results into popular languages like Spanish, Hindi, Chinese, and more.  

### 📜 **Sidebar Navigation**  
- **Original Text**: View the unaltered text for reference.  
- **Summary & Translation**: Generate summaries and translate them seamlessly.  
- **NER**: Extract named entities with descriptions.  
- **Interactive Q&A**: Engage with a chatbot for personalized medical insights.  
- **Sentiment Analysis**: Visualize the sentiment breakdown with interactive charts.  

---

## 🔍 Key Functionalities  

1. **Summarization**  
   - Uses PubMedBERT to condense large medical texts into clear, concise summaries.  
   - Scoring algorithms ensure the most relevant sentences are included.  

2. **Translation**  
   - Integrates with Google Cloud Translation API to deliver accurate and fast translations.  

3. **Named Entity Recognition (NER)**  
   - Identifies key terms like diseases, drugs, and procedures using a specialized biomedical model.  

4. **Interactive Q&A**  
   - Chatbot retains conversation context for better accuracy.  
   - Queries are answered in multiple languages with seamless translation.  

5. **Sentiment Analysis**  
   - Visualize the sentiment distribution (positive, negative, neutral) with **Plotly** bar charts.  
   - Supports chunked analysis for lengthy text inputs.  

---

## 📸 Image Showcase  

- **Main Interface**: Navigate through the sidebar to select functionalities.  
- **Summary and Translation**: View concise summaries with real-time translation.  
- **NER Outputs**: Extracted entities displayed with their types.  
- **Sentiment Charts**: Interactive visualizations for sentiment analysis.  
- **Chatbot Conversations**: Engage with a responsive medical chatbot for queries.  

---

## 🛠️ Packages and Tools Used  

- **Streamlit**: Web app framework for creating interactive user interfaces.  
- **Transformers (Hugging Face)**: For leveraging PubMedBERT, DistilBERT, and biomedical NER models.  
- **PyMuPDF**: Efficient PDF text extraction.  
- **Google Cloud Translation API**: For multilingual support.  
- **Plotly**: Interactive charts for visualizing sentiment analysis.  

---

## 🛠️ Setup Instructions  

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-repo/medi-text-summarizer.git
   cd medi-text-summarizer
   ```

2. **Install Required Libraries**  
   ```bash
streamlit
transformers
nltk
pymupdf
requests
plotly
torch
pandas
google-generativeai
openai
   ```

3. **Run the Application**  
   ```bash
   streamlit run app.py
   ```

4. **Configure API Keys**  
   - Add your Google Cloud API key in the code (`API_KEY` field).  
   - Ensure Ollama's local server is running for chatbot functionality.  

---

## 🔄 Versioning  

- **v1.0.0**: Initial release with features for summarization, translation, NER, sentiment analysis, and chatbot integration.  

---

## 🤝 Contributing  

Contributions are welcome! Fork the repository and submit a pull request with your improvements. For major changes, please open an issue first to discuss what you would like to change.  

---

## 📜 License  

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute it with attribution.  

---

## 👩‍💻 Author  

**Satheesh Meadi**  
Master's Student in Data Science | NLP Enthusiast  
📧 Email: smeadi1@umbc.edu  
🌐 GitHub: [GitHub](https://github.com/SATHEESH-MEADI)
📚 LinkedIn: [Satheesh Meadi](https://www.linkedin.com/in/satheesh-meadi/)  

---
