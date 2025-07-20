# app.py
import os
from dotenv import load_dotenv
import gradio as gr
from deep_translator import GoogleTranslator
from rag_pipeline import create_vectorstore, get_qa_chain

# Load .env
load_dotenv()

# Build vector DB if not already there
if not os.path.exists("hygiene_vector_db/index"):
    create_vectorstore()

qa_chain = get_qa_chain()

def hygiene_bot(user_input):
    try:
        # Translate to English
        translated_input = GoogleTranslator(source='auto', target='en').translate(user_input)

        # Run RAG chain
        english_response = qa_chain.run(translated_input)

        # Translate back to Bangla
        bangla_response = GoogleTranslator(source='auto', target='bn').translate(english_response)

        return f"English:\n{english_response}\n\nBangla:\n{bangla_response}"

    except Exception as e:
        return f"Error: {e}"

gr.Interface(
    fn=hygiene_bot,
    inputs="text",
    outputs="text",
    title="Hygiene Assistant (RAG + GPT-4o)",
    description="Ask hygiene-related questions in Bangla or English. Answers come from your hygiene knowledge base!",
).launch(share=True)
