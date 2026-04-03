import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import gradio as gr
from src.retrieval.query_data import search_db , build_context
from src.retrieval.prompt import generate_answer

def chat(query):
    
    search_results = search_db(query=query)
    
    context , source = build_context(search_results)
    
    answer = generate_answer(context=context , query=query)


    return answer, source


interface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(lines=2, placeholder="Ask a telecom question..."),
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Top Sources")
    ],
    title="📡 TeleRAG Scholar",
    description="Ask questions about telecom research papers (RAG system)"
)

if __name__ == "__main__":
    interface.launch()