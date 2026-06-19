import gradio as gr
import os
import time

from embeddings.embedder import get_embeddings
from vectorstore.faiss_store import FAISSVectorStore
from reranker.cross_encoder import CrossEncoderReranker
from rag.retriever import AdvancedRetriever
from rag.memory import ConversationMemory
from rag.chain import RAGChain

# ==========================================
# Shared RAG Core Initialization
# ==========================================
embeddings_model = get_embeddings()
vectorstore = FAISSVectorStore(embeddings_model)
vectorstore.load_index()

reranker = CrossEncoderReranker()
memory = ConversationMemory()
retriever = AdvancedRetriever(vectorstore, reranker)

rag_chain = RAGChain(retriever, memory)
# ==========================================

custom_css = """
body, .gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #020617 100%) !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

.glass-card {
    background: rgba(30, 41, 59, 0.4) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5) !important;
    padding: 20px !important;
    margin-bottom: 20px !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}

.glass-card:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 40px rgba(0, 0, 0, 0.6) !important;
}

.gradient-text {
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.badge {
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #38bdf8;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 8px;
    display: inline-block;
}

/* Progress bar for similarity score */
.sim-bar-container {
    width: 100%;
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    margin-top: 5px;
    margin-bottom: 15px;
}
.sim-bar {
    height: 8px;
    border-radius: 8px;
    background: linear-gradient(90deg, #38bdf8, #c084fc);
}

.chunk-text {
    background: rgba(15, 23, 42, 0.5);
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #818cf8;
    font-style: italic;
}

/* Fix chat height and remove default borders */
#chatbot {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
    height: 600px !important;
}
"""

def update_metrics_html(time_taken=0.0):
    num_docs = len(rag_chain.indexed_documents)
    vs = rag_chain.retriever.vectorstore.vectorstore
    num_vectors = vs.index.ntotal if vs else 0
    
    html = f"""
    <div class="glass-card">
        <h3 style='margin-top: 0; color: #38bdf8;'>System Metrics</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
            <div>
                <p style="color: #94a3b8; font-size: 0.9em; margin-bottom: 2px;">Documents Indexed</p>
                <h2 style="margin: 0; font-size: 1.8em;">{num_docs}</h2>
            </div>
            <div>
                <p style="color: #94a3b8; font-size: 0.9em; margin-bottom: 2px;">Total Vectors</p>
                <h2 style="margin: 0; font-size: 1.8em;">{num_vectors}</h2>
            </div>
            <div>
                <p style="color: #94a3b8; font-size: 0.9em; margin-bottom: 2px;">Embedding Model</p>
                <h3 style="margin: 0; font-size: 1.2em;">BGE Embeddings</h3>
            </div>
            <div>
                <p style="color: #94a3b8; font-size: 0.9em; margin-bottom: 2px;">LLM Model</p>
                <h3 style="margin: 0; font-size: 1.2em;">Groq Llama 3.1 8B</h3>
            </div>
            <div>
                <p style="color: #94a3b8; font-size: 0.9em; margin-bottom: 2px;">Memory State</p>
                <h3 style="margin: 0; font-size: 1.2em;">{len(rag_chain.memory.history)} messages</h3>
            </div>
            <div>
                <p style="color: #94a3b8; font-size: 0.9em; margin-bottom: 2px;">Last Response Time</p>
                <h3 style="margin: 0; font-size: 1.2em;">{time_taken:.2f}s</h3>
            </div>
        </div>
    </div>
    """
    return html

def upload_files(files):
    if not files:
        return "No files selected.", update_metrics_html()
        
    try:
        filepaths = [f.name for f in files]
        result = rag_chain.index_documents(filepaths)
        msg = f"✅ {result['message']}"
        return msg, update_metrics_html()
    except Exception as e:
        return f"❌ Error: {str(e)}", update_metrics_html()

def clear_index():
    try:
        rag_chain.retriever.vectorstore.vectorstore = None
        rag_chain.indexed_documents = []
        return "✅ Index cleared from memory.", update_metrics_html()
    except Exception as e:
        return f"❌ Error: {str(e)}", update_metrics_html()

def ask_question(query, history):
    if not query.strip():
        yield history, "<div class='glass-card'>Please enter a question.</div>", update_metrics_html()
        return
        
    if rag_chain.retriever.vectorstore.vectorstore is None:
        history.append((query, "❌ Error: No documents indexed. Please upload documents first."))
        yield history, "<div class='glass-card'>No documents indexed.</div>", update_metrics_html()
        return
        
    history.append((query, ""))
    
    try:
        start_time = time.time()
        response_data = rag_chain.ask(query, stream=True)
        
        context_docs = response_data["context_docs"]
        sq = response_data["standalone_query"]
        
        # Format Explainability
        context_display = f"""
        <div class='glass-card'>
            <h3 style='margin-top: 0; color: #38bdf8;'>Retrieval Context</h3>
            <p><strong>Rewritten Query:</strong> {sq}</p>
            <p><strong>Chunks Retrieved:</strong> {len(context_docs)}</p>
        </div>
        """
        
        for i, (doc, score) in enumerate(context_docs):
            source = doc.metadata.get('filename', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            # Assuming score is cosine similarity or we normalize it roughly. 
            # We'll just display a bar based on the score * 100 if it's <=1, else bounded.
            bar_width = max(0, min(100, score * 100)) if score <= 1 else 100
            
            context_display += f"""
            <div class='glass-card'>
                <h4>Chunk {i+1} <span style='color: #94a3b8; font-weight: normal;'>(Source: {source}, Page: {page})</span></h4>
                <p style='margin-bottom: 2px;'>Similarity Score: {score:.4f}</p>
                <div class='sim-bar-container'><div class='sim-bar' style='width: {bar_width}%;'></div></div>
                <div class='chunk-text'>{doc.page_content}</div>
            </div>
            """
            
        full_answer = ""
        streamer = response_data["streamer"]
        
        for text in streamer:
            full_answer += text
            history[-1] = (query, full_answer)
            yield history, context_display, update_metrics_html(time.time() - start_time)
            
        rag_chain.memory.add_user_message(query)
        rag_chain.memory.add_assistant_message(full_answer.strip())
        
        time_taken = time.time() - start_time
        yield history, context_display, update_metrics_html(time_taken)
                        
    except Exception as e:
        history[-1] = (query, f"Error: {str(e)}")
        yield history, f"<div class='glass-card'>An error occurred: {str(e)}</div>", update_metrics_html()

def generate_summary_ui():
    try:
        summary = rag_chain.summarize()
        summary_html = summary.replace('\n', '<br>')
        html = f"""
        <div class="glass-card">
            <h2 style='margin-top: 0; color: #38bdf8;'>Executive Summary</h2>
            <div style="line-height: 1.6;">
                {summary_html}
            </div>
        </div>
        """
        return html
    except Exception as e:
        return f"<div class='glass-card' style='color: #ef4444;'>Error: {str(e)}</div>"

def clear_chat():
    rag_chain.memory.clear()
    return [], update_metrics_html()

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_width="0px",
)

with gr.Blocks(theme=theme, css=custom_css, title="MedRAG AI") as demo:
    # Header
    gr.HTML("""
    <div style='text-align: center; margin-bottom: 30px; margin-top: 20px;'>
        <h1 class='gradient-text' style='font-size: 3.5rem; font-weight: 800; margin-bottom: 10px; line-height: 1.2;'>🏥 MedRAG AI</h1>
        <h3 style='color: #94a3b8; font-weight: 400; margin-top: 0;'>Production-Grade Medical Report Understanding System</h3>
        <div style='margin-top: 20px;'>
            <span class='badge'>🤖 Groq API</span>
            <span class='badge'>🔍 Semantic Search</span>
            <span class='badge'>⚡ Reranking</span>
            <span class='badge'>📚 Multi-PDF</span>
            <span class='badge'>🧠 History-Aware Retrieval</span>
            <span class='badge'>🔬 Explainable AI</span>
        </div>
    </div>
    """)
    
    with gr.Tabs():
        with gr.TabItem("📄 Upload Documents"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Index your medical documents securely")
                    file_upload = gr.File(file_count="multiple", file_types=[".pdf"], label="Drag and drop PDFs here")
                    with gr.Row():
                        upload_button = gr.Button("Index Documents", variant="primary")
                        clear_index_button = gr.Button("Clear Index")
                    upload_status = gr.Textbox(label="Status", interactive=False)
                with gr.Column(scale=1):
                    upload_metrics_panel = gr.HTML(value=update_metrics_html())
            
        with gr.TabItem("💬 Chat Assistant"):
            chatbot = gr.Chatbot(elem_id="chatbot")
            msg = gr.Textbox(label="Ask questions about your medical reports...", placeholder="e.g., What does the report say about pneumonia?")
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear Chat")
                
        with gr.TabItem("📋 Report Summary"):
            gr.Markdown("### Comprehensive Summary")
            summary_button = gr.Button("Generate Full Summary", variant="primary")
            summary_output = gr.HTML("<div class='glass-card'>Click the button above to generate a full executive summary.</div>")
            
        with gr.TabItem("🔬 Explainability"):
            gr.Markdown("### Retrieval Confidence & Context")
            gr.Markdown("*Note: Similarity scores represent retrieval similarity and not true model confidence.*")
            context_panel = gr.HTML(value="<div class='glass-card'>Ask a question to see retrieved chunks here.</div>")
            
        with gr.TabItem("📊 Metrics"):
            metrics_panel = gr.HTML(value=update_metrics_html())

    # Footer
    gr.HTML("""
    <div style='text-align: center; margin-top: 50px; padding: 20px; border-top: 1px solid rgba(255,255,255,0.1); color: #64748b;'>
        <small><strong>Medical Disclaimer:</strong> This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.</small>
    </div>
    """)

    # Event Handlers
    upload_button.click(upload_files, inputs=[file_upload], outputs=[upload_status, upload_metrics_panel])
    upload_button.click(update_metrics_html, outputs=[metrics_panel])
    
    clear_index_button.click(clear_index, outputs=[upload_status, upload_metrics_panel])
    clear_index_button.click(update_metrics_html, outputs=[metrics_panel])
    
    # We update metrics on both the chat page and metrics page
    msg.submit(ask_question, inputs=[msg, chatbot], outputs=[chatbot, context_panel, metrics_panel])
    submit_btn.click(ask_question, inputs=[msg, chatbot], outputs=[chatbot, context_panel, metrics_panel])
    
    clear_btn.click(clear_chat, outputs=[chatbot, metrics_panel])
    
    summary_button.click(generate_summary_ui, outputs=[summary_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
