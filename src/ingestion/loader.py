import os
import fitz  # PyMuPDF
from langchain_core.documents import Document

def load_documents(data_path="data"):
    docs = []
    files = [f for f in os.listdir(data_path) if f.endswith(".pdf")]
    print("Found PDF files:", files)

    for file in files:
        pdf_path = os.path.join(data_path, file)
        pdf = fitz.open(pdf_path)
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text()
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": file, "page": page_num}
                    )
                )
        pdf.close()

    print(f"Loaded {len(docs)} pages")
    return docs
