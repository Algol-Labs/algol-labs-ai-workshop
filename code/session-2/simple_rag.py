#!/usr/bin/env python3
"""
Session 2: Simple RAG Implementation
Building a basic RAG system for document-based Q&A
"""

import os
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re

load_dotenv()

class SimpleRAG:
    """
    A simple RAG system for document-based Q&A
    """

    def __init__(self):
        """Initialize the RAG system"""
        print("🔧 Initializing RAG system...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []  # Store document chunks
        self.embeddings = []  # Store embeddings
        self.metadata = []   # Store metadata for each chunk
        print("✅ RAG system ready!")

    def chunk_text(self, text, chunk_size=500, overlap=50):
        """
        Split text into overlapping chunks

        Args:
            text: The text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks
        """
        chunks = []

        # Simple chunking by character count
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk.strip()) > 100:  # Only keep substantial chunks
                chunks.append(chunk.strip())

        return chunks

    def add_document(self, text, metadata=None):
        """
        Add a document to the RAG system

        Args:
            text: Document content
            metadata: Additional information about the document
        """
        print(f"📄 Processing document ({len(text)} characters)...")

        # Chunk the text
        chunks = self.chunk_text(text)

        # Create embeddings for each chunk
        for chunk in chunks:
            chunk_embedding = self.embedder.encode(chunk)
            self.documents.append(chunk)
            self.embeddings.append(chunk_embedding)
            self.metadata.append(metadata or {})

        print(f"✅ Added {len(chunks)} chunks from document")

    def load_text_file(self, file_path):
        """Load a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None

    def load_pdf_file(self, file_path):
        """Load a PDF file"""
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
            return content
        except Exception as e:
            print(f"❌ Error loading PDF {file_path}: {e}")
            return None

    def find_similar_chunks(self, query, top_k=3):
        """
        Find the most similar document chunks to a query

        Args:
            query: User question
            top_k: Number of chunks to return
        """
        # Encode the query
        query_embedding = self.embedder.encode(query)

        # Calculate similarities
        similarities = cosine_similarity(
            [query_embedding], self.embeddings
        )[0]

        # Get indices of most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [(self.documents[i], self.metadata[i], similarities[i])
                for i in top_indices]

    def query(self, question, top_k=3):
        """
        Answer a question using RAG

        Args:
            question: User's question
            top_k: Number of relevant chunks to use
        """
        print(f"🤔 Answering: {question}")

        # Find similar chunks
        similar_chunks = self.find_similar_chunks(question, top_k)

        if not similar_chunks:
            return "❌ No relevant information found in documents"

        # Build context from similar chunks
        context = "\n\n".join([chunk for chunk, _, _ in similar_chunks])

        # Create the prompt for LLM
        prompt = f"""
بناءً على المعلومات التالية، أجب على السؤال باللغة العربية:

المعلومات المتاحة:
{context}

السؤال: {question}

يرجى تقديم إجابة مفيدة ودقيقة بناءً على المعلومات المتاحة فقط.
"""

        # Call the LLM
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "أنت مساعد مفيد يجيب باللغة العربية بناءً على المعلومات المقدمة فقط."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for factual responses
                max_tokens=300
            )

            answer = response.choices[0].message.content

            # Add source information
            sources = [metadata.get('source', 'Unknown') for _, metadata, _ in similar_chunks]
            answer += f"\n\n📚 المصادر: {', '.join(set(sources))}"

            return answer

        except Exception as e:
            return f"❌ خطأ في الاتصال بالذكاء الاصطناعي: {str(e)}"

# Test the RAG system
if __name__ == "__main__":
    # Initialize RAG system
    rag = SimpleRAG()

    # Add a sample document (you can replace this with your own files)
    sample_text = """
دليل خدمة العملاء - الأردن

كيفية التعامل مع العملاء الغاضبين:
1. استمع بعناية لشكوى العميل دون مقاطعة
2. اعتذر عن أي إزعاج تسببنا فيه
3. اسأل أسئلة لفهم المشكلة بوضوح
4. قدم حلول عملية وسريعة
5. تابع مع العميل للتأكد من رضاه

سياسة الإرجاع:
- يمكن إرجاع المنتجات خلال 14 يوم من الشراء
- يجب أن تكون المنتجات في حالتها الأصلية
- سيتم استرداد المبلغ خلال 3-5 أيام عمل

خدمة التوصيل:
- توصيل مجاني للطلبات فوق 50 دينار
- نفس اليوم في عمان والزرقاء
- 1-2 يوم عمل لباقي المحافظات
"""

    rag.add_document(sample_text, {"source": "customer_service_guide.txt"})

    # Test queries
    test_questions = [
        "كيف أتعامل مع عميل غاضب؟",
        "ما هي سياسة الإرجاع؟",
        "كم يستغرق التوصيل في إربد؟"
    ]

    for question in test_questions:
        print("\n" + "="*60)
        answer = rag.query(question)
        print(f"❓ السؤال: {question}")
        print(f"💡 الإجابة: {answer}")
        print("="*60)
