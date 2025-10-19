# Session 2: Building with LLMs - RAG Implementation

## Welcome to Session 2!

Today we'll build on what you learned in Session 1 and implement **RAG (Retrieval-Augmented Generation)** - one of the most powerful and immediately valuable AI techniques for businesses. RAG allows AI to answer questions based on your specific documents and knowledge, not just general training data.

## What You'll Build

By the end of this session, you'll have a working RAG system that can:
- Load and process local documents (PDFs, text files)
- Answer questions based on your specific content
- Provide accurate, contextual responses for business use cases

## Jordan Business Applications

RAG is perfect for Jordanian businesses because it can:
- **Customer Service**: Answer questions from product manuals or FAQs
- **Document Processing**: Analyze contracts, reports, or legal documents
- **Knowledge Management**: Make company policies and procedures searchable
- **Content Creation**: Generate responses based on your specific brand guidelines

## Prerequisites

- **Session 1 completed** (you know how to make LLM API calls)
- **Python environment** set up from Session 1
- **Sample documents** (we'll provide some, or use your own)

## Step 1: Understanding RAG

Before we code, let's understand how RAG works:

### The Problem with Basic LLM Calls
```python
# Basic LLM call (from Session 1)
response = call_llm("كيف أتعامل مع عميل غاضب؟")
# Response: General advice, may not fit your business context
```

### The RAG Solution
```python
# RAG approach
documents = ["customer_service_guide.txt", "company_policy.txt"]
response = rag_system.query("كيف أتعامل مع عميل غاضب؟", documents)
# Response: Specific advice based on YOUR documents
```

### How RAG Works
1. **Load Documents**: Read your files (PDFs, text, etc.)
2. **Chunk Text**: Break documents into smaller pieces
3. **Create Embeddings**: Convert text to numerical vectors
4. **Find Similar Content**: Match user questions to relevant chunks
5. **Generate Response**: Use similar chunks as context for LLM

## Step 2: Setting Up RAG Components

First, let's create the basic RAG structure:

```bash
# Create Session 2 directory
mkdir -p code/session-2
code code/session-2/simple_rag.py
```

## Step 3: Basic RAG Implementation

Copy this code to start building your RAG system:

```python
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
```

## Step 4: Testing with Your Own Documents

Let's create a sample business document for testing:

```bash
# Create sample documents
cat > sample_documents/customer_service_guide.txt << 'EOF'
دليل خدمة العملاء - متجر إلكتروني أردني

التعامل مع العملاء الغاضبين:
1. ابدأ بالترحيب وتقديم الاعتذار الفوري
2. استمع للمشكلة كاملة دون مقاطعة
3. أعد صياغة المشكلة للتأكد من الفهم
4. قدم حلول متعددة للعميل
5. اختر الحل الأنسب مع العميل
6. تابع النتيجة خلال 24 ساعة

سياسة الإرجاع والاستبدال:
- إرجاع مجاني خلال 30 يوم من الشراء
- المنتجات يجب أن تكون في العبوة الأصلية
- استثناءات: المنتجات المخصصة والكتب المستعملة
- استرداد خلال 5-7 أيام عمل

خدمة التوصيل:
- مجاني للطلبات فوق 25 دينار
- نفس اليوم في عمان (قبل الساعة 2 ظهراً)
- 1 يوم عمل في الزرقاء ومادبا
- 2 يوم عمل في إربد والكرك
- 3 أيام عمل لباقي المحافظات
EOF

cat > sample_documents/company_faq.txt << 'EOF'
الأسئلة الشائعة - متجرنا الإلكتروني

س: كيف أتبع طلبي؟
ج: استخدم رقم الطلب في صفحة "تتبع الطلب" على موقعنا

س: هل تقبلون الدفع عند التسليم؟
ج: نعم، نقبل الدفع النقدي عند التسليم في جميع المدن الرئيسية

س: ما هي طرق الدفع المتاحة؟
ج: نقبل فيزا، ماستركارد، والدفع عند التسليم

س: هل لديكم ضمان على المنتجات؟
ج: نعم، ضمان لمدة عام على جميع المنتجات الإلكترونية

س: كيف أتواصل مع خدمة العملاء؟
ج: هاتف: 0791234567
واتساب: 0777654321
بريد إلكتروني: support@store.jo
ساعات العمل: 9 صباحاً - 9 مساءً يومياً
EOF
```

## Step 5: Advanced RAG Features

Let's enhance your RAG system with more features:

```python
#!/usr/bin/env python3
"""
Advanced RAG Features
"""

# ... existing code ...

def add_multiple_documents(self, file_paths):
    """Add multiple documents at once"""
    for file_path in file_paths:
        if file_path.endswith('.txt'):
            content = self.load_text_file(file_path)
        elif file_path.endswith('.pdf'):
            content = self.load_pdf_file(file_path)
        else:
            print(f"⚠️  Unsupported file type: {file_path}")
            continue

        if content:
            metadata = {"source": file_path}
            self.add_document(content, metadata)

def query_with_confidence(self, question, top_k=3, threshold=0.3):
    """Query with confidence scoring"""
    similar_chunks = self.find_similar_chunks(question, top_k)

    # Filter by similarity threshold
    confident_chunks = [(chunk, meta, sim) for chunk, meta, sim in similar_chunks if sim > threshold]

    if not confident_chunks:
        return "❌ لا توجد معلومات كافية للإجابة على هذا السؤال"

    # Rest of the query logic...
```

## Step 6: Integration with Your Existing Systems

RAG can enhance your current PHP/.NET applications:

```python
# Example: Add RAG to existing customer service system
def integrate_with_php_system(rag_system, question):
    """
    Example of how to integrate RAG with PHP/.NET backend
    """
    # Get answer from RAG
    answer = rag_system.query(question)

    # Format for your existing system
    response = {
        "answer": answer,
        "confidence": "high",  # You can calculate this
        "sources": ["customer_guide", "faq"],
        "timestamp": "2025-10-18"
    }

    return response

# Use in your PHP backend:
# $rag_answer = call_rag_api($question);
// return $rag_answer to frontend
```

## You're Ready for Session 2! 🎉

You now have:
- ✅ Basic RAG system implementation
- ✅ Document loading and processing
- ✅ Similarity search and context building
- ✅ LLM integration with document context
- ✅ Sample documents for testing
- ✅ Ideas for business integration

## Next Steps for the Workshop

1. **Test with your own documents** (company policies, product manuals, etc.)
2. **Experiment with different chunk sizes** and see how they affect results
3. **Try different types of questions** and observe the quality of responses
4. **Think about integration points** with your existing systems

See you in the workshop where we'll dive deeper into these concepts and build more advanced features!
