# Algol AI Workshop - Detailed Plan

## Workshop Overview

**Two 2-hour hands-on sessions** designed for Jordanian engineers with PHP, .NET, or basic Python experience. Focus on practical, immediately applicable AI skills that add value to local businesses.

### Target Audience
- Engineers familiar with PHP, .NET, or Python
- No prior AI/LLM experience required
- Comfortable with basic programming concepts
- Interested in adding AI capabilities to existing applications

### Learning Philosophy
- **Hands-on first**: Every concept includes practical exercises
- **Jordan-focused**: Examples relevant to local business needs
- **Progressive difficulty**: Start simple, build to complex applications
- **Production-ready**: Use modern, maintainable patterns

## Session 1: AI/LLM Fundamentals (2 hours)

### Hour 1: Understanding AI/LLMs (Theory + Concepts)

**Objective**: Build mental models for how AI/LLMs work

**Topics Covered:**
- What is LLM AI vs Machine Learning vs Deep Learning?
- How Large Language Models work (tokens, transformers, attention)
- Prompt engineering basics
- Temperature and creativity vs consistency
- API parameters that matter (temperature, topK, max_tokens)

**Activities:**
1. **Interactive Demo** (15 min): Show same prompt with different temperatures
2. **Group Discussion** (10 min): "How could this help my current projects?"
3. **Parameter Exploration** (20 min): Hands-on experimentation with parameters

**Key Takeaway**: "AI is a tool I can control with the right parameters"

### Hour 2: First API Calls (Hands-on Practice)

**Objective**: Make participants' first successful LLM API call

**Topics Covered:**
- OpenAI API setup and authentication
- Basic API call structure
- Understanding responses and errors
- Parameter tuning in practice
- Cost awareness and free tier limits

**Hands-on Exercises:**
1. **Setup Exercise** (10 min): Get API key and make first call
2. **Parameter Playground** (30 min): Experiment with temperature, topK, max_tokens
3. **Real-world Application** (20 min): Apply to a business scenario

**Code Example Structure:**
```python
# Simple, clear examples that work immediately
import openai
import os
from dotenv import load_dotenv

load_dotenv()

def call_llm(prompt, temperature=0.7):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=150
    )

    return response.choices[0].message.content

# Jordan-specific example
business_prompt = "كيف يمكنني تحسين خدمة العملاء في متجري الإلكتروني؟"
result = call_llm(business_prompt, temperature=0.3)
```

**Expected Outcomes:**
- ✅ Successful API call without errors
- ✅ Understanding of key parameters (temperature, topK)
- ✅ Confidence to experiment independently

---

## Session 2: Building with LLMs - RAG Implementation (2 hours)

### Hour 1: RAG Concepts and Setup (Theory + Setup)

**Objective**: Understand why and how RAG improves AI applications

**Topics Covered:**
- What is RAG (Retrieval-Augmented Generation)?
- Why RAG matters for business applications
- Document processing and chunking
- Embedding and similarity search
- Local vs cloud approaches

**Activities:**
1. **RAG Demo** (15 min): Show same question with/without relevant context
2. **Document Analysis** (15 min): Break down how RAG processes documents
3. **Setup Exercise** (30 min): Install and configure RAG components

**Key Takeaway**: "RAG makes AI answer from my specific knowledge, not just general training data"

### Hour 2: Building a Simple RAG System (Hands-on Implementation)

**Objective**: Build a working RAG system with local documents

**Topics Covered:**
- Document loading and preprocessing
- Text chunking strategies
- Embedding generation
- Similarity search implementation
- RAG query processing

**Hands-on Exercises:**
1. **Document Processing** (15 min): Load and chunk local files
2. **Embedding Setup** (20 min): Generate embeddings for documents
3. **Query System** (25 min): Build and test RAG queries

**Code Example Structure:**
```python
# Building on Session 1 patterns
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimpleRAG:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = []

    def add_document(self, text, metadata=None):
        # Chunk text intelligently
        chunks = self.chunk_text(text)
        for chunk in chunks:
            self.documents.append({"text": chunk, "metadata": metadata})
            self.embeddings.append(self.embedder.encode(chunk))

    def query(self, question, top_k=3):
        # Embed question and find similar chunks
        question_embedding = self.embedder.encode(question)
        similarities = cosine_similarity(
            [question_embedding], self.embeddings
        )[0]

        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Use chunks as context for LLM call
        context = "\n".join([self.documents[i]["text"] for i in top_indices])
        return self.call_llm_with_context(question, context)

# Jordan business example - customer service knowledge base
rag = SimpleRAG()
with open("customer_service_guide.txt", "r", encoding="utf-8") as f:
    rag.add_document(f.read(), {"source": "customer_guide"})

answer = rag.query("كيف أتعامل مع عميل غاضب؟")
```

**Expected Outcomes:**
- ✅ Working RAG system with local documents
- ✅ Understanding of embedding and similarity concepts
- ✅ Ability to add new documents and query them

---

## Technical Approach

### Why This Works for Jordanian Engineers

1. **Familiar Patterns**: Uses Python like their existing work
2. **Immediate Value**: RAG can enhance existing PHP/.NET applications
3. **Practical Focus**: Every concept ties to business use cases
4. **Production Ready**: Uses modern, maintainable patterns

### Technology Choices

- **Python 3.8+**: Widely available, familiar to many
- **OpenAI API**: Free tier, excellent for learning
- **Local Processing**: Privacy-friendly, works offline
- **VS Code**: Already used by participants
- **No Complex Dependencies**: Everything installs with pip

### Business Applications for Jordan

1. **Customer Service**: RAG-powered FAQ systems
2. **Document Processing**: Automated contract analysis
3. **Content Creation**: Marketing copy generation
4. **Code Assistance**: Technical documentation queries
5. **Business Intelligence**: Report analysis and insights

---

## Success Metrics

### For Participants
- **Technical**: Can make LLM API calls and build basic RAG systems
- **Business**: Can identify 2-3 use cases for their current work
- **Confidence**: Comfortable experimenting with AI parameters

### For the Workshop
- **Engagement**: Active participation in exercises
- **Comprehension**: Questions show understanding of concepts
- **Application**: Participants start applying concepts to their work

---

## Materials and Resources

### Repository Structure
- `setup/` - Environment setup scripts
- `code/session-1/` - API call examples and exercises
- `code/session-2/` - RAG implementation examples
- `docs/guides/` - Step-by-step instructions
- `docs/slides/` - Presentation materials

### Additional Resources
- OpenAI API documentation
- Python packaging guides
- VS Code AI extensions
- Local Jordanian tech community resources

---

## Timeline

- **Session 1**: Before mid-November 2025
- **Session 2**: Early January 2026
- **Follow-up**: Optional office hours for continued support

This workshop design leverages your expertise in modern AI systems while making the content accessible and immediately valuable for Jordanian engineers. The focus on RAG addresses the "virgin AI market" you mentioned, providing a technology that can add immediate value to local businesses.
