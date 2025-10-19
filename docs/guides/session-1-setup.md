# Session 1: AI/LLM Fundamentals - Setup Guide

## Welcome to Session 1!

Today we'll explore the fundamentals of AI and Large Language Models through hands-on experimentation. By the end of this session, you'll be comfortable making API calls to LLMs and understand how to control their behavior with parameters.

## What You'll Learn

- **AI/LLM Basics**: Understanding how these systems work
- **API Integration**: Making your first LLM API calls
- **Parameter Tuning**: Controlling creativity vs consistency
- **Practical Applications**: Real-world use cases for Jordanian businesses

## Prerequisites

- **Python 3.8+** installed on your system
- **VS Code** (or your preferred editor)
- **Git** (for cloning this repository)
- **OpenAI API Key** (free tier available)

## Step 1: Environment Setup

Let's get your development environment ready:

### 1.1 Clone the Repository

```bash
git clone <repository-url>
cd algol-ai-workshop
```

### 1.2 Set Up Python Environment

Run the automated setup script:

```bash
cd setup
python setup.py
```

This will:
- ✅ Check your Python version
- ✅ Create a virtual environment
- ✅ Install all required packages
- ✅ Create a `.env` file template

### 1.3 Get Your OpenAI API Key

1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Click "Create new secret key"
3. Copy the key (you won't see it again!)
4. Edit the `.env` file in your project root:
   ```bash
   # Open in VS Code
   code .env
   ```
5. Replace `your_openai_api_key_here` with your actual key:
   ```env
   OPENAI_API_KEY=sk-your-actual-key-here
   ```

## Step 2: Your First API Call

Let's make your first LLM API call! Open VS Code and create a new Python file:

```bash
code code/session-1/first_api_call.py
```

Copy and paste this code:

```python
#!/usr/bin/env python3
"""
Session 1: First API Call Example
Making your first LLM API call and understanding responses
"""

import openai
import os
from dotenv import load_dotenv

# Load environment variables (including your API key)
load_dotenv()

def call_llm(prompt, temperature=0.7):
    """
    Make a call to OpenAI's GPT model

    Args:
        prompt: The text prompt to send to the model
        temperature: Controls randomness (0.0 = consistent, 1.0 = creative)

    Returns:
        The model's response text
    """
    try:
        # Initialize the OpenAI client
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Make the API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using the affordable model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=150  # Limit response length for cost control
        )

        # Extract and return the response text
        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

# Test your first API call
if __name__ == "__main__":
    # Simple test prompt
    test_prompt = "Hello! Can you tell me what AI is in 2 sentences?"

    print("🤖 Making your first API call...")
    print(f"Prompt: {test_prompt}")
    print("-" * 50)

    # Try with low temperature (more consistent)
    response = call_llm(test_prompt, temperature=0.1)
    print(f"Low temperature (0.1): {response}")
    print()

    # Try with high temperature (more creative)
    response = call_llm(test_prompt, temperature=0.9)
    print(f"High temperature (0.9): {response}")
    print()

    # Jordan-specific business example
    business_prompt = "كيف يمكنني تحسين خدمة العملاء في متجري الإلكتروني في الأردن؟"
    print(f"Business prompt: {business_prompt}")
    print("-" * 50)
    response = call_llm(business_prompt, temperature=0.3)
    print(f"Response: {response}")
```

## Step 3: Run Your Code

In VS Code terminal (or your terminal):

```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Run your first API call
python code/session-1/first_api_call.py
```

## Expected Output

You should see something like:
```
🤖 Making your first API call...
Prompt: Hello! Can you tell me what AI is in 2 sentences?
--------------------------------------------------
Low temperature (0.1): AI is a branch of computer science that focuses on creating machines capable of performing tasks that typically require human intelligence, such as understanding natural language, recognizing patterns, and making decisions.

High temperature (0.9): Artificial Intelligence represents the cutting-edge intersection of computer science and cognitive simulation, where machines learn to interpret, reason, and interact with their environment in ways that mimic human thought processes.

Business prompt: كيف يمكنني تحسين خدمة العملاء في متجري الإلكتروني في الأردن؟
--------------------------------------------------
Response: لتحسين خدمة العملاء في متجرك الإلكتروني في الأردن، يمكنك:
1. توفير دعم عملاء متعدد القنوات (الهاتف، البريد الإلكتروني، الدردشة المباشرة)
2. ضمان استجابة سريعة للاستفسارات (خلال 24 ساعة)
3. تقديم سياسة إرجاع مرنة وواضحة
4. تخصيص تجربة التسوق حسب احتياجات العملاء المحليين
```

## Step 4: Understanding the Parameters

Let's experiment with different parameters. Create another file:

```bash
code code/session-1/parameter_experiment.py
```

```python
#!/usr/bin/env python3
"""
Understanding LLM Parameters Through Experimentation
"""

import openai
import os
from dotenv import load_dotenv

load_dotenv()

def experiment_with_parameters(prompt, parameter_name, values):
    """Test the same prompt with different parameter values"""

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"\n🔬 Experimenting with {parameter_name}")
    print(f"Prompt: {prompt}")
    print("-" * 60)

    for value in values:
        try:
            # Build parameters dynamically
            params = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.7  # Default temperature
            }

            # Override the parameter we're testing
            if parameter_name == "temperature":
                params["temperature"] = value
            elif parameter_name == "top_p":
                params["top_p"] = value
            elif parameter_name == "max_tokens":
                params["max_tokens"] = value

            response = client.chat.completions.create(**params)
            result = response.choices[0].message.content.strip()

            print(f"{parameter_name}={value}: {result}")

        except Exception as e:
            print(f"{parameter_name}={value}: ERROR - {str(e)}")

if __name__ == "__main__":
    # Test prompt for experimentation
    test_prompt = "اكتب جملة قصيرة باللغة العربية عن التكنولوجيا"

    # Experiment with temperature (0.0 to 1.0)
    temperatures = [0.0, 0.3, 0.7, 1.0]
    experiment_with_parameters(test_prompt, "temperature", temperatures)

    # Experiment with max_tokens
    max_tokens_values = [20, 50, 100]
    experiment_with_parameters(test_prompt, "max_tokens", max_tokens_values)
```

## Step 5: Cost Awareness

Understanding API costs is important:

```python
# In your code, you can check token usage:
response = client.chat.completions.create(...)
print(f"Tokens used: {response.usage.total_tokens}")
print(f"Estimated cost: ${response.usage.total_tokens * 0.002 / 1000:.6f}")
```

**Cost Guidelines:**
- **gpt-3.5-turbo**: $0.002 per 1,000 tokens
- **gpt-4**: $0.03 per 1,000 tokens (use sparingly)
- **Free tier**: $5 credit for new accounts

## Step 6: Error Handling

Let's improve your code with proper error handling:

```python
def call_llm_with_error_handling(prompt, temperature=0.7):
    """Improved LLM call with comprehensive error handling"""

    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        if not client.api_key:
            return "❌ Error: OPENAI_API_KEY not found in environment variables"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=150
        )

        # Check for successful response
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return "❌ Error: No response generated"

    except openai.AuthenticationError:
        return "❌ Error: Invalid API key"
    except openai.RateLimitError:
        return "❌ Error: Rate limit exceeded. Please try again later"
    except openai.APIError as e:
        return f"❌ Error: API error - {str(e)}"
    except Exception as e:
        return f"❌ Error: Unexpected error - {str(e)}"
```

## Next Steps

✅ **You've completed the basic setup!**

📚 **What to explore next:**
1. Try different prompts in Arabic and English
2. Experiment with business scenarios from your work
3. Read through the parameter explanations
4. Prepare questions for the interactive session

## Common Issues & Solutions

**"Invalid API Key"**
- Double-check your API key in the `.env` file
- Make sure there are no extra spaces or characters

**"Rate limit exceeded"**
- You're making too many requests too quickly
- Wait a minute and try again
- Consider using shorter prompts

**Import errors**
- Make sure you've activated the virtual environment
- Run `pip install -r requirements.txt` again if needed

## You're Ready for Session 1! 🎉

You now have:
- ✅ Working Python environment
- ✅ OpenAI API access
- ✅ First successful API call
- ✅ Understanding of key parameters
- ✅ Error handling in place

See you in the workshop session where we'll dive deeper into these concepts and explore real-world applications!
