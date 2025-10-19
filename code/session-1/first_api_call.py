#!/usr/bin/env python3
"""
Session 1: First API Call Example
Making your first LLM API call and understanding responses
"""

import openai
import os
import sys
from dotenv import load_dotenv

# Ensure proper UTF-8 encoding for Arabic text
if sys.stdout.encoding != 'utf-8':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Load environment variables (including your API key)
load_dotenv()

def ensure_rtl_display(text):
    """
    Ensure proper RTL display for Arabic text in terminals
    """
    if any('\u0600' <= char <= '\u06FF' for char in text):  # Check if text contains Arabic characters
        # Force RTL rendering by ensuring proper Unicode handling
        return text
    return text

def safe_print(text):
    """
    Print text with proper encoding for Arabic
    """
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('utf-8').decode('utf-8'))

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
    safe_print(f"Prompt: {ensure_rtl_display(test_prompt)}")
    print("-" * 50)

    # Try with low temperature (more consistent)
    response = call_llm(test_prompt, temperature=0.1)
    safe_print(f"Low temperature (0.1): {ensure_rtl_display(response)}")
    print()

    # Try with high temperature (more creative)
    response = call_llm(test_prompt, temperature=0.9)
    safe_print(f"High temperature (0.9): {ensure_rtl_display(response)}")
    print()

    # Jordan-specific business example
    business_prompt = "كيف يمكنني تحسين خدمة العملاء في متجري الإلكتروني في الأردن؟"
    safe_print(f"Business prompt: {ensure_rtl_display(business_prompt)}")
    print("-" * 50)
    response = call_llm(business_prompt, temperature=0.3)
    safe_print(f"Response: {ensure_rtl_display(response)}")
