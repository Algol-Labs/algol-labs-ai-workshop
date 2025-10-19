#!/usr/bin/env python3
"""
Understanding LLM Parameters Through Experimentation
"""

import openai
import os
import sys
from dotenv import load_dotenv

# Ensure proper UTF-8 encoding for Arabic text
if sys.stdout.encoding != 'utf-8':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

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

def experiment_with_parameters(prompt, parameter_name, values):
    """Test the same prompt with different parameter values"""

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"\n🔬 Experimenting with {parameter_name}")
    safe_print(f"Prompt: {ensure_rtl_display(prompt)}")
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

            safe_print(f"{parameter_name}={value}: {ensure_rtl_display(result)}")

        except Exception as e:
            print(f"{parameter_name}={value}: ERROR - {str(e)}")

def experiment_temperature_effects():
    """Show how temperature affects response creativity"""

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = "اكتب وصف قصير لتطبيق جوال يساعد الناس في الأردن"

    print("\n🎨 Temperature Effects on Creativity")
    safe_print(f"Prompt: {ensure_rtl_display(prompt)}")
    print("-" * 60)

    for temp in [0.0, 0.3, 0.7, 1.0]:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=80
            )
            result = response.choices[0].message.content.strip()
            safe_print(f"Temperature {temp}: {ensure_rtl_display(result)}")
        except Exception as e:
            print(f"Temperature {temp}: ERROR - {str(e)}")

def experiment_token_limits():
    """Show how max_tokens affects response length"""

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = "شرح مفهوم الذكاء الاصطناعي باللغة العربية"

    print("\n📏 Token Limit Effects")
    safe_print(f"Prompt: {ensure_rtl_display(prompt)}")
    print("-" * 60)

    for tokens in [20, 50, 100, 200]:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=tokens,
                temperature=0.3
            )
            result = response.choices[0].message.content.strip()
            actual_tokens = len(result.split())  # Rough token count
            safe_print(f"Max tokens {tokens}: {ensure_rtl_display(result)} ({actual_tokens} words)")
        except Exception as e:
            print(f"Max tokens {tokens}: ERROR - {str(e)}")

if __name__ == "__main__":
    # Test prompt for experimentation
    test_prompt = "اكتب جملة قصيرة باللغة العربية عن التكنولوجيا"
    safe_print(f"\n🧪 Testing with: {ensure_rtl_display(test_prompt)}")

    # Experiment with temperature (0.0 to 1.0)
    temperatures = [0.0, 0.3, 0.7, 1.0]
    experiment_with_parameters(test_prompt, "temperature", temperatures)

    # Experiment with max_tokens
    max_tokens_values = [20, 50, 100]
    experiment_with_parameters(test_prompt, "max_tokens", max_tokens_values)

    # Additional experiments
    experiment_temperature_effects()
    experiment_token_limits()

    print("\n✅ Parameter experimentation complete!")
    print("\n💡 Key insights:")
    print("- Lower temperature (0.0-0.3): More consistent, focused responses")
    print("- Higher temperature (0.7-1.0): More creative, varied responses")
    print("- max_tokens limits response length")
    print("- Balance creativity vs consistency based on your use case")
