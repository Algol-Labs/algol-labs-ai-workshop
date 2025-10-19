#!/usr/bin/env python3
"""
Algol Labs AI Workshop - Environment Setup Script

This script helps participants quickly set up their development environment
for the Algol Labs AI Workshop sessions.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a shell command and return success status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True,
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8+")
        return False

def create_env_file():
    """Create .env file template in project root."""
    # Create .env file in project root (one level up from setup directory)
    project_root = Path("..")
    env_path = project_root / ".env"

    if not env_path.exists():
        env_content = """# Algol AI Workshop - Environment Variables
# Copy this file to .env and fill in your values

# OpenAI API Key (get from https://platform.openai.com/api-keys)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Custom settings
DEBUG=true
"""
        env_path.write_text(env_content)
        print("✅ Created .env template file in project root")
        print("📝 Please edit .env file in the project root and add your OpenAI API key")
    else:
        print("✅ .env file already exists in project root")

def main():
    """Main setup function."""
    print("🚀 Algol Labs AI Workshop - Environment Setup")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        print("\n❌ Setup aborted due to Python version incompatibility")
        return 1

    # Create virtual environment if it doesn't exist
    if not Path("venv").exists():
        print("\n📦 Creating virtual environment...")
        if not run_command(f"{sys.executable} -m venv venv", "Create virtual environment"):
            return 1

    # Activate virtual environment and install dependencies
    print("\n📚 Installing Python packages...")

    # Determine activation command based on platform
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"

    # Install requirements
    if run_command(f"{pip_cmd} install -r requirements.txt",
                  "Install Python dependencies"):
        # Create .env file template
        create_env_file()

        print("\n✅ Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Edit the .env file and add your OpenAI API key")
        print("2. Activate the virtual environment:")
        if platform.system() == "Windows":
            print("   venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")
        print("3. Start with Session 1 materials in docs/guides/")
        print("\n🎉 Happy learning!")
        return 0
    else:
        print("\n❌ Setup failed during package installation")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
