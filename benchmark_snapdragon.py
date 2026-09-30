import os
from dotenv import load_dotenv
import qai_hub as hub

# Load credentials from .env file
load_dotenv()
api_token = os.getenv("QAI_HUB_API_TOKEN")

if not api_token:
    raise ValueError("❌ QAI_HUB_API_TOKEN not found in .env file!")

print("📡 Connecting to Qualcomm AI Hub Workbench...")
devices = hub.get_devices()
print(f"✅ Connected successfully! Found {len(devices)} device targets.")

# Target Snapdragon X-series PC hardware for the hackathon
target_device = hub.Device("Snapdragon X Elite CRD")
print(f"🎯 Target Device Selected: {target_device.name}")

# List available models or check pipeline readiness
print("🚀 Qualcomm AI Hub is ready to compile and benchmark your local models.")
