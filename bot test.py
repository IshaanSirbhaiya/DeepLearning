import requests

# Put your credentials here, safely inside the quotation marks!
TOKEN = "8622784083:AAG9eU9XMSZQJ0_MC90RYcmqn-P31FAKiPE"
CHAT_ID = "5808527465" # I removed the stray 'E' from the end of your number

# Now Python will safely plug the variables in below
url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
payload = {
    "chat_id": CHAT_ID,
    "text": "🚨 SYSTEM TEST: If you are reading this, your Deep Learning Week evacuation bot is officially online and ready to route! 🚨"
}

print("Firing message to Telegram...")
response = requests.post(url, json=payload)

if response.status_code == 200:
    print("✅ SUCCESS! Check your phone right now.")
else:
    print(f"❌ FAILED. Telegram said: {response.text}")