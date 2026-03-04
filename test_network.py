
import requests
sym = "btc"
url = f"https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/32/color/{sym}.png"
try:
    resp = requests.get(url, timeout=5)
    print(f"Status: {resp.status_code}")
    print(f"Content length: {len(resp.content)}")
except Exception as e:
    print(f"Error: {e}")
