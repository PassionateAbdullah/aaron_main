"""Test the file loading fix."""
import json
import os

path = r"C:\Users\mamun\Downloads\Project\Aeron\aaron_data.json"

# Try multiple encodings to handle BOM and other issues
with open(path, "rb") as f:
    raw = f.read()

print(f"File size: {len(raw)} bytes")
print(f"First 10 bytes (hex): {raw[:10].hex()}")

# Try to decode with different encodings
for encoding in ("utf-8-sig", "utf-8", "latin-1"):
    try:
        content = raw.decode(encoding)
        data = json.loads(content)
        print(f"\n✓ Successfully loaded with {encoding}")
        print(f"  Keys in dataset: {list(data.keys())[:5]}")
        break
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        print(f"✗ Failed with {encoding}: {type(e).__name__}")
        continue
