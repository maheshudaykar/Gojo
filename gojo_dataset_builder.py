import os
import urllib.request
import zipfile
import csv
import random

# ============================
# CONFIGURATION
# ============================

PHISHING_TARGET = 25000
BENIGN_TARGET = 25000
OUTPUT_FILE = "gojo_dataset_v1.csv"
TEMP_DIR = "gojo_temp"

os.makedirs(TEMP_DIR, exist_ok=True)

# ============================
# DOWNLOAD FUNCTIONS
# ============================

def download_file(url, filename):
    path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(path):
        print(f"[+] Already exists: {filename}")
        return path

    print(f"[+] Downloading: {url}")
    
    # Needs user agent for Tranco
    req = urllib.request.Request(
        url, 
        data=None, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    )
    
    with urllib.request.urlopen(req, timeout=60) as response, open(path, 'wb') as out_file:
        out_file.write(response.read())

    return path

# ============================
# PHISHTANK
# ============================

def load_phishtank():
    url = "https://data.phishtank.com/data/online-valid.csv"
    path = download_file(url, "phishtank.csv")

    urls = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        try:
            next(reader) # skip header
        except:
            pass
        for row in reader:
            if len(row) > 1:
                urls.append(row[1])

    print(f"[+] PhishTank URLs: {len(urls)}")
    return urls

# ============================
# OPENPHISH
# ============================

def load_openphish():
    url = "https://openphish.com/feed.txt"
    path = download_file(url, "openphish.txt")

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"[+] OpenPhish URLs: {len(urls)}")
    return urls

# ============================
# URLHAUS
# ============================

def load_urlhaus():
    url = "https://urlhaus.abuse.ch/downloads/csv_online/"
    path = download_file(url, "urlhaus.csv")

    urls = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and not row[0].startswith('#') and len(row) > 2:
                urls.append(row[2])

    print(f"[+] URLHaus URLs: {len(urls)}")
    return urls

# ============================
# TRANCO BENIGN
# ============================

def load_tranco():
    url = "https://tranco-list.eu/top-1m.csv.zip"
    path = download_file(url, "tranco.zip")

    urls = []
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            lines = f.read().decode('utf-8', errors='ignore').splitlines()
            for line in lines:
                parts = line.split(',')
                if len(parts) > 1:
                    urls.append("http://" + parts[1].strip())

    print(f"[+] Tranco domains: {len(urls)}")
    return urls

# ============================
# CLEANING
# ============================

def clean_urls(urls):
    cleaned = set()
    print("Cleaning URLs...")
    for url in urls:
        if isinstance(url, str) and len(url) > 5:
            cleaned.add(url.strip().lower())

    return list(cleaned)

# ============================
# BUILD DATASET
# ============================

def build_dataset():

    phishing = []
    benign = []

    try:
        phishing.extend(load_phishtank())
    except Exception as e:
        print(f"Failed PhishTank: {e}")
        
    try:
        phishing.extend(load_openphish())
    except Exception as e:
        print(f"Failed OpenPhish: {e}")
        
    try:
        phishing.extend(load_urlhaus())
    except Exception as e:
        print(f"Failed URLHaus: {e}")

    try:
        benign.extend(load_tranco())
    except Exception as e:
        print(f"Failed Tranco: {e}")

    phishing = clean_urls(phishing)
    benign = clean_urls(benign)

    print(f"[+] Unique phishing URLs: {len(phishing)}")
    print(f"[+] Unique benign URLs: {len(benign)}")

    # Remove overlap
    benign = list(set(benign) - set(phishing))

    print(f"[+] Benign after overlap removal: {len(benign)}")

    # Sample target size
    phishing_sample = random.sample(
        phishing,
        min(PHISHING_TARGET, len(phishing))
    )

    benign_sample = random.sample(
        benign,
        min(BENIGN_TARGET, len(benign))
    )

    data = []

    for url in phishing_sample:
        data.append((url, "phish"))

    for url in benign_sample:
        data.append((url, "legit"))

    random.shuffle(data)

    print(f"[+] Writing dataset to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["url", "verdict"])
        for row in data:
            writer.writerow(row)

    print(f"[+] Dataset saved: {OUTPUT_FILE}")
    print(f"[+] Total samples: {len(data)}")


# ============================
# MAIN
# ============================

if __name__ == "__main__":
    build_dataset()
