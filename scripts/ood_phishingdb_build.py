from __future__ import annotations

import csv
import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path

random.seed(42)
out_dir = Path("data/ood")
out_dir.mkdir(parents=True, exist_ok=True)

headers = {"User-Agent": "gojo-ood-eval"}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


phishing_urls: list[str] = []
try:
    data = fetch("https://phish.co.za/latest/phishing-links-ACTIVE.txt")
    phishing_urls = [
        line.strip()
        for line in data.decode("utf-8", "ignore").splitlines()
        if line.strip().startswith(("http://", "https://"))
    ]
except Exception as exc:
    print("Phishing.Database download failed:", exc)

tranco_domains: list[str] = []
try:
    data = fetch("https://tranco-list.eu/top-1m.csv.zip")
    zf = zipfile.ZipFile(io.BytesIO(data))
    name = zf.namelist()[0]
    for line in zf.read(name).decode("utf-8", "ignore").splitlines():
        parts = line.split(",")
        if len(parts) >= 2:
            tranco_domains.append(parts[1].strip())
except Exception as exc:
    print("Tranco download failed:", exc)

phishing_urls = list(dict.fromkeys(phishing_urls))
max_phish = min(len(phishing_urls), 5000)
phishing_urls = phishing_urls[:max_phish]

benign_urls = ["https://" + domain for domain in tranco_domains[:max_phish]]

rows: list[dict[str, int | str]] = [{"url": url, "label": 1, "time": ""} for url in phishing_urls]
rows.extend({"url": url, "label": 0, "time": ""} for url in benign_urls)

with (out_dir / "phishingdb_ood.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["url", "label", "time"])
    writer.writeheader()
    writer.writerows(rows)

meta = {
    "phishingdb_count": len(phishing_urls),
    "tranco_count": len(tranco_domains),
    "phish_used": len(phishing_urls),
    "benign_used": len(benign_urls),
}
(out_dir / "phishingdb_ood_sources.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("OOD dataset written:", out_dir / "phishingdb_ood.csv", "rows", len(rows))
