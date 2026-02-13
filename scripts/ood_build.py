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


openphish_urls: list[str] = []
try:
    data = fetch("https://openphish.com/feed.txt")
    openphish_urls = [
        line.strip()
        for line in data.decode("utf-8", "ignore").splitlines()
        if line.strip().startswith(("http://", "https://"))
    ]
except Exception as exc:
    print("OpenPhish download failed:", exc)

phishtank_rows: list[dict[str, str]] = []
try:
    data = fetch("http://data.phishtank.com/data/online-valid.csv")
    reader = csv.DictReader(io.StringIO(data.decode("utf-8", "ignore")))
    for row in reader:
        url = (row.get("url") or "").strip()
        if not url:
            continue
        phishtank_rows.append({"url": url, "time": row.get("submission_time", "")})
except Exception as exc:
    print("PhishTank download failed:", exc)

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

phish_urls = list(dict.fromkeys(openphish_urls + [row["url"] for row in phishtank_rows]))
max_phish = min(len(phish_urls), 5000)
phish_urls = phish_urls[:max_phish]

benign_urls = ["https://" + domain for domain in tranco_domains[:max_phish]]

rows: list[dict[str, int | str]] = [{"url": url, "label": 1, "time": ""} for url in phish_urls]
rows.extend({"url": url, "label": 0, "time": ""} for url in benign_urls)

with (out_dir / "ood_combined.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["url", "label", "time"])
    writer.writeheader()
    writer.writerows(rows)

meta = {
    "openphish_count": len(openphish_urls),
    "phishtank_count": len(phishtank_rows),
    "tranco_count": len(tranco_domains),
    "phish_used": len(phish_urls),
    "benign_used": len(benign_urls),
}
(out_dir / "ood_sources.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("OOD dataset written:", out_dir / "ood_combined.csv", "rows", len(rows))
