# Gojo 🛡️

<p align="center">
    <img src="webapp/static/favicon.png" alt="Gojo Logo" width="200" />
</p>

<div align="center">

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-Personal--Use--Only-orange.svg)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

**Gojo is a production-grade phishing URL detection system powered by machine learning and reinforcement learning.**

[Features](#-features) •
[Quick Start](#-quick-start) •
[Documentation](#-documentation) •
[Contributing](#-contributing)

</div>

---

## 🎯 Overview

**Gojo** is an advanced, production-ready system that identifies phishing URLs using a hybrid approach combining:

- **Heuristic Rules**: Fast, interpretable pattern matching (12 rules)
- **Machine Learning**: Ensemble of lexical features and character n-grams
- **Reinforcement Learning**: Thompson Sampling for adaptive weight selection
- **Contextual Bandits**: Online learning from user feedback

Built for **real-world deployment**, this tool achieves **96% accuracy** while remaining explainable and adaptive.

---

## ✨ Features

### Core Capabilities
- ✅ **Hybrid Detection**: Combines rules, ML, and RL for optimal accuracy
- ✅ **Real-time Analysis**: Process URLs in ~20ms
- ✅ **Bulk Processing**: Handle 10,000+ URLs efficiently
- ✅ **Web Interface**: Beautiful dark-mode UI with detailed explanations
- ✅ **CLI Tool**: Command-line interface for automation
- ✅ **Python API**: Integrate into your applications

### Machine Learning
- 🧠 **Ensemble Models**: Lexical + Character n-gram classifiers
- 🎯 **99%+ Accuracy**: Trained on 50,000 rigorous phishing/legitimate URLs
- 📊 **20 Features**: Entropy, homoglyphs, URL structure, token analysis
- 🔄 **Calibrated Probabilities**: Reliable confidence scores

### Reinforcement Learning (NEW!)
- 🎰 **Thompson Sampling**: Bayesian approach for exploration/exploitation
- 📈 **Context-Aware**: 25 contextual states for fine-grained adaptation
- 📊 **Comprehensive Metrics**: Regret tracking, optimal action rates
- 🔍 **Explainable**: See why the agent chose specific weights

### Production Features
- 🔐 **Input Validation**: Sanitization and security checks
- 📝 **Structured Logging**: JSON logs for monitoring
- 🏥 **Health Endpoints**: `/health` and `/metrics` for observability
- 🔄 **Auto-Shutdown**: Browser lifecycle management
- 🌐 **WSGI Ready**: Deploy with Waitress/Gunicorn
- 🧭 **Domain Enrichment**: RDAP-based age/ASN + DNS volatility signals
- ⚖️ **Cost-Sensitive Rewards**: Tunable FN/FP costs for policy updates

### Advanced Detection Features (NEW!)
- 🌐 **Content Analysis**: HTML/page inspection for credential forms, suspicious JavaScript, and brand impersonation
- 📊 **Drift Detection**: Monitors feature distribution shifts using Population Stability Index (PSI)
- 🎯 **Dynamic TLD Learning**: Real-time learning of suspicious TLD patterns from production traffic
- 🕵️ **Attack Pattern Recognition**: Fingerprints and tracks emerging phishing techniques
- 🔍 **Adaptive Scoring**: Automatically adjusts detection thresholds based on drift signals
- 🛡️ **Multi-Layer Defense**: Combines URL analysis, content inspection, and behavioral patterns

**Note**: Content analysis requires live URLs. Historical/dead phishing URLs return limited content data.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- 8GB RAM recommended

### 1. Clone Repository
```bash
git clone https://github.com/maheshudaykar/Gojo.git
cd gojo
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements_production.txt
```

For development (tests/type/lint):
```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Download Dataset (Optional)
```bash
# Clone dataset repository
git clone https://github.com/Priyanshu88/DatasetWebFraudDetection.git data/DatasetWebFraudDetection
```

### 5. Train Models
```bash
python -m phish_detector.train --data data/DatasetWebFraudDetection/dataset.csv --url-col url --label-col verdict
```

### 6. Launch Web UI
```bash
# Windows
run_gojo.bat

# Linux/Mac
python -m webapp.app
```

Visit **http://127.0.0.1:5000** 🎉

### 7. Verify (optional but recommended)
```bash
python -m pytest
python -m mypy .
python -m flake8
```

---

## 🔬 Reproducibility (Research Paper)

To reproduce the exact metrics reported in the associated research paper, follow these strict execution steps:

### 1. Dataset Construction (Frozen Checkpoint)
The benchmark requires the mathematically balanced 50,000-sample dataset.
1. Ensure the `gojo_dataset_v1.csv` checkpoint (frozen prior to evaluation) is placed in the project root.
2. This dataset has been pre-filtered for exact-string deduplication and cross-split host leakage.

### 2. Training Baseline Models
All ML models and Thompson Sampling priors are initialized using a fixed random seed (`seed=42`) to guarantee deterministic cross-validation splits.
```bash
# Ensure the models are trained prior to validation
python -m phish_detector.train --data gojo_dataset_v1.csv --url-col url --label-col label
```

### 3. Claims Validation & Evaluation
To run the full suite of paper claims (Accuracy, ROC-AUC, Latency Microbenchmarks, McNemar Statistical Tests, Drift Degradation, and Imbalanced FPR) on the identical 80/20 test split:
```bash
python validate_claims.py
```

### 4. Random Seeds
All stochastic operations (Random Forest bootstrap sampling, TS context multi-armed bandit, temporal dataset shuffling, and validation splits) are globally locked to `seed=42` throughout the validation pipeline.

---

## 💻 Usage

### Web Interface

The easiest way to use the detector:

```bash
run_gojo.bat  # Windows
python -m webapp.app  # Linux/Mac
```

Features:
- Single URL analysis
- Bulk CSV upload (up to 10,000 URLs)
- Detailed breakdowns (features, signals, ML scores)
- Policy insights (RL agent decisions)
- Download results as CSV

### Command Line

#### Analyze Single URL
```bash
python -m phish_detector.cli --url "https://suspicious-site.com/login" --ml-mode ensemble
```

#### Bulk Analysis
```bash
python -m phish_detector.cli --input-csv urls.csv --url-col url --output results.csv --output-format csv --ml-mode ensemble
```

### Python API

```python
from phish_detector.analyze import AnalysisConfig, analyze_url, load_ml_context
from phish_detector.policy_v2 import ThompsonSamplingPolicy

# Configure
config = AnalysisConfig(
    ml_mode="ensemble",
    lexical_model="models/lexical_model.joblib",
    char_model="models/char_model.joblib",
    policy_path="models/policy.json",
    feedback_store="models/feedback.json",
    # Advanced features (optional)
    enable_content_analysis=True,      # HTML/page inspection
    enable_advanced_detection=True,    # Drift detection, TLD learning
    models_dir="models"
)

# Analyze
ml_context = load_ml_context(config)
policy = ThompsonSamplingPolicy("models/policy.json")
report, extra = analyze_url("https://suspicious-url.com", config, ml_context, policy)

print(f"Score: {report['summary']['score']}")
print(f"Label: {report['summary']['label']}")  # green/yellow/red

# Advanced enhancements (if enabled)
if 'advanced_enhancements' in extra:
    adv = extra['advanced_enhancements']
    print(f"Dynamic TLD: {adv.get('dynamic_tld', {}).get('tld')}")
    print(f"Attack Pattern: {adv.get('attack_pattern', {}).get('technique')}")
    
# Content analysis (if URL is live)
if extra.get('content_analysis'):
    ca = extra['content_analysis']
    print(f"Has credential form: {ca.get('has_credential_form')}")
    print(f"Content risk score: {ca.get('content_risk_score')}")
```

### Enrichment Notes

- Domain age/ASN are fetched via RDAP (public, no key required).
- DNS volatility is computed using public resolvers.
- Optional Google Safe Browsing: set `GOJO_GSB_API_KEY` to enable reputation checks.

---

## 📚 Documentation

- [Production Deployment Guide](PRODUCTION_README.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [Architecture Overview](#architecture)
- [API Documentation](#python-api)

### Architecture

```
Input URL → Parser → Feature Extraction → Rules + ML → RL Policy → Final Verdict
                ↓                             ↓
            Content Fetch              Thompson Sampling
         (HTML Analysis)           (25 contexts × 4 actions)
                ↓                             ↓
         Advanced Detection             Dynamic TLD Learning
      (Drift + Attack Pattern)        Attack Pattern Recognition
```

**Detection Layers**:
1. **URL Analysis**: Parse and extract structural features (21 metrics)
2. **Rule Engine**: Fast heuristic pattern matching (12 rules)
3. **ML Ensemble**: Lexical + Character n-gram classifiers
4. **Content Inspection**: HTML forms, JavaScript, visual brand matching (optional)
5. **Drift Detection**: Monitor distribution shifts and concept drift
6. **RL Policy**: Thompson Sampling for adaptive weight selection
7. **Final Verdict**: Multi-signal aggregation with confidence scores

---

## 📈 Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Rules Only | 74% | 68% | 81% | 0.74 |
| Lexical | 91% | 89% | 93% | 0.91 |
| Char N-gram | 94% | 92% | 96% | 0.94 |
| **Ensemble (Gojo)** | **99.1%** | **99.0%** | **99.2%** | **0.99** |

**Latency**: ~20ms per URL | **Throughput**: ~50 URLs/second

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Quick start:
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and test
4. Commit: `git commit -m 'Add amazing feature'`
5. Push: `git push origin feature/amazing-feature`
6. Open Pull Request

---

## 📁 Project Structure

```
gojo/
├── phish_detector/           # Core detection logic
│   ├── analyze.py            # Main orchestration
│   ├── features.py           # Feature extraction (21 features)
│   ├── rules.py              # Heuristic rules (12 rules)
│   ├── ml_*.py               # ML models (lexical, char, ensemble)
│   ├── policy_v2.py          # Thompson Sampling RL agent
│   ├── advanced_detection.py # Multi-layer advanced detection
│   ├── content_analysis.py   # HTML/page inspection
│   ├── drift_detection.py    # Distribution shift monitoring
│   ├── dynamic_tld_learning.py # Real-time TLD risk learning
│   └── cli.py                # Command-line interface
├── webapp/                   # Web interface
│   ├── app.py                # Production Flask app (heartbeat, validation)
│   └── templates/            # HTML templates (dark mode)
├── data/                     # Training datasets
├── models/                   # Trained models (generated)
├── tests/                    # Unit tests
└── _LOCAL_DOCS/              # Internal documentation (gitignored)
```

---

## 📄 License

Personal-use license (non-commercial). See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Dataset**: [DatasetWebFraudDetection](https://github.com/Priyanshu88/DatasetWebFraudDetection)
- **Scikit-learn**: Machine learning framework
- **Flask**: Web framework
- **Thompson Sampling**: Research by Agrawal & Goyal (2012)

---

<div align="center">

**Made with ❤️ for cybersecurity**

⭐ Star us on GitHub if you find this useful!

[Report Bug](https://github.com/maheshudaykar/Gojo/issues) •
[Request Feature](https://github.com/maheshudaykar/Gojo/issues)

</div>
