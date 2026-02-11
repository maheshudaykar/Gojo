# Contributing to Gojo 🤝

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [Areas for Contribution](#areas-for-contribution)

---

## 📜 Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- **Be respectful** and inclusive
- **Be patient** with newcomers
- **Be constructive** in criticism
- **Focus on what's best** for the community
- **Show empathy** towards other community members

---

## 🎯 How Can I Contribute?

### Reporting Bugs 🐛

Before creating bug reports, please check existing issues to avoid duplicates.

**Good bug reports include:**
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Error messages/screenshots
- Environment details (OS, Python version)
- Sample URLs that trigger the issue

**Template:**
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots/Logs**
If applicable, add screenshots or log output.

**Environment:**
 - OS: [e.g., Windows 11]
 - Python Version: [e.g., 3.10.5]
 - Package Version: [e.g., 2.0.0]
```

### Suggesting Enhancements ✨

Enhancement suggestions are tracked as GitHub issues.

**Good enhancement suggestions include:**
- Clear use case
- Why existing features don't address this
- Mockups/examples if applicable
- Potential implementation approach

**Template:**
```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
What you want to happen.

**Describe alternatives you've considered**
Other solutions or features you've considered.

**Additional context**
Any other context or screenshots.
```

### Contributing Code 💻

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/gojo.git`
3. **Create** a branch: `git checkout -b feature/your-feature-name`
4. **Make** changes
5. **Test** thoroughly
6. **Commit** with good messages
7. **Push** to your fork
8. **Submit** a Pull Request

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Git
- Virtual environment tool (venv, conda)
- Text editor/IDE (VS Code, PyCharm recommended)

### Setup Development Environment

```bash
# 1. Clone your fork
git clone https://github.com/YOUR_USERNAME/gojo.git
cd gojo

# 2. Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/gojo.git

# 3. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# 4. Install development dependencies
pip install -r requirements.txt
pip install -r requirements_production.txt

# 5. Install in editable mode
pip install -e .

# 6. Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

### Download Test Data

```bash
git clone https://github.com/Priyanshu88/DatasetWebFraudDetection.git data/DatasetWebFraudDetection
```

### Train Models

```bash
python -m phish_detector.train \
    --dataset data/DatasetWebFraudDetection/dataset.csv \
    --url-col url \
    --label-col verdict
```

---

## 🔄 Development Workflow

### Branching Strategy

We use **GitHub Flow** (simplified Git Flow):

- `main` - Production-ready code (protected)
- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `hotfix/*` - Urgent production fixes
- `docs/*` - Documentation updates

**Branch Naming:**
```
feature/add-neural-network-model
bugfix/fix-url-parsing-error
hotfix/critical-security-patch
docs/update-contributing-guide
```

### Workflow Steps

1. **Sync with upstream**
   ```bash
   git checkout main
   git pull upstream main
   ```

2. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make changes**
   - Write code
   - Add tests
   - Update documentation

4. **Test locally**
   ```bash
   # Run tests
   python -m pytest tests/
   
   # Check types
   python -m mypy phish_detector/
   
   # Check style
   python -m pylint phish_detector/
   ```

5. **Commit changes**
   ```bash
   git add .
   git commit -m "feat: add neural network model"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create Pull Request** on GitHub

---

## 📝 Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line length**: 100 characters max
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings
- **Type hints**: Required for all functions
- **Docstrings**: Required for all public functions (Google style)

**Example:**

```python
def analyze_url(
    url: str,
    config: AnalysisConfig,
    ml_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    \"""
    Analyze URL for phishing indicators.
    
    Args:
        url: The URL to analyze
        config: Analysis configuration
        ml_context: Pre-loaded ML models (optional)
    
    Returns:
        Tuple of (report dict, extra metadata dict)
    
    Raises:
        ValueError: If URL is invalid
    
    Example:
        >>> config = AnalysisConfig(ml_mode="ensemble", ...)
        >>> report, extra = analyze_url("https://example.com", config)
        >>> print(report['summary']['score'])
        15
    \"""
    # Implementation
    pass
```

### Code Organization

- **One class per file** (exceptions for small helpers)
- **Group imports**: stdlib → third-party → local
- **Separate concerns**: parse → extract → analyze → report
- **Avoid circular imports**
- **Use type hints** everywhere

### Documentation

- **Docstrings**: All public functions, classes, modules
- **Comments**: Explain *why*, not *what*
- **Type hints**: Use modern syntax (`dict[str, Any]` not `Dict`)
- **Examples**: Include in docstrings for complex functions

---

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── test_parsing.py
├── test_features.py
├── test_rules.py
├── test_ml_models.py
├── test_policy.py
├── test_integration.py
└── fixtures/
    ├── sample_urls.txt
    └── test_dataset.csv
```

### Writing Tests

```python
import pytest
from phish_detector.parsing import parse_url

def test_parse_url_http():
    \"\"\"Test that HTTP URLs are parsed correctly.\"\"\"
    parsed = parse_url("http://example.com/path")
    assert parsed.scheme == "http"
    assert parsed.host == "example.com"
    assert parsed.path == "/path"

def test_parse_url_no_scheme():
    \"\"\"Test that URLs without scheme get http:// added.\"\"\"
    parsed = parse_url("example.com")
    assert parsed.scheme == "http"
    assert parsed.original == "example.com"
    assert parsed.normalized == "http://example.com"

@pytest.mark.parametrize("url,expected_score", [
    ("http://example.com", 0),  # Clean URL
    ("http://paypa1.com", 85),  # Typosquatting
    ("http://192.168.1.1/login", 45),  # IP address
])
def test_url_scoring(url, expected_score):
    \"\"\"Test URL scoring with various inputs.\"\"\"
    # Test implementation
    pass
```

### Running Tests

```bash
# All tests
python -m pytest tests/

# Specific test file
python -m pytest tests/test_parsing.py

# With coverage
python -m pytest tests/ --cov=phish_detector --cov-report=html

# Verbose mode
python -m pytest tests/ -v

# Stop on first failure
python -m pytest tests/ -x
```

### Test Coverage

- Aim for **80%+ coverage**
- Focus on **critical paths** (parsing, feature extraction, ML inference)
- Include **edge cases** (empty strings, special characters, long URLs)
- Test **error handling** (invalid inputs, missing files)

---

## 🎯 Commit Message Guidelines

We follow **Conventional Commits** specification:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style (formatting, no logic change)
- **refactor**: Code refactoring
- **test**: Add/update tests
- **chore**: Maintenance tasks (dependencies, build)
- **perf**: Performance improvements
- **ci**: CI/CD changes

### Examples

```
feat(ml): add LSTM-based URL classifier

Implemented bidirectional LSTM model for sequence-based
URL analysis. Achieves 97% accuracy on test set.

Closes #42

---

fix(parsing): handle URLs with unicode characters

URLs containing non-ASCII characters now properly
normalized using IDNA encoding.

Fixes #123

---

docs(readme): update installation instructions

Added troubleshooting section for Windows users.

---

refactor(policy): simplify Thompson Sampling implementation

Reduced code complexity while maintaining functionality.
```

### Best Practices

- **Use imperative mood**: "add" not "added"
- **Keep subject line <50 chars**
- **Capitalize subject**
- **No period at end of subject**
- **Separate subject and body** with blank line
- **Wrap body at 72 chars**
- **Reference issues/PRs** in footer

---

## 🔀 Pull Request Process

### Before Creating PR

- [ ] Code follows project style guide
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Type hints added
- [ ] No extraneous changes (formatting, whitespace)
- [ ] Commits follow message guidelines
- [ ] Branch is up to date with `main`

### PR Description Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## How Has This Been Tested?
Describe tests you ran.

## Checklist
- [ ] My code follows the project style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code where necessary
- [ ] I have updated the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix/feature works
- [ ] New and existing tests pass locally

## Screenshots (if applicable)
Add screenshots for UI changes.

## Related Issues
Closes #123
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **At least 1 review** required from maintainers
3. **Address feedback** promptly
4. **Squash commits** if requested
5. **Maintainers merge** after approval

### After Merge

- Delete your branch
- Update your fork:
  ```bash
  git checkout main
  git pull upstream main
  git push origin main
  ```

---

## 💡 Areas for Contribution

### High Priority

- [ ] **Unit Tests**: Increase coverage to 80%+
- [ ] **Documentation**: API docs, tutorials, examples
- [ ] **Performance**: Optimize feature extraction
- [ ] **Security**: Input validation, sanitization
- [ ] **Datasets**: Labeled phishing URLs for training

### Feature Requests

- [ ] **Neural Networks**: LSTM, Transformer models
- [ ] **Multi-language**: Internationalization
- [ ] **Browser Extension**: Chrome/Firefox addon
- [ ] **REST API**: FastAPI service with auth
- [ ] **Real-time**: Stream processing with Kafka
- [ ] **Explainability**: SHAP/LIME integration
- [ ] **Distributed**: Multi-node training/inference

### Bug Fixes

Check the [issue tracker](https://github.com/yourusername/gojo/issues) for bugs labeled `good first issue`.

### Documentation

- Improve README with more examples
- Add architecture diagrams
- Create video tutorials
- Write blog posts about the project
- Translate documentation

---

## 🎓 Learning Resources

### For New Contributors

- [GitHub Flow Guide](https://guides.github.com/introduction/flow/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Pytest Documentation](https://docs.pytest.org/)

### Project-Specific

- [Thompson Sampling Paper](https://arxiv.org/abs/1111.1797)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

## 📧 Questions?

- **GitHub Discussions**: [Ask questions](https://github.com/yourusername/gojo/discussions)
- **Issues**: [Report bugs/request features](https://github.com/yourusername/gojo/issues)
- **Email**: your.email@example.com

---

## 🙏 Thank You!

Thank you for contributing to Gojo! Every contribution, no matter how small, helps make this project better.

**Contributors will be acknowledged** in our [Contributors](#) section.

---

<div align="center">

**Happy Contributing! 🚀**

[Back to README](README.md)

</div>
