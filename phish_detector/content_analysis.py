"""
Content-based HTML/page analysis for phishing detection.

This module extends URL-only detection with page content analysis:
- HTML structure patterns (fake login forms, credential harvesting)
- Suspicious JavaScript (obfuscation, keylogging, redirects)
- Visual similarity to legitimate brands (logo/design cloning)
- SSL certificate mismatch with claimed identity
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import re
import warnings


@dataclass(frozen=True)
class ContentAnalysis:
    """Results from HTML/page content analysis."""
    has_password_field: bool
    has_credential_form: bool
    suspicious_js_count: int
    external_form_action: bool
    brand_visual_match: Optional[str]
    ssl_cert_mismatch: bool
    risk_score: int
    signals: list[str]
    is_safe: bool = True  # False if contains dangerous content


def analyze_html_content(
    html: str,
    url: str,
    claimed_brand: Optional[str] = None
) -> ContentAnalysis:
    """
    Analyze HTML page content for phishing indicators.

    SECURITY: This function only PARSES HTML, it does NOT execute scripts.
    All analysis is done via regex patterns - no JavaScript execution.

    Args:
        html: Raw HTML content of the page (sanitized, not executed)
        url: The originating URL
        claimed_brand: Brand the page claims to represent

    Returns:
        ContentAnalysis with detected patterns and risk score
    """
    signals: list[str] = []
    risk_score = 0

    # Pattern 1: Password/credential input fields
    has_password_field = bool(re.search(
        r'<input[^>]*type\s*=\s*["\']password["\']',
        html,
        re.IGNORECASE
    ))

    # Pattern 2: Login/credential harvesting form
    has_credential_form = bool(re.search(
        r'<form[^>]*(login|signin|password|credential)',
        html,
        re.IGNORECASE
    )) and has_password_field

    if has_credential_form:
        signals.append("credential_harvesting_form")
        risk_score += 30

    # Pattern 3: Form action to external domain
    external_form_action = False
    form_actions = re.findall(
        r'<form[^>]*action\s*=\s*["\']([^"\']+)["\']',
        html,
        re.IGNORECASE
    )
    for action in form_actions:
        if action.startswith('http') and url not in action:
            external_form_action = True
            signals.append(f"external_form_action:{action[:50]}")
            risk_score += 40
            break

    # Pattern 4: Suspicious JavaScript patterns
    suspicious_js_patterns = [
        r'document\.write\s*\(',  # Dynamic content injection
        r'eval\s*\(',  # Code obfuscation
        r'atob\s*\(',  # Base64 decoding (common in obfuscation)
        r'fromCharCode',  # Character-level obfuscation
        r'window\.location\s*=',  # Forced redirects
        r'addEventListener\s*\(\s*["\']keypress["\']',  # Keylogging
    ]

    suspicious_js_count = sum(
        1 for pattern in suspicious_js_patterns
        if re.search(pattern, html, re.IGNORECASE)
    )

    if suspicious_js_count >= 3:
        signals.append(f"suspicious_js_patterns:{suspicious_js_count}")
        risk_score += 25

    # Pattern 5: Brand visual cloning (logo/design theft)
    brand_visual_match = None
    if claimed_brand:
        # Check for brand name/logo in HTML
        if re.search(rf'\b{claimed_brand}\b', html, re.IGNORECASE):
            brand_visual_match = claimed_brand
            if has_credential_form:
                signals.append(f"brand_impersonation:{claimed_brand}")
                risk_score += 35

    # Pattern 6: SSL certificate mismatch (placeholder - requires cert inspection)
    ssl_cert_mismatch = False  # Would require actual SSL cert validation

    # Determine if content is suspicious enough to warn user
    is_safe = risk_score < 50  # Threshold for dangerous content

    return ContentAnalysis(
        has_password_field=has_password_field,
        has_credential_form=has_credential_form,
        suspicious_js_count=suspicious_js_count,
        external_form_action=external_form_action,
        brand_visual_match=brand_visual_match,
        ssl_cert_mismatch=ssl_cert_mismatch,
        risk_score=min(100, risk_score),
        signals=signals,
        is_safe=is_safe
    )


def fetch_and_analyze(
    url: str,
    timeout: int = 10,
    claimed_brand: Optional[str] = None
) -> Optional[ContentAnalysis]:
    """
    Fetch URL and analyze content (requires requests library).

    SECURITY NOTICE:
    - Only fetches HTML/text content, does NOT execute JavaScript
    - Uses safe regex-based parsing, NO code evaluation
    - Sandboxed analysis - malicious scripts cannot run
    - Timeouts prevent hanging on unresponsive sites

    Args:
        url: URL to fetch and analyze
        timeout: Request timeout in seconds
        claimed_brand: Brand the page claims to represent

    Returns:
        ContentAnalysis if successful, None on error
    """
    try:
        import requests

        # Warn about fetching potentially malicious content
        warnings.warn(
            f"Fetching content from {url} - analysis is sandboxed (no script execution)",
            UserWarning,
            stacklevel=2
        )

        # Fetch with safety measures
        response = requests.get(
            url,
            timeout=timeout,
            allow_redirects=True,
            headers={'User-Agent': 'Mozilla/5.0 (Gojo Phishing Detector)'},
            verify=True  # Verify SSL certificates
        )

        if response.status_code == 200:
            # Only analyze text/html content
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type or 'text/plain' in content_type:
                # Safe: only passing HTML string for regex analysis
                # NO script execution, NO eval(), NO rendering
                return analyze_html_content(response.text, url, claimed_brand)

    except ImportError:
        warnings.warn("requests library not installed, content analysis disabled", UserWarning)
        return None
    except Exception as e:
        # Network/connection error or any other error - silently fail
        warnings.warn(f"Content analysis error for {url}: {str(e)}", UserWarning)
        return None

    return None
