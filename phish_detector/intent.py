from __future__ import annotations

INTENT_TOKENS = {
    "secure",
    "verify",
    "account",
    "login",
    "update",
    "password",
    "billing",
    "confirm",
    "wallet",
    "signin",
    "sign-in",
    "payment",
    "invoice",
}


def find_intent_tokens(text: str) -> list[str]:
    lowered = text.lower()
    return sorted({token for token in INTENT_TOKENS if token in lowered})


def has_intent(text: str) -> bool:
    return bool(find_intent_tokens(text))
