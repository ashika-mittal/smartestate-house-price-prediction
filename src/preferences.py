"""Normalize optional preferences to fields supported by the dataset."""

import re


_UNVERIFIABLE_PATTERNS = (
    (r"\b(?:pollution(?:[- ]free)?|air quality|clean air)\b", "pollution or air quality"),
    (r"\b(?:safe|safety|crime|secure|security)\b", "safety or crime levels"),
    (
        r"\b(?:school|college|hospital|metro|bus stop|station|amenit(?:y|ies))\b",
        "nearby amenities or transport",
    ),
    (r"\b(?:parking|car park)\b", "parking"),
    (r"\b(?:furnish(?:ed|ing)?)\b", "furnishing"),
    (r"\b(?:view|views|scenic)\b", "views"),
    (r"\b(?:layout|ventilation|sunlight)\b", "layout, ventilation, or sunlight"),
    (
        r"\b(?:available|availability|ready[- ]to[- ]move|possession)\b",
        "current availability",
    ),
    (r"\b(?:vastu|vaastu)\b", "Vastu"),
)


def find_unverifiable_preferences(text):
    """Return user-friendly labels for preferences absent from the dataset."""
    found = []
    for pattern, label in _UNVERIFIABLE_PATTERNS:
        if re.search(pattern, text or "", flags=re.IGNORECASE) and label not in found:
            found.append(label)
    return found


def contains_unverifiable_response_claim(text):
    """Identify AI responses that claim knowledge of unsupported features."""
    return bool(find_unverifiable_preferences(text))


def _near(text, field_pattern, direction_pattern):
    return bool(
        re.search(
            rf"(?:{direction_pattern}).{{0,24}}(?:{field_pattern})|"
            rf"(?:{field_pattern}).{{0,24}}(?:{direction_pattern})",
            text,
        )
    )


def normalize_matching_priorities(text):
    """Convert free text into a controlled set of supported ranking priorities."""
    text = (text or "").lower()
    priorities = []

    area = r"square feet|sq\.?\s*ft|area|space|size"
    balconies = r"balcon(?:y|ies)"
    bathrooms = r"bath(?:room)?s?"
    prices = r"historical price|recorded price|price|budget"

    if _near(text, area, r"larger|bigger|more|spacious"):
        priorities.append("Prefer larger square feet.")
    elif _near(text, area, r"smaller|compact|less|fewer"):
        priorities.append("Prefer smaller square feet.")
    elif _near(text, area, r"closest|similar|near|requested"):
        priorities.append("Prefer square feet closest to the request.")

    if _near(text, balconies, r"more|extra|additional"):
        priorities.append("Prefer more balconies.")
    elif _near(text, balconies, r"fewer|less"):
        priorities.append("Prefer fewer balconies.")
    elif _near(text, balconies, r"closest|similar|requested") or re.search(
        r"\b(?:with|has|having)\b.{0,20}\bbalcon(?:y|ies)\b", text
    ):
        priorities.append("Prefer balconies closest to the request.")

    if _near(text, bathrooms, r"more|extra|additional"):
        priorities.append("Prefer more bathrooms.")
    elif _near(text, bathrooms, r"fewer|less"):
        priorities.append("Prefer fewer bathrooms.")
    elif _near(text, bathrooms, r"closest|similar|requested"):
        priorities.append("Prefer bathrooms closest to the request.")

    if _near(text, prices, r"lower|cheaper|cheap|affordable"):
        priorities.append("Prefer lower historical prices.")
    elif _near(text, prices, r"closest|similar|near|estimate|predicted"):
        priorities.append("Prefer historical prices closest to the estimate.")

    return " ".join(dict.fromkeys(priorities))
