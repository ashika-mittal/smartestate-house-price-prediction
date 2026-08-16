import re

from src.config import GROQ_MODEL_NAME


def _has_response_violation(response_text, properties):
    """Return True when Groq contradicts grounded retrieval output."""
    if re.search(
        r"\b(?:available|availability)\b",
        response_text,
        flags=re.IGNORECASE
    ):
        return True

    match_labels = (
        "Exact match",
        "Close match",
        "Alternative match"
    )

    for rank, (_, row) in enumerate(properties.iterrows(), start=1):
        section_pattern = re.compile(
            rf"Property\s+{rank}\b"
            rf"(?:(?!Property\s+\d+\b).)*",
            flags=re.IGNORECASE | re.DOTALL
        )

        for section_match in section_pattern.finditer(response_text):
            section = section_match.group(0)

            for label in match_labels:
                if label == row["match_type"]:
                    continue

                if re.search(
                    rf"\b{re.escape(label)}\b",
                    section,
                    flags=re.IGNORECASE
                ):
                    return True

    return False


def _build_safe_summary(predicted_price, properties):
    """Create a deterministic response using only retrieval output."""
    lines = [
        (
            "The regression model estimates the property price at "
            f"**{predicted_price:.2f} lakhs**. This is an estimate and "
            "may not reflect the current market value."
        ),
        "",
        "Historical property comparison:"
    ]

    for rank, (_, row) in enumerate(properties.iterrows(), start=1):
        summary = (
            f"- **Property {rank} — {row['match_type']}:** "
            f"{row['location']}, {row['total_sqft']} square feet, "
            f"{int(row['bhk'])} BHK, {int(row['bath'])} bathrooms, "
            f"{int(row['balcony'])} balconies; historical recorded "
            f"price: {row['price']} lakhs."
        )

        if row["differences"] != "None":
            summary += f" Differences from the request: {row['differences']}."

        lines.append(summary)

    lines.extend(
        [
            "",
            (
                "Property 1 is the highest-ranked overall match from the "
                "retrieval system. These are historical records and may "
                "not represent current listings."
            )
        ]
    )

    return "\n".join(lines)


def format_retrieved_context(properties):
    context_parts = []

    for rank, (_, row) in enumerate(
        properties.iterrows(),
        start=1
    ):
        context_parts.append(
            f"Property {rank}:\n"
            f"- Location: {row['location']}\n"
            f"- Total area: {row['total_sqft']} square feet\n"
            f"- BHK: {int(row['bhk'])}\n"
            f"- Bathrooms: {int(row['bath'])}\n"
            f"- Balconies: {int(row['balcony'])}\n"
            f"- Match category: {row['match_type']}\n"
            f"- Differs from request: {row['differences']}\n"
            f"- Retrieval score: {row['match_score']:.3f}\n"
            f"- Historical recorded price: "
            f"{row['price']} lakhs"
        )

    return "\n\n".join(context_parts)


def generate_groq_response(
    client,
    user_query,
    location,
    area_type,
    total_sqft,
    bath,
    balcony,
    bhk,
    max_price,
    predicted_price,
    retrieved_properties
):
    retrieved_context = format_retrieved_context(
        retrieved_properties
    )

    prompt = f"""
User preferences:
{user_query}

Property submitted for price estimation:
- Location: {location}
- Area type: {area_type}
- Total area: {total_sqft} square feet
- BHK: {bhk}
- Bathrooms: {bath}
- Balconies: {balcony}
- Maximum budget: {max_price:.2f} lakhs

Regression model result:
The estimated property price is {predicted_price:.2f} lakhs.

FAISS-ranked historical properties:
{retrieved_context}

Ranking rules:
- The properties are already ordered from best to worst overall match.
- Property 1 is the highest-ranked and best overall match.
- Retrieval order is authoritative even when another property's historical price is closer to the estimate or budget.

Instructions:
- Clearly state that the regression value is an estimated price.
- Call dataset prices "historical recorded prices".
- Compare the estimate with the retrieved historical properties.
- Area type is used only for the regression estimate. Do not discuss it when comparing historical properties.
- The dataset contains historical records, not current listing-status information. Do not make claims about whether a property can be obtained now.
- Distinguish exact matches from close or alternative matches.
- Treat the supplied match category and "Differs from request" value as authoritative.
- Copy each property's supplied match category verbatim.
- Use the phrase "exact match" only for a property whose supplied match category is "Exact match".
- Never use wording such as "exact match except" or "exact match apart from".
- Clearly mention every listed difference before recommending an alternative.
- Do not add a difference that is not present in "Differs from request".
- Never say a close or alternative match meets all requirements.
- If recommending one property, recommend Property 1.
- Lower-ranked properties may be mentioned only as alternatives.
- Never describe a lower-ranked property as closer or better overall than Property 1.
- If a lower-ranked property's historical price is numerically closer to the estimate, describe only its price as closer, not the property itself.
- Recommend only properties present in the retrieved context.
- Do not invent properties, prices, features, or market information.
- Keep the estimated price separate from historical recorded prices.
- Mention that the records may not represent current listings.
- Keep the response concise and easy to understand.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a grounded Bengaluru house-price "
                "and property-comparison assistant. Use only "
                "the regression estimate and retrieved historical "
                "properties provided in the prompt. Never invent "
                "a property, price, feature, or market fact."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=messages,
        temperature=0.0
    )

    draft = response.choices[0].message.content

    if not _has_response_violation(
        draft,
        retrieved_properties
    ):
        return draft

    correction = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=messages + [
            {
                "role": "assistant",
                "content": draft
            },
            {
                "role": "user",
                "content": (
                    "Correct the response because it violates at least one "
                    "grounding rule. Copy every supplied Match category "
                    "verbatim. The phrase 'exact match' is allowed only for "
                    "rows labeled 'Exact match'. Remove every claim or "
                    "condition about whether a historical property can be "
                    "obtained now; the dataset cannot establish that. Do not "
                    "use the words 'available' or 'availability'. Return only "
                    "the corrected comparison."
                )
            }
        ],
        temperature=0.0
    )

    corrected_draft = correction.choices[0].message.content

    if not _has_response_violation(
        corrected_draft,
        retrieved_properties
    ):
        return corrected_draft

    return _build_safe_summary(
        predicted_price,
        retrieved_properties
    )
