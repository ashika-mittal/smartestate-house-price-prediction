import re

from src.config import GROQ_MODEL_NAME


def _price_comparison_facts(properties, predicted_price):
    """Return authoritative price gaps and the nearest historical price."""
    facts = []

    for rank, (_, row) in enumerate(properties.iterrows(), start=1):
        historical_price = float(row["price"])
        facts.append(
            {
                "rank": rank,
                "price": historical_price,
                "difference": abs(historical_price - predicted_price),
            }
        )

    closest = min(facts, key=lambda fact: (fact["difference"], fact["rank"]))
    return facts, closest


def _has_response_violation(response_text, properties, predicted_price):
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

    price_facts, closest = _price_comparison_facts(
        properties,
        predicted_price,
    )
    price_gaps = {fact["rank"]: fact["difference"] for fact in price_facts}

    closer_claim_pattern = re.compile(
        r"Property\s*(\d+)(?:['’]s)?"
        r"(?:(?!Property\s*\d+).){0,300}?\bcloser\b"
        r".{0,160}?\bthan\s+Property\s*(\d+)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for claim in closer_claim_pattern.finditer(response_text):
        claimed_closer = int(claim.group(1))
        compared_with = int(claim.group(2))

        if claimed_closer not in price_gaps or compared_with not in price_gaps:
            return True

        if not price_gaps[claimed_closer] < price_gaps[compared_with]:
            return True

    closest_claim_pattern = re.compile(
        r"Property\s*(\d+)(?:['’]s)?"
        r"(?:(?!Property\s*\d+).){0,240}?\bclosest\b"
        r".{0,120}?\bestimate\b",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for claim in closest_claim_pattern.finditer(response_text):
        if int(claim.group(1)) != closest["rank"]:
            return True

    if re.search(
        r"(?:^|\n)\s*(?:#{1,6}\s*)?"
        r"(?:recommendation|alternatives?|conclusion)\s*:?\s*$",
        response_text,
        flags=re.IGNORECASE,
    ):
        return True

    return False


def build_safe_summary(predicted_price, properties):
    """Create a deterministic response using only retrieval output."""
    price_facts, closest = _price_comparison_facts(
        properties,
        predicted_price,
    )
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
                f"Property {closest['rank']} has the historical recorded price "
                f"closest to the estimate: **{closest['price']:.2f} lakhs**, "
                f"a difference of **{closest['difference']:.2f} lakhs**."
            ),
            "",
            (
                "Property 1 is the highest-ranked overall match from the "
                "retrieval system. These are historical records and may "
                "not represent current listings."
            )
        ]
    )

    return "\n".join(lines)


def format_retrieved_context(properties, predicted_price):
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
            f"{row['price']} lakhs\n"
            f"- Absolute difference from estimate: "
            f"{abs(float(row['price']) - predicted_price):.2f} lakhs"
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
        retrieved_properties,
        predicted_price,
    )
    price_facts, closest_price = _price_comparison_facts(
        retrieved_properties,
        predicted_price,
    )
    price_fact_lines = "\n".join(
        (
            f"- Property {fact['rank']}: |{fact['price']:.2f} - "
            f"{predicted_price:.2f}| = {fact['difference']:.2f} lakhs"
        )
        for fact in price_facts
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

Authoritative price-proximity calculation:
{price_fact_lines}
- Property {closest_price['rank']} has the historical recorded price closest
  to the estimate, with a difference of {closest_price['difference']:.2f} lakhs.

Ranking rules:
- The properties are already ordered from best to worst overall match.
- Property 1 is the highest-ranked and best overall match.
- Retrieval order is authoritative even when another property's historical price is closer to the estimate or budget.
- The supplied price-proximity calculation is authoritative. Do not redo this arithmetic.

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
- Never claim that one price is closer than another unless that ordering agrees with the supplied absolute differences.
- Recommend only properties present in the retrieved context.
- Do not invent properties, prices, features, or market information.
- Keep the estimated price separate from historical recorded prices.
- Mention that the records may not represent current listings.
- Return exactly three short sections: "Estimate", "Best match", and "Alternatives and conclusion".
- Do not reproduce the full property table; it is already visible above the response.
- Complete every section, use no more than 180 words, and end with the historical-records disclaimer.
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
        temperature=0.0,
        max_completion_tokens=2000,
    )

    draft = response.choices[0].message.content

    if not draft or not draft.strip():
        return build_safe_summary(
            predicted_price,
            retrieved_properties,
        )

    if not _has_response_violation(
        draft,
        retrieved_properties,
        predicted_price,
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
                    "the corrected comparison. Use the supplied absolute price "
                    f"differences: Property {closest_price['rank']} has the "
                    "historical price closest to the estimate. Return all three "
                    "required sections, stay under 180 words, and finish the "
                    "conclusion and disclaimer."
                )
            }
        ],
        temperature=0.0,
        max_completion_tokens=2000,
    )

    corrected_draft = correction.choices[0].message.content

    if not corrected_draft or not corrected_draft.strip():
        return build_safe_summary(
            predicted_price,
            retrieved_properties,
        )

    if not _has_response_violation(
        corrected_draft,
        retrieved_properties,
        predicted_price,
    ):
        return corrected_draft

    return build_safe_summary(
        predicted_price,
        retrieved_properties
    )
