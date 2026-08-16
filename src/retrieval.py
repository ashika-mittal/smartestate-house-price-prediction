import faiss
import numpy as np

from src.config import DEFAULT_TOP_K


AREA_MATCH_TOLERANCE = 0.05


def create_property_text(row):
    return (
        f"Property located in {row['location']}. "
        f"It has {int(row['bhk'])} BHK, "
        f"{int(row['bath'])} bathrooms, and "
        f"{int(row['balcony'])} balconies. "
        f"Total area is {row['total_sqft']} square feet. "
        f"Historical recorded price is {row['price']} lakhs."
    )


def hybrid_retrieve(
    data,
    embedding_model,
    query,
    location,
    bhk,
    max_price,
    total_sqft=None,
    bath=None,
    balcony=None,
    target_price=None,
    k=DEFAULT_TOP_K
):
    # Required structured filters
    matches = data[
        (data["location"] == location)
        & (data["bhk"] == bhk)
        & (data["price"] <= max_price)
    ].copy()

    if matches.empty:
        return matches

    if embedding_model is None:
        ranked_matches = matches.copy()
        ranked_matches["semantic_score"] = 0.0
    else:
        matches["property_text"] = matches.apply(
            create_property_text,
            axis=1
        )

        documents = matches["property_text"].tolist()

        document_embeddings = embedding_model.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        query_embedding = embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        document_embeddings = np.ascontiguousarray(
            document_embeddings
        )

        query_embedding = np.ascontiguousarray(
            query_embedding
        )

        index = faiss.IndexFlatIP(
            document_embeddings.shape[1]
        )

        index.add(document_embeddings)

        # Retrieve all eligible records before final reranking.
        scores, indices = index.search(
            query_embedding,
            k=len(matches)
        )

        ranked_matches = matches.iloc[indices[0]].copy()
        ranked_matches["semantic_score"] = scores[0]

    if total_sqft is not None:
        ranked_matches["area_similarity"] = (
            1 / (
                1
                + (
                    ranked_matches["total_sqft"]
                    - total_sqft
                ).abs() / max(total_sqft, 1)
            )
        )
    else:
        ranked_matches["area_similarity"] = 1.0

    if balcony is not None:
        ranked_matches["balcony_similarity"] = (
            1 / (
                1
                + (
                    ranked_matches["balcony"]
                    - balcony
                ).abs()
            )
        )
    else:
        ranked_matches["balcony_similarity"] = 1.0

    if bath is not None:
        ranked_matches["bath_similarity"] = (
            1 / (
                1
                + (
                    ranked_matches["bath"]
                    - bath
                ).abs()
            )
        )
    else:
        ranked_matches["bath_similarity"] = 1.0

    if target_price is not None:
        ranked_matches["price_similarity"] = (
            1 / (
                1
                + (
                    ranked_matches["price"]
                    - target_price
                ).abs() / max(target_price, 1)
            )
        )
    else:
        ranked_matches["price_similarity"] = 1.0

    # Form fields other than location, BHK, and budget are soft
    # preferences. They affect ranking without removing useful records.
    ranked_matches["match_score"] = (
        0.40 * ranked_matches["semantic_score"]
        + 0.25 * ranked_matches["area_similarity"]
        + 0.15 * ranked_matches["balcony_similarity"]
        + 0.10 * ranked_matches["bath_similarity"]
        + 0.10 * ranked_matches["price_similarity"]
    )

    def describe_match(row):
        differences = []

        if total_sqft is not None:
            area_difference = row["total_sqft"] - total_sqft
            area_difference_ratio = (
                abs(area_difference) / max(total_sqft, 1)
            )

            if area_difference_ratio > AREA_MATCH_TOLERANCE:
                direction = (
                    "more" if area_difference > 0 else "less"
                )
                differences.append(
                    f"square feet ({abs(area_difference):g} "
                    f"{direction})"
                )

        if bath is not None and row["bath"] != bath:
            bath_difference = row["bath"] - bath
            direction = "more" if bath_difference > 0 else "less"
            differences.append(
                f"bathrooms ({abs(bath_difference):g} {direction})"
            )

        if balcony is not None and row["balcony"] != balcony:
            balcony_difference = row["balcony"] - balcony
            direction = (
                "more" if balcony_difference > 0 else "less"
            )
            differences.append(
                f"balconies ({abs(balcony_difference):g} "
                f"{direction})"
            )

        if not differences:
            return "Exact match", "None"

        if len(differences) == 1:
            match_type = "Close match"
        else:
            match_type = "Alternative match"

        return match_type, ", ".join(differences)

    match_descriptions = ranked_matches.apply(
        describe_match,
        axis=1,
        result_type="expand"
    )

    ranked_matches[["match_type", "differences"]] = (
        match_descriptions
    )

    match_priority = {
        "Exact match": 0,
        "Close match": 1,
        "Alternative match": 2
    }

    ranked_matches["match_priority"] = (
        ranked_matches["match_type"].map(match_priority)
    )

    # Exact matches are shown first. If there are fewer than k exact
    # matches, the remaining positions are filled with close alternatives.
    ranked_matches = ranked_matches.sort_values(
        by=["match_priority", "match_score"],
        ascending=[True, False]
    ).head(k)

    return ranked_matches[
        [
            "location",
            "total_sqft",
            "bhk",
            "bath",
            "balcony",
            "price",
            "match_type",
            "differences",
            "match_score"
        ]
    ]
