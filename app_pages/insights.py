import streamlit as st

from src.assistant import generate_groq_response
from src.resources import load_groq_client, load_reference_data


result = st.session_state.get("prediction_result")

st.markdown(":blue-badge[Step 2 of 2] :violet-badge[Grounded insights]")
st.title("Your property intelligence report")
st.write(
    "Review the model estimate, ranked historical matches, and grounded GenAI "
    "explanation without cluttering the prediction form."
)

if not result:
    with st.container(border=True, key="empty_card", horizontal_alignment="center"):
        st.space("small")
        st.subheader(":material/analytics: No estimate yet")
        st.write("Complete the property details first to unlock this report.")
        if st.button(
            "Create an estimate",
            type="primary",
            icon=":material/arrow_back:",
            width="stretch",
        ):
            st.switch_page("app_pages/predict.py")
        st.space("small")
    st.stop()

house_data = load_reference_data()
comparables = result["comparables"]
predicted_price = result["predicted_price"]

local_market = house_data[
    (house_data["location"] == result["location"])
    & (house_data["bhk"] == result["bhk"])
].copy()
if local_market.empty:
    local_market = house_data[house_data["location"] == result["location"]].copy()

local_median = float(local_market["price"].median())
median_price_per_sqft = float(local_market["price_per_sqft"].median())

with st.container(border=True, key="result_hero", gap="small"):
    heading_column, action_column = st.columns(
        [4, 1],
        vertical_alignment="center",
    )
    with heading_column:
        st.subheader(f":material/home_pin: {result['location']} · {result['bhk']} BHK")
        st.caption(
            f"{result['total_sqft']:,} sq ft · {result['bath']} bathrooms · "
            f"{result['balcony']} balconies · {result['area_type'].strip()}"
        )
    with action_column:
        if st.button(
            "Edit inputs",
            icon=":material/edit:",
            width="stretch",
            key="edit_inputs",
        ):
            st.switch_page("app_pages/predict.py")

    with st.container(horizontal=True):
        with st.container(key="result_price"):
            st.metric(
                "Model estimate",
                f"₹{predicted_price:,.1f} lakh",
                border=True,
                icon=":material/currency_rupee:",
            )
        st.metric(
            "Historical median",
            f"₹{local_median:,.1f}L",
            border=True,
            icon=":material/payments:",
        )
        st.metric(
            "Median / sq ft",
            f"₹{median_price_per_sqft:,.0f}",
            border=True,
            icon=":material/square_foot:",
        )
        if comparables.empty:
            st.metric("Top match", "None", border=True, icon=":material/search_off:")
        else:
            top_score = max(0, min(100, round(comparables.iloc[0]["match_score"] * 100)))
            st.metric(
                "Top match score",
                f"{top_score}%",
                border=True,
                icon=":material/target:",
                help="Historical-match similarity, not model confidence.",
            )

with st.container(border=True, key="comparables_card", gap="small"):
    st.subheader(":material/apartment: Ranked comparables")

    if result["embedding_warning"]:
        st.warning(
            "Semantic ranking was unavailable; structured ranking was used.",
            icon=":material/wifi_off:",
        )

    if comparables.empty:
        st.warning(
            "No historical property matches the location, BHK, and budget.",
            icon=":material/search_off:",
        )
        st.caption("Increase the budget or edit the location or BHK.")
    else:
        display = comparables.rename(
            columns={
                "location": "Location",
                "total_sqft": "Square feet",
                "bhk": "BHK",
                "bath": "Bathrooms",
                "balcony": "Balconies",
                "price": "Historical price",
                "match_type": "Match category",
                "differences": "Differences",
                "match_score": "Match score",
            }
        )
        display.insert(0, "Rank", range(1, len(display) + 1))

        st.dataframe(
            display,
            hide_index=True,
            height=330,
            width="stretch",
            column_config={
                "Rank": st.column_config.NumberColumn("#", width="small"),
                "Historical price": st.column_config.NumberColumn(
                    "Historical price (L)", format="₹%.2f"
                ),
                "Match score": st.column_config.ProgressColumn(
                    "Match score", min_value=0, max_value=1, format="percent"
                ),
            },
        )
        st.caption(
            "Property 1 is the highest-ranked overall match—not necessarily "
            "the closest on every individual measure."
        )

if not comparables.empty:
    with st.container(border=True, key="ai_card", gap="small"):
        st.subheader(":material/auto_awesome: GenAI RAG Assistant")
        st.caption(
            "Grounded only in the model estimate and ranked historical records."
        )

        if not result["assistant_attempted"]:
            groq_client = load_groq_client()
            result["assistant_attempted"] = True

            if groq_client is None:
                result["assistant_error"] = "GROQ_API_KEY is missing from .env."
            else:
                try:
                    with st.spinner("Generating your grounded comparison..."):
                        result["assistant_response"] = generate_groq_response(
                            client=groq_client,
                            user_query=result["matching_priorities"],
                            location=result["location"],
                            area_type=result["area_type"],
                            total_sqft=result["total_sqft"],
                            bath=result["bath"],
                            balcony=result["balcony"],
                            bhk=result["bhk"],
                            max_price=result["max_price"],
                            predicted_price=predicted_price,
                            retrieved_properties=comparables,
                        )
                except Exception:
                    result["assistant_error"] = (
                        "The AI comparison could not be generated. The estimate "
                        "and historical matches remain available."
                    )

            st.session_state.prediction_result = result

        if result.get("assistant_response"):
            st.markdown(result["assistant_response"])
        else:
            st.error(
                result.get("assistant_error", "AI comparison is unavailable."),
                icon=":material/cloud_off:",
            )

with st.container(horizontal=True, horizontal_alignment="right"):
    if st.button(
        "Start a new estimate",
        type="primary",
        icon=":material/refresh:",
    ):
        st.session_state.prediction_result = None
        st.switch_page("app_pages/predict.py")
