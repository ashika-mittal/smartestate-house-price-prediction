import streamlit as st

from src.preferences import (
    find_unverifiable_preferences,
    normalize_matching_priorities,
)
from src.prediction import predict_house_price
from src.resources import (
    load_embedding_model,
    load_model,
    load_reference_data,
)
from src.retrieval import hybrid_retrieve
from src.ui import range_control


model, model_columns = load_model()
house_data = load_reference_data()

locations = sorted(house_data["location"].dropna().unique())
area_types = sorted(house_data["area_type"].dropna().unique())

area_min = max(300, int(house_data["total_sqft"].min()))
area_max = int(house_data["total_sqft"].max())
bhk_max = int(house_data["bhk"].max())
bath_max = int(house_data["bath"].max())
balcony_max = int(house_data["balcony"].max())
budget_min = int(house_data["price"].min())
budget_max = int(house_data["price"].max())

st.markdown(":blue-badge[Step 1 of 2] :green-badge[Live estimate]")
st.title("Build your property estimate")
st.write(
    "Tune the verified property details below. Every change refreshes the "
    "estimate and neighborhood snapshot instantly."
)

form_column, preview_column = st.columns(
    [0.98, 1.08],
    gap="large",
    vertical_alignment="top",
)

with form_column:
    with st.container(border=True, key="property_card", gap="small"):
        st.subheader(":material/tune: Property details")

        location = st.selectbox(
            "Location",
            locations,
            key="property_location",
            help="Bengaluru locations represented in the historical dataset.",
            persist_state="session",
        )
        area_type = st.selectbox(
            "Area type",
            area_types,
            index=area_types.index("Super built-up  Area"),
            key="property_area_type",
            help="The construction-area definition used by the model.",
            persist_state="session",
        )

        total_sqft = range_control(
            "Total area (sq ft)",
            "property_sqft",
            area_min,
            area_max,
            1200,
            50,
            "%d",
            "Observed dataset range; use the number box for exact entry.",
            ":material/square_foot:",
        )
        bhk = range_control(
            "BHK (rooms)",
            "property_bhk",
            1,
            bhk_max,
            2,
            1,
            "%d",
            "Bedrooms, hall, and kitchen configuration.",
            ":material/bed:",
        )
        bath = range_control(
            "Bathrooms (count)",
            "property_bath",
            1,
            bath_max,
            2,
            1,
            "%d",
            None,
            ":material/bathtub:",
        )
        balcony = range_control(
            "Balconies (count)",
            "property_balcony",
            0,
            balcony_max,
            1,
            1,
            "%d",
            None,
            ":material/balcony:",
        )
        max_price = range_control(
            "Maximum budget (₹ lakh)",
            "property_budget",
            budget_min,
            budget_max,
            150,
            5,
            "%d",
            "Budget range follows historical prices in the dataset.",
            ":material/currency_rupee:",
        )

        user_query = st.text_area(
            "Matching priorities (optional)",
            placeholder=(
                "For example: prefer larger square feet, fewer balconies, "
                "or a historical price closest to the estimate"
            ),
            key="property_priorities",
            help=(
                "Supported priorities: square feet, bathrooms, balconies, "
                "and historical price."
            ),
            persist_state="session",
        )

        matching_priorities = normalize_matching_priorities(user_query)
        unsupported_preferences = find_unverifiable_preferences(user_query)

        if unsupported_preferences:
            st.warning(
                "Not used for matching: "
                + ", ".join(unsupported_preferences)
                + ". These details are absent from the dataset.",
                icon=":material/info:",
            )
        elif user_query.strip() and not matching_priorities:
            st.caption(
                ":material/info: No supported priority found; property details "
                "will drive the ranking."
            )


input_is_valid = True
validation_message = None
if total_sqft / bhk < 300:
    input_is_valid = False
    validation_message = "Area must be at least 300 square feet per BHK."
elif bath > bhk + 2:
    input_is_valid = False
    validation_message = "Bathrooms appear unusually high for the selected BHK."

predicted_price = None
if input_is_valid:
    predicted_price = predict_house_price(
        model=model,
        model_columns=model_columns,
        location=location,
        area_type=area_type,
        total_sqft=total_sqft,
        bath=bath,
        balcony=balcony,
        bhk=bhk,
    )

local_market = house_data[
    (house_data["location"] == location) & (house_data["bhk"] == bhk)
].copy()
if local_market.empty:
    local_market = house_data[house_data["location"] == location].copy()

local_median = float(local_market["price"].median())
median_price_per_sqft = float(local_market["price_per_sqft"].median())

with preview_column:
    with st.container(border=True, key="live_card", gap="small"):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader(":material/bolt: Live estimate")
            st.space("stretch")
            if predicted_price is not None and predicted_price <= max_price:
                st.badge("Within budget", icon=":material/check_circle:", color="green")
            elif predicted_price is not None:
                st.badge("Above budget", icon=":material/trending_up:", color="orange")

        if predicted_price is None:
            st.error(validation_message, icon=":material/error:")
            st.metric("Estimated property price", "—")
        else:
            delta = predicted_price - local_median
            with st.container(key="price_hero"):
                st.metric(
                    "Estimated property price",
                    f"₹{predicted_price:,.2f} lakh",
                    delta=(
                        f"₹{abs(delta):,.2f}L "
                        + ("above" if delta >= 0 else "below")
                        + " local median"
                    ),
                    delta_color="off",
                    delta_arrow="off",
                    help="Regression estimate—not statistical certainty.",
                )

        st.caption("Updates instantly when a slider or exact-value box changes.")

    with st.container(border=True, key="market_card", gap="small"):
        st.subheader(":material/location_city: Neighborhood pulse")
        with st.container(horizontal=True):
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
            st.metric(
                "Local records",
                f"{len(local_market):,}",
                border=True,
                icon=":material/database:",
            )

        st.progress(
            min(1.0, len(local_market) / 25),
            text=f"Historical coverage: {len(local_market)} relevant records",
        )

    if st.button(
        "Find matches and open insights",
        type="primary",
        icon=":material/arrow_forward:",
        width="stretch",
        key="primary_cta",
        disabled=not input_is_valid,
    ):
        semantic_query = (
            f"{matching_priorities or 'No additional ranking priority.'} "
            f"Location: {location}. Required area: {total_sqft} square feet. "
            f"BHK: {bhk}. Bathrooms: {bath}. Balconies: {balcony}. "
            f"Maximum budget: {max_price} lakhs."
        )

        embedding_warning = False
        try:
            embedding_model = load_embedding_model()
        except Exception:
            embedding_model = None
            embedding_warning = True

        with st.spinner("Ranking historical properties..."):
            comparable_properties = hybrid_retrieve(
                data=house_data,
                embedding_model=embedding_model,
                query=semantic_query,
                location=location,
                bhk=bhk,
                max_price=max_price,
                total_sqft=total_sqft,
                bath=bath,
                balcony=balcony,
                target_price=predicted_price,
                k=5,
            )

        st.session_state.prediction_result = {
            "location": location,
            "area_type": area_type,
            "total_sqft": total_sqft,
            "bath": bath,
            "balcony": balcony,
            "bhk": bhk,
            "max_price": max_price,
            "predicted_price": predicted_price,
            "matching_priorities": matching_priorities,
            "comparables": comparable_properties,
            "embedding_warning": embedding_warning,
            "assistant_response": None,
            "assistant_attempted": False,
        }
        st.switch_page("app_pages/insights.py")
