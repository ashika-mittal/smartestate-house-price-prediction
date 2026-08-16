import numpy as np
import pandas as pd


def predict_house_price(
    model,
    model_columns,
    location,
    area_type,
    total_sqft,
    bath,
    balcony,
    bhk
):
    input_data = pd.DataFrame(
        np.zeros((1, len(model_columns))),
        columns=model_columns
    )

    input_data.loc[
        0,
        ["total_sqft", "bath", "balcony", "bhk"]
    ] = [total_sqft, bath, balcony, bhk]

    location_column = f"location_{location}"
    area_type_column = f"area_type_{area_type}"

    if location_column in model_columns:
        input_data.loc[0, location_column] = 1
    elif "location_other" in model_columns:
        input_data.loc[0, "location_other"] = 1

    if area_type_column in model_columns:
        input_data.loc[0, area_type_column] = 1

    prediction = model.predict(input_data)[0]

    return float(prediction)