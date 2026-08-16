# House Price Prediction – Data Cleaning & Preprocessing Report

## Dataset Overview

The Bengaluru House Price dataset initially contained **13,320 records** and **9 features** describing residential properties, including location, size, total area, number of bathrooms, balconies, and price.

### Initial Features

- area_type
- availability
- location
- size
- society
- total_sqft
- bath
- balcony
- price

---

# Data Quality Issues Identified

The dataset contained several challenges:

1. Missing values in size, society, bath, balcony, and location.
2. Mixed formats in the total_sqft column.
3. High-cardinality categorical features such as location and society.
4. Inconsistent property configurations.
5. Price and area outliers affecting model reliability.

---

# Data Cleaning Steps Performed

## 1. Handling Missing Size Values

Rows with missing values in the size column were removed because BHK information could not be derived from them.

**Rows removed:** 16

Dataset size:

- Before: 13,320
- After: 13,304

---

## 2. Feature Engineering – BHK Extraction

A new feature called **bhk** was created by extracting the numerical component from the size column.

Examples:

| Size | BHK |
|--------|--------|
| 2 BHK | 2 |
| 4 Bedroom | 4 |

This feature was later used for outlier detection and model training.

---

## 3. Standardizing Total Area

The total_sqft column contained:

- Single numeric values
- Area ranges (e.g., 2100 - 2850)
- Different units (Sq. Meter, Sq. Yards, Acres, Guntha, etc.)

Processing performed:

- Area ranges were replaced with their average value.
- Alternative units were converted into square feet.
- Invalid entries were removed.

Result:

A standardized numeric total_sqft feature was obtained for all records.

---

## 4. Price Per Square Foot Feature

A derived feature was created:

price_per_sqft = (price × 100000) / total_sqft

This metric was used to identify unusual pricing patterns and detect outliers.

---

## 5. Area Per Bedroom Feature

A second derived feature was created:

sqft_per_bhk = total_sqft / bhk

This feature was used to identify unrealistic property configurations.

---

## 6. Removing Unrealistic Area Configurations

Properties with:

sqft_per_bhk < 300

were considered unrealistic and removed.

Examples:

- 8 BHK in 600 sqft
- 6 BHK in 1020 sqft

Dataset size:

- Before: 13,304
- After: 12,556

---

## 7. Handling High Cardinality Locations

The location feature originally contained more than 1,200 unique values.

Locations appearing fewer than 10 times were grouped into a common category called:

other

This reduced noise and prevented excessive dimensionality during encoding.

---

## 8. Location-wise Price Outlier Removal

Property prices vary significantly across locations.

Instead of using a global threshold, outliers were removed separately within each location using:

mean ± standard deviation

of price_per_sqft.

Dataset size:

- Before: 12,556
- After: 10,339

---

## 9. BHK-wise Price Consistency Check

Within the same location, larger properties should generally not be significantly cheaper than smaller properties.

Example:

- 3 BHK priced below the typical 2 BHK price
- 4 BHK priced below the typical 3 BHK price

Such anomalous records were removed.

Dataset size:

- Before: 10,339
- After: 7,302

---

## 10. Manual Extreme Outlier Removal

Two properties exhibited unrealistic area values:

- 653,400 sqft
- 26,460 sqft

These records produced abnormally low price_per_sqft values and were removed.

Dataset size:

- Before: 7,302
- After: 7,300

---

## 11. Bathroom Imputation

The bath column contained 32 missing values.

Instead of deleting these records, missing bathrooms were filled using:

Median bathroom count within the same BHK category.

Examples:

- Missing bath in a 2 BHK → filled using median bath of 2 BHK properties.
- Missing bath in a 3 BHK → filled using median bath of 3 BHK properties.

This preserved useful observations while maintaining consistency.

---

## 12. Bathroom Outlier Removal

Properties satisfying:

bath > bhk + 2

were treated as anomalies and removed.

Examples:

- 3 BHK with 6 bathrooms
- 4 BHK with 8 bathrooms

Dataset size:

- Before: 7,300
- After: 7,296

---

## 13. Society Feature Analysis

The society feature contained:

- 2,503 missing values (~34%)
- 1,715 unique society names

Additionally, 87% of society names appeared fewer than five times.

Due to extreme sparsity and high cardinality, the feature was excluded from the traditional machine learning model.

However, it may be retained for future recommendation and GenAI-based retrieval systems.

---

# Final Dataset

Final records: **7,296**

The dataset is fully cleaned and ready for machine learning model development.

Key features retained:

- area_type
- availability
- location
- total_sqft
- bath
- balcony
- bhk
- price

Additional helper features used during preprocessing:

- price_per_sqft
- sqft_per_bhk

---

# Key Insights

1. Location is the strongest factor influencing house prices.
2. Rare locations introduce noise and require grouping.
3. Price per square foot is highly effective for detecting pricing anomalies.
4. Bathroom and bedroom counts show strong correlation with property size and value.
5. Careful outlier removal reduced the dataset from 13,320 to 7,296 high-quality observations, improving reliability for model training.