# SmartEstate – House Price Prediction & GenAI RAG Assistant

SmartEstate is a two-page Streamlit application that estimates Bengaluru house
prices, ranks comparable historical properties, and produces a grounded
AI-assisted explanation of the result.

The application combines a trained regression model with structured filtering,
semantic retrieval, and a Groq-hosted language model. Retrieved properties are
historical dataset records—not live listings.

## Features

- Bengaluru house-price estimation using Linear Regression
- Instant estimate updates as property inputs change
- Synchronized sliders and exact-value number inputs
- Realistic, dataset-informed ranges for area, BHK, bathrooms, balconies, and budget
- Mandatory comparable-property filtering by location, BHK, and maximum budget
- Hybrid ranking using property differences, predicted-price proximity,
  SentenceTransformer embeddings, and FAISS
- Full-width ranked-comparables table with explicit match categories and differences
- Grounded GenAI RAG Assistant powered by Groq
- Bright and dark appearance modes with responsive desktop and mobile layouts
- Graceful fallbacks when embeddings, the Groq API, or matching records are unavailable

## Application Pages

### 1. Predict

The prediction page contains:

- Location and area-type selectors
- Total area, BHK, bathroom, balcony, and maximum-budget controls
- Optional natural-language matching priorities
- Live model estimate
- Historical neighborhood median, median price per square foot, and record count

Submitting the form ranks eligible historical records and opens the report page.

### 2. Insights & Assistant

The insights page contains:

- Property and model-estimate summary
- Historical median and median price-per-square-foot metrics
- Top historical-match score
- A full-width, expandable table of ranked comparable properties
- A grounded GenAI RAG Assistant explanation

Property 1 is always the highest-ranked overall match. It is not necessarily the
closest property on every individual measure.

## Model Performance

The final Linear Regression model achieved:

- R² score: 0.8411
- Mean Absolute Error: ₹18.52 lakh
- Root Mean Squared Error: ₹35.72 lakh

Linear Regression was selected because it demonstrated more consistent
cross-validation performance than the evaluated alternatives.

## Application Pipeline

```text
Property details
      |
      v
Linear Regression price estimate
      |
      v
Location + BHK + budget filtering
      |
      v
Structured and semantic ranking
      |
      v
Ranked historical comparables
      |
      v
Grounded Groq explanation
```

## Project Structure

```text
House Rate Prediction/
├── app.py                         # Streamlit router and shared page shell
├── app_pages/
│   ├── predict.py                 # Property inputs and live estimate
│   └── insights.py                # Comparables and GenAI report
├── src/
│   ├── __init__.py
│   ├── assistant.py               # Guarded Groq prompting and responses
│   ├── config.py                  # Project paths and model names
│   ├── prediction.py              # Regression inference
│   ├── preferences.py             # Natural-language preference parsing
│   ├── resources.py               # Cached models, data, and API client
│   ├── retrieval.py               # Filtering and hybrid ranking
│   └── ui.py                      # Shared controls and visual theme
├── Data/
│   ├── raw/
│   └── processed/
├── Models/
│   ├── house_price_model.pkl
│   └── model_columns.json
├── notebooks/
├── reports/
├── tests/
│   └── test_pipeline.py
├── .streamlit/
│   └── config.toml
├── .gitignore
├── requirements.txt
└── README.md
```

The local `.env` file is intentionally excluded from version control.

## Local Setup

### 1. Open the project directory

```bash
cd "/path/to/House Rate Prediction"
```

### 2. Activate the environment

The project was developed and validated with Python 3.10 in the
`houseprice_automl` Conda environment:

```bash
conda activate houseprice_automl
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a local `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
```

Never commit `.env`, `.streamlit/secrets.toml`, or an API key.

### 5. Start SmartEstate

Run Streamlit from the repository root so local and cloud file paths behave the
same way:

```bash
python -m streamlit run app.py
```

Open the local address printed by Streamlit, normally:

```text
http://localhost:8501
```

## How Retrieval Works

The retrieval pipeline first applies mandatory filters:

- Exact location
- Exact BHK
- Maximum budget

Square feet, bathrooms, and balconies are ranking preferences rather than hard
exclusion filters. An exact match must match the requested bathrooms and
balconies and fall within 5% of the requested square feet. Remaining positions
are filled with clearly labelled close or alternative matches, with their
differences displayed explicitly.

Eligible records are converted into natural-language descriptions.
SentenceTransformer and FAISS contribute semantic preference similarity, while
structured differences and proximity to the model estimate contribute to the
final score. Current listing status is intentionally excluded because the
source data is historical.

The `all-MiniLM-L6-v2` SentenceTransformer model is downloaded on first use. If
it cannot be loaded—for example, while offline—the application continues with
deterministic structured ranking.

## How the GenAI Assistant Works

The regression estimate and ranked historical records are passed to Groq. The
guarded prompt requires the assistant to:

- Separate the model estimate from historical recorded prices
- Preserve the retrieval order and recommend the highest-ranked record first
- Recommend only properties returned by the retrieval pipeline
- Avoid inventing property information or current-listing status
- Explain that historical records may not represent current listings

The generated response is validated before display. Unsafe or unsupported
claims are corrected, and repeated validation failures use a deterministic
fallback. When retrieval returns no records, SmartEstate skips the Groq request.

## Validation

Compile the application files:

```bash
python -m py_compile app.py app_pages/predict.py app_pages/insights.py src/*.py
```

Run the automated pipeline tests:

```bash
python -m unittest discover -s tests -v
```

The current test suite covers prediction, mandatory retrieval filters, ranking,
fallback retrieval, AI-response validation, and unsupported-claim handling.

## Deploying to Streamlit Community Cloud

Push the project to GitHub, then create an app at
[share.streamlit.io](https://share.streamlit.io/) with:

- Repository: your SmartEstate GitHub repository
- Branch: `main`
- Entrypoint file: `app.py`
- Python version: match the version used to validate the project

In **Advanced settings → Secrets**, add the Groq key using TOML syntax:

```toml
GROQ_API_KEY = "your_actual_groq_api_key"
```

Do not upload the local `.env` file. Root-level Streamlit secrets are exposed as
environment variables, so the existing Groq client configuration can read the
deployed key.

After deployment, verify prediction, page navigation, ranked comparables,
fullscreen table controls, bright/dark mode, mobile responsiveness, and the
GenAI response.

## Technology Stack

- Python
- Streamlit
- pandas and NumPy
- scikit-learn
- SentenceTransformers
- FAISS
- Groq API

## Important Limitations

- Predictions are estimates based on historical Bengaluru housing data.
- Comparable properties are historical records, not current listings.
- The application does not provide professional real-estate or financial advice.
- Accuracy may decrease for unusual or previously unseen property combinations.
- Market conditions may have changed since the dataset was collected.

## Future Improvements

- Precomputed embeddings for faster cold starts
- Broader and typo-tolerant location matching
- Model-drift and data-quality monitoring
- Feature-level prediction explanations
- Automated deployment checks in GitHub Actions
