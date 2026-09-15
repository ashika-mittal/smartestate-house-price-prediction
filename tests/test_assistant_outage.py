from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest


class TestAssistantOutage(unittest.TestCase):
    def test_outage_keeps_summary_and_retry_recovers(self):
        records = pd.DataFrame([dict(
            location="1st Phase JP Nagar", total_sqft=1200, bhk=2,
            bath=2, balcony=1, price=95.0, price_per_sqft=7916.67,
            match_type="Exact match", differences="None", match_score=0.9,
        )])
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app_pages/insights.py"))
        app.session_state["prediction_result"] = dict(
            location="1st Phase JP Nagar", area_type="Super built-up Area",
            total_sqft=1200, bhk=2, bath=2, balcony=1, max_price=150,
            predicted_price=98.5, comparables=records,
            matching_priorities="", embedding_warning=False,
            assistant_attempted=False,
        )
        with patch("src.resources.load_reference_data", return_value=records), \
             patch("src.resources.load_groq_client", return_value=object()), \
             patch("src.assistant.generate_groq_response", side_effect=[
                 RuntimeError("Simulated service outage"), "Recovered AI comparison"
             ]):
            app.run(timeout=15)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.error), 0)
            self.assertTrue(any("Historical property comparison" in m.value
                                for m in app.markdown))
            self.assertTrue(app.session_state["prediction_result"]["assistant_is_fallback"])
            app.button(key="retry_ai_comparison").click().run(timeout=15)
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("Recovered AI comparison" in m.value
                                for m in app.markdown))
            self.assertFalse(app.session_state["prediction_result"]["assistant_is_fallback"])


if __name__ == "__main__":
    unittest.main()
