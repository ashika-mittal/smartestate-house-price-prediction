import unittest

import numpy as np

from src.assistant import generate_groq_response
from src.prediction import predict_house_price
from src.resources import load_model, load_reference_data
from src.retrieval import hybrid_retrieve


class FakeEmbeddingModel:
    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ):
        embeddings = []

        for text in texts:
            vector = np.array(
                [
                    len(text),
                    text.lower().count("balcony") + 1,
                    text.lower().count("bathroom") + 1
                ],
                dtype="float32"
            )

            if normalize_embeddings:
                vector = vector / np.linalg.norm(vector)

            embeddings.append(vector)

        return np.array(
            embeddings,
            dtype="float32"
        )


class FakeGroqClient:
    def __init__(self, responses=None):
        self.last_request = None
        self.requests = []
        self.responses = list(responses or ["Test response"])
        self.chat = self.Chat(self)

    class Chat:
        def __init__(self, client):
            self.completions = FakeGroqClient.Completions(client)

    class Completions:
        def __init__(self, client):
            self.client = client

        def create(self, **kwargs):
            self.client.last_request = kwargs
            self.client.requests.append(kwargs)

            if self.client.responses:
                content = self.client.responses.pop(0)
            else:
                content = "Test response"

            message = type(
                "Message",
                (),
                {"content": content}
            )()

            choice = type(
                "Choice",
                (),
                {"message": message}
            )()

            return type(
                "Response",
                (),
                {"choices": [choice]}
            )()


class TestHousePricePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model, cls.model_columns = load_model()
        cls.data = load_reference_data()
        cls.embedding_model = FakeEmbeddingModel()

    def test_known_price_prediction(self):
        price = predict_house_price(
            model=self.model,
            model_columns=self.model_columns,
            location="1st Phase JP Nagar",
            area_type="Super built-up  Area",
            total_sqft=1200,
            bath=2,
            balcony=2,
            bhk=2
        )

        self.assertEqual(
            round(price, 2),
            98.53
        )

    def test_retrieval_respects_mandatory_filters(self):
        results = hybrid_retrieve(
            data=self.data,
            embedding_model=self.embedding_model,
            query=(
                "I want a home around "
                "1200 square feet with two balconies."
            ),
            location="1st Phase JP Nagar",
            total_sqft=1200,
            bhk=2,
            bath=2,
            balcony=2,
            max_price=150,
            k=5
        )

        self.assertFalse(results.empty)

        self.assertTrue(
            (
                results["location"]
                == "1st Phase JP Nagar"
            ).all()
        )

        self.assertTrue(
            (results["bhk"] == 2).all()
        )

        self.assertTrue(
            (results["price"] <= 150).all()
        )

    def test_retrieval_accepts_prediction_as_ranking_input(self):
        results = hybrid_retrieve(
            data=self.data,
            embedding_model=self.embedding_model,
            query="Prefer historical prices closest to the estimate.",
            location="1st Phase JP Nagar",
            total_sqft=1200,
            bhk=2,
            bath=2,
            balcony=2,
            max_price=150,
            target_price=98.53,
            k=5
        )

        self.assertFalse(results.empty)
        self.assertTrue((results["price"] <= 150).all())

    def test_closest_area_ranks_first(self):
        results = hybrid_retrieve(
            data=self.data,
            embedding_model=self.embedding_model,
            query=(
                "Property around "
                "1200 square feet"
            ),
            location="1st Phase JP Nagar",
            total_sqft=1200,
            bhk=2,
            bath=2,
            balcony=2,
            max_price=150,
            k=5
        )

        self.assertFalse(results.empty)

        self.assertEqual(
            results.iloc[0]["total_sqft"],
            1180
        )

        self.assertEqual(
            results.iloc[0]["match_type"],
            "Exact match"
        )

    def test_soft_difference_returns_alternatives(self):
        results = hybrid_retrieve(
            data=self.data,
            embedding_model=self.embedding_model,
            query="Property with ten balconies",
            location="1st Phase JP Nagar",
            total_sqft=1200,
            bhk=2,
            bath=2,
            balcony=10,
            max_price=150,
            k=5
        )

        self.assertFalse(results.empty)

        self.assertTrue(
            results["differences"].str.contains("balconies").all()
        )

        self.assertTrue(
            (results["match_type"] != "Exact match").all()
        )

    def test_area_outside_tolerance_is_not_exact(self):
        results = hybrid_retrieve(
            data=self.data,
            embedding_model=self.embedding_model,
            query="Property around 1000 square feet",
            location="Arekere",
            total_sqft=1000,
            bhk=2,
            bath=2,
            balcony=1,
            max_price=150,
            k=1000
        )

        nine_hundred_sqft = results[
            (results["total_sqft"] == 900)
            & (results["bath"] == 2)
            & (results["balcony"] == 1)
        ]

        self.assertFalse(nine_hundred_sqft.empty)

        self.assertTrue(
            (
                nine_hundred_sqft["match_type"]
                == "Close match"
            ).all()
        )

        self.assertTrue(
            nine_hundred_sqft["differences"]
            .str.contains("square feet")
            .all()
        )

    def test_no_mandatory_match_returns_empty_result(self):
        results = hybrid_retrieve(
            data=self.data,
            embedding_model=None,
            query="Property under twenty lakhs",
            location="1st Phase JP Nagar",
            total_sqft=1200,
            bhk=2,
            bath=2,
            balcony=2,
            max_price=20,
            k=5
        )

        self.assertTrue(results.empty)

    def test_retrieval_falls_back_without_embedding_model(self):
        results = hybrid_retrieve(
            data=self.data,
            embedding_model=None,
            query="Property around 1200 square feet",
            location="1st Phase JP Nagar",
            total_sqft=1200,
            bhk=2,
            bath=2,
            balcony=2,
            max_price=150,
            target_price=98.53,
            k=5
        )

        self.assertFalse(results.empty)
        self.assertEqual(results.iloc[0]["match_type"], "Exact match")
        self.assertTrue((results["price"] <= 150).all())

    def test_ai_prompt_preserves_retrieval_ranking(self):
        results = hybrid_retrieve(
            data=self.data,
            embedding_model=self.embedding_model,
            query="Property with three balconies",
            location="Basaveshwara Nagar",
            total_sqft=1100,
            bhk=3,
            bath=3,
            balcony=3,
            max_price=150,
            k=5
        )

        client = FakeGroqClient()

        response = generate_groq_response(
            client=client,
            user_query="I prefer a property with three balconies.",
            location="Basaveshwara Nagar",
            area_type="Super built-up  Area",
            total_sqft=1100,
            bath=3,
            balcony=3,
            bhk=3,
            max_price=150,
            predicted_price=101.17,
            retrieved_properties=results
        )

        prompt = client.last_request["messages"][1]["content"]

        self.assertEqual(response, "Test response")
        self.assertIn(
            "Property 1 is the highest-ranked",
            prompt
        )
        self.assertIn(
            "If recommending one property, recommend Property 1",
            prompt
        )
        self.assertIn("Retrieval score:", prompt)

        historical_context = prompt.split(
            "FAISS-ranked historical properties:",
            maxsplit=1
        )[1].split("Ranking rules:", maxsplit=1)[0]

        self.assertNotIn("- Area type:", historical_context)
        self.assertNotIn("- Availability:", historical_context)
        self.assertNotIn("availability", results.columns)

    def test_ai_response_corrects_false_exact_match(self):
        results = hybrid_retrieve(
            data=self.data,
            embedding_model=self.embedding_model,
            query="Property with a balcony",
            location="Banashankari Stage III",
            total_sqft=1200,
            bhk=2,
            bath=4,
            balcony=1,
            max_price=150,
            k=5
        )

        self.assertFalse(results.empty)
        self.assertNotEqual(
            results.iloc[0]["match_type"],
            "Exact match"
        )

        client = FakeGroqClient(
            responses=[
                (
                    "Property 1 is an exact match except for "
                    "square feet and bathrooms."
                ),
                (
                    "Property 1 is an Alternative match and differs "
                    "from the request in square feet and bathrooms."
                )
            ]
        )

        response = generate_groq_response(
            client=client,
            user_query="I prefer a property with a balcony.",
            location="Banashankari Stage III",
            area_type="Super built-up  Area",
            total_sqft=1200,
            bath=4,
            balcony=1,
            bhk=2,
            max_price=150,
            predicted_price=75.09,
            retrieved_properties=results
        )

        self.assertEqual(len(client.requests), 2)
        self.assertNotIn("exact match except", response.lower())
        self.assertIn(
            results.iloc[0]["match_type"],
            response
        )

    def test_ai_response_removes_current_listing_status_claim(self):
        results = hybrid_retrieve(
            data=self.data,
            embedding_model=self.embedding_model,
            query="Property with a balcony",
            location="Banashankari Stage III",
            total_sqft=1200,
            bhk=2,
            bath=4,
            balcony=1,
            max_price=150,
            k=5
        )

        client = FakeGroqClient(
            responses=[
                (
                    "Property 1 is an Alternative match. "
                    "Property 2 is available."
                ),
                (
                    "Property 1 is an Alternative match. "
                    "Property 2 is another Alternative match."
                )
            ]
        )

        response = generate_groq_response(
            client=client,
            user_query="I prefer a property with a balcony.",
            location="Banashankari Stage III",
            area_type="Super built-up  Area",
            total_sqft=1200,
            bath=4,
            balcony=1,
            bhk=2,
            max_price=150,
            predicted_price=75.09,
            retrieved_properties=results
        )

        self.assertEqual(len(client.requests), 2)
        self.assertNotRegex(
            response.lower(),
            r"\b(?:available|availability)\b"
        )
        self.assertIn(
            "Do not use the words 'available' or 'availability'",
            client.requests[1]["messages"][-1]["content"]
        )

    def test_ai_response_uses_safe_fallback_after_two_violations(self):
        results = hybrid_retrieve(
            data=self.data,
            embedding_model=self.embedding_model,
            query="Property with a balcony",
            location="Banashankari Stage III",
            total_sqft=1200,
            bhk=2,
            bath=4,
            balcony=1,
            max_price=150,
            k=5
        )

        client = FakeGroqClient(
            responses=[
                "Property 1 is an exact match except for bathrooms.",
                "Property 1 remains an exact match except for bathrooms."
            ]
        )

        response = generate_groq_response(
            client=client,
            user_query="I prefer a property with a balcony.",
            location="Banashankari Stage III",
            area_type="Super built-up  Area",
            total_sqft=1200,
            bath=4,
            balcony=1,
            bhk=2,
            max_price=150,
            predicted_price=75.09,
            retrieved_properties=results
        )

        self.assertEqual(len(client.requests), 2)
        self.assertNotIn("exact match except", response.lower())
        self.assertIn(
            f"Property 1 — {results.iloc[0]['match_type']}",
            response
        )


if __name__ == "__main__":
    unittest.main()
