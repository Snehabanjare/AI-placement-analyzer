import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rag import analyze_resume, chunk_text, extract_pdf_text, normalize_analysis


class ChunkTextTests(unittest.TestCase):
    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(chunk_text(""), [])

    def test_chunks_overlap_and_cover_all_words(self):
        words = [f"word{i}" for i in range(10)]
        chunks = chunk_text(" ".join(words), chunk_words=4, overlap=1)

        self.assertEqual(chunks, ["word0 word1 word2 word3", "word3 word4 word5 word6", "word6 word7 word8 word9"])

    def test_rejects_invalid_chunk_settings(self):
        with self.assertRaises(ValueError):
            chunk_text("some words", chunk_words=3, overlap=3)


class NormalizeAnalysisTests(unittest.TestCase):
    def test_accepts_empty_categories_and_trims_strings(self):
        result = normalize_analysis(
            {
                "overall_summary": "  A good fit.  ",
                "matching_skills": [],
                "missing_skills": [],
                "experience_gaps": [],
                "recommendations": [],
            }
        )

        self.assertEqual(result["overall_summary"], "A good fit.")
        self.assertEqual(result["matching_skills"], [])

    def test_rejects_malformed_category_entry(self):
        with self.assertRaises(ValueError):
            normalize_analysis(
                {
                    "overall_summary": "Summary",
                    "matching_skills": [{"skill": "Python"}],
                    "missing_skills": [],
                    "experience_gaps": [],
                    "recommendations": [],
                }
            )


class ExtractPdfTextTests(unittest.TestCase):
    def test_ocr_fallback_runs_when_pdf_text_is_empty(self):
        with patch("pypdf.PdfReader") as pdf_reader, patch("pdf2image.convert_from_bytes") as convert, patch(
            "pytesseract.image_to_string", return_value="OCR text"
        ):
            pdf_reader.return_value.pages = [SimpleNamespace(extract_text=lambda: "")]
            convert.return_value = ["image-1"]

            self.assertEqual(extract_pdf_text(b"dummy"), "OCR text")


class AnalyzeResumeTests(unittest.TestCase):
    def test_retrieves_resume_and_job_chunks_and_cleans_up_collection(self):
        response_data = {
            "overall_summary": "Relevant evidence was found.",
            "matching_skills": [],
            "missing_skills": [],
            "experience_gaps": [],
            "recommendations": [],
        }
        collection = Mock()
        collection.query.side_effect = [
            {
                "documents": [["Resume evidence"]],
                "metadatas": [[{"chunk": 0, "source": "resume"}]],
                "distances": [[0.1]],
            },
            {
                "documents": [["Job requirement"]],
                "metadatas": [[{"chunk": 0, "source": "job_description"}]],
                "distances": [[0.1]],
            },
        ]
        vector_client = Mock()
        vector_client.create_collection.return_value = collection

        model_client = Mock()
        model_client.embeddings.create.side_effect = lambda **kwargs: SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(index)]) for index, _ in enumerate(kwargs["input"])]
        )
        model_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response_data)))]
        )

        with patch("chromadb.Client", return_value=vector_client), patch(
            "openai.OpenAI", return_value=model_client
        ):
            result = analyze_resume("Resume evidence", "Job requirement", "test-key")

        self.assertEqual(result["overall_summary"], response_data["overall_summary"])
        self.assertEqual(collection.add.call_args.kwargs["metadatas"][0]["source"], "resume")
        self.assertEqual(collection.add.call_args.kwargs["metadatas"][1]["source"], "job_description")
        self.assertEqual(collection.query.call_args_list[0].kwargs["where"], {"source": "resume"})
        self.assertEqual(collection.query.call_args_list[1].kwargs["where"], {"source": "job_description"})
        vector_client.delete_collection.assert_called_once()


if __name__ == "__main__":
    unittest.main()
