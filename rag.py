import json
import re
import uuid
from typing import Any


EMBEDDING_MODEL = "text-embedding-3-small"
ANALYSIS_MODEL = "gpt-4o-mini"
CHUNK_WORDS = 450
CHUNK_OVERLAP = 80


def extract_pdf_text(pdf_bytes: bytes) -> str:
    from io import BytesIO

    from pypdf import PdfReader

    reader = PdfReader(BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages).strip()
    if text:
        return text

    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except ImportError as exc:
        raise ValueError(
            "No text could be extracted from this PDF. Install the OCR dependencies "
            "(pdf2image and pytesseract) and make sure Tesseract is available on your system."
        ) from exc

    try:
        images = convert_from_bytes(pdf_bytes, dpi=200)
        ocr_text = "\n\n".join(
            pytesseract.image_to_string(image, config="--psm 6") for image in images
        )
    except Exception as exc:  # pragma: no cover - depends on local OCR tooling
        raise ValueError(
            "No text could be extracted from this PDF. If it is a scanned document, "
            "install Tesseract OCR and upload a searchable or scanned PDF."
        ) from exc

    ocr_text = ocr_text.strip()
    if not ocr_text:
        raise ValueError(
            "No text could be extracted from this PDF. If it is a scanned document, "
            "install Tesseract OCR and upload a searchable or scanned PDF."
        )
    return ocr_text


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if chunk_words <= 0 or overlap < 0 or overlap >= chunk_words:
        raise ValueError("Chunk size must be positive and overlap must be smaller than the chunk size.")

    step = chunk_words - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_words]))
        if start + chunk_words >= len(words):
            break
    return chunks


def _embed_texts(client: Any, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def analyze_resume(resume_text: str, job_description: str, api_key: str) -> dict[str, Any]:
    from chromadb import Client
    from openai import OpenAI

    resume_chunks = chunk_text(resume_text)
    job_chunks = chunk_text(job_description)
    if not resume_chunks or not job_chunks:
        raise ValueError("The resume and job description must both contain readable text.")

    client = OpenAI(api_key=api_key)
    collection_name = "resume_analysis_" + re.sub(r"[^a-z0-9]", "", uuid.uuid4().hex)
    vector_client = Client()
    collection = vector_client.create_collection(name=collection_name)
    try:
        documents = resume_chunks + job_chunks
        metadatas = (
            [{"source": "resume", "chunk": index} for index in range(len(resume_chunks))]
            + [{"source": "job_description", "chunk": index} for index in range(len(job_chunks))]
        )
        embeddings = _embed_texts(client, documents)
        collection.add(
            ids=[f"chunk_{index}" for index in range(len(documents))],
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        query_embeddings = _embed_texts(client, job_chunks)
        resume_results = collection.query(
            query_embeddings=query_embeddings,
            n_results=min(8, len(resume_chunks)),
            where={"source": "resume"},
            include=["documents", "metadatas", "distances"],
        )
        job_results = collection.query(
            query_embeddings=query_embeddings,
            n_results=min(2, len(job_chunks)),
            where={"source": "job_description"},
            include=["documents", "metadatas", "distances"],
        )
        retrieved_resume = _select_chunks(resume_results, max_chunks=12)
        retrieved_job = _select_chunks(job_results, max_chunks=8)

        if not retrieved_resume:
            raise ValueError("No resume passages were retrieved. Try a longer or more detailed resume.")

        analysis = _request_analysis(
            client,
            job_description,
            retrieved_resume,
            retrieved_job,
        )
        return normalize_analysis(analysis)
    finally:
        vector_client.delete_collection(name=collection_name)


def _request_analysis(
    client: Any,
    job_description: str,
    resume_chunks: dict[int, str],
    job_chunks: dict[int, str],
) -> Any:
    schema = {
        "overall_summary": "A concise, balanced summary of the candidate's fit.",
        "matching_skills": [
            {
                "skill": "Skill or qualification",
                "resume_evidence": "Direct evidence from the resume",
                "jd_requirement": "Related requirement from the job description",
            }
        ],
        "missing_skills": [
            {
                "skill": "Skill not evidenced in the resume",
                "jd_requirement": "Requirement from the job description",
                "suggestion": "Practical way to address the gap",
            }
        ],
        "experience_gaps": [
            {
                "requirement": "Experience requirement from the job description",
                "resume_evidence": "Relevant resume evidence, or state that none was found",
                "gap": "Specific difference between the requirement and evidence",
            }
        ],
        "recommendations": [
            {
                "priority": "High, Medium, or Low",
                "action": "Concrete improvement action",
                "reason": "Why this action addresses an identified gap",
                "timeframe": "Suggested timeframe",
            }
        ],
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful resume-to-job-description analyst. Use only facts in the "
                "provided job description and retrieved resume passages. Do not infer skills, "
                "credentials, achievements, or years of experience that are not evidenced. "
                "Distinguish a missing resume mention from proof that the candidate lacks a "
                "skill. Cite concise resume evidence for every claimed match. Make recommendations "
                "specific and actionable. Return only a JSON object following this schema: "
                + json.dumps(schema)
            ),
        },
        {
            "role": "user",
            "content": (
                "JOB DESCRIPTION:\n"
                + job_description
                + "\n\nRETRIEVED JOB-DESCRIPTION PASSAGES:\n"
                + _format_chunks(job_chunks)
                + "\n\nRETRIEVED RESUME PASSAGES (chunk numbers are source references):\n"
                + _format_chunks(resume_chunks)
                + "\n\nAnalyze the evidence. In each resume_evidence field, include the relevant "
                "resume chunk number(s), such as [Resume chunk 2]. Return empty arrays when a "
                "category has no evidence-based entries."
            ),
        },
    ]
    response = client.chat.completions.create(
        model=ANALYSIS_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("The language model returned an empty analysis.")
    try:
        return json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("The language model returned invalid JSON. Please retry the analysis.") from error


def _format_chunks(chunks: dict[int, str]) -> str:
    if not chunks:
        return "(No passages retrieved.)"
    return "\n\n".join(f"[Chunk {index}]\n{text}" for index, text in sorted(chunks.items()))


def _select_chunks(results: dict[str, Any], max_chunks: int) -> dict[int, str]:
    matches: dict[int, tuple[float, str]] = {}
    for matched_documents, matched_metadatas, distances in zip(
        results["documents"], results["metadatas"], results["distances"]
    ):
        for document, metadata, distance in zip(matched_documents, matched_metadatas, distances):
            chunk_index = metadata["chunk"]
            if chunk_index not in matches or distance < matches[chunk_index][0]:
                matches[chunk_index] = (distance, document)
    return {
        index: text
        for index, (_, text) in sorted(matches.items(), key=lambda item: item[1][0])[:max_chunks]
    }


def normalize_analysis(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("The analysis response was not a JSON object.")

    list_fields = ("matching_skills", "missing_skills", "experience_gaps", "recommendations")
    required_fields = {
        "matching_skills": ("skill", "resume_evidence", "jd_requirement"),
        "missing_skills": ("skill", "jd_requirement", "suggestion"),
        "experience_gaps": ("requirement", "resume_evidence", "gap"),
        "recommendations": ("priority", "action", "reason", "timeframe"),
    }
    normalized: dict[str, Any] = {
        "overall_summary": _require_string(data.get("overall_summary"), "overall_summary")
    }
    for field in list_fields:
        entries = data.get(field)
        if not isinstance(entries, list):
            raise ValueError(f"The analysis response is missing a valid '{field}' list.")
        normalized[field] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"Entry {index + 1} in '{field}' is not an object.")
            normalized[field].append(
                {
                    key: _require_string(entry.get(key), f"{field}.{key}")
                    for key in required_fields[field]
                }
            )
    return normalized


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"The analysis response is missing a valid '{field}' value.")
    return value.strip()
