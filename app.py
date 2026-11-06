import json
import os
import sqlite3
from datetime import datetime, timezone

import streamlit as st
from chromadb.errors import ChromaError
from openai import APIConnectionError, APIError, AuthenticationError, RateLimitError
from pypdf.errors import PdfReadError

from rag import analyze_resume, extract_pdf_text


MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_TEXT_LENGTH = 30_000
MAX_RESUME_TEXT_LENGTH = 100_000
HISTORY_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_history.db")


def init_history_db() -> None:
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                overall_summary TEXT NOT NULL,
                matching_skills TEXT NOT NULL,
                missing_skills TEXT NOT NULL,
                experience_gaps TEXT NOT NULL,
                recommendations TEXT NOT NULL
            )
            """
        )


def save_analysis_history(filename: str, result: dict) -> None:
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO analysis_history (
                created_at, filename, overall_summary,
                matching_skills, missing_skills, experience_gaps, recommendations
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                filename,
                result["overall_summary"],
                json.dumps(result["matching_skills"]),
                json.dumps(result["missing_skills"]),
                json.dumps(result["experience_gaps"]),
                json.dumps(result["recommendations"]),
            ),
        )


def get_recent_history(limit: int = 5) -> list[dict]:
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT created_at, filename, overall_summary, matching_skills, missing_skills,
                   experience_gaps, recommendations
            FROM analysis_history
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "created_at": row[0],
            "filename": row[1],
            "overall_summary": row[2],
            "matching_skills": json.loads(row[3]),
            "missing_skills": json.loads(row[4]),
            "experience_gaps": json.loads(row[5]),
            "recommendations": json.loads(row[6]),
        }
        for row in rows
    ]


init_history_db()

st.set_page_config(page_title="Resume Gap Analyzer", page_icon="🎯", layout="wide")

st.markdown(
    """
    <style>
    .block-container { max-width: 1100px; padding-top: 2rem; }
    .hero { padding: 1.5rem 1.75rem; border-radius: 16px; background: linear-gradient(120deg, #10243a, #176b70); color: white; margin-bottom: 1.5rem; }
    .hero h1 { margin: 0 0 .45rem; color: white; }
    .hero p { margin: 0; color: #e0f2f1; font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>Resume-to-JD Gap Analyzer</h1>'
    "<p>Find evidence-backed skill matches, gaps, and practical next steps.</p></div>",
    unsafe_allow_html=True,
)
st.caption(
    "Your resume is processed for this analysis only. Its text is not saved by this app; "
    "the extracted text is sent to OpenAI for embeddings and analysis."
)

with st.sidebar:
    st.header("Model configuration")
    api_key = st.text_input(
        "OpenAI API key",
        value=os.environ.get("OPENAI_API_KEY", ""),
        type="password",
        help="Used for embeddings and analysis. You can also set OPENAI_API_KEY in your environment.",
    )
    st.caption("Embedding: text-embedding-3-small · Analysis: gpt-4o-mini")

    st.subheader("Recent analyses")
    history = get_recent_history()
    if not history:
        st.info("No saved analyses yet.")
    else:
        for item in history:
            st.markdown(f"**{item['filename']}**")
            st.caption(item["created_at"][:19].replace("T", " "))
            st.caption(item["overall_summary"])
            st.divider()

left, right = st.columns(2)
with left:
    uploaded_file = st.file_uploader("Resume PDF", type=["pdf"])
with right:
    job_description = st.text_area(
        "Target job description",
        height=240,
        placeholder="Paste the full job description here…",
    )

analyze_button = st.button("Analyze gaps", type="primary", use_container_width=True)

if analyze_button:
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("analysis_filename", None)
    if not api_key.strip():
        st.error("Enter an OpenAI API key in the sidebar to run an analysis.")
    elif uploaded_file is None:
        st.error("Upload a resume PDF to continue.")
    elif uploaded_file.size > MAX_PDF_BYTES:
        st.error("The PDF exceeds the 10 MB limit. Please upload a smaller file.")
    elif not job_description.strip():
        st.error("Paste a job description to continue.")
    elif len(job_description) > MAX_TEXT_LENGTH:
        st.error("The job description exceeds the 30,000-character limit.")
    else:
        try:
            resume_text = extract_pdf_text(uploaded_file.getvalue())
            if len(resume_text) > MAX_RESUME_TEXT_LENGTH:
                raise ValueError("The extracted resume text exceeds the 100,000-character limit.")
            with st.spinner("Embedding documents, retrieving relevant evidence, and analyzing…"):
                result = analyze_resume(resume_text, job_description.strip(), api_key.strip())
            st.session_state["analysis_result"] = result
            st.session_state["analysis_filename"] = uploaded_file.name
            save_analysis_history(uploaded_file.name, result)
        except AuthenticationError:
            st.error("OpenAI rejected the API key. Check the key and try again.")
        except RateLimitError:
            st.error("The OpenAI API rate limit or quota was reached. Try again later.")
        except APIConnectionError:
            st.error("Could not connect to OpenAI. Check your connection and try again.")
        except APIError as error:
            st.error(f"The OpenAI request failed: {error}")
        except PdfReadError as error:
            st.error(f"The uploaded file is not a readable PDF: {error}")
        except ChromaError as error:
            st.error(f"The temporary vector database failed: {error}")
        except ValueError as error:
            st.error(str(error))


def render_items(title: str, items: list[dict[str, str]], fields: list[tuple[str, str]]) -> None:
    st.subheader(title)
    if not items:
        st.info("No items identified.")
        return
    for item in items:
        heading = item.get(fields[0][0], title)
        with st.container(border=True):
            st.markdown(f"**{heading}**")
            for label, key in fields[1:]:
                st.markdown(f"**{label}:** {item[key]}")


result = st.session_state.get("analysis_result")
if result:
    st.divider()
    st.caption(f"Analysis for {st.session_state.get('analysis_filename', 'uploaded resume')}")
    st.subheader("Overall assessment")
    st.write(result["overall_summary"])

    render_items(
        "Matching skills",
        result["matching_skills"],
        [("skill", "Skill"), ("Resume evidence", "resume_evidence"), ("Job requirement", "jd_requirement")],
    )
    render_items(
        "Missing or unconfirmed skills",
        result["missing_skills"],
        [("skill", "Skill"), ("Job requirement", "jd_requirement"), ("How to address it", "suggestion")],
    )
    render_items(
        "Experience gaps",
        result["experience_gaps"],
        [("requirement", "Requirement"), ("Resume evidence", "resume_evidence"), ("Gap", "gap")],
    )
    render_items(
        "Recommendations",
        result["recommendations"],
        [("action", "Action"), ("Priority", "priority"), ("Why it helps", "reason"), ("Timeframe", "timeframe")],
    )
