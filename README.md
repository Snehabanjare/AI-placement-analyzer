# Resume-to-JD Gap Analyzer

A Streamlit MVP that extracts text from a resume PDF, embeds resume and job-description chunks, retrieves relevant passages from an in-memory Chroma vector database, and asks an OpenAI model for a grounded gap analysis.

## Features

- Extracts selectable text from PDF resumes and falls back to OCR for scanned PDFs.
- Chunks both documents and embeds them with `text-embedding-3-small`.
- Stores embeddings in a temporary, in-memory Chroma collection and retrieves relevant passages.
- Uses `gpt-4o-mini` to report matching skills, missing or unconfirmed skills, experience gaps, and prioritized recommendations.
- Requires resume evidence for matches and distinguishes an unmentioned skill from a confirmed lack of skill.
- Saves analysis results locally in SQLite for quick review; the history includes the uploaded filename and generated evidence, but not the original PDF or raw job description.
- Deletes the vector collection after each analysis; the app does not persist uploaded files or extracted text.

## Run locally

Requires Python 3.10 or later, an OpenAI API key with access to the configured models, and a working Tesseract installation for scanned PDFs.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Install Tesseract from https://github.com/tesseract-ocr/tesseract
$env:OPENAI_API_KEY = "your-openai-api-key"
python -m streamlit run app.py
```

Alternatively, enter the API key in the app's sidebar. The key is not written to a file by the app.

## Tests

```powershell
python -m unittest discover -s tests
```

## Data and privacy

Resume text is sent to OpenAI for embedding and analysis. Review the provider's data policies before uploading sensitive personal information. The vector database is in-memory and its collection is deleted when processing finishes; no resume content is saved by this application.

The app supports PDFs up to 10 MB (with extracted text limited to 100,000 characters) and job descriptions up to 30,000 characters. OCR for scanned PDFs requires Tesseract and Poppler to be installed separately. Analysis history is stored in `analysis_history.db` in the project directory and is excluded from Git; delete that database file to remove saved history. The app does not include authentication, so keep it on a trusted local machine unless you add appropriate deployment controls.
