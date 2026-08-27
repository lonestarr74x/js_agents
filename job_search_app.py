import streamlit as st
import os
import csv
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
os.environ["USER_AGENT"] = "job-search-agent/1.0"

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import WebBaseLoader
from pypdf import PdfReader
from docx import Document
from typing import TypedDict, Optional
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Works locally (.env via load_dotenv) AND on Streamlit Cloud (st.secrets)
api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")

if not api_key:
    st.error("No ANTHROPIC_API_KEY found. Add it to your .env file locally, "
              "or to Settings > Secrets in Streamlit Cloud.")
    st.stop()

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.3, api_key=api_key)


# --- Shared state ---
class JobSearchState(TypedDict):
    job_url: str
    job_description: Optional[str]
    resume_base: str
    job_title: Optional[str]
    company_name: Optional[str]
    work_mode: Optional[str]
    pay_range: Optional[str]
    fit_critique: Optional[str]
    fit_level: Optional[str]
    tailored_resume: Optional[str]
    cover_letter: Optional[str]


# --- Agent 1: Job Scanner (fetches posting + extracts structured details) ---
def job_scanner_agent(state: JobSearchState) -> JobSearchState:
    # If a job description was already provided manually (fallback), skip scraping
    if state.get('job_description'):
        job_description = state['job_description']
    else:
        try:
            loader = WebBaseLoader(state['job_url'])
            docs = loader.load()
            job_description = docs[0].page_content
        except Exception as e:
            state['job_description'] = ""
            state['job_title'] = "SCRAPE_FAILED"
            return state

        # Some sites return a bot-block/login page instead of the real posting.
        # A real job posting is normally at least a few hundred characters.
        if not job_description or len(job_description.strip()) < 200:
            state['job_description'] = ""
            state['job_title'] = "SCRAPE_FAILED"
            return state

    state['job_description'] = job_description

    prompt = f"""Extract these details from the job posting below. If something isn't
listed, write "not listed" — do not guess or infer.

Job Posting:
{job_description}

Respond in EXACTLY this format (nothing else):
TITLE: [job title]
COMPANY: [company name]
WORK_MODE: [remote / hybrid / onsite / not listed]
PAY_RANGE: [as listed, e.g. "$120,000-$150,000" or "not listed"]"""

    response = llm.invoke(prompt)
    for line in response.content.splitlines():
        if line.upper().startswith("TITLE:"):
            state['job_title'] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("COMPANY:"):
            state['company_name'] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("WORK_MODE:"):
            state['work_mode'] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("PAY_RANGE:"):
            state['pay_range'] = line.split(":", 1)[1].strip()
    return state


# --- Agent 2: Fit Critique (skeptical, honest, outputs a structured verdict) ---
def fit_critique_agent(state: JobSearchState) -> JobSearchState:
    prompt = f"""You are a skeptical, honest hiring manager reviewing this candidate for a real role.
Do NOT soften or oversell. Identify:
1. Genuine strengths that match the role
2. Real gaps or missing qualifications
3. Risk areas where this candidate might get screened out

Job Description:
{state['job_description']}

Candidate Resume:
{state['resume_base']}

After your analysis, end your response with a line EXACTLY in this format (nothing else on that line):
FIT_LEVEL: strong
or
FIT_LEVEL: moderate
or
FIT_LEVEL: weak"""
    response = llm.invoke(prompt)
    content = response.content
    state['fit_critique'] = content

    fit_level = "moderate"  # safe default if parsing fails
    for line in content.splitlines():
        if line.strip().upper().startswith("FIT_LEVEL:"):
            fit_level = line.split(":", 1)[1].strip().lower()
    state['fit_level'] = fit_level
    return state


# --- Routing function: decide whether to proceed to tailoring ---
def route_after_critique(state: JobSearchState) -> str:
    if state['fit_level'] == "weak":
        return "skip"
    return "proceed"


# --- Agent 3: Tailoring (resume + cover letter) ---
def tailoring_agent(state: JobSearchState) -> JobSearchState:
    prompt = f"""You are a resume and cover letter writer. Using ONLY skills and experience
that actually appear in the base resume below, tailor the resume bullets and write a cover
letter for this specific job. NEVER invent or embellish skills not present in the base resume.

For context, here is an honest fit critique of this candidate for the role — use it to
understand where to focus (genuine strengths) and be honest about, not to fabricate around:
{state['fit_critique']}

Job Description:
{state['job_description']}

Base Resume:
{state['resume_base']}

Return in this format:
### TAILORED RESUME BULLETS
[bullets here]

### COVER LETTER
[cover letter here]"""
    response = llm.invoke(prompt)
    state['tailored_resume'] = response.content
    return state


# --- Agent 4: Humanizer / Tone Agent ---
def humanizer_agent(state: JobSearchState) -> JobSearchState:
    prompt = f"""You are an expert editor who specializes in making professional writing sound
genuinely human — not AI-generated. Review the tailored resume and cover letter below and rewrite
them to:

1. Remove filler phrases and corporate buzzwords ("passionate about," "proven track record,"
   "leverage," "synergy," "results-driven," "dynamic," etc.)
2. Cut overly dramatic or inflated language — replace with grounded, specific, concrete phrasing
3. Vary sentence structure and length — avoid the repetitive rhythm that reads as AI-generated
4. Keep it professional, but let it sound like a real person wrote it — natural word choice,
   no forced enthusiasm.
5. Do NOT change any facts, skills, experience, or claims — only adjust language, tone, and phrasing
6. Do NOT add new accomplishments or embellish — if anything, prefer understatement over overselling
7. Keep the letter short, about 150-250 words.

Original tailored resume and cover letter:
{state['tailored_resume']}

Return in the same format:
### TAILORED RESUME BULLETS
[revised bullets here]

### COVER LETTER
[revised cover letter here]"""
    response = llm.invoke(prompt)
    state['tailored_resume'] = response.content
    return state


# --- Build the graph ---
graph = StateGraph(JobSearchState)
graph.add_node("scan", job_scanner_agent)
graph.add_node("critique", fit_critique_agent)
graph.add_node("tailor", tailoring_agent)
graph.add_node("humanize", humanizer_agent)

graph.set_entry_point("scan")
graph.add_edge("scan", "critique")
graph.add_conditional_edges(
    "critique",
    route_after_critique,
    {
        "proceed": "tailor",
        "skip": END
    }
)
graph.add_edge("tailor", "humanize")
graph.add_edge("humanize", END)

app = graph.compile()


# --- Helper: extract resume text from an uploaded PDF, DOCX, or TXT file ---
def load_resume_text(uploaded_file) -> str:
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    elif filename.endswith(".docx"):
        doc = Document(uploaded_file)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)

    elif filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")

    else:
        raise ValueError(f"Unsupported file type: {filename}")


# --- Tracker: append every run to a CSV ---
def log_to_tracker(state, filename="applications_tracker.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "title", "company", "work_mode", "pay_range", "fit_level", "url"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d"),
            state.get('job_title', ''),
            state.get('company_name', ''),
            state.get('work_mode', ''),
            state.get('pay_range', ''),
            state.get('fit_level', ''),
            state.get('job_url', '')
        ])


# --- UI ---
st.title("Job Search Agent")

resume_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])
job_url = st.text_input("Paste the job posting URL")

manual_jd = st.text_area(
    "If the URL fails to load (common on LinkedIn/Indeed), paste the job description here instead",
    height=150
)

if st.button("Run Analysis") and resume_file and job_url:
    with st.spinner("Running agents..."):
        resume_text = load_resume_text(resume_file)

        result = app.invoke({
            "job_url": job_url,
            "job_description": manual_jd.strip() if manual_jd.strip() else None,
            "resume_base": resume_text,
            "job_title": None,
            "company_name": None,
            "work_mode": None,
            "pay_range": None,
            "fit_critique": None,
            "fit_level": None,
            "tailored_resume": None,
            "cover_letter": None,
        })

        if result.get('job_title') == "SCRAPE_FAILED":
            st.error(
                "Couldn't load that job posting automatically — this site likely blocks "
                "automated scraping (common for LinkedIn, Indeed, and some ATS platforms). "
                "Paste the job description into the box above and click Run Analysis again."
            )
            st.stop()

        log_to_tracker(result)

    st.subheader(f"{result.get('job_title', 'Unknown title')} — {result.get('company_name', 'Unknown company')}")
    st.write(f"**Work mode:** {result.get('work_mode', 'not listed')}  |  **Pay range:** {result.get('pay_range', 'not listed')}")

    st.subheader("Fit Critique")
    st.write(result['fit_critique'])
    st.write(f"**Fit level:** {result['fit_level']}")

    if result['fit_level'] == "weak":
        st.warning("Fit assessed as weak — tailoring skipped.")
    else:
        st.subheader("Tailored Resume + Cover Letter")
        st.write(result['tailored_resume'])
