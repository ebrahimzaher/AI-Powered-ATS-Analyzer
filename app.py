import streamlit as st
import os
import tempfile
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import re

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

def extract_text_from_pdf(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        text = "\n".join([doc.page_content for doc in docs])

        os.remove(tmp_path)
        return text

    except Exception:
        return ""

def extract_json_with_regex(text: str):
    """
    Extract the first valid JSON object from LLM output using regex.
    Returns dict or None.
    """
    try:
        text = re.sub(r"```json|```", "", text).strip()

        match = re.search(r"\{.*\}", text, re.DOTALL)

        if not match:
            return None

        json_str = match.group(0)

        return json.loads(json_str)

    except Exception:
        return None

def evaluate_cv(cv_text, job_description):

    prompt = ChatPromptTemplate.from_template("""
You are an ATS (Applicant Tracking System).

Evaluate strictly.

Job Description:
{job_description}

Candidate CV:
{cv_text}

Scoring Rules:
- Skills Match (0-40)
- Experience Relevance (0-30)
- Tools & Technologies (0-20)
- Education & Certifications (0-10)

Return ONLY valid JSON:

{{
  "skills_score": number,
  "experience_score": number,
  "tools_score": number,
  "education_score": number,
  "total_score": number,
  "strong_keywords": ["keyword1"],
  "missing_keywords": ["keyword1"],
  "improvement_suggestions": ["suggestion1"],
  "summary": "short evaluation summary"
}}
""")

    chain = prompt | llm

    try:
        raw = chain.invoke({
            "cv_text": cv_text,
            "job_description": job_description
        })

        parsed = extract_json_with_regex(raw.content)

        if parsed:
            return parsed
        else:
            return {"total_score": 0, "summary": "Parsing error"}

    except Exception as e:
        return {"total_score": 0, "summary": f"LLM Error: {str(e)}"}


def main():

    st.set_page_config(page_title="Match-Your-CV Pro", page_icon="👔")
    st.title("Match-Your-CV Pro 🚀")
    st.caption("AI-Powered ATS CV Analyzer")


    job_description = st.text_area("Paste Job Description (JD)", height=300)

    uploaded_files = st.file_uploader(
        "Upload CVs (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Analyze Candidates"):

        if not job_description or not uploaded_files:
            st.error("Please provide both a JD and at least one CV.")
            return

        results = []
        progress_text = st.empty()
        bar = st.progress(0)

        for i, file in enumerate(uploaded_files):

            progress_text.text(f"Processing {file.name}...")
            file.seek(0)

            cv_text = extract_text_from_pdf(file)

            if not cv_text or len(cv_text) < 100:
                results.append({
                    "Filename": file.name,
                    "Skills": 0,
                    "Experience": 0,
                    "Tools": 0,
                    "Education": 0,
                    "Total Score": 0,
                    "Summary": "⚠️ No readable text found",
                    "Strong Keywords": "",
                    "Missing Keywords": "",
                    "Suggestions": ""
                })
                continue

            data = evaluate_cv(cv_text, job_description)

            try:
                total_score = int(data.get("total_score", 0))
            except:
                total_score = 0

            results.append({
                "Filename": file.name,
                "Skills": data.get("skills_score", 0),
                "Experience": data.get("experience_score", 0),
                "Tools": data.get("tools_score", 0),
                "Education": data.get("education_score", 0),
                "Total Score": total_score,
                "Summary": data.get("summary", ""),
                "Strong Keywords": ", ".join(data.get("strong_keywords", [])),
                "Missing Keywords": ", ".join(data.get("missing_keywords", [])),
                "Suggestions": " | ".join(data.get("improvement_suggestions", []))
            })

            bar.progress((i + 1) / len(uploaded_files))

        progress_text.empty()
        bar.empty()

        df = pd.DataFrame(results).sort_values(by="Total Score", ascending=False)

        if not df.empty:
            best = df.iloc[0]
            st.success(f"🏆 Top Candidate: {best['Filename']} — {best['Total Score']}%")
            st.balloons()

        st.subheader("Leaderboard")

        st.dataframe(
            df,
            width="stretch",
            column_config={
                "Total Score": st.column_config.ProgressColumn(
                    "Match Score",
                    min_value=0,
                    max_value=100,
                    format="%d%%",
                )
            }
        )

        st.subheader("Detailed Analysis")

        for _, row in df.iterrows():
            with st.expander(f"{row['Filename']} — {row['Total Score']}%"):
                st.markdown("### 📊 Category Scores")
                st.write(f"- Skills: {row['Skills']}/40")
                st.write(f"- Experience: {row['Experience']}/30")
                st.write(f"- Tools: {row['Tools']}/20")
                st.write(f"- Education: {row['Education']}/10")

                st.markdown("### 🧠 Summary")
                st.write(row["Summary"])

                st.markdown("### ✅ Strong Keywords")
                st.write(row["Strong Keywords"])

                st.markdown("### ❌ Missing Keywords")
                st.write(row["Missing Keywords"])

                st.markdown("### 🚀 Improvement Suggestions")
                st.write(row["Suggestions"])


if __name__ == "__main__":
    main()
