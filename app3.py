# app.py

import streamlit as st
import pandas as pd
import docx2txt
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title='Resume Ninja',
    page_icon='🧲️'
)
# Set up Streamlit app
st.title("🧲️ :violet[Resume Ninja]")
st.write("Upload resumes (PDF or DOCX) and screen based on required skills.")

# Sidebar input for required skills
st.sidebar.header("Required Skills")
required_skills = st.sidebar.text_area(
    "Enter required skills, separated by commas", 
    "Python, Machine Learning, Data Analysis, Web Development, html, css, javascript, web3, UIUX Design,Data Analytics,internship, projects, Blockchain", 
    height=250
)

# Upload resumes (PDF or DOCX)
uploaded_files = st.file_uploader("Upload resumes (.docx or .pdf)", type=["docx", "pdf"], accept_multiple_files=True)

# Function to extract text from PDF
def read_pdf(file):
    text = ""
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text += page.get_text("text")
    return text

# Process resumes if files are uploaded
if uploaded_files and required_skills:
    st.subheader(":green[Screening Results]")

    # Convert required skills to vectorizable format
    required_skills_list = required_skills.lower().split(',')
    required_skills_text = ' '.join(required_skills_list)

    results = []

    for file in uploaded_files:
        # Extract text based on file type
        if file.type == "application/pdf":
            resume_text = read_pdf(file).lower()
        else:
            resume_text = docx2txt.process(file).lower()

        # Convert text to vector format
        text_data = [required_skills_text, resume_text]
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(text_data)

        # Calculate cosine similarity
        similarity_score = cosine_similarity(count_matrix)[0][1]

        # Save results
        results.append({
            "Resume": file.name,
            "Match Percentage": round(similarity_score * 100, 2)
        })

    # Display results as a DataFrame
    results_df = pd.DataFrame(results)

    # Sidebar filtering options
    st.sidebar.subheader("Filter Screening Results")

    # Adjust the maximum value for top_n based on the number of resumes
    max_top_n = len(results_df) if len(results_df) > 0 else 1
    top_n = st.sidebar.number_input("Select top N candidates", min_value=1, max_value=max_top_n, value=max_top_n, step=1)

    # Filter option for minimum match percentage
    min_match_percentage = st.sidebar.slider("Minimum Match Percentage", min_value=0, max_value=100, value=50)

    # Apply filters
    filtered_results = results_df[results_df["Match Percentage"] >= min_match_percentage].sort_values(by="Match Percentage", ascending=False)
    filtered_results = filtered_results.head(top_n)

    # Display filtered results
    st.subheader("🎯 Filtered Screening Results")
    st.dataframe(filtered_results)

    # Download button for filtered results
    st.download_button(
        label="Download Filtered Results as CSV",
        data=filtered_results.to_csv(index=False),
        file_name="filtered_resume_screening_results.csv",
        mime="text/csv"
    )
else:
    st.warning("Please upload resumes and enter required skills.")
