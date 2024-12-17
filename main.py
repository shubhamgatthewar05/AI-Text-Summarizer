import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please provide your GOOGLE_API_KEY in the .env file.")
    st.stop()

# Initialize Gogole generative AI
llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3, google_api_key=GOOGLE_API_KEY)

# llm1 =ChatGoogleGenerativeAI(model="gemini-pro", google_api_key = GOOGLE_API_KEY)

# Defining the refined Prompts
short_summary_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        """Analyze the following text thoroughly and extract its core message. 
        Ensure the summary is highly precise, context-aware, and captures the essence of the content in one impactful sentence:

{text}

Core Insight (1 Sentence):"""
    ),
)


long_summary_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        """Perform an in-depth analysis of the following text and generate a structured summary in 3-5 sentences. 
        Highlight the most important points, maintain logical flow, and avoid including irrelevant details. 
        Focus on clarity, coherence, and completeness:

{text}

Structured Summary:"""
    ),
)




keyword_summary_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        """From the text below, identify the 3-5 most critical keywords that encapsulate its main ideas. 
        Use these keywords to frame a sharp and contextually accurate summary. 
        Ensure the keywords are distinct and relevant, and the summary integrates them seamlessly:

{text}

Key Concepts:  
1.  
2.  
3.  

Summary (with Keywords):"""
    ),
)




action_item_summary_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        """Review the text below and extract all actionable items, decisions, or next steps. 
        Provide a prioritized and categorized list of actions with clear descriptions to guide implementation:

{text}

Prioritized Action Items:  
1.  
2.  
3.  

Category Breakdown (if applicable):  
- Immediate Actions:  
- Follow-ups:  
- Long-term Considerations:"""
    ),
)







# Store prompts in a dictionary for easy selection
prompts = {
    "Short Summary": short_summary_prompt,
    "Long Summary": long_summary_prompt,
    "Keyword Summary": keyword_summary_prompt,
    "Action Items": action_item_summary_prompt,
    # "conceptual Items": contextual_summary_prompt
}

def summarize_text(text, prompt_type):
    """
    Summarizes the input text using the selected prompt.

    Args:
        text (str): Input text to summarize.
        prompt_type (str): The type of prompt to use (keys from `prompts` dictionary).

    Returns:
        str: Generated summary.
    """
    if prompt_type not in prompts:
        raise ValueError("Invalid prompt_type. Choose from: 'Short Summary', 'Long Summary', 'Keyword Summary', 'Action Items'.")

    selected_prompt = prompts[prompt_type]
    formatted_prompt = selected_prompt.format(text=text)

    response = llm.invoke(formatted_prompt)
    return response.content





# Streamlit UI
st.title("AI Text Summarizer")
st.write("Generate summaries and extract actionable insights from your text with the power of Google Generative AI.")

# Text input
input_text = st.text_area("Enter your text here:", height=200)

# Prompt type selection
prompt_type = st.selectbox(
    "Choose the type of summary:", list(prompts.keys()), index=0
)

# Generate button
if st.button("Generate Summary"):
    if input_text.strip():
        with st.spinner("Generating summary..."):
            try:
                summary = summarize_text(input_text, prompt_type)
                st.subheader("Generated Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter some text to summarize.")

# Footer
st.markdown("---")
st.write("Developed with ❤️ using [Streamlit](https://streamlit.io) and LangChain Google Generative AI.")
