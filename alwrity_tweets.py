import time #Iwish
import os
import json
import requests
import streamlit as st
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import google.generativeai as genai


def main():
    # Set page configuration
    st.set_page_config(
        page_title="Alwrity - AI X Tweets Generator (Beta)",
        layout="wide",
    )
    # Remove the extra spaces from margin top.
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)
    st.markdown(f"""
      <style>
      [class="st-emotion-cache-7ym5gk ef3psqc12"]{{
            display: inline-block;
            padding: 5px 20px;
            background-color: #4681f4;
            color: #FBFFFF;
            width: 300px;
            height: 35px;
            text-align: center;
            text-decoration: none;
            font-size: 16px;
            border-radius: 8px;’
      }}
      </style>
    """
    , unsafe_allow_html=True)

    # Hide top header line
    hide_decoration_bar_style = '<style>header {visibility: hidden;}</style>'
    st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

    # Hide footer
    hide_streamlit_footer = '<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>'
    st.markdown(hide_streamlit_footer, unsafe_allow_html=True)

    with st.expander("**PRO-TIP** - Read the instructions below.", expanded=True):
        col1, col2 = st.columns([5, 5])
        with col1:
            hook = st.text_input(
                    label="What's the tweet about:(Hook)",
                    placeholder="e.g., Discover the future of tech today!",
                    help="Provide a compelling opening statement or question to grab attention."
            )

        with col2:
            # Collect user inputs with placeholders and help text
            target_audience = st.text_input(
                label="Target Audience",
                placeholder="e.g., technology enthusiasts, travel lovers",
                help="Describe the audience you want to target with this tweet."
            )

    if st.button('**Write Tweets**'):
        with st.status("Assigning AI professional to write your Google Ads copy..", expanded=True) as status:
            if not target_audience or not hook:
                st.error("🚫 Please provide all required inputs.")
            else:
                response = tweet_writer(target_audience, hook)
                if response:
                    st.subheader(f'**🧕👩: Your Final Tweets!**')
                    st.write(response)
                    st.write("\n\n\n\n\n\n")
                else:
                    st.error("💥**Failed to write Letter. Please try again!**")


def tweet_writer(target_audience, hook):
    """ Email project_update_writer """

    prompt = f"""
    You are a social media expert creating tweets for an audience interested in {target_audience}. 
    Write 5 engaging, concise, and visually appealing tweets that each:

    1. Start with a compelling hook based on the following keywords: "{hook}"
    2. Include a compelling call to action.
    3. Use 2-3 relevant hashtags. 
    4. Adopt a tone that matches the following options: 
        - Humorous 
        - Informative 
        - Inspirational 
        - Serious 
        - Casual
    5. Be under 100 characters (including spaces and punctuation). 

    Here are some examples of call-to-actions to include:
    - Retweet this if you agree!
    - Share your thoughts in the comments!
    - Learn more at [link] 
    - Follow for more [topic] content
    - Like if you're excited about [topic]

    Output each tweet separated by a newline.
    """

    try:
        response = generate_text_with_exception_handling(prompt)
        return response
    except Exception as err:
        st.error(f"Exit: Failed to get response from LLM: {err}")
        exit(1)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_text_with_exception_handling(prompt):
    """
    Generates text using the Gemini model with exception handling.

    Args:
        api_key (str): Your Google Generative AI API key.
        prompt (str): The prompt for text generation.

    Returns:
        str: The generated text.
    """

    try:
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 8192,
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]

        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)

        convo = model.start_chat(history=[])
        convo.send_message(prompt)
        return convo.last.text

    except Exception as e:
        st.exception(f"An unexpected error occurred: {e}")
        return None


if __name__ == "__main__":
    main()
