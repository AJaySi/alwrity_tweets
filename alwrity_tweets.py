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
                ::-webkit-scrollbar-track {
        background: #e1ebf9;
        }

        ::-webkit-scrollbar-thumb {
            background-color: #90CAF9;
            border-radius: 10px;
            border: 3px solid #e1ebf9;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #64B5F6;
        }

        ::-webkit-scrollbar {
            width: 16px;
        }
        div.stButton > button:first-child {
            background: #1565C0;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 10px 2px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
            font-weight: bold;
        }
      </style>
    """
    , unsafe_allow_html=True)

    # Hide top header line
    hide_decoration_bar_style = '<style>header {visibility: hidden;}</style>'
    st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

    # Hide footer
    hide_streamlit_footer = '<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>'
    st.markdown(hide_streamlit_footer, unsafe_allow_html=True)

    st.title("✍️  Alwrity - AI Tweet/X Generator")

    # Input section
    with st.expander("**PRO-TIP** - Read the instructions below.", expanded=True):
        col1, col2 = st.columns([5, 5])
        with col1:
            hook = st.text_input(
                label="**What's the tweet about:(Hook)**✨",
                placeholder="e.g., Discover the future of tech today! 🚀",
                help="Provide a compelling opening statement or question to grab attention."
            )
    
        with col2:
            target_audience = st.text_input(
                label="**Target Audience 🎯**",
                placeholder="e.g., technology enthusiasts, travel lovers",
                help="Describe the audience you want to target with this tweet."
            )
        
        col3, col4 = st.columns([5, 5])
        with col3:
            tone_style = st.selectbox(
                label="**Tone & Style 🎨**",
                options=['Humorous', 'Informative', 'Inspirational', 'Serious', 'Casual'],
                help="Choose the tone and style of the tweet."
            )
            
        with col4:
            hashtags = st.text_input(
                label="**Hashtags 📢**",
                placeholder="e.g., #TechTrends, #TravelGoals",
                help="Add 2-3 relevant hashtags to enhance visibility."
            )
    
        call_to_action = st.text_input(
            label="**Call to Action (CTA)** 🚀",
            placeholder="e.g., Share, Retweet, Like. Retweet if you agree! Share your thoughts in the comments!",
            help="Include a strong call to action to encourage engagement."
        )
    
    # Button to generate tweets
    if st.button('**Write Tweets**'):
        with st.status("Assigning AI professional to write your tweets..", expanded=True) as status:
            if not target_audience or not hook or not tone_style or not hashtags or not call_to_action:
                st.error("🚫 Please provide all required inputs.")
            else:
                response = tweet_writer(hook, target_audience, tone_style, hashtags, call_to_action)
                if response:
                    st.subheader(f'**🧕👩: Your Final Tweets!**')
                    st.write(response)
                    st.write("\n\n\n\n\n\n")
                else:
                    st.error("💥**Failed to generate tweets. Please try again!**")



def tweet_writer(hook, target_audience, tone_style, hashtags, call_to_action):
    """ Function to generate tweets using user inputs """

    prompt = f"""
    You are a social media expert creating tweets for an audience interested in {target_audience}. 
    Write 5 engaging, concise, and visually appealing tweets that each:

    1. Start with a compelling hook based on the following keywords: "{hook}"
    2. Include a compelling call to action: "{call_to_action}"
    3. Use 2-3 relevant hashtags: "{hashtags}" 
    4. Adopt a tone that matches: {tone_style}
    5. Be under 100 characters (including spaces and punctuation). 

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
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 2096,
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

        model = genai.GenerativeModel(model_name="gemini-1.5-flash",
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
