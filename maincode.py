import streamlit as st
import json
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.preview.vision_models import ImageGenerationModel
import google.generativeai as genai
import google.cloud.texttospeech as tts
from io import BytesIO
import time
import logging # Import logging module
import re

# Configure logging to see more details in the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# !!! SECURITY WARNING: REPLACE THIS WITH YOUR REAL API KEY AND DETAILS !!!
API_KEY = ""
PROJECT_ID = "smartstoryteller2006"
LOCATION = "us-central1"

genai.configure(api_key=API_KEY)
vertexai.init(project=PROJECT_ID, location=LOCATION)

text_model = GenerativeModel("gemini-2.5-pro")
image_model = ImageGenerationModel.from_pretrained("imagegeneration@006")


@st.cache_data(show_spinner="Analyzing story and creating scenes...")
def ai_split_story_into_scenes(story_text: str, num_scenes: int = 12):
    """
    Splits a long story text into a specified number of coherent scenes using a generative AI model.
    """
    prompt = (
        "You are an expert story analyst. Read the full story below and divide it into "
        f"{num_scenes} distinct and logical scenes. For each scene, provide a concise summary. "
        "The scene breaks should be based on changes in character, setting, or plot. "
        "Your response must be a JSON array, with each object containing a 'scene' number and a 'summary'. "
        "Do not include any other text, dialogue, or explanation outside of the JSON array."
        "\n\nStory:\n\n"
        f"{story_text}"
        "\n\nJSON Output:\n"
    )
    
    try:
        model = GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(prompt)
        
        json_string = response.text.strip()
        json_string = re.sub(r"```json\s*", "", json_string)
        json_string = re.sub(r"\s*```", "", json_string)
        
        scene_list = json.loads(json_string)
        
        if len(scene_list) != num_scenes:
            st.warning(f"AI returned {len(scene_list)} scenes, not the expected {num_scenes}. Using what was returned.")
        
        return {str(item['scene']): item for item in scene_list}
        
    except Exception as e:
        st.error(f"Failed to generate scenes using AI. Please check the model's response format or try a different prompt. Error: {e}")
        return {}

@st.cache_data(show_spinner="Generating a beautiful image...")
def generate_image_for_scene(prompt: str):
    """
    Generates an image from a text prompt using Vertex AI with a retry mechanism.
    """
    for i in range(3):
        try:
            images_response = image_model.generate_images(prompt=prompt)
            if images_response.images and hasattr(images_response.images[0], '_image_bytes'):
                return images_response.images[0]._image_bytes
            else:
                st.warning(f"Attempt {i+1}: Image generation successful but no image bytes found. Retrying...")
                time.sleep(2)
        except Exception as e:
            st.error(f"Attempt {i+1}: Failed to generate image: {e}. Retrying...")
            time.sleep(2)

    st.error("Failed to generate image after multiple attempts.")
    return None

def get_character_summary(summary: str):
    """Generates a character summary using a large language model."""
    prompt = f"Based on the following scene, identify and describe the key characters and their roles in a clear, concise manner: {summary}"
    response = text_model.generate_content(prompt)
    return response.text

def get_reflection_response(summary: str, user_answer: str):
    """Generates a reflection response based on the user's answer."""
    prompt = (
        f"The scene is: '{summary}'. The user's action/answer is: '{user_answer}'. "
        f"Describe what will happen next if the story were to follow this choice. Be creative and detailed."
    )
    response = text_model.generate_content(prompt)
    return response.text

def get_genz_version(summary: str):
    """Generates a Gen Z style comparison table for the scene."""
    prompt = (
        f"Create a markdown table with two columns: 'Situation' and 'Gen Z Vibe'. "
        f"Compare the theme of the following scene to a relatable, modern-day Gen Z scenario. "
        f"Use Gen Z slang. Scene: {summary}"
    )
    response = text_model.generate_content(prompt)
    return response.text
    
def get_elongated_summary(summary: str):
    """Generates a detailed summary from a brief one."""
    prompt = (
        f"Elaborate on the following scene summary, adding more detail and a richer narrative. "
        f"Expand it into a full descriptive paragraph. Do not include a title. and also keep the vocabulary really simple as explaining to a 10 year old"
        f"Scene summary:\n{summary}"
    )
    response = text_model.generate_content(prompt)
    return response.text

@st.cache_data(show_spinner="Refining prompt for a richer image...")
def get_hyper_detailed_image_prompt(summary: str):
    """
    Refines a short summary into a highly detailed image generation prompt,
    with an emphasis on safe and positive word choices.
    """
    prompt = (
        "You are an expert AI image prompt engineer. Your task is to take a brief scene summary "
        "also remember that the whole thing should have south indian aesthetics because its a local south indian folklore"
        "and expand it into a highly detailed, single-sentence image generation prompt. "
        "The prompt must be no more than 60 words and should be crafted to be safe and positive. "
        "Do not use words related to violence, injury, negativity, or despair. "
        "Instead, focus on conveying mood and action through lighting, color, and safe descriptors."
        "also create image prompt safe for vertex ai"
        "Also avoid words like: beggar, rags, poor, torn, sick, dead, injure, collapsed, dramatic drop and replace with alternate words."
        "if the scence is explicit,violent or inappropriate make an appropriate version of it"
        f"Now, create a prompt for the following scene: {summary}"
    )
    response = text_model.generate_content(prompt)
    
    return ' '.join(response.text.split()[:60])

@st.cache_data(show_spinner="Generating audio...")
def generate_tts_audio(text: str):
    """Generates TTS audio from the given text with an Indian accent."""
    try:
        client = tts.TextToSpeechClient()
        synthesis_input = tts.SynthesisInput(text=text)

        # Updated to a more expressive Neural2 voice
        voice = tts.VoiceSelectionParams(
            language_code="en-IN",
            name="en-IN-Neural2-A", 
            ssml_gender=tts.SsmlVoiceGender.FEMALE,
        )

        audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        return BytesIO(response.audio_content)
    except Exception as e:
        st.error(f"Text-to-Speech generation failed. Please check your credentials: {e}")
        return None

# --- Streamlit App Layout ---
def main():
    st.title("Smart Cultural Storyteller")
    st.markdown("Upload a **TXT** file containing a story, and the app will automatically divide it into scenes!")
    
    uploaded_txt_file = st.file_uploader("Choose a TXT file", type="txt")

    scenes = {}
    if uploaded_txt_file is not None:
        file_content = uploaded_txt_file.read().decode("utf-8")
        scenes = ai_split_story_into_scenes(file_content)

    if not scenes:
        st.info("Please upload a valid TXT file to start the story.")
        return

    num_scenes = len(scenes)

    if 'current_scene' not in st.session_state or st.session_state.current_scene > num_scenes:
        st.session_state.current_scene = 1
        st.session_state.twist_output = {}
        st.session_state.elongated_summary_output = {}

    current_scene_number = st.session_state.current_scene
    summary = scenes.get(str(current_scene_number), {}).get('summary', 'Summary not found.')

    st.header(f"Scene {current_scene_number}")
    
    # Mandatorily use the hyper-detailed prompt for image generation
    refined_image_prompt = get_hyper_detailed_image_prompt(summary)
    st.write(f"**Image Prompt:** *{refined_image_prompt}*")

    image_bytes = generate_image_for_scene(refined_image_prompt)
    if image_bytes:
        st.image(image_bytes, caption=summary, use_container_width=True)

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⏪ Back", disabled=current_scene_number == 1):
            st.session_state.current_scene -= 1
            st.rerun()
    with col2:
        if st.button("Next ⏩", disabled=current_scene_number == num_scenes):
            st.session_state.current_scene += 1
            st.rerun()

    st.markdown("---")
    
    # Mode selection tabs
    tab_titles = ["Characters", "Reflection", "Gen Z Version", "Twist Mode", "Elongated Summary"]
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        st.subheader("Characters & Roles")
        with st.spinner("Generating character summary..."):
            st.write(get_character_summary(summary))

    with tabs[1]:
        st.subheader("Reflection Mode")
        reflection_question = f"Given the scene: '{summary}', how would you handle the main challenge?"
        st.write(reflection_question)
        user_answer = st.text_area("Your answer here:")
        if st.button("See What Happens"):
            if user_answer:
                with st.spinner("Reflecting on your choice..."):
                    st.write(get_reflection_response(summary, user_answer))
            else:
                st.warning("Please provide an answer to see the outcome.")

    with tabs[2]:
        st.subheader("Gen Z Version")
        with st.spinner("Generating Gen Z comparison..."):
            st.markdown(get_genz_version(summary))

    with tabs[3]:
        st.subheader("Twist Mode")
        
        what_if_input = st.text_input(
            "Enter a major decision change", 
            key=f"what_if_{st.session_state.current_scene}"
        )
        tone = st.selectbox(
            "Choose storytelling tone:", 
            ["Neutral", "Tragic", "Redemptive", "Comedic"], 
            index=0,
            key=f"twist_tone_{st.session_state.current_scene}"
        )

        if st.button("Generate Twist", key=f"twist_button_{st.session_state.current_scene}") and what_if_input:
            with st.spinner("Generating twist..."):
                twist_prompt = (
                    "You are a master storyteller. Rewrite the following story scene based on a major decision change. "
                    "Focus on the current scene. "
                    "Include vivid descriptions, emotions, and consequences. "
                    f"Scene summary:\n{summary}\n\n"
                    f"Decision change: '{what_if_input}'\n"
                    f"Rewrite the outcome in 2-3 sentences, keeping it dramatic and aligned with the {tone.lower()} tone. Use simpler vocabulary."
                )
                twist_response = text_model.generate_content(twist_prompt)
                twist_text = twist_response.text
                
                st.session_state.twist_output[st.session_state.current_scene] = twist_text

        if st.session_state.current_scene in st.session_state.twist_output:
            st.markdown(f"**Twist Outcome:** {st.session_state.twist_output[st.session_state.current_scene]}")
            
    with tabs[4]:
        st.subheader("Elongated Summary")
        
        col_gen, col_audio = st.columns([0.7, 0.3])
        
        with col_gen:
            if st.button("Generate Elongated Summary", key=f"elongate_button_{st.session_state.current_scene}"):
                with st.spinner("Elongating summary..."):
                    elongated_text = get_elongated_summary(summary)
                    st.session_state.elongated_summary_output[st.session_state.current_scene] = elongated_text

        if st.session_state.current_scene in st.session_state.elongated_summary_output:
            elongated_text = st.session_state.elongated_summary_output[st.session_state.current_scene]
            st.markdown(elongated_text)
            
            with col_audio:
                if st.button("▶️ Listen to Summary", key=f"listen_button_{st.session_state.current_scene}"):
                    if elongated_text:
                        audio_file_bytes = generate_tts_audio(elongated_text)
                        if audio_file_bytes:
                            st.audio(audio_file_bytes, format='audio/mp3')

if __name__ == "__main__":

    main()
