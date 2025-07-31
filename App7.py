import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image
import urllib.parse # Import for URL encoding
import random # Import for shuffling

# --- Configuration and Paths ---
BASE_DIR = os.path.dirname(__file__)
EXCEL_FILE_PATH = os.path.join(BASE_DIR, "harry_potter_quiz_training_data.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "random_forest_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "saved_model", "label_encoders.pkl")
TARGET_ENCODER_PATH = os.path.join(BASE_DIR, "saved_model", "target_encoder.pkl")
IMAGE_FOLDER_PATH = os.path.join(BASE_DIR, "character_images")

# --- Helper Function for Loading Pickled Files ---
@st.cache_resource
def load_resources():
    """Loads models and encoders, with error handling."""
    resources = {}
    try:
        resources['model'] = joblib.load(MODEL_PATH)
        resources['label_encoders'] = joblib.load(ENCODER_PATH)
        resources['target_le'] = joblib.load(TARGET_ENCODER_PATH)
        return resources
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}. Please ensure 'saved_model' directory and its contents exist.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or encoders: {e}")
        st.stop()

# --- Helper Function for Loading Quiz Data ---
@st.cache_data
def load_quiz_data():
    """Loads quiz questions and answers, with error handling."""
    try:
        df = pd.read_excel(EXCEL_FILE_PATH, sheet_name="questions_and_answers")
        df.columns = ['Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Option E']
        return df
    except FileNotFoundError:
        st.error(f"Quiz data Excel file not found at '{EXCEL_FILE_PATH}'.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading quiz data from '{EXCEL_FILE_PATH}': {e}. Ensure it's a valid Excel file.")
        st.stop()

# --- Helper Function for Updating User Submissions (Careful with concurrent writes!) ---
def update_user_submissions(new_entry_df):
    """Reads, updates, and writes the user_submissions sheet."""
    try:
        all_sheets = pd.read_excel(EXCEL_FILE_PATH, sheet_name=None)
        user_sub_df = all_sheets.get("user_submissions", pd.DataFrame(columns=new_entry_df.columns))
        
        user_sub_df = pd.concat([user_sub_df, new_entry_df], ignore_index=True)
        all_sheets["user_submissions"] = user_sub_df

        with pd.ExcelWriter(EXCEL_FILE_PATH, engine="openpyxl", mode='w') as writer:
            for sheet_name, df in all_sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        return True
    except Exception as e:
        st.error(f"Failed to save your submission: {e}. Please try again.")
        return False

# --- Callback function to reset quiz state ---
def reset_quiz_state():
    st.session_state.quiz_submitted = False
    st.session_state.user_name_input = "" # Clear name input
    st.session_state.quiz_selections_display = {} # Clear quiz answers
    # This forces a rerun and clears the displayed content
    if 'shuffled_questions_df' in st.session_state:
        del st.session_state.shuffled_questions_df # Clear shuffled questions to re-shuffle on next run
    if 'shuffled_options_map' in st.session_state:
        del st.session_state.shuffled_options_map # Clear shuffled options to re-shuffle on next run
    
    # Remove the following line:
    # st.rerun() # Force a rerun to immediately switch views and re-shuffle


# --- Main App Logic ---
st.set_page_config(page_title="Harry Potter Character Quiz", layout="centered")

# Load all resources at the start
with st.spinner("Loading magical artifacts..."):
    resources = load_resources()
    model = resources['model']
    label_encoders = resources['label_encoders']
    target_le = resources['target_le']
    original_questions_df = load_quiz_data() # Load original data

feature_cols = ["A1", "A2", "A3", "A4", "A5"] 

character_descriptions = {
    "Harry Potter": "With unshakable courage and a heart that leaps to protect, Harry faces danger head-on, leading the way into every adventure, always guided by the unwavering light of doing what’s right.",
    "Hermione Granger": "With brilliance in her mind and honesty in her heart, Hermione finds answers in books and logic, creating thoughtful plans that transform curiosity into power",
    "Ron Weasley": "With loyalty as his compass and humor as his shield, Ron may stumble in panic, but he always stands by those he loves, offering the warmth of friendship above all.",
    "Draco Malfoy": "Driven by ambition and influence, Draco plays life like a chessboard, crafting strategies to turn every challenge into an opportunity for power and success.",
    "Neville Longbottom": "Gentle yet steadfast, Neville grows stronger with every struggle, turning kindness and perseverance into quiet acts of bravery that prove doubters wrong."
}

st.title("🧙‍♀️ Which Harry Potter Character Are You?")
st.markdown("---")

# Initialize session state for quiz answers and submission status
if 'quiz_selections_display' not in st.session_state:
    st.session_state.quiz_selections_display = {} 
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False
if 'user_name_input' not in st.session_state: # To control visibility and clearing
    st.session_state.user_name_input = ""

# --- Shuffling Logic ---
# Only shuffle questions and options if the quiz hasn't been submitted
# and if they haven't been shuffled yet for the current session.
if not st.session_state.quiz_submitted:
    if 'shuffled_questions_df' not in st.session_state:
        # Shuffle questions (rows of the DataFrame)
        st.session_state.shuffled_questions_df = original_questions_df.sample(frac=1, random_state=random.randint(0, 10000)).reset_index(drop=True)
        # Store how options were shuffled for each question to map back correctly
        st.session_state.shuffled_options_map = {}
    
    questions_df_for_display = st.session_state.shuffled_questions_df
    shuffled_options_map = st.session_state.shuffled_options_map
else:
    # If quiz submitted, use the stored shuffled data to maintain state if re-displaying
    questions_df_for_display = st.session_state.shuffled_questions_df
    shuffled_options_map = st.session_state.shuffled_options_map


# --- Conditional Display of Quiz or Results ---

# If the quiz has NOT been submitted, show the quiz form
if not st.session_state.quiz_submitted:
    user_name = st.text_input("Enter your name:", key="user_name_input") # Persist key

    if user_name:
        st.markdown("### Answer these questions to reveal your character:")
        st.markdown("---") 

        current_answers_for_model = [] 
        quiz_is_complete = True 

        # Use the shuffled DataFrame here
        for i, row in questions_df_for_display.iterrows():
            question = row['Question']
            
            # Extract options and their corresponding model codes
            original_options = {
                'A': row['Option A'], 
                'B': row['Option B'], 
                'C': row['Option C'], 
                'D': row['Option D'], 
                'E': row['Option E']
            }
            
            # Create a list of (model_code, option_text) tuples
            option_pairs = list(original_options.items())
            
            # Shuffle the option pairs IF NOT ALREADY SHUFFLED FOR THIS QUESTION IN THIS SESSION
            if f"q_{i}" not in shuffled_options_map:
                random.shuffle(option_pairs)
                shuffled_options_map[f"q_{i}"] = option_pairs # Store the shuffled order
            else:
                option_pairs = shuffled_options_map[f"q_{i}"] # Use the stored shuffled order

            display_options_with_placeholder = ["--- Please Select ---"]
            # Map shuffled options back to display labels (A, B, C, D, E) for the user
            # and maintain a lookup for the actual model input
            current_question_option_mapping = {} # To map display string back to model code
            
            for idx, (model_code, option_text) in enumerate(option_pairs):
                display_label = chr(ord('A') + idx) # Dynamically assign A, B, C...
                display_string = f"{display_label}. {option_text}"
                display_options_with_placeholder.append(display_string)
                current_question_option_mapping[display_string] = model_code
            
            # Store this mapping for later retrieval if the user refreshes/reruns
            # This is important for "st.radio" to correctly restore the selected option after a rerun
            if f"q_{i}_mapping" not in st.session_state:
                st.session_state[f"q_{i}_mapping"] = current_question_option_mapping

            stored_display_string = st.session_state.quiz_selections_display.get(f"q_{i}", display_options_with_placeholder[0])
            
            # Find the index of the stored option based on the *current* shuffled display options
            try:
                current_selection_index = display_options_with_placeholder.index(stored_display_string)
            except ValueError:
                current_selection_index = 0 

            selected_display_option = st.radio(
                f"**Q{i+1}: {question}**",
                display_options_with_placeholder,
                index=current_selection_index,
                key=f"q_{i}" 
            )
            
            st.session_state.quiz_selections_display[f"q_{i}"] = selected_display_option

            # Get the actual coded answer (A, B, C, D, E) for the model based on the selected display option
            if selected_display_option == "--- Please Select ---":
                coded_answer = "INVALID"
            else:
                # Use the mapping generated earlier for this specific question
                coded_answer = current_question_option_mapping.get(selected_display_option, "INVALID")
                
            current_answers_for_model.append(coded_answer)
            
            if coded_answer == "INVALID":
                quiz_is_complete = False

        st.markdown("---")
        if st.button("Submit My Answers", use_container_width=True):
            if not quiz_is_complete:
                st.warning("Please answer all questions before submitting!")
            else:
                try:
                    encoded_input_dict = {}
                    for col_name, coded_ans in zip(feature_cols, current_answers_for_model):
                        if col_name not in label_encoders:
                            st.error(f"Error: Label encoder for '{col_name}' not found. Model features might not match.")
                            st.stop()
                        le = label_encoders[col_name]
                        
                        try:
                            encoded_val = le.transform([coded_ans])[0]
                        except ValueError:
                            st.error(f"Error: Unseen answer '{coded_ans}' for question '{col_name}'. Check quiz data and encoders.")
                            st.stop()
                            
                        encoded_input_dict[col_name] = encoded_val

                    input_df = pd.DataFrame([encoded_input_dict])

                    predicted_code = model.predict(input_df)[0]
                    predicted_character = target_le.inverse_transform([predicted_code])[0]

                    submission_columns = ['Name'] + feature_cols + ['Predicted_Character']
                    new_entry_data = [user_name] + current_answers_for_model + [predicted_character]
                    new_entry_df = pd.DataFrame([new_entry_data], columns=submission_columns)

                    if update_user_submissions(new_entry_df):
                        # Set the flag to True to hide the quiz and show results
                        st.session_state.quiz_submitted = True
                        st.session_state.predicted_character = predicted_character
                        st.session_state.user_name_display = user_name # Store name for result display
                        st.rerun() # Force a rerun to immediately switch views
                    else:
                        st.error("Failed to save your quiz results. Please contact support.")

                except Exception as e:
                    st.error(f"An unexpected error occurred during prediction or result display: {e}")
                    st.info("Please check your input or try refreshing the page.")
    else:
        st.info("Please enter your name to start the quiz!")

# If the quiz HAS been submitted, show the results
elif st.session_state.quiz_submitted:
    if 'predicted_character' in st.session_state and 'user_name_display' in st.session_state:
        predicted_character = st.session_state.predicted_character
        user_name_display = st.session_state.user_name_display
        
        st.success(f"🎉 {user_name_display}, you are most like **{predicted_character}**!")
        desc = character_descriptions.get(predicted_character, "A truly unique magical being!")

        img_path_jpg = os.path.join(IMAGE_FOLDER_PATH, f"{predicted_character}.jpg")
        img_path_png = os.path.join(IMAGE_FOLDER_PATH, f"{predicted_character}.png")

        final_image_path = None
        if os.path.exists(img_path_jpg):
            final_image_path = img_path_jpg
        elif os.path.exists(img_path_png):
            final_image_path = img_path_png

        if final_image_path:
            try:
                img = Image.open(final_image_path)
                st.image(img, caption=f"{predicted_character}: {desc}", use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load image for {predicted_character} from '{final_image_path}': {e}")
        else:
            st.warning(f"Image not found for '{predicted_character}' in '{IMAGE_FOLDER_PATH}'. Tried: {predicted_character}.jpg, {predicted_character}.png")
            st.markdown(f"**{desc}**")
    else:
        st.error("Something went wrong displaying results. Please try again.")
    
    st.markdown("---")
    
    # --- LinkedIn Share Link ---
    # The URL of your Streamlit app (where the quiz is hosted)
    # IMPORTANT: Replace "YOUR_STREAMLIT_APP_URL" with the actual URL of your deployed app!
    app_url = "https://harry-potter-5efxzu8rjmh8kyeepnpbuz.streamlit.app/" # Placeholder: Replace with your actual deployed app URL
    
    # Text to pre-fill in the LinkedIn share dialog
    share_text = f"I just found out I'm a {predicted_character} in the Harry Potter universe with this fun quiz! Check it out and see who you are! #HarryPotter #Quiz #CharacterQuiz"
    
    # Encode the text for URL safety
    encoded_share_text = urllib.parse.quote(share_text)
    encoded_app_url = urllib.parse.quote(app_url)

    linkedin_share_url = f"https://www.linkedin.com/sharing/share-offsite/?url={encoded_app_url}&summary={encoded_share_text}"

    st.markdown(f"""
        <div style="text-align: center; margin-top: 20px;">
            <a href="{linkedin_share_url}" target="_blank" style="
                display: inline-block;
                padding: 10px 20px;
                background-color: #0077B5; 
                color: white; 
                text-align: center; 
                text-decoration: none;
                border-radius: 5px;
                font-size: 16px;
                font-weight: bold;
            ">
                Share My Result on LinkedIn
            </a>
        </div>
    """, unsafe_allow_html=True)

    st.button("Take Quiz Again", on_click=reset_quiz_state, use_container_width=True)

st.markdown("---")
st.markdown("Hope you enjoyed discovering your inner Harry Potter character!")
