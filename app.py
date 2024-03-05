import streamlit as st
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder, speech_to_text
import tooling1

# Initialize OpenAI client
client = OpenAI()

st.header("Convert your audio into P&ID data")

# Option for users to either upload a file or record directly
option = st.radio("Choose an option:", ('Upload Audio File', 'Record Audio'))

if 'text_output' not in st.session_state:
    st.session_state['text_output'] = ''

# Function to convert voice to text
def convert_voice_to_text(audio_file):
    transcript = client.audio.translations.create(
        model="whisper-1", 
        file=audio_file,
        response_format="text"
    )
    return transcript

if option == 'Upload Audio File':
    audio_file = st.file_uploader("Upload an audio file", type=['m4a', 'wav', 'mp3', 'mp4'])
    if audio_file and st.button('Convert Audio to Text'):
        text_output = convert_voice_to_text(audio_file)  # Convert audio file to text
        st.session_state['text_output'] = text_output  # Store text output in session state for later use

        processed_output = tooling1.process_text_from_streamlit(st.session_state['text_output'])

        # Display the processed output in Streamlit
        st.text_area("Processed Output", value=processed_output, height=300)

elif option == 'Record Audio':
    st.write("Record yourself describing the plant, and play the recorded audio:")
    audio = mic_recorder(start_prompt="Start Recording ⏺️", stop_prompt="Stop Recording ⏹️", key='recorder')
    if audio is not None and 'bytes' in audio:
        # Save the recorded audio to a file
        with open("recorded_audio.wav", "wb") as f:
            f.write(audio['bytes'])  # Write the bytes to a file
        st.success("Audio recorded and saved successfully.")
        # Convert recorded audio to text
        if st.button('Convert Recorded Audio to Text'):
            with open("recorded_audio.wav", "rb") as f:
                text_output = convert_voice_to_text(f)  # Convert recorded audio file to text
                st.session_state['text_output'] = text_output  # Store text output in session state for later use

                processed_output = tooling1.process_text_from_streamlit(st.session_state['text_output'])
                
                # Display the processed output in Streamlit
                st.text_area("Processed Output", value=processed_output, height=300)

# Additional functionality (translation and text-to-speech conversion) remains the same
if 'text_output' in st.session_state and st.session_state['text_output']:
    language_input = st.text_input("Enter the language you would like to translate it to:", "")

    if language_input and st.button('Translate further if required'):
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You are a universal translator."},
                {"role": "user", "content": f"Translate into {language_input}: {st.session_state['text_output']}"}
            ]
        )
        
        text_to_voice = response.choices[0].message.content
        st.text_area("Translated Text Output", value=text_to_voice, height=300)
        
        answer = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text_to_voice,
        )
        answer.stream_to_file("translated_audio.mp3")

        # Play the synthesized audio
        st.audio("translated_audio.mp3", format='audio/mp3', start_time=0)
