import streamlit as st
from PIL import Image
from image.bird_image.species_from_image import get_species_from_image
from image.bird_image.detect_and_annotate import get_bbox_and_species
from image.feather_image.species_from_feather import get_species_from_feather
from image.leaf_image.inference_leaf import get_species_from_leaf
from audio.species.mtl_species_classi import mtl_species_classi
from audio.call.inference_call import predict_audio_class
from llm.generate_info import initial_prompt, get_llm_response_as_gen, get_llm_response_as_text 
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

BOT_ICON = "assistant"
USER_ICON = "😎"

# Function to handle user prompt
def user_submits_prompt():
    """
    Function to handle the user prompt submission.
    """
    user_prompt = st.session_state["user_prompt"]
    i = st.session_state["show_chat"] = st.session_state["last_chat"]
    ai_response = get_llm_response_as_text(i, user_prompt)
    st.session_state["history"][i][0].append(HumanMessage(user_prompt))
    st.session_state["history"][i][0].append(AIMessage(ai_response))


def show_previous_history():
    i = st.session_state["show_chat"]
    st.session_state["show_chat"] = -1

    with st.chat_message(USER_ICON):
        format = st.session_state["history"][i][1][0]
        resource = st.session_state["history"][i][1][1]
        if format == "image":
            _, col, _ = st.columns([1, 2, 1])
            col.write(resource)
        elif format == "audio":
            st.audio(resource)
    
    for j in range(1, len(st.session_state["history"][i][0])):
        msg = st.session_state["history"][i][0][j]
        icon = BOT_ICON if isinstance(msg, AIMessage) else USER_ICON
        with st.chat_message(icon):
            st.write(msg.content)

def show_image_and_gen(image_of):
    img = Image.open(st.session_state["file_uploaded"])
    img = img.resize((300, 300))

    # Creating a space for storing message for this chat
    i = len(st.session_state["history"])
    st.session_state["history"].append([[], []])

    # Displaying the image
    with st.chat_message(USER_ICON):
        with st.spinner("Analyzing image..."):
            if image_of == "bird":
                # species = get_species_from_image(img)
                img, species = get_bbox_and_species(img)
            elif image_of == "feather":
                species = get_species_from_feather(img)
            elif image_of == "leaf":
                species = get_species_from_leaf(img) 
                print(species)
        _, center_col, _  = st.columns([1, 2, 1])
        center_col.write(img)


    with st.chat_message(BOT_ICON):
        # Showing spinner till inference is done
        with st.spinner("LLM is thinking...."):

            st.session_state["history"][-1][0].append(
                SystemMessage(initial_prompt)
            )
            info = get_llm_response_as_gen(i, "Give me a brief summary about " + species)
        info = st.write_stream(info)
        st.session_state["history"][-1][0].append(AIMessage(info))


    st.session_state["chat_names"].append(f"{species} • Image")
    st.session_state["history"][-1][1].append("image")
    st.session_state["history"][-1][1].append(img)

def show_audio_and_gen():
    audio = st.session_state["file_uploaded"]

    i = len(st.session_state["history"])
    st.session_state["history"].append([[], []])

    with st.chat_message(USER_ICON):
        st.audio(audio)

    with st.chat_message(BOT_ICON):
        with st.spinner("Analyzing the audio..."):
            species, _ = mtl_species_classi(audio)
            type_of_call = predict_audio_class(audio)
            st.session_state["history"][-1][0].append(
                SystemMessage(initial_prompt)
            )
            info = get_llm_response_as_gen(i, f"You are given a audio of {species}, performing {type_of_call} type of sound. Give brief summary about this bird." )
        info = st.write_stream(info)
        st.session_state["history"][-1][0].append(AIMessage(info))

    st.session_state["chat_names"].append(f"{species} • Audio")
    st.session_state["history"][-1][1].append("audio")
    st.session_state["history"][-1][1].append(audio)


############################ PAGE LOGIC STARTS HERE ###################################



# Below statement is only for debugging purposes
# st.write("written from result.py")

# Checking if the user wants to see a previous chat
if st.session_state["show_chat"] != -1:
    st.session_state["last_chat"] = st.session_state["show_chat"]
    show_previous_history()
elif st.session_state["file_uploaded"] is None:
    st.switch_page("./pages/home.py")
# User must have uploaded a file
# If it is image
# elif st.session_state["file_uploaded"].type.find("image") != -1:
elif st.session_state["model_type"] == "Bird Image":
    st.session_state["last_chat"] = len(st.session_state["history"])
    show_image_and_gen("bird")
    st.session_state["file_uploaded"] = None

# If it is audio
elif st.session_state["model_type"] == "Bird Audio":
    st.session_state["last_chat"] = len(st.session_state["history"])
    show_audio_and_gen()
    st.session_state["file_uploaded"] = None

elif st.session_state["model_type"] == "Feather Image":
    st.session_state["last_chat"] = len(st.session_state["history"])
    show_image_and_gen("feather")
    st.session_state["file_uploaded"] = None

elif st.session_state["model_type"] == "Leaf Image":
    st.session_state["last_chat"] = len(st.session_state["history"])
    show_image_and_gen("leaf")
    st.session_state["file_uploaded"] = None

# Not audio or image
else:
    st.error("Unsupported file Type")

user_prompt = st.chat_input("Ask your follow up question!", key="user_prompt", on_submit=user_submits_prompt)


