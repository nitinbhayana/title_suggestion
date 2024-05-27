import streamlit as st
import time
import random
import requests
import re


def api_1(title):
    # Simulate a delay for the API call
    time.sleep(2)  # Simulate a 2-second delay
    first_word ="{" + '"'+ title.split()[0].lower()
    
    B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"
    B_INST, E_INST = "[INST]", "[/INST]"
    B_in, E_in = "[Title]", "[/Title]"
    prompt = f"""{B_INST} {B_SYS} You are a helpful assistant that provides accurate and concise responses. {E_SYS}
    \nExtract named entities from the given product title. Provide the output in JSON format.
    \n{B_in} {title.strip()} {E_in}\n{E_INST}
    \n### NER Response:\n {first_word}"""
    
    API_URL_ner = "https://api-inference.huggingface.co/models/shivanikerai/TinyLlama-1.1B-Chat-v1.0-sku-title-ner-generation-reversed-v1.0"
    
    headers = {"Authorization": "Bearer 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"}
    payload={
    "inputs": prompt,
    "parameters": {"return_full_text":False, "max_new_tokens": 1024},
    "options":{"wait_for_model": True}
    }
    response = requests.post(API_URL_ner, headers=headers, json=payload)
    generated_text = response.json()[0]["generated_text"]
    
    output = first_word + " " + generated_text
    output = re.sub(' ": "', '": "', output)
    output_dict = eval(output)
    return output_dict

                   
def api_2(title):
    # Simulate a delay for the API call
    time.sleep(2)  # Simulate a 2-second delay
    return "Grocery & Gourmet Foods->Cereal & Muesli->Oats & Porridege"

def api_3(category):
    # Simulate a delay for the API call
    time.sleep(2)  # Simulate a 2-second delay
    return [f"keyword_{i}" for i in range(1, 11)]

def api_4(selected_dict, selected_keywords):
    # Simulate a delay for the API call
    time.sleep(2)  # Simulate a 2-second delay
    return [f"New Title {i}" for i in range(1, 6)]

def api_5(titles):
    # Simulate a delay for the API call and return a list of scores between 0 and 1
    time.sleep(2)  # Simulate a 2-second delay
    return [round(random.uniform(0, 1), 2) for _ in titles]

def title_annotated(title_input, ner_result):
    annotated_title = title_input

    # Sort entities by their start position to handle them in the correct sequence
    entities = sorted(ner_result, key=lambda x: title_input.lower().find(x.lower()))

    # Apply HTML tags to each entity found in the title
    for entity in entities:
        start_index = title_input.lower().find(entity.lower())
        if start_index != -1:  # Only proceed if the entity is found
            original_text = title_input[start_index:start_index + len(entity)]
            # Replace the original text with the annotated version in the title
            annotated_title = annotated_title.replace(original_text,
                                                      f"{original_text}<span style='color:green;'>({ner_result[entity]})</span>",
                                                      1)

    return annotated_title

def attribute_dict(data_dict):
    try:
        inverted_dict = {}
        for key, value in data_dict.items():
            if value in inverted_dict:
                if not isinstance(inverted_dict[value], list):
                    inverted_dict[value] = [inverted_dict[value]]
                inverted_dict[value].append(key)
            else:
                inverted_dict[value] = [key]
        return inverted_dict
    except Exception as e:
        return ({})

def reset_state():
    st.session_state.clear()
    st.session_state.step = 1

def main():
    st.title("Product Title Generation")

    # Step 1: User inputs the product title
    if 'step' not in st.session_state:
        st.session_state.step = 1

    if st.session_state.step == 1:
        title = st.text_input("Enter the product title:")
        if st.button("Submit"):
            with st.spinner("Analysing Title..."):
                st.session_state.title = title
                st.session_state.api_1_response = api_1(title)
                st.session_state.api_1_response1 = attribute_dict(st.session_state.api_1_response)
            st.session_state.step = 2

    # Step 2: Display title and API 1 response
    if st.session_state.step == 2:
        st.write(f"Title: {st.session_state.title}")
        annotated_title = title_annotated(st.session_state.title, st.session_state.api_1_response)
        st.markdown(annotated_title, unsafe_allow_html=True)
        with st.spinner("Checking Product Category..."):
            st.session_state.category = api_2(st.session_state.title)
        with st.spinner("Collecting AMS search_terms..."):
            st.session_state.keywords = api_3(st.session_state.category)
        st.session_state.step = 3

    # Step 3: Display keywords and dictionary response with checkboxes
    if st.session_state.step == 3:
        st.write("This product is categorized under <span style='font-weight:bold; color:blue'>", st.session_state.category, "</span>", unsafe_allow_html=True)

        st.write("Select Product Attributes:")
        selected_dict = {}
        st.write(st.session_state.api_1_response1)
        for key, value in st.session_state.api_1_response1.items():
            selected_dict[key] = []
            for item in value:
                if st.checkbox(f"{key}: {item}", key=f"{key}-{item}"):
                    selected_dict[key].append(item)

        col1, col2 ,col3 = st.columns(3)
        with col1:
            new_key = st.text_input("",placeholder="Attribute", label_visibility ='collapsed')
        with col2:
            new_value = st.text_input("",placeholder="Attribute Value", label_visibility ='collapsed')
        with col3:
            
            if st.button("Add"):
                if new_key and new_value:
                    if new_key in st.session_state.api_1_response1 and isinstance(st.session_state.api_1_response1[key], list):
                        st.session_state.api_1_response1[new_key].append(new_value)
                    else:
                        st.session_state.api_1_response1[new_key] = [new_value]
                    st.experimental_rerun()

        st.write("Select AMS search_terms:")
        col1, col2=st.columns(2)
        with col1:
            selected_keywords = st.multiselect("search_terms", options=st.session_state.keywords ,placeholder ="Select maximum of Six Search_terms", max_selections =6)
        with col2:
            new_keyword = st.text_input("Add keyword:")
            if st.button("Add Keyword"):
                if new_keyword:
                    st.session_state.keywords.append(new_keyword)
                    st.experimental_rerun()

        if st.button("Suggest Titles"):
            with st.spinner("Calling API 4..."):
                new_selected_dict = {key: value for key, value in selected_dict.items() if len(value) != 0}
                st.session_state.selected_dict = new_selected_dict
                st.session_state.selected_keywords = selected_keywords
                st.session_state.new_titles = api_4(selected_dict, selected_keywords)
            st.session_state.step = 4

    # Step 4: Display original and new product titles only after clicking "Suggest Titles"
    if st.session_state.step == 4:
        st.write(st.session_state.selected_dict)
        st.write(st.session_state.selected_keywords)
        st.write(f"Original Title: {st.session_state.title}")
        st.write("New Product Titles:")
        titles = [st.session_state.title] + st.session_state.new_titles
        for title in titles:
            st.write(title)

        if st.button("Get Scores"):
            with st.spinner("Calling API 5..."):
                st.session_state.scores = api_5(titles)
            st.session_state.step = 5

    # Step 5: Display scores for each title on a new screen
    if st.session_state.step == 5:
        st.write("Product Titles with Scores:")
        titles = [st.session_state.title] + st.session_state.new_titles
        scores = st.session_state.scores
        for title, score in zip(titles, scores):
            st.write(f"{title}: {score}")
            st.progress(score)

    # Add Reset button
    if st.button("Reset"):
        reset_state()
        st.experimental_rerun()

if __name__ == "__main__":
    main()
