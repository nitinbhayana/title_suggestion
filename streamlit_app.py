import streamlit as st
import time
import random
import requests
import re
from openai import OpenAI
import json

BASE_URL = "http://91.203.132.18:8085"
client = OpenAI(
    api_key="Z0M9UX80FZ1I94DIRMG2OGXB1B8WNDS1YXWG4JUF",
    base_url="https://api.runpod.ai/v2/1anply7r9bug3m/openai/v1",
    )

def api_1(title):
    # Simulate a delay for the API call
    # time.sleep(2)  # Simulate a 2-second delay
    first_word ="{" + '"'+ title.split()[0].lower()
    
    B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"
    B_INST, E_INST = "[INST]", "[/INST]"
    B_in, E_in = "[Title]", "[/Title]"
    prompt = f"""{B_INST} {B_SYS} You are a helpful assistant that provides accurate and concise responses. {E_SYS}
    Extract named entities from the given product title. Provide the output in JSON format.
    {B_in} {title.strip()} {E_in}\n{E_INST}
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
    output_dict = convert_to_dictionary(output)
    return output_dict

                   
def api_2(title, count):
    # Simulate a delay for the API call
    # time.sleep(2)  # Simulate a 2-second delay
    search_category_input_data = {
        "product": title.strip(),
        "count": count
        }
    response = requests.post(f"{BASE_URL}/searchcat/", json=search_category_input_data)
    category = response.json()
    my_cat = category[0]
    return my_cat

def api_3(category, count):
    # Simulate a delay for the API call
    # time.sleep(2)  # Simulate a 2-second delay
    search_keywords_input_data = {
        "category": category.strip(),
        "count": count
        }
    response = requests.post(f"{BASE_URL}/searchkeywords/", json=search_keywords_input_data)
    keywords = response.json()
    return keywords



def get_title_suggestions(prompt):
    response_stream = client.completions.create(
    model="shivanikerai/Llama-2-7b-chat-hf-seo-optimised-title-suggestion-v1.0",
    prompt = prompt,
    temperature=0,
    max_tokens=512
    )
    return response_stream


def api_4(selected_dict, selected_keywords):
    # Simulate a delay for the API call
    # time.sleep(2)  # Simulate a 2-second delay
    prompt = f"""[INST] <<SYS>> You are a helpful, respectful and honest assistant for ecommerce product title creation. <</SYS>>
    Create a SEO optimized e-commerce product title for the keywords:{selected_keywords}
    [Product Details]{selected_dict}[/Product Details]\n[/INST]
    \n[Suggested Titles]"""
    
    response = get_title_suggestions(prompt)
    print(response)
    title_suggestions = response.choices[0].text
    title_suggestions = title_suggestions.replace('[/Suggested Titles]', '')
    title_suggestions = title_suggestions.strip()
    titles = title_suggestions.split('\n')
    unique_titles = list(set(titles))
    print(unique_titles)
    return unique_titles


def api_5(titles, keywords):
    # Simulate a delay for the API call and return a list of scores between 0 and 1
    # time.sleep(2)  # Simulate a 2-second delay
    relevance_score_input_data = {
        "titles": titles,
        "keywords": keywords
        }
    response = requests.post(f"{BASE_URL}/get_relevance_score/", json=relevance_score_input_data)
    scores = response.json()
    return scores

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

def convert_to_dictionary(input_string):
    try:
        input_string = input_string.replace('</s>', '')
        input_string = input_string.replace("\n ","\n")
        input_string = input_string.replace(" :",":")
        input_string = input_string.replace("\n"," ")
        data_dict = {}
        for item in input_string.split('", "'):
            key, value = item.split('": "')
            key = key.strip('{}"')
            value = value.strip('{}"')
            data_dict[key] = value
        # inverted_dict = {}
        # for key, value in data_dict.items():
        #     if value in inverted_dict:
        #         if not isinstance(inverted_dict[value], list):
        #             inverted_dict[value] = [inverted_dict[value]]
        #         inverted_dict[value].append(key)
        #     else:
        #         inverted_dict[value] = [key]
        return data_dict
    except Exception as e:
        print(f"\nAn error occurred: {e}\n{input_string}")
        pass

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
        # title = title.replace('"', "'")
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
            st.session_state.category = api_2(st.session_state.title, count = 5)
            # sub_category = st.session_state.category.split("->")[-2:]
            # st.session_state.sub_category = "->".join(sub_category)
        with st.spinner("Collecting AMS search_terms..."):
            st.session_state.keywords_with_sv = api_3(st.session_state.category, count = 100)
            st.session_state.keywords = [index['input_search_term'] for index in st.session_state.keywords_with_sv]
        st.session_state.step = 3

    # Step 3: Display keywords and dictionary response with checkboxes
    if st.session_state.step == 3:
        st.write("This product is categorized under <span style='font-weight:bold; color:blue'>", st.session_state.category, "</span>", unsafe_allow_html=True)

        st.write("Select Product Attributes:")
        selected_dict = {}
        #st.write(st.session_state.api_1_response1)
        for key, value in st.session_state.api_1_response1.items():
            selected_dict[key] = []
            for item in value:
                if st.checkbox(f"{key}: {item}", key=f"{key}-{item}", value=True):
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
            sorted_keywords_with_sv = sorted(st.session_state.keywords_with_sv, key=lambda x: x['search_volume'], reverse=True)
            st.sesstion_state.display_keywords = [f"{index['input_search_term']} ({index['search_volume']})" for index in sorted_keywords_with_sv]
            selected_display_keywords = st.multiselect("search_terms", options=st.session_state.display_keywords ,placeholder ="Select maximum of Six Search_terms", max_selections =6)
            keyword_pattern = re.compile(r"^(.*) \(\d+\)$")
            selected_keywords = [keyword_pattern.match(opt).group(1) for opt in selected_display_keywords if keyword_pattern.match(opt)]
        with col2:
            new_keyword = st.text_input("Add keyword:")
            if st.button("Add Keyword"):
                if new_keyword:
                    st.session_state.keywords.append(new_keyword)
                    st.experimental_rerun()

        if st.button("Suggest Titles"):
            with st.spinner("Suggesting New Titles..."):
                new_selected_dict = {key: value for key, value in selected_dict.items() if len(value) != 0}
                st.session_state.selected_dict = new_selected_dict
                st.session_state.selected_keywords = selected_keywords
                st.session_state.new_titles = api_4(selected_dict, selected_keywords)
            st.session_state.step = 4

    # Step 4: Display original and new product titles only after clicking "Suggest Titles"
    if st.session_state.step == 4:
        # st.write(st.session_state.selected_dict)
        # st.write(st.session_state.selected_keywords)
        st.write(f"Original Title:\n{st.session_state.title}")
        st.write("New Product Titles:")
        new_titles = st.session_state.new_titles
        for index,title in enumarate(new_titles):
            st.write(f"{index+1}. {title}")

        titles = [st.session_state.title] + st.session_state.new_titles
        keywords = st.session_state.keywords

        if st.button("Get Scores"):
            with st.spinner("Getting Scores..."):
                st.session_state.scores = api_5(titles, keywords)
            st.session_state.step = 5

# Step 5: Display scores for each title on a new screen
    if st.session_state.step == 5:
        st.write("Product Titles with Scores:")
        # titles = [st.session_state.title] + st.session_state.new_titles
        title_scores = st.session_state.scores
        for title_score in title_scores:
            st.write(f"{title_score['title']}: {title_score['score']}")
            st.progress(title_score['score'])

    # Add Reset button
    if st.button("Reset"):
        reset_state()
        st.experimental_rerun()

if __name__ == "__main__":
    main()
