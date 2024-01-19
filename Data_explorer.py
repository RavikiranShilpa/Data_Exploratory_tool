import pandas as pd
import requests
from nl2query import PandasQuery
import pandasql as psql
import matplotlib.pyplot as plt

from datetime import datetime
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
import streamlit as st
import pandas as pd
import os
import seaborn as sns


# Set the framework to PyTorch explicitly
from transformers import pipeline
# Load a pre-trained model for question answering from Hugging Face
import spacy
nlp = spacy.load("en_core_web_sm")
#openai_api_key = " "

agriculture_df=pd.read_csv("agriculture_df.csv")
weather_df=pd.read_csv("weather_df.csv")
economy_df=pd.read_csv("economy_df.csv")
merged_df = pd.read_csv("merged_dataset.csv")

# Display the merged dataset

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}
# Function to initialize session state
def init_session_state():
    st.session_state.messages = []

# Check if session state is already initialized
if 'messages' not in st.session_state:
    init_session_state()
    
    
def clear_submit():
    st.session_state["submit"] = False
    
    
@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None
    
    
def answer_question_opeanai(data, query):
    # Use LangChain agents for chat-based interactions
    llm = ChatOpenAI(
        temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=openai_api_key, streaming=True
    )

    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        data,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
    )

    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    
    # Check if session state is already initialized before using it
    if 'messages' not in st.session_state:
        init_session_state()

    response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])

    return response



def answer_question(data, query):
    try:
        # Tokenize the user's query
        tokens = [token.text.lower() for token in nlp(query)]
        
        # Display tokens for debugging
        #st.write("Tokens:", tokens)

        # Find matching columns based on tokenized query
        matching_columns = [col for col in data.columns if any(token in col.lower() for token in tokens)]

        # Display matching columns for debugging
        #st.write("Matching Columns:", matching_columns)

        # If there are no matching columns, return a message
        if not matching_columns:
            return "No matching columns found in the dataset."

        # Retrieve information from the matching columns
        result = data[matching_columns].to_dict(orient='records')

        # Display information about the matching columns
        st.write("Information about Matching Columns:")
        st.write(data[matching_columns].describe())

    

        return result

    except Exception as e:
        return f"Error: {str(e)}"


def clean_data(data, actions):
    cleaned_data = data.copy()

    for action in actions:
        if action[0] == "remove_nulls":
            cleaned_data = cleaned_data.dropna()
        elif action[0] == "remove_duplicates":
            cleaned_data = cleaned_data.drop_duplicates()
        elif action[0] == "drop_column":
            cleaned_data = cleaned_data.drop(columns=[action[1]])
        elif action[0] == "fill_missing":
            fill_column, fill_strategy = action[1], action[2]
            if fill_strategy == "forward_fill":
                cleaned_data[fill_column] = cleaned_data[fill_column].ffill()
            elif fill_strategy == "mean":
                cleaned_data[fill_column] = cleaned_data[fill_column].fillna(cleaned_data[fill_column].mean())
            elif fill_strategy == "median":
                cleaned_data[fill_column] = cleaned_data[fill_column].fillna(cleaned_data[fill_column].median())
            elif fill_strategy == "mode":
                cleaned_data[fill_column] = cleaned_data[fill_column].fillna(cleaned_data[fill_column].mode().iloc[0])
            elif fill_strategy == "custom_value":
                cleaned_data[fill_column] = cleaned_data[fill_column].fillna(custom_value)
        elif action[0] == "convert_to_lowercase":
            cleaned_data[action[1]] = cleaned_data[action[1]].str.lower()
        elif action[0] == "remove_outliers":
            column_to_remove_outliers = action[1]
    
            # Calculate Z-scores for the specified column
            z_scores = (cleaned_data[column_to_remove_outliers] - cleaned_data[column_to_remove_outliers].mean()) / cleaned_data[column_to_remove_outliers].std()
    
            # Set a Z-score threshold (e.g., 3) beyond which values are considered outliers
            z_score_threshold = 3
    
            # Remove rows with outliers
            cleaned_data = cleaned_data[abs(z_scores) <= z_score_threshold]

        elif action[0] == "standardize_column":
            column_to_standardize = action[1]

            # Standardize values in the specified column using Z-score
            cleaned_data[column_to_standardize] = (cleaned_data[column_to_standardize] - cleaned_data[column_to_standardize].mean()) / cleaned_data[column_to_standardize].std()

        elif action[0] == "drop_rows_with_nulls":
            cleaned_data = cleaned_data.dropna(axis=0, how="any")
        elif action[0] == "filter_rows":
            condition = action[1]

            # Example: condition = "Age > 30"
            # Split the condition into column and comparison
            column, comparison, value = condition.split()
    
            # Apply the filter based on the condition
            if comparison == ">":
                cleaned_data = cleaned_data[cleaned_data[column] > float(value)]
            elif comparison == "<":
                cleaned_data = cleaned_data[cleaned_data[column] < float(value)]
            elif comparison == ">=":
                cleaned_data = cleaned_data[cleaned_data[column] >= float(value)]
            elif comparison == "<=":
                cleaned_data = cleaned_data[cleaned_data[column] <= float(value)]
            elif comparison == "==":
                cleaned_data = cleaned_data[cleaned_data[column] == float(value)]
            elif comparison == "!=":
                cleaned_data = cleaned_data[cleaned_data[column] != float(value)]

        elif action[0] == "rename_column":
            cleaned_data = cleaned_data.rename(columns={action[1]: action[2]})
        elif action[0] == "remove_spaces":
            cleaned_data[action[1]] = cleaned_data[action[1]].str.strip()
        elif action[0] == "extract_substring":
            column_to_extract, start_index = action[1], action[2]

            # Example: Extract substring from "Name" column starting at index 2
            cleaned_data[column_to_extract] = cleaned_data[column_to_extract].str[start_index:]

        elif action[0] == "encode_categorical":
            column_to_encode = action[1]

            # Use pandas' get_dummies function to one-hot encode categorical values
            cleaned_data = pd.get_dummies(cleaned_data, columns=[column_to_encode])

        elif action[0] == "impute_missing":
            impute_column, impute_strategy = action[1], action[2]

            if impute_strategy == "mean":
                cleaned_data[impute_column] = cleaned_data[impute_column].fillna(cleaned_data[impute_column].mean())
            elif impute_strategy == "median":
                cleaned_data[impute_column] = cleaned_data[impute_column].fillna(cleaned_data[impute_column].median())
            elif impute_strategy == "mode":
                cleaned_data[impute_column] = cleaned_data[impute_column].fillna(cleaned_data[impute_column].mode().iloc[0])
            elif impute_strategy == "custom_value":
                # Ask the user for the custom value
                custom_value = st.text_input(f"Enter custom value for imputing missing values in {impute_column}:", key=f"custom_value_{impute_column}")

        # Fill missing values with the user-specified custom value
                cleaned_data[impute_column] = cleaned_data[impute_column].fillna(custom_value)


    # Add more cleaning actions as needed...

    return cleaned_data



# Function to process user's query
def process_user_query_tab1(data, query):
    actions = []

    query_map = {
        "remove nulls": ("remove_nulls",),
        "remove duplicates": ("remove_duplicates",),
        "drop column": ("drop_column",),
        "fill missing": ("fill_missing",),
        "convert to lowercase": ("convert_to_lowercase",),
        "remove outliers": ("remove_outliers",),
        "standardize column": ("standardize_column",),
        "drop rows with any nulls": ("drop_rows_with_nulls",),
        "filter rows": ("filter_rows",),
        "rename column": ("rename_column",),
        "remove leading/trailing spaces": ("remove_spaces",),
        "extract substring": ("extract_substring",),
        "encode categorical": ("encode_categorical",),
        "impute missing values": ("impute_missing",),
    }

    # Check if the query is in the query_map
    if query.lower() in query_map:
        actions.append(query_map[query.lower()])
    else:
        # Check if the query contains information about filling missing values
        if "fill missing values in" in query.lower():
            # Example: "Fill missing values in Age with forward_fill"
            parts = query.split("fill missing values in ")[1].split(" with ")
            fill_column, fill_strategy = parts[0].strip(), parts[1].strip()
            actions.append(("fill_missing", fill_column, fill_strategy))
        elif "impute missing values in" in query.lower():
            # Example: "Impute missing values in Age with mean"
            parts = query.split("impute missing values in ")[1].split(" with ")
            impute_column, impute_strategy = parts[0].strip(), parts[1].strip()
            actions.append(("impute_missing", impute_column, impute_strategy))
        else:
            st.warning(f"Unsupported query: {query}")

    return actions



# Function to handle user queries
def handle_user_queries_tab1(data):
    st.subheader("Tab 1: Data Cleaning and Exploration")
    
    # Placeholder for cleaning actions
    cleaning_actions = []

    # Text input for user query
    user_query = st.text_area("What data operations would you like to perform?:")

    # Submit button
    submit_button = st.button("Submit")

    if submit_button and user_query:
        identified_actions = process_user_query_tab1(data, user_query)

        st.subheader("User Query Result (Tab 1):")
        st.write("Your query:", user_query)

        if identified_actions:
            for action in identified_actions:
                data = clean_data(data, action)
                cleaning_actions.append((action[0], *action[1:]))

            # Display a sample of the cleaned data
            st.write("Data Sample after Actions:")
            st.write(data.head(10))

            # Display basic statistics about the cleaned data
            st.subheader("Basic Statistics about the Cleaned Data:")
            st.write(data.describe())

            # Display text description of cleaning actions
            st.subheader("Cleaning Actions:")
            for clean_action in cleaning_actions:
                st.write(f"- {clean_action[0]}: {clean_action[1:]}")
       # Allow users to ask questions
    user_question = st.text_input("Ask anything about the data:")
    
    if user_question and data is not None:

        response = answer_question_opeanai(data, user_question)

        #st.subheader("Answer:")
        #st.write(response)


# Function to handle user queries in Tab 2
def handle_user_queries_tab2():
    st.subheader("Data Analysis and Prediction")
    data=None
    # Display information about available datasets
    st.write("### Available Datasets:")
    st.write("- Agriculture Data")
    st.write("- Weather Data")
    st.write("- Economy Data")
    st.write("- Merged Data (Agriculture + Weather + Economy)")

    # Allow users to ask questions
    selected_dataset = st.selectbox("Select a dataset:", ["Agriculture Data", "Weather Data", "Economy Data", "Merged Data"])
    user_question = st.text_input("Ask a question about the selected dataset:")

    if user_question and selected_dataset:
        # Load the corresponding DataFrame based on the selected dataset
        if selected_dataset == "Agriculture Data":
            response = answer_question_opeanai(agriculture_df, user_question)
        elif selected_dataset == "Weather Data":
            response = answer_question_opeanai(weather_df, user_question)
        elif selected_dataset == "Economy Data":
            response = answer_question_opeanai(economy_df, user_question)
        elif selected_dataset == "Merged Data":
            response = answer_question_opeanai(merged_df, user_question)
        else:
            st.warning("Invalid dataset selection")

        # Now you can use the loaded DataFrame for answering questions
        response = answer_question_opeanai(data, user_question)

        st.subheader("Answer:")
        st.write(response)

        # Now you can use the loaded DataFrame for answering questions
        #response = answer_question(data, user_question)

        #st.subheader("Answer:")
        #st.write(response)

def main():
    st.set_page_config(page_title="Data Explorer Tool", page_icon="🔍")
    st.title("Data Explorer Tool")
    data=None
    # Tab selection
    selected_tab = st.selectbox("Select a tab:", ["Data Cleaning and Exploration", "Data Analysis and Prediction"])

    # Tab content
    if selected_tab == "Data Cleaning and Exploration":
        # Display dataset upload option specific to Tab 1
        uploaded_file = st.file_uploader("Upload a Data file",type=list(file_formats.keys()),help="Various File formats are Support",on_change=clear_submit,)

        if not uploaded_file:
            st.warning("This app uses LangChain's `PythonAstREPLTool` which is vulnerable to arbitrary code execution. Please use caution in deploying and sharing this app.")        
        if uploaded_file:
            
            uploaded_file.seek(0)
            data = load_data(uploaded_file) 
            if data is not None:
                st.success("Data uploaded successfully!")

                handle_user_queries_tab1(data)

    elif selected_tab == "Data Analysis and Prediction":
        handle_user_queries_tab2()

# Run the app
if __name__ == "__main__":
    main()
