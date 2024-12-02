import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
from git import Repo
import shutil
from pathlib import Path
import openai

# Function to clone a GitHub repository
def clone_repository(repo_url):
    """Clones a GitHub repository to a temporary directory."""
    repo_name = repo_url.split("/")[-1]  # Extract repository name from URL
    repo_path = f"./{repo_name}"  # Use the current working directory instead of /content/

    # Check if the directory exists and delete it
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)  # Remove the directory and its contents

    # Clone the repository into the specified path
    Repo.clone_from(repo_url, str(repo_path))
    return str(repo_path)

# Function to index a cloned codebase into Pinecone
def index_codebase(repo_path, namespace):
    """Indexes the codebase by reading files, encoding content, and storing in Pinecone."""
    repo_files = Path(repo_path).rglob("*.*")  # Get all files in the repo

    # Initialize embeddings
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    index = pc.Index("codebase-rag")

    for file_path in repo_files:
        if file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Skip empty files
                if not content.strip():
                    print(f"Skipped empty file: {file_path}")
                    continue
                # Embed content and upsert into Pinecone
                vector = embedding_model.encode(content).tolist()
                index.upsert([{
                    "id": str(file_path),
                    "values": vector,
                    "metadata": {"text": content}
                }], namespace=namespace)
                print(f"Indexed file: {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    print(f"Codebase indexed successfully in namespace: {namespace}")
    # Debugging: Check index stats
    stats = index.describe_index_stats()
    print("Index Stats:", stats)

# Initialize Pinecone using the secrets API
pc = Pinecone(
    api_key=st.secrets["PINECONE"]["API_KEY"],  # Use Pinecone API Key from secrets
    environment=st.secrets["PINECONE"]["ENVIRONMENT"]  # Use Pinecone environment from secrets
)

# Set OpenAI API key from secrets
openai.api_key = st.secrets["GROQ"]["API_KEY"]  # Use Groq API key from secrets

# Function to get Hugging Face embeddings
def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

# Function to perform retrieval-augmented generation
def perform_rag(query, namespace):
    raw_query_embedding = get_huggingface_embeddings(query)

    # Query Pinecone index
    top_matches = pc.Index("codebase-rag").query(
        vector=raw_query_embedding.tolist(),
        top_k=5,
        include_metadata=True,
        namespace=namespace,
    )

    if not top_matches["matches"]:  # Check if there are no matches
        return "No relevant information found in the codebase. The namespace may be empty, or the query may not match any content. Please ensure the repository is properly indexed."

    contexts = [item['metadata']['text'] for item in top_matches['matches']]

    augmented_query = (
        "<CONTEXT>\n" +
        "\n\n-------\n\n".join(contexts[:10]) +
        "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" +
        query
    )

    system_prompt = (
    "You are a highly skilled and experienced Senior Software Engineer with expertise in TypeScript. "
    "Your role is to assist with understanding, analyzing, and providing insights about the codebase. "
    "Always use the provided code context to form accurate and well-reasoned responses. "
    "When answering questions, ensure your explanations are clear, concise, and tailored to the user's query. "
    "If relevant, provide examples or suggestions to improve code quality or resolve potential issues."
    )

    # Make the API call to the correct Groq chat endpoint
    llm_response = openai.ChatCompletion.create(
        model="llama-3.1-8b-instant",  # specify the model you want to use
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query},
        ],
        api_base="https://api.groq.com/openai/v1"  # Correct API endpoint
    )

    return llm_response['choices'][0]['message']['content']

# Streamlit UI
st.set_page_config(page_title="Codebase Chat Assistant", page_icon="💬")

# Ensure session state for chat history and namespace
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize 'namespace' if not already present in session state
if "namespace" not in st.session_state:
    st.session_state.namespace = None

# Streamlit Sidebar for navigation
st.sidebar.title("Codebase Chat Assistant")
st.sidebar.markdown("### Choose an option below:")

# User selects between pre-indexed or new repository
option = st.sidebar.radio("Choose how you want to interact:", 
                          ["Provide a new GitHub repository", "Select a pre-indexed codebase"])

# Handle namespace reset and chat clearing
def reset_chat(namespace):
    if st.session_state.namespace != namespace:
        st.session_state.namespace = namespace
        st.session_state.messages = []  # Clear chat history

# Fetch pre-indexed namespaces dynamically from Pinecone
index = pc.Index("codebase-rag")
preindexed_codebases = index.describe_index_stats().get("namespaces", {})

# Handle pre-indexed codebase selection
if option == "Select a pre-indexed codebase":
    if preindexed_codebases:
        selected_codebase = st.selectbox("Select a codebase:", list(preindexed_codebases.keys()))
        if selected_codebase:
            reset_chat(selected_codebase)
            st.success(f"Selected codebase: {selected_codebase}")
    else:
        st.warning("No pre-indexed codebases found.")

# Handle new GitHub repository input
if option == "Provide a new GitHub repository":
    github_url = st.text_input("Paste a public GitHub link:")
    if st.button("Index and Chat") and github_url:
        st.info(f"Cloning and indexing the repository: {github_url} (this will take a few minutes)")
        try:
            with st.spinner("Cloning and indexing repository..."):
                # Clone the repository
                repo_path = clone_repository(github_url)
                reset_chat(github_url)

                # Index the codebase
                index_codebase(repo_path, st.session_state.namespace)
                st.success("Repository indexed successfully! You can now start chatting.")
        except Exception as e:
            st.error(f"Error indexing the repository: {e}")

# Chat functionality
if st.session_state.namespace:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask something about the codebase"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            response = perform_rag(prompt, st.session_state.namespace)
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
