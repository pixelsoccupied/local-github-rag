import streamlit as st
import os
import logging
from pathlib import Path
from git import Repo
from github import Github
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama
from typing import Optional, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "qwen2.5-coder:14b"
EMBEDDING_MODEL = "nomic-embed-text"
PERSIST_DIRECTORY = "./code_rag_db"
VECTOR_STORE_NAME = "code-rag"


def list_processed_repositories() -> List[str]:
    """List all repositories that have been processed."""
    if os.path.exists(PERSIST_DIRECTORY):
        return [d for d in os.listdir(PERSIST_DIRECTORY)
                if os.path.isdir(os.path.join(PERSIST_DIRECTORY, d))]
    return []


def list_downloaded_repositories(target_dir: str = "./repos") -> List[str]:
    """List all repositories that have been downloaded."""
    if os.path.exists(target_dir):
        return [d for d in os.listdir(target_dir)
                if os.path.isdir(os.path.join(target_dir, d)) and not d.startswith('.')]
    return []


def init_session_state():
    if 'repo_processed' not in st.session_state:
        st.session_state.repo_processed = False
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'chain' not in st.session_state:
        st.session_state.chain = None
    if 'current_repo' not in st.session_state:
        st.session_state.current_repo = None
    if 'processed_repos' not in st.session_state:
        st.session_state.processed_repos = list_processed_repositories()
    if 'downloaded_repos' not in st.session_state:
        st.session_state.downloaded_repos = list_downloaded_repositories()


def clone_repository(github_url: str, force_download: bool = False, target_dir: str = "./repos") -> Path:
    if not github_url.startswith(('http://', 'https://')):
        raise ValueError("Invalid GitHub URL")

    os.makedirs(target_dir, exist_ok=True)
    repo_name = github_url.split("/")[-1].replace(".git", "")
    repo_path = Path(target_dir) / repo_name

    try:
        if repo_path.exists() and not force_download:
            st.info(f"Using existing repository at {repo_path}")
            return repo_path

        if repo_path.exists():
            st.info(f"Force downloading repository...")
            import shutil
            shutil.rmtree(repo_path)

        st.info(f"Cloning repository {github_url}")
        Repo.clone_from(github_url, repo_path)
        return repo_path

    except Exception as e:
        st.error(f"Failed to clone/update repository: {e}")
        raise


def fetch_github_issues(repo_url: str, github_token: Optional[str] = None) -> List[Document]:
    """Fetch GitHub issues if credentials are provided."""
    try:
        g = Github(github_token) if github_token else Github()
        _, _, _, owner, repo_name = repo_url.rstrip('/').split('/')
        repo = g.get_repo(f"{owner}/{repo_name}")

        documents = []
        rate_limit = g.get_rate_limit()

        if rate_limit.core.remaining < 100:
            st.warning(f"GitHub API rate limit low: {rate_limit.core.remaining} remaining")
            return []

        with st.spinner("Fetching GitHub issues..."):
            progress_bar = st.progress(0)
            issues = list(repo.get_issues(state='all'))
            total_issues = len(issues)

            for idx, issue in enumerate(issues):
                try:
                    comments = list(issue.get_comments())
                    content = f"""Title: {issue.title}
Description: {issue.body or ''}
State: {issue.state}
Labels: {', '.join([label.name for label in issue.labels])}
Comments:
{chr(10).join([comment.body for comment in comments])}
"""
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": f"issue_{issue.number}",
                                "type": "issue",
                                "url": issue.html_url,
                                "created_at": issue.created_at.isoformat()
                            }
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error processing issue {issue.number}: {e}")
                    continue

                progress = min(1.0, (idx + 1) / total_issues)
                progress_bar.progress(progress)

        return documents

    except Exception as e:
        st.warning(f"Failed to fetch GitHub issues: {e}")
        return []


def load_repository_files(repo_path: Path) -> List[Document]:
    """Load code files from the repository."""
    documents = []
    extensions = ['.py', '.md', '.txt', '.go', '.js', '.java', '.cpp', '.h', '.rs', '.adoc']

    with st.spinner("Loading repository files..."):
        progress_bar = st.progress(0)
        file_paths = list(repo_path.rglob("*"))
        total_files = len(file_paths)

        for idx, file_path in enumerate(file_paths):
            if file_path.suffix in extensions and '.git' not in str(file_path).split(os.sep):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():
                            relative_path = str(file_path.relative_to(repo_path))
                            documents.append(
                                Document(
                                    page_content=content,
                                    metadata={
                                        "source": relative_path,
                                        "file_type": file_path.suffix,
                                        "type": "code"
                                    }
                                )
                            )
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")

            progress = min(1.0, (idx + 1) / total_files)
            progress_bar.progress(progress)

    return documents


def load_vector_store(repo_name: str) -> Optional[Chroma]:
    """Load an existing vector store for a repository if it exists."""
    repo_persist_directory = os.path.join(PERSIST_DIRECTORY, repo_name)
    collection_name = f"{VECTOR_STORE_NAME}-{repo_name}"

    if os.path.exists(repo_persist_directory):
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        return Chroma(
            persist_directory=repo_persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name
        )
    return None


def setup_vector_store(_documents: List[Document], repo_name: str) -> Chroma:
    """Set up the vector store with the provided documents."""
    # Create repository-specific paths
    repo_persist_directory = os.path.join(PERSIST_DIRECTORY, repo_name)
    collection_name = f"{VECTOR_STORE_NAME}-{repo_name}"

    # Clear existing vector store if it exists
    if os.path.exists(repo_persist_directory):
        import shutil
        shutil.rmtree(repo_persist_directory)

    with st.spinner("Setting up embedding model..."):
        ollama.pull(EMBEDDING_MODEL)
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    with st.spinner("Processing documents..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(_documents)

    with st.spinner("Creating vector store..."):
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=repo_persist_directory
        )
        vector_store.persist()

    return vector_store


def create_chain(vector_store):
    """Create the QA chain."""
    llm = ChatOllama(model=MODEL_NAME)

    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI code assistant. Generate five different versions of the
given question to help retrieve relevant code documents and GitHub issues. Focus on technical and
implementation aspects, as well as related discussions and problem reports.

Original question: {question}"""
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_store.as_retriever(),
        llm,
        prompt=query_prompt
    )

    template = """You are a helpful code assistant. Answer the question based on the
following code context and GitHub issues. If the context doesn't contain relevant information, say so.
Be specific and include code examples when appropriate. When referencing GitHub issues, include their numbers and URLs.

Context: {context}
Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain


def process_repository_files(repo_name: str, repo_path: Path, include_issues: bool = False,
                             github_token: Optional[str] = None):
    """Process repository files and optionally GitHub issues."""
    try:
        # Load code files
        code_docs = load_repository_files(repo_path)

        # Fetch issues if requested
        issue_docs = []
        if include_issues and github_token:
            repo_url = f"https://github.com/{repo_name}"  # This is an approximation
            issue_docs = fetch_github_issues(repo_url, github_token)
            if issue_docs:
                st.success(f"Loaded {len(issue_docs)} issues")
            else:
                st.warning("No issues were loaded")

        # Combine documents
        all_documents = code_docs + issue_docs
        if not all_documents:
            st.error("No documents were loaded from the repository")
            return

        st.success(f"Loaded {len(code_docs)} code files")

        # Set up vector store and chain
        st.session_state.vector_store = setup_vector_store(all_documents, repo_name)
        st.session_state.chain = create_chain(st.session_state.vector_store)
        st.session_state.repo_processed = True
        st.session_state.current_repo = repo_name

        # Update list of processed repositories
        if repo_name not in st.session_state.processed_repos:
            st.session_state.processed_repos.append(repo_name)

    except Exception as e:
        st.error(f"Error processing repository files: {e}")
        raise


def main():
    init_session_state()
    st.title("Code Repository Assistant")

    # Create three tabs for different repository operations
    tab1, tab2, tab3 = st.tabs(["Process New Repository", "Use Downloaded Repository", "Switch Processed Repository"])

    with tab1:
        st.subheader("Process New Repository")
        github_token = st.text_input("GitHub Token (optional):", type="password", key="github_token_new")
        repo_url = st.text_input(
            "Enter GitHub Repository URL:",
            placeholder="https://github.com/username/repository"
        )
        include_issues = st.checkbox("Include GitHub Issues", key="include_issues_new")
        process_new = st.button("Process New Repository")

    with tab2:
        st.subheader("Use Downloaded Repository")
        st.write("Select from repositories in ./repos directory:")

        # Refresh the list of downloaded repos
        st.session_state.downloaded_repos = list_downloaded_repositories()

        if st.session_state.downloaded_repos:
            selected_downloaded = st.selectbox(
                "Available repositories:",
                options=[""] + st.session_state.downloaded_repos,
                key="downloaded_repo_selector"
            )

            if selected_downloaded:
                st.write(f"Selected: {selected_downloaded}")
                col1, col2 = st.columns(2)
                with col1:
                    include_issues_downloaded = st.checkbox("Include GitHub Issues", key="include_issues_downloaded")
                    if include_issues_downloaded:
                        github_token = st.text_input("GitHub Token:", type="password", key="github_token_downloaded")
                with col2:
                    process_downloaded = st.button("Process Selected Repository", type="primary")
        else:
            st.info("No downloaded repositories found in ./repos directory. Please download a repository first.")
            process_downloaded = False
            selected_downloaded = ""

    with tab3:
        st.subheader("Switch Processed Repository")
        if st.session_state.processed_repos:
            selected_processed = st.selectbox(
                "Select a previously processed repository:",
                [""] + st.session_state.processed_repos,
                key="processed_repo_selector"
            )
            if selected_processed and selected_processed != st.session_state.current_repo:
                if st.button("Switch to Selected Repository"):
                    vector_store = load_vector_store(selected_processed)
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.session_state.chain = create_chain(vector_store)
                        st.session_state.current_repo = selected_processed
                        st.session_state.repo_processed = True
                        st.success(f"Switched to repository: {selected_processed}")
        else:
            st.info("No processed repositories available")

    # Process new repository from URL
    if process_new and repo_url:
        try:
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            repo_path = clone_repository(repo_url, force_download=False)
            st.session_state.downloaded_repos = list_downloaded_repositories()

            process_repository_files(
                repo_name,
                repo_path,
                include_issues=include_issues,
                github_token=github_token if include_issues else None
            )

        except Exception as e:
            st.error(f"Error processing repository: {e}")

    # Process downloaded repository
    if 'process_downloaded' in locals() and process_downloaded and selected_downloaded:
        try:
            repo_path = Path("./repos") / selected_downloaded
            process_repository_files(
                selected_downloaded,
                repo_path,
                include_issues=include_issues_downloaded,
                github_token=github_token if include_issues_downloaded else None
            )
        except Exception as e:
            st.error(f"Error processing downloaded repository: {e}")

    # Question answering interface
    if st.session_state.repo_processed:
        st.markdown("---")
        st.subheader(f"Ask questions about: {st.session_state.current_repo}")

        question = st.text_input("Enter your question:", placeholder="How does this code work?")

        if question:
            try:
                with st.spinner("Generating response..."):
                    response = st.session_state.chain.invoke(input=question)

                st.markdown("### Response:")
                st.markdown(response)

            except Exception as e:
                st.error(f"Error generating response: {e}")

    else:
        st.info("Please process a repository first to start asking questions.")


if __name__ == "__main__":
    main()