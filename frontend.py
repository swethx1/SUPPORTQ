
import requests
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 180

st.set_page_config(
    page_title="SupportIQ",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# Session state
# --------------------------------------------------

DEFAULTS = {
    "document_id": None,
    "document_name": None,
    "messages": [],
    "upload_error": None,
    "api_error": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# Custom styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    /* Overall page */
    .stApp {
        background: #10141C;
    }

    .block-container {
        max-width: 1050px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
}

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 0.5rem;
        margin-bottom: 1.2rem;
        padding: 6px 0;
}

    .brand-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 44px;
        height: 44px;
        border-radius: 13px;
        background: #263956;
        border: 1px solid #3B5276;
        font-size: 23px;
    }

    .brand-name {
        color: #E7ECF5;
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .brand-tagline {
        color: #96A4BB;
        font-size: 0.88rem;
        margin-top: 2px;
    }

    /* Document status panel */
    .document-panel {
        background: #191F2B;
        border: 1px solid #2B3547;
        border-radius: 15px;
        padding: 15px 18px;
        margin: 12px 0 20px 0;
    }

    .muted {
        color: #9AA9C0;
        font-size: 0.88rem;
    }

    .document-name {
        color: #E7ECF5;
        font-weight: 600;
        overflow-wrap: anywhere;
    }

    /* Intro panel */
    .welcome {
        background: linear-gradient(
            135deg,
            #1B2638 0%,
            #202E45 100%
        );
        border: 1px solid #344967;
        border-radius: 20px;
        padding: 30px;
        margin: 16px 0 20px 0;
    }

    .welcome h2 {
        color: #F0F4FC;
        font-size: 1.8rem;
        letter-spacing: -0.6px;
        margin: 0 0 10px 0;
    }

    .welcome p {
        color: #B9C7DD;
        font-size: 1rem;
        line-height: 1.7;
        margin-bottom: 0;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 550;
        transition: border-color 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #7896C8;
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        border: 1px solid #263143;
        border-radius: 15px;
        padding: 12px 15px;
        background: #151B25;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        border-color: #34445D;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: #151B25;
        border: 1px dashed #40516B;
        border-radius: 12px;
        padding: 10px;
    }

    /* Expandable source sections */
    [data-testid="stExpander"] {
        background: #151B25;
        border: 1px solid #2B3547;
        border-radius: 12px;
    }

    /* Small footer */
    .footer {
        text-align: center;
        color: #718098;
        font-size: 0.8rem;
        padding-top: 22px;
    }

    /* Hide the default Streamlit decoration */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Reusable functions
# --------------------------------------------------

def clear_active_document():
    """Clear the active document and its visible conversation."""
    st.session_state.document_id = None
    st.session_state.document_name = None
    st.session_state.messages = []
    st.session_state.upload_error = None
    st.session_state.api_error = None


def upload_pdf(uploaded_file):
    """Send the selected PDF to the existing FastAPI endpoint."""
    response = requests.post(
        f"{API_URL}/upload-document",
        files={
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf",
            )
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def ask_question(document_id, question):
    """Ask a question about the active document."""
    response = requests.post(
        f"{API_URL}/ask-document",
        json={
            "document_id": document_id,
            "question": question,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def display_sources(sources):
    """Display the retrieved passages used to support an answer."""
    if not sources:
        return

    with st.expander(
        f"📚 View sources · {len(sources)} passage(s)",
        expanded=False,
    ):
        for index, source in enumerate(sources, start=1):
            page = source.get("page", "?")
            content = source.get("content", "")

            st.markdown(f"**Source {index} · Page {page}**")
            st.write(content)

            if index < len(sources):
                st.divider()


def display_message(message):
    """Render a saved message and any sources attached to it."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            display_sources(message["sources"])


def error_details(error):
    """Extract a readable API error when one is available."""
    if isinstance(error, requests.HTTPError) and error.response is not None:
        try:
            data = error.response.json()
            detail = data.get("detail")
            if detail:
                return str(detail)
        except ValueError:
            pass

    return str(error)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    """
    <div class="brand">
        <div class="brand-icon">📘</div>
        <div>
            <div class="brand-name">SupportIQ</div>
            <div class="brand-tagline">
                Your documents, understood.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# --------------------------------------------------
# Compact document controls
# --------------------------------------------------

with st.container():
    left, right = st.columns([5, 2], vertical_alignment="center")

    with left:
        st.markdown("#### Your document")

        if st.session_state.document_id:
            st.markdown(
                f"""
                <div class="document-panel">
                    <div class="muted">ACTIVE DOCUMENT</div>
                    <div class="document-name">
                        📄 {st.session_state.document_name}
                    </div>
                    <div class="muted">
                        Ready to answer questions
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.caption("Upload a PDF to start a conversation.")

    with right:
        if st.session_state.document_id:
            if st.button(
                "＋ Change document",
                use_container_width=True,
            ):
                clear_active_document()
                st.rerun()


# --------------------------------------------------
# Upload workflow
# --------------------------------------------------

if not st.session_state.document_id:
    st.markdown(
        """
        <div class="welcome">
            <h2>Ask your documents anything.</h2>
            <p>
                Understand policies, warranties, manuals and other
                documents. Get answers with supporting page references.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Drop your PDF here, or browse your files",
        type=["pdf"],
        help="Maximum size: 10 MB. Text-based PDFs work best.",
        key="pdf_upload",
    )

    if uploaded_file is not None:
        st.caption(f"Selected: {uploaded_file.name}")

        if st.button(
            "Process document  →",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Reading your PDF and preparing it for questions..."
                ):
                    result = upload_pdf(uploaded_file)

                st.session_state.document_id = result["document_id"]
                st.session_state.document_name = result["filename"]
                st.session_state.messages = []
                st.session_state.upload_error = None
                st.session_state.api_error = None

                st.rerun()

            except requests.RequestException as error:
                st.session_state.upload_error = error_details(error)

    if st.session_state.upload_error:
        st.error("We couldn't process this document.")
        st.caption(st.session_state.upload_error)

    st.markdown(
        """
        <div class="footer">
            Your documents, your questions, clearer answers.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Do not show the chat until a document has been processed.
    st.stop()


# --------------------------------------------------
# Chat interface
# --------------------------------------------------

st.markdown("#### Conversation")
st.caption(
    "Ask naturally. SupportIQ searches your active document "
    "and shows supporting passages when available."
)

# Suggested prompts for first-time users.
if not st.session_state.messages:
    st.markdown("**Not sure where to start?**")

    suggestions = [
        "What is this document about?",
        "Summarise the key points",
        "What are the important conditions?",
    ]

    columns = st.columns(len(suggestions))

    for index, suggestion in enumerate(suggestions):
        with columns[index]:
            if st.button(
                suggestion,
                key=f"suggestion_{index}",
                use_container_width=True,
            ):
                st.session_state.pending_question = suggestion
                st.rerun()


# Restore conversation after Streamlit reruns.
for saved_message in st.session_state.messages:
    display_message(saved_message)


# Retrieve a suggestion selected above, or accept normal chat input.
pending_question = st.session_state.pop("pending_question", None)
typed_question = st.chat_input(
    "Ask a question about your document..."
)

question = pending_question or typed_question

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Finding the relevant passages..."):
                result = ask_question(
                    st.session_state.document_id,
                    question,
                )

            answer = result.get(
                "answer",
                "No answer was returned by the assistant.",
            )
            sources = result.get("sources", [])

            st.markdown(answer)
            display_sources(sources)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )

    except requests.RequestException as error:
        details = error_details(error)
        friendly_error = (
            "I couldn't retrieve an answer. Please check the backend "
            "and try again."
        )

        with st.chat_message("assistant"):
            st.error(friendly_error)

            with st.expander("Technical details"):
                st.code(details)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": friendly_error,
            }
        )


# Footer
st.markdown(
    """
    <div class="footer">
        SupportIQ · AI-powered document assistance
    </div>
    """,
    unsafe_allow_html=True,
)
