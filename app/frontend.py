
import re
import requests
import streamlit as st


import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


# --------------------------------------------------
# Page setup
# --------------------------------------------------

st.set_page_config(
    page_title="EngiDoc AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.5rem;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 16px;
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 1.9rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        opacity: 0.75;
        font-size: 1rem;
    }

    .source-title {
        font-size: 0.95rem;
        font-weight: 600;
    }

    .source-meta {
        font-size: 0.85rem;
        opacity: 0.7;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Text helpers
# --------------------------------------------------

def clean_passage(text):
    """Clean common PDF extraction formatting artifacts."""
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=[a-z])", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_excerpt(text, answer, max_chars=650):
    """Prefer sentences that relate to the answer."""
    text = clean_passage(text)

    if len(text) <= max_chars:
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text)

    answer_words = {
        word.lower()
        for word in re.findall(r"\b[a-zA-Z]{4,}\b", answer)
    }

    stop_words = {
        "this", "that", "with", "from", "have", "does",
        "into", "their", "there", "these", "those", "only",
        "provided", "passages", "exact", "value", "values",
        "motor", "manual", "page", "pages",
    }

    answer_words -= stop_words

    scored_sentences = []

    for index, sentence in enumerate(sentences):
        sentence_words = {
            word.lower()
            for word in re.findall(r"\b[a-zA-Z]{4,}\b", sentence)
        }

        score = len(answer_words & sentence_words)
        scored_sentences.append((index, score, sentence))

    relevant = [
        item for item in scored_sentences if item[1] > 0
    ]

    if not relevant:
        return text[:max_chars].rsplit(" ", 1)[0] + "…"

    # Select relevant sentences, then preserve their original order.
    selected = []
    total_chars = 0

    for index, score, sentence in sorted(
        relevant,
        key=lambda item: (-item[1], item[0]),
    ):
        if total_chars + len(sentence) > max_chars:
            continue

        selected.append((index, sentence))
        total_chars += len(sentence)

    if not selected:
        return text[:max_chars].rsplit(" ", 1)[0] + "…"

    selected.sort(key=lambda item: item[0])
    excerpt = " ".join(sentence for _, sentence in selected)

    if len(excerpt) < len(text):
        excerpt += " …"

    return excerpt


def display_evidence(contexts, answer):
    """Show cited sources and concise excerpts, without technical scores."""
    cited_ids = set(
        re.findall(r"\[(S\d+)\]", answer)
    )

    if not cited_ids:
        st.caption("No source citations were provided in this answer.")
        return

    cited_contexts = []

    for index, context in enumerate(contexts, start=1):
        source_id = f"S{index}"

        if source_id in cited_ids:
            cited_contexts.append((source_id, context))

    if not cited_contexts:
        st.caption(
            "The answer contains citations, but matching source passages "
            "were not returned."
        )
        return

    with st.expander(
        f"Sources and supporting passages ({len(cited_contexts)})",
        expanded=False,
    ):
        st.caption(
            "Open a source to read the supporting excerpt. "
            "Use the page reference to verify it in the original manual."
        )

        for source_id, context in cited_contexts:
            document = context.get(
                "document_name", "Engineering document"
            )
            page = context.get("page_number", "Unknown")

            with st.container(border=True):
                st.markdown(f"**{document}**")
                st.caption(f"Page {page}")

                excerpt = make_excerpt(
                    context.get("text", ""),
                    answer,
                )

                st.markdown(excerpt)

                full_text = clean_passage(
                    context.get("text", "")
                )

                if full_text and full_text != excerpt.rstrip(" …"):
                    with st.popover("Read full passage"):
                        st.write(full_text)


# --------------------------------------------------
# Backend helpers
# --------------------------------------------------

def check_backend():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=3,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def ask_backend(question):
    response = requests.post(
        f"{API_URL}/ask",
        json={"question": question},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ EngiDoc AI")
    st.caption("Engineering knowledge assistant")

    st.divider()

    st.markdown("### How it helps")
    st.markdown(
        """
        - Find information in engineering manuals
        - Summarize relevant technical details
        - Show source pages for verification
        """
    )

    st.divider()

    st.markdown("### Built with")
    st.caption(
        "Python · FastAPI · Streamlit · ChromaDB"
    )
    st.caption(
        "Sentence Transformers · Cross-Encoder · Groq"
    )

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "Always verify critical specifications against "
        "the original engineering documentation."
    )


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            Engineering Documentation Assistant
        </div>
        <div class="hero-subtitle">
            Get clear answers from your engineering manuals,
            with page references to verify the details.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Backend status
# --------------------------------------------------

health = check_backend()

if health is None:
    st.error(
        "Cannot connect to the backend. Start FastAPI in another "
        "terminal with `python -m uvicorn app.api:app --reload`."
    )
    st.stop()

if health.get("status") != "ok":
    st.warning("The backend is starting. Please try again shortly.")
    st.stop()

st.caption(
    f"Connected · {health.get('indexed_chunks', 0):,} document passages indexed"
)


# --------------------------------------------------
# Conversation state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Conversation history
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_evidence(
                message.get("contexts", []),
                message["content"],
            )


# --------------------------------------------------
# Ask a question
# --------------------------------------------------

question = st.chat_input(
    "Ask about torque, maintenance, operating limits..."
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Checking the engineering manual..."
        ):
            try:
                result = ask_backend(question)

                answer = result["answer"]
                contexts = result.get("contexts", [])

                st.markdown("#### Answer")
                st.markdown(answer)

                st.divider()

                display_evidence(contexts, answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "contexts": contexts,
                    }
                )

            except requests.HTTPError as exc:
                st.error(
                    "The backend returned an error. "
                    "Check the FastAPI terminal for details."
                )
                st.caption(str(exc))

            except requests.RequestException as exc:
                st.error(
                    "Could not reach the backend. "
                    "Check that FastAPI is running and try again."
                )
                st.caption(str(exc))
