"""Streamlit interface for the local financial RAG pipeline."""

import os
import sys
from pathlib import Path
from typing import Any, Dict

import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
DOTENV_FILE = PROJECT_ROOT / "src" / ".env"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(DOTENV_FILE)


@st.cache_resource(show_spinner="Загрузка embedding-модели...")
def load_rag_module() -> Any:
    """Import the RAG module once per Streamlit process."""
    from rag import answer_question

    return answer_question


def render_sources(sources: list[Dict[str, Any]]) -> None:
    """Render retrieved chunks used to build the answer."""
    if not sources:
        return

    with st.expander(f"Источники ({len(sources)})"):
        for source in sources:
            similarity = float(source.get("similarity", 0))
            document = source.get("document_id", "Неизвестный документ")
            page = source.get("page_number", "?")
            st.markdown(f"**{document}**, стр. {page} | similarity: `{similarity:.3f}`")
            st.caption(source.get("text", ""))


def render_message(message: Dict[str, Any]) -> None:
    """Render one stored chat message."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))


def main() -> None:
    st.set_page_config(
        page_title="Financial RAG",
        page_icon="📊",
        layout="centered",
    )
    st.title("Financial RAG")
    st.caption("Вопросы к финансовым отчетам через локальную LLM")

    with st.sidebar:
        st.header("Настройки")
        model_name = st.selectbox(
            "Модель Ollama",
            options=["qwen2.5:7b", "llama3.1:8b", "mistral"],
            index=0,
        )
        top_k = st.slider("Количество фрагментов", min_value=1, max_value=10, value=5)
        similarity_threshold = st.slider(
            "Порог similarity",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
            help="Если лучший фрагмент ниже порога, приложение не передает его в LLM.",
        )
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            st.success("Подключение к базе настроено")
        else:
            st.error("Не задан DATABASE_URL")
        if st.button("Очистить историю", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        render_message(message)

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Ваш вопрос",
            placeholder="Например: Какова выручка компании за последний год?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Отправить", use_container_width=True)

    if not submitted or not question.strip():
        return

    if not database_url:
        st.error("Добавьте DATABASE_URL в .env перед отправкой вопроса.")
        return

    question = question.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Ищу ответ в документах..."):
            try:
                answer_question = load_rag_module()
                result = answer_question(
                    question=question,
                    database_url=database_url,
                    model_name=model_name,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                )
                answer = result.get("answer", "Ответ не найден.")
                sources = result.get("sources", [])
            except Exception as error:
                st.error(f"Не удалось получить ответ: {error}")
                return

        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


if __name__ == "__main__":
    main()
