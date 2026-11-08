"""Task 4: Streamlit chat UI with RAG, token streaming and session memory.

streamlit run task4_app.py
"""
from pathlib import Path

import streamlit as st
from src.config import DOCS_DIR, CHROMA_DIR, require_key
from src.rag import (
    NO_ANSWER, chat_model, index_documents, make_messages,
    retrieve, validate_source_ids,
)
from src.utils import message_text

st.set_page_config(page_title='Datheon AI | Document Assistant', page_icon='📚', layout='centered')
st.title('📚 Datheon Document Q&A')
st.caption('Task 4 · RAG chatbot powered by Gemini, LangChain, ChromaDB, and Streamlit')

if 'messages' not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader('Knowledge base')
    st.write('Unggah PDF digital atau TXT. Semua dokumen diindeks ulang ketika tombol dipilih.')
    uploaded_files = st.file_uploader(
        'Tambah dokumen', type=['pdf', 'txt'], accept_multiple_files=True
    )
    if st.button('Index / Re-index documents', use_container_width=True):
        try:
            require_key('GOOGLE_API_KEY')
            DOCS_DIR.mkdir(parents=True, exist_ok=True)
            for uploaded in uploaded_files or []:
                name = Path(uploaded.name).name
                if Path(name).suffix.lower() not in ('.pdf', '.txt'):
                    raise ValueError(f'Unsupported extension: {name}')
                # Files remain local in data/docs; app is intended for local use only.
                (DOCS_DIR / name).write_bytes(uploaded.getvalue())
            with st.spinner('Loading, chunking, embedding, and indexing...'):
                document_count, chunks = index_documents()
            st.session_state.messages = []
            st.success(f'Indexed {document_count} document pages/TXT records; {chunks} chunks.')
        except Exception as exc:
            st.error(f'Indexing failed: {exc}')

    if st.button('Clear chat history', use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    files = sorted(p.name for p in DOCS_DIR.glob('*')
                   if p.is_file() and p.suffix.lower() in ('.pdf', '.txt'))
    st.write('**Files in data/docs:**')
    for name in files:
        st.caption('• ' + name)
    st.info('Minta hanya informasi yang ada dalam PDF/TXT. PDF hasil scan membutuhkan OCR.')

for entry in st.session_state.messages:
    with st.chat_message(entry['role']):
        st.markdown(entry['content'])
        if entry.get('references'):
            with st.expander('Retrieved references'):
                for source in entry['references']:
                    st.caption(source)

if question := st.chat_input('Contoh: Berapa bobot kualitas kode?'):
    prior = st.session_state.messages.copy()
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)

    with st.chat_message('assistant'):
        try:
            require_key('GOOGLE_API_KEY')
            # Resolve embeddings/retrieval before displaying token-level output.
            retrieved = retrieve(question)
            if not retrieved.sources:
                answer = NO_ANSWER
                st.markdown(answer)
                citations = []
            else:
                llm = chat_model()

                def token_stream():
                    for chunk in llm.stream(make_messages(question, retrieved, prior)):
                        text = message_text(chunk)
                        if text:
                            yield text

                # Streamlit renders each token/chunk as it arrives.
                answer = st.write_stream(token_stream())
                if not isinstance(answer, str):
                    answer = str(answer)
                if not validate_source_ids(answer, retrieved.sources):
                    st.warning(
                        'Jawaban tidak lolos pemeriksaan ID sitasi. '
                        'Jangan gunakan tanpa verifikasi; coba tanyakan lebih spesifik.'
                    )
                    answer = NO_ANSWER
                citations = [source.citation for source in retrieved.sources]
                if answer != NO_ANSWER:
                    with st.expander('Retrieved references'):
                        for source in retrieved.sources:
                            st.markdown(f'**{source.citation}**')
                            st.caption(source.snippet)
        except Exception as exc:
            answer = f'Error: {exc}'
            citations = []
            st.error(answer)
    st.session_state.messages.append({
        'role': 'assistant', 'content': answer,
        'references': citations if answer != NO_ANSWER else [],
    })
