from src.rag import Source, validate_source_ids, NO_ANSWER, split_documents
from langchain_core.documents import Document


def test_source_page_label():
    source = Source('S1', 'doc.pdf', 2, 'foo')
    assert source.citation == '[S1] doc.pdf (halaman 2)'


def test_citations():
    sources = [Source('S1', 'doc.pdf', 1, 'foo')]
    assert validate_source_ids('Berdasarkan dokumen, A [S1].', sources)
    assert validate_source_ids(NO_ANSWER, sources)
    assert not validate_source_ids('Berdasarkan dokumen, A [S2].', sources)
    assert not validate_source_ids('Jawabannya A.', sources)


def test_split_preserves_page_metadata():
    docs = split_documents([Document(page_content='Hello knowledge base',
                                     metadata={'source': 'test.pdf', 'page': 2})])
    assert docs[0].metadata['source'] == 'test.pdf'
    assert docs[0].metadata['page'] == 2
