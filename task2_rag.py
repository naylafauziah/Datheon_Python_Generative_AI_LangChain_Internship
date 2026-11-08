"""Task 2: Index and query PDFs/TXT with citations.

python task2_rag.py index
python task2_rag.py ask "Berapa bobot penilaian kualitas kode?"
"""
import argparse
from src.rag import ask, index_documents, NO_ANSWER


def main():
    parser = argparse.ArgumentParser(description='Document RAG assistant')
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('index', help='Rebuild Chroma index from data/docs')
    query = sub.add_parser('ask', help='Ask a question using indexed docs')
    query.add_argument('question', nargs='+', help='Your question')
    args = parser.parse_args()

    if args.action == 'index':
        pages, chunks = index_documents()
        print(f'Index successfully created: {pages} pages/TXT records, {chunks} chunks.')
    else:
        text, sources = ask(' '.join(args.question))
        print('\nANSWER:\n' + text)
        if sources and not text.startswith(NO_ANSWER):
            print('\nRETRIEVED REFERENCES:')
            for src in sources:
                print('  ' + src.citation)


if __name__ == '__main__':
    main()
