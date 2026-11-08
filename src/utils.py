"""Small shared utilities without optional LangChain/vector dependencies."""


def message_text(message) -> str:
    """Extract displayable text from LangChain message string/content blocks."""
    content = getattr(message, 'content', message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(block.get('text', '') for block in content
                       if isinstance(block, dict) and block.get('type') == 'text')
    return str(content)
