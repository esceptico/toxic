from typing import Any, Dict, Iterator, List, Sequence, Tuple


def simplify_fasttext_label(label: str) -> str:
    """Simplifies fasttext-like label representation

    Args:
        label (str): Raw fasttext-like label (with `__label__` prefix).

    Examples:
        >>> simplify_fasttext_label('__label__NORMAL')
        'normal'

    Returns:
        str: Simplified label.
    """
    prefix = '__label__'
    if label.startswith(prefix):
        return label[len(prefix):].lower()
    return label


def iter_fasttext_data(path: str) -> Iterator[Tuple[str, List[str]]]:
    """Iterate over given fasttext-like dataset

    Args:
        path (str): Path to data.

    Yields:
        Tuple[text, labels]
    """
    with open(path, encoding='utf-8') as file:
        for line in file:
            sep = line.index(' ')
            text, labels = line[sep + 1:].strip(), line[:sep].split(",")
            labels = list(map(simplify_fasttext_label, labels))
            yield text, labels


def read_fasttext_data(
    path: str,
    label_set: Sequence[str]
) -> List[Dict[str, Any]]:
    data_list = []
    for text, labels in iter_fasttext_data(path):
        labels = [int(label in labels) for label in label_set]
        data_list.append({
            'text': text,
            'target': labels
        })
    return data_list
