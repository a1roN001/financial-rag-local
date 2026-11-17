"""In-memory BM25 index over document chunks (lexical branch)."""
import logging
import re
from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from config import DATABASE_URL, RECALL_K
from db import connect

logger = logging.getLogger(__name__)

# Компактные английские стоп-слова. Числа и тикеры сюда НЕ входят и
# вырезаются отдельным правилом ниже. Для 892 чанков хардкод честнее,
# чем тянуть nltk (скачивание corpora на винде = та самая боль).
_STOP = frozenset("""
a an the is are was were be been being am in on at to for of and or by with
as that this these those it its we our their you your he she him her they
them not no yes from during will would can may might must shall should has
have had do does did doing so such than then there here when where why how
all any both each few more most other some only own same too very just also
""".split())

# Токен = латиница/цифры, опционально с десятичным разделителем.
# Кириллица не матчится regexom -> отсекается естественно.
_TOKEN_RE = re.compile(r"\b[A-Za-z0-9]+(?:[.,][0-9]+)?\b")
_NUM_COMMA_RE = re.compile(r"^(\d+),(\d+)$")


def tokenize(text: str) -> List[str]:
    """Единая токенизация для документов И запроса.

    - lowercase;
    - извлечение латиницы/цифр (кириллица отсекается -> кросс-лингва
      лексической ветки сведена к общим tokenам: тикеры, числа, лат. термины);
    - нормализация '2,4' -> '2.4' (русский запрос vs английский документ);
    - удаление английских стоп-слов; числа/тикеры сохраняются всегда.
    """
    out: List[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        m = _NUM_COMMA_RE.match(raw)
        tok = f"{m.group(1)}.{m.group(2)}" if m else raw
        if tok in _STOP:
            continue
        out.append(tok)
    return out


class BM25Index:
    def __init__(self, chunk_ids: List[str], corpus: List[List[str]]):
        self.chunk_ids = chunk_ids
        self.bm25 = BM25Okapi(corpus)

    def rank(self, query: str, top_k: int) -> List[Tuple[str, int]]:
        """Возвращает [(chunk_id, rank)] по убыванию BM25-скора, rank 1 = лучший."""
        q_tokens = tokenize(query)
        if not q_tokens:
            return []  # чисто кириллический запрос -> лексическая ветка пуста
        scores = self.bm25.get_scores(q_tokens)
        order = np.argsort(-scores)  # индексы в порядке убывания скора
        ranked: List[Tuple[str, int]] = []
        for r, idx in enumerate(order[:top_k], start=1):
            if scores[idx] <= 0:
                break  # хвост нулевых совпадений не несёт сигнала в RRF
            ranked.append((self.chunk_ids[idx], r))
        return ranked


_INDEX: BM25Index | None = None


def _load_from_db() -> BM25Index:
    with connect(DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute("SELECT chunk_id, text FROM document_chunks ORDER BY chunk_id")
        rows = cur.fetchall()
    chunk_ids = [r[0] for r in rows]
    corpus = [tokenize(r[1]) for r in rows]
    logger.info("BM25 index built over %d chunks", len(chunk_ids))
    return BM25Index(chunk_ids, corpus)


def get_bm25_index() -> BM25Index:
    global _INDEX
    if _INDEX is None:
        _INDEX = _load_from_db()
    return _INDEX


def rebuild_bm25_index() -> None:
    """Вызывать после переиндексации чанков (после embeddings.py)."""
    global _INDEX
    _INDEX = _load_from_db()