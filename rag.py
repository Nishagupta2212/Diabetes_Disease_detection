"""Retrieval for the assistant: BM25 keyword search over the Markdown files in knowledge/.

Each knowledge file starts with "# Title" and a "Source: name | url" line, followed by
"## Section" blocks. Every section becomes one searchable passage.
"""
import math
import re
from collections import Counter
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"

STOPWORDS = set(
    "a an and are as at be been but by can do does for from has have how i if in into is it its "
    "me my of on or so than that the their them there these this to was what when which who why "
    "will with would you your should could about am".split()
)


def stem(word):
    for suffix in ("ing", "ies", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def tokenize(text):
    return [stem(w) for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if w not in STOPWORDS]


def load_passages(directory=KNOWLEDGE_DIR):
    passages = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = re.search(r"^# (.+)$", text, re.M).group(1).strip()
        source_line = re.search(r"^Source: (.+)$", text, re.M).group(1)
        source, _, url = (part.strip() for part in source_line.partition("|"))
        for section in re.split(r"^## ", text, flags=re.M)[1:]:
            heading, _, body = section.partition("\n")
            passages.append({
                "title": f"{title}: {heading.strip()}",
                "text": " ".join(body.split()),
                "source": source,
                "url": url,
            })
    return passages


class BM25Index:
    def __init__(self, passages, k1=1.5, b=0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        # The heading is counted twice so a matching section title ranks higher.
        self.docs = [Counter(tokenize((p["title"] + " ") * 2 + p["text"])) for p in passages]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = sum(self.lengths) / len(self.lengths)
        doc_freq = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def search(self, query, k=3):
        terms = set(tokenize(query))
        scored = []
        for passage, doc, length in zip(self.passages, self.docs, self.lengths):
            score = 0.0
            for term in terms & doc.keys():
                tf = doc[term]
                score += self.idf[term] * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * length / self.avg_length)
                )
            if score > 0:
                scored.append((score, passage))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [dict(p, score=round(s, 2)) for s, p in scored[:k]]


index = BM25Index(load_passages())
