from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path(__file__).parent / "docs"


def load_docs(docs_dir=DOCS_DIR):
    """docs 폴더의 .md 파일을 모두 읽어 [{source, text}] 로 돌려준다."""
    return [
        {"source": p.name, "text": p.read_text(encoding="utf-8")}
        for p in sorted(Path(docs_dir).glob("*.md"))
    ]


def split_docs(docs, chunk_size, chunk_overlap=None):
    """문서를 조각으로 자른다. 조각마다 출처(파일 이름)를 붙인다."""
    if chunk_overlap is None:
        chunk_overlap = chunk_size // 10
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return [
        {"source": d["source"], "text": t}
        for d in docs
        for t in splitter.split_text(d["text"])
    ]


import json
import os
import unicodedata

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

CHUNK_SIZE = 400
CHUNK_OVERLAP = 40
INDEX_PATH = Path(__file__).parent / "index.json"


def count_tokens(texts):
    """대략 토큰 수 (OpenAI 표준 토큰 규칙으로 센 값)."""
    enc = tiktoken.get_encoding("cl100k_base")
    return sum(len(enc.encode(t)) for t in texts)


def embed_texts(texts, batch_size=100):
    """글자를 숫자(벡터)로 바꾼다. 키·모델 이름이 없으면 한국어 한 줄로 알리고 끝낸다."""
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(".env 에 OPENAI_API_KEY 가 없어요. 키를 넣고 다시 실행해 주세요.")
    model = os.getenv("OPENAI_EMBED_MODEL")
    if not model:
        raise SystemExit(".env 에 OPENAI_EMBED_MODEL 이 없어요. 임베딩 모델 이름을 넣어 주세요.")
    client = OpenAI()
    vectors = []
    for i in range(0, len(texts), batch_size):
        res = client.embeddings.create(model=model, input=texts[i:i + batch_size])
        vectors += [d.embedding for d in res.data]
    return vectors


def save_index(chunks, vectors, path=INDEX_PATH):
    """조각 + 출처 + 벡터를 index.json 으로 저장한다."""
    items = [{**c, "embedding": v} for c, v in zip(chunks, vectors)]
    data = {"model": os.environ["OPENAI_EMBED_MODEL"], "chunks": items}
    Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def load_index(path=INDEX_PATH):
    """index.json 을 읽어 (모델 이름, 조각 목록) 으로 돌려준다."""
    if not Path(path).exists():
        raise SystemExit("index.json 이 없어요. 먼저 python3 r3_index.py 를 실행해 주세요.")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["model"], data["chunks"]


def cosine(a, b):
    """코사인 유사도: 1 에 가까울수록 비슷하고, 0 에 가까울수록 관련이 없다."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)


def search(question, chunks, top_k=3):
    """질문과 가장 비슷한 조각 top_k 개를 [(점수, 조각)] 로 돌려준다. 점수 큰 순서."""
    q_vec = embed_texts([question])[0]
    scored = [(cosine(q_vec, c["embedding"]), c) for c in chunks]
    return sorted(scored, key=lambda s: s[0], reverse=True)[:top_k]


SYSTEM_PROMPT = """당신은 '클래스로 마켓' 문서 안내 도우미입니다. 아래 규칙을 반드시 지키세요.

1) 아래 [자료] 안에 있는 내용만으로 답하세요. 원래 알던 지식은 쓰지 마세요.
2) 자료에 답이 없으면 다른 말 없이 이 문장만 쓰세요: 문서에서 찾지 못했어요
3) 숫자, 기간, 조건은 자료에 적힌 그대로 쓰세요. 바꾸거나 반올림하거나 추측하지 마세요.
4) 답의 마지막 줄에 출처를 쓰세요. 형식: 출처: 파일이름"""


def build_prompt(question, results):
    """검색 결과 [(점수, 조각)] 로 모델에 보낼 messages 를 만든다."""
    material = "\n\n".join(f"[출처: {c['source']}]\n{c['text']}" for _, c in results)
    user = f"[자료]\n{material}\n\n[질문]\n{question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def answer(messages):
    """모델을 불러 답을 받는다. temperature 0 으로 매번 같은 답이 나오게 한다."""
    res = OpenAI().chat.completions.create(
        model=os.environ["OPENAI_MODEL"],
        messages=messages,
        temperature=0,
    )
    return res.choices[0].message.content


NOT_FOUND = "문서에서 찾지 못했어요"


def _nfc(name):
    """macOS 는 한글 파일 이름을 낱자(NFD)로 저장한다. 비교 전에 완성형(NFC)으로 맞춘다."""
    return unicodedata.normalize("NFC", name)


def check_source(reply, results):
    """답 마지막 줄의 출처가 찾아온 조각의 파일 이름인지 검사한다.
    돌려주는 값: (최종 답, 경고 문구 또는 None)"""
    text = reply.strip()
    if text == NOT_FOUND:  # 못 찾았다는 답에는 출처 줄이 없는 게 맞다
        return text, None

    allowed = [c["source"] for _, c in results]
    allowed_nfc = [_nfc(a) for a in allowed]
    lines = text.splitlines()
    last = lines[-1].strip() if lines else ""

    if not last.startswith("출처:"):
        problem = "답 마지막 줄에 '출처:' 가 없어요"
        body = lines
    else:
        names = [n.strip() for n in last[len("출처:"):].split(",") if n.strip()]
        bad = [n for n in names if _nfc(n) not in allowed_nfc]
        if names and not bad:
            return text, None
        problem = f"찾아오지 않은 파일 이름이에요: {', '.join(bad) or '(비어 있음)'}"
        body = lines[:-1]

    fixed = "\n".join(body).rstrip() + f"\n출처: {allowed[0]}"
    return fixed, f"{problem}. 가장 점수가 높은 조각의 파일({allowed[0]})로 코드가 바꿨어요."
