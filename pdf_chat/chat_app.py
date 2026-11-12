import io
import secrets
import unicodedata
from collections import OrderedDict

import openai
from flask import Flask, abort, redirect, render_template, request, url_for
from pypdf.errors import PyPdfError

from rag_chain import build_store, make_chain, pdf_to_docs

MAX_LEN = 300          # 질문 최대 글자 수
MAX_PAGES = 100        # 업로드 PDF 최대 쪽수
MAX_DOCS = 20          # 메모리에 두는 문서 수 (넘으면 가장 오래된 것부터 지움)

app = Flask(__name__)
app.json.ensure_ascii = False
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB

DOCS = OrderedDict()   # {문서 id: {"name", "pages", "chunks", "chain"}}


def nfc(text):
    return unicodedata.normalize("NFC", text)


def show(**kw):
    return render_template("chat.html", max_len=MAX_LEN, **kw)


@app.errorhandler(413)
def too_big(_):
    return show(error="파일이 너무 커요. 20MB 이하 PDF 를 올려 주세요."), 413


@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return show()

    f = request.files.get("pdf")
    if not f or not f.filename:
        return show(error="PDF 파일을 골라 주세요.")
    data = f.read()
    if not f.filename.lower().endswith(".pdf") or not data.startswith(b"%PDF"):
        return show(error="PDF 파일만 올릴 수 있어요.")

    name = nfc(f.filename)
    try:
        docs = pdf_to_docs(io.BytesIO(data), name)
        if not docs:
            return show(error="글자를 읽을 수 없는 PDF예요. 스캔한 이미지 PDF는 지원하지 않아요.")
        if len(docs) > MAX_PAGES:
            return show(error=f"{MAX_PAGES}쪽 이하 PDF 만 올릴 수 있어요. 지금은 {len(docs)}쪽이에요.")
        store, n_chunks = build_store(docs)
        chain = make_chain(store)
    except PyPdfError:
        return show(error="PDF 를 읽지 못했어요. 암호가 걸렸거나 깨진 파일일 수 있어요.")
    except SystemExit as e:
        return show(error=str(e))
    except openai.AuthenticationError:
        return show(error="OPENAI_API_KEY 가 틀렸어요. .env 의 키를 다시 확인해 주세요.")
    except openai.NotFoundError:
        return show(error="OPENAI_MODEL 또는 OPENAI_EMBED_MODEL 이름이 틀렸어요. .env 를 다시 확인해 주세요.")
    except openai.OpenAIError as e:
        return show(error=f"OpenAI 호출에 실패했어요: {type(e).__name__}")

    doc_id = secrets.token_urlsafe(8)
    DOCS[doc_id] = {"name": name, "pages": len(docs), "chunks": n_chunks, "chain": chain}
    while len(DOCS) > MAX_DOCS:
        DOCS.popitem(last=False)
    return redirect(url_for("chat", doc_id=doc_id))


@app.route("/chat/<doc_id>", methods=["GET", "POST"])
def chat(doc_id):
    doc = DOCS.get(doc_id)
    if not doc:
        return show(error="올린 문서를 찾을 수 없어요. 서버가 다시 켜졌거나 오래된 문서일 수 있어요. PDF 를 다시 올려 주세요.")
    info = {"name": doc["name"], "pages": doc["pages"], "chunks": doc["chunks"]}
    if request.method == "GET":
        return show(doc=info)

    question = request.form.get("question", "").strip()
    if not question:
        return show(doc=info, error="질문을 입력해 주세요.")
    if len(question) > MAX_LEN:
        return show(doc=info, question=question,
                    error=f"질문은 {MAX_LEN}자 이하로 써 주세요. 지금은 {len(question)}자예요. 조금 줄여서 다시 물어보세요.")

    try:
        out = doc["chain"].invoke(question)
    except SystemExit as e:
        return show(doc=info, question=question, error=str(e))
    except openai.AuthenticationError:
        return show(doc=info, question=question, error="OPENAI_API_KEY 가 틀렸어요. .env 의 키를 다시 확인해 주세요.")
    except openai.OpenAIError as e:
        return show(doc=info, question=question, error=f"OpenAI 호출에 실패했어요: {type(e).__name__}")

    result = {
        "answer": out["answer"],
        "hits": [{"rank": i, "score": f"{score:.3f}", "source": d.metadata["source"],
                  "page": d.metadata["page"], "head": d.page_content[:150].replace("\n", " ")}
                 for i, (d, score) in enumerate(out["hits"], 1)],
    }
    return show(doc=info, question=question, result=result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=False)
