import os
import unicodedata

import openai
from flask import Flask, render_template, request

from rag import load_index, search, build_prompt, answer, check_source

MAX_LEN = 300

app = Flask(__name__)
app.json.ensure_ascii = False  # JSON 으로 내보낼 때도 한글이 \uXXXX 로 깨지지 않게

# index.json 은 앱이 켜질 때 딱 한 번만 읽는다
INDEX_MODEL, CHUNKS = load_index()
if os.getenv("OPENAI_EMBED_MODEL") != INDEX_MODEL:
    raise SystemExit(f"index.json 은 다른 임베딩 모델({INDEX_MODEL})로 만들어졌어요. r3_index.py 를 다시 실행해 주세요.")


def nfc(text):
    """macOS 의 낱자 한글 파일 이름을 완성형으로 바꿔 화면에 보여 준다."""
    return unicodedata.normalize("NFC", text)


def render(question="", error=None, result=None):
    return render_template("ask.html", question=question, error=error,
                           result=result, max_len=MAX_LEN)


@app.route("/", methods=["GET", "POST"])
def ask():
    if request.method == "GET":
        return render()

    question = request.form.get("question", "").strip()

    if not question:
        return render(error="질문을 입력해 주세요.")
    if len(question) > MAX_LEN:
        return render(question=question,
                      error=f"질문은 {MAX_LEN}자 이하로 써 주세요. 지금은 {len(question)}자예요. 조금 줄여서 다시 물어보세요.")

    try:
        found = search(question, CHUNKS, top_k=3)
        reply = answer(build_prompt(question, found))
        reply, warning = check_source(reply, found)
    except SystemExit as e:  # rag.py 가 키 없음 등을 한국어 한 줄로 알려 주는 경우
        return render(question=question, error=str(e))
    except openai.AuthenticationError:
        return render(question=question, error="OPENAI_API_KEY 가 틀렸어요. .env 의 키를 다시 확인해 주세요.")
    except openai.NotFoundError:
        return render(question=question, error="OPENAI_MODEL 이름이 틀렸어요. .env 를 다시 확인해 주세요.")
    except openai.OpenAIError as e:
        return render(question=question, error=f"OpenAI 호출에 실패했어요: {type(e).__name__}")

    result = {
        "reply": reply,
        "warning": warning,
        "chunks": [
            {"rank": i, "score": f"{score:.3f}", "source": nfc(c["source"]),
             "head": c["text"][:150].replace("\n", " ")}
            for i, (score, c) in enumerate(found, 1)
        ],
    }
    return render(question=question, result=result)


if __name__ == "__main__":
    # 5000 번은 최신 macOS 의 AirPlay 가 쓰는 경우가 있어 5001 로 한다
    app.run(host="127.0.0.1", port=5001, debug=False)
