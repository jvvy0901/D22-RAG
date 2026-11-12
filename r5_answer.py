import os
import sys

import openai

from rag import load_index, search, build_prompt, answer, check_source

if len(sys.argv) < 2:
    raise SystemExit('사용법: python3 r5_answer.py "질문"')
question = " ".join(sys.argv[1:])

if not os.getenv("OPENAI_MODEL"):
    raise SystemExit(".env 에 OPENAI_MODEL 이 없어요. 답변 모델 이름을 넣어 주세요.")

model, chunks = load_index()
if os.getenv("OPENAI_EMBED_MODEL") != model:
    raise SystemExit(f"index.json 은 다른 임베딩 모델({model})로 만들어졌어요. r3_index.py 를 다시 실행해 주세요.")

try:
    print(f"질문: {question}\n")

    print("━━ 1단계 · 검색 ━━")
    results = search(question, chunks, top_k=3)
    for rank, (score, c) in enumerate(results, 1):
        print(f"{rank}위 | 점수 {score:.3f} | {c['source']}")

    print("\n━━ 2단계 · 프롬프트 만들기 ━━")
    messages = build_prompt(question, results)
    for m in messages:
        print(f"[{m['role']}]\n{m['content']}\n")

    print(f"━━ 3단계 · 모델 부르기 ({os.environ['OPENAI_MODEL']}, temperature 0) ━━")
    reply = answer(messages)
except openai.AuthenticationError:
    raise SystemExit("OPENAI_API_KEY 가 틀렸어요. .env 의 키를 다시 확인해 주세요.")
except openai.NotFoundError:
    raise SystemExit("OPENAI_MODEL 이름이 틀렸어요. .env 를 다시 확인해 주세요.")
except openai.OpenAIError as e:
    raise SystemExit(f"OpenAI 호출에 실패했어요: {type(e).__name__}")

print("\n━━ 4단계 · 출처 검사 ━━")
reply, warning = check_source(reply, results)
print(f"⚠️ {warning}" if warning else "통과: 출처가 찾아온 조각의 파일 이름과 맞아요")

print("\n━━ 답 ━━")
print(reply)

print("\n━━ 찾아온 조각 ━━")
for rank, (score, c) in enumerate(results, 1):
    print(f"{rank}위 | 점수 {score:.3f} | {c['source']}")
