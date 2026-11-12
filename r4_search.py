import os
import sys

import openai

from rag import load_index, search

if len(sys.argv) < 2:
    raise SystemExit('사용법: python3 r4_search.py "질문"')
question = " ".join(sys.argv[1:])

model, chunks = load_index()
if os.getenv("OPENAI_EMBED_MODEL") != model:
    raise SystemExit(f"index.json 은 다른 임베딩 모델({model})로 만들어졌어요. r3_index.py 를 다시 실행해 주세요.")

try:
    results = search(question, chunks, top_k=3)
except openai.AuthenticationError:
    raise SystemExit("OPENAI_API_KEY 가 틀렸어요. .env 의 키를 다시 확인해 주세요.")
except openai.OpenAIError as e:
    raise SystemExit(f"OpenAI 호출에 실패했어요: {type(e).__name__}")

print(f"질문: {question}\n")
for rank, (score, c) in enumerate(results, 1):
    head = c["text"][:150].replace("\n", " ")
    print(f"{rank}위 | 점수 {score:.3f} | {c['source']}")
    print(f"    {head}\n")
