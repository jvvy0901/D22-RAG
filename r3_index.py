import openai

from rag import (CHUNK_SIZE, CHUNK_OVERLAP, INDEX_PATH, load_docs, split_docs,
                 count_tokens, embed_texts, save_index)

chunks = split_docs(load_docs(), CHUNK_SIZE, CHUNK_OVERLAP)
texts = [c["text"] for c in chunks]

print(f"전체 조각 수: {len(chunks)}개 (크기 {CHUNK_SIZE}자, 겹침 {CHUNK_OVERLAP}자)")
print(f"숫자로 바꿀 글자 수: {sum(len(t) for t in texts)}자")
print(f"대략 토큰 수: {count_tokens(texts)}토큰")

try:
    vectors = embed_texts(texts)
except openai.AuthenticationError:
    raise SystemExit("OPENAI_API_KEY 가 틀렸어요. .env 의 키를 다시 확인해 주세요.")
except openai.NotFoundError:
    raise SystemExit("OPENAI_EMBED_MODEL 이름이 틀렸어요. .env 를 다시 확인해 주세요.")
except openai.OpenAIError as e:
    raise SystemExit(f"OpenAI 호출에 실패했어요: {type(e).__name__}")

save_index(chunks, vectors)
print(f"벡터 길이: {len(vectors[0])}")
print(f"저장 완료: {INDEX_PATH.name}")
