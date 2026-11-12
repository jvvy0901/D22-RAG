# r1_alone.py - 문서 없이 OpenAI API에 질문 3개를 그대로 묻는다.
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

here = Path(__file__).parent
load_dotenv(here / ".env") or load_dotenv(here / "docs" / ".env")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = os.environ["OPENAI_MODEL"]

QUESTIONS = [
    "클래스로 마켓 환불 규정은? 수업 3일 전에 취소하면 얼마나 돌려받아?",
    "클래스로 마켓에서 사진 리뷰를 쓰면 포인트를 얼마나 줘?",
    "클래스로 마켓 고객센터 전화번호 알려 줘.",
]

for i, q in enumerate(QUESTIONS, 1):
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": q}],  # 문서 없이 질문만
    )
    print(f"[질문 {i}] {q}")
    print(f"[답변 {i}] {res.choices[0].message.content}\n")
