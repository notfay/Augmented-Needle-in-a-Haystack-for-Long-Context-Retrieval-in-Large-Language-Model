import json
import pandas as pd
import time
from pathlib import Path
from google import genai
import re
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()


# === CONFIG ===
INPUT_PATH = Path("qwen3coder.json")
CSV_OUT = Path("qwen3coder.csv")
XLSX_OUT = Path("LLM_Review.xlsx")

# === Your LLM API call placeholder ===
def review_with_llm(needle: str, question: str, answer: str) -> str:
    """
    Replace this function with your actual API call.
    It receives Needle, Question, and Answer, and should return a review string.
    """
    # Example payload if you’re using a REST API:
    prompt = f"""
============================================================
STRICT FACT RETRIEVAL EVALUATOR
============================================================

You are an uncompromising evaluator for a retrieval experiment.

Your ONLY job:
    Determine whether the candidate Answer contains the SINGLE
    UNIQUE factual element from the Needle (a specific concept,
    key detail, unique phrasing, number, or entity).

This is a binary retrieval check — NOT reasoning, NOT opinion,
NOT topic similarity.

============================================================
RULES
============================================================

1. SPECIFIC FACT REQUIREMENT
   - The Answer must contain the unique factual detail from the Needle.
   - This may be an exact match OR a precise paraphrase that preserves
     the full meaning and structure (e.g., "per-run totals" ↔ "run-by-run totals").

2. NO GENERALIZATIONS
   - If the Answer speaks generally or vaguely without the specific detail,
     mark: No 0–2.

3. NO HALLUCINATIONS
   - If the Answer invents facts, numbers, names, or details not present
     in the Needle, mark: No 0–2.

4. FACT MUST APPEAR IN CORRECT RELATION
   - If the Answer contains the word(s) from the Needle only because it
     copied them from the question or restated the question, mark: No.
   - The fact must be used correctly and meaningfully as part of the answer.

5. "NOT FOUND BUT MENTIONS" RULE (REFINED)
   - If the Answer explicitly references the specific fact from the Needle,
     showing awareness of the fact even if stating uncertainty:
         → Mark: Yes 5
     Example: "I couldn't find the exact value, but I did find the study 
               about per-run totals and stability."
   - BUT if the Answer merely says "I didn't find the information" or copies
     words from the question without showing the fact:
         → Mark: No.

6. LONG PARAGRAPH RULE (REFINED)
   - If the Answer is long AND it clearly includes the specific fact
     in a correct factual relationship:
         → Mark: Yes 6
   - If the Answer is long but only contains the words from the question
     without actual retrieval:
         → Mark: No.

7. NO PARTIAL CREDIT FOR "VIBES"
   - If the Answer is "logically correct" or seems contextually related
     but does NOT include the unique fact:
         → Mark: No.

============================================================
SCORING
============================================================

YES (retrieval succeeded)
    - Yes 10 : Exact fact recovered
    - Yes 8–9 : Accurate paraphrase of the exact fact
    - Yes 6   : Long paragraph containing the fact in correct relation
    - Yes 5   : Fact mentioned while admitting uncertainty ("Not found but mentions")

NO (retrieval failed)
    - No 0–2 : Fact missing, vague, wrong, hallucinated, or restated question

============================================================
INPUT
============================================================

Needle: "{needle}"
Question: "{question}"
Candidate Answer: "{answer}"

============================================================
OUTPUT FORMAT
============================================================

Return ONLY one of the following formats:

    Yes 10
    Yes 9
    Yes 8
    Yes 6
    Yes 5
    No 2
    No 1
    No 0

No explanations. No extra text.
"""

    # --- Example structure (commented out) ---
    # response = requests.post("https://api.yourllm.com/v1/review", json=payload)
    # review_text = response.json().get("review", "")
    # return review_text
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )
    # For now, just simulate it:
    return response.text

# === Helper to parse LLM output ===
def parse_review(review_text: str):
    """
    Parse something like:
      'Yes 7', 'No 0', 'YES:10', 'no - 3', etc.
    into ('Yes', 7)
    """
    if not isinstance(review_text, str):
        return None, None

    # Try to find yes/no
    match_resp = re.search(r"\b(yes|no)\b", review_text, re.IGNORECASE)
    response = match_resp.group(1).capitalize() if match_resp else None

    # Try to find numeric score
    match_score = re.search(r"(-?\d+(?:\.\d+)?)", review_text)
    score = float(match_score.group(1)) if match_score else None

    return response, score

# === MAIN SCRIPT ===
def main():
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for idx, item in enumerate(data, start=1):
        cfg = item.get("config", {})
        needle = cfg.get("needle", {})
        question = cfg.get("question", {})

        pos = needle.get("custom_position_percent", None)
        haystack_sim = needle.get("haystack_similarity", None)
        q_sim = question.get("question_similarity", None)
        needle_text = needle.get("text", "")
        question_text = question.get("text", "")
        answer_text = item.get("response", "")
        latency = item.get("latency_seconds", item.get("latency", None))

        # === Send to your LLM review API ===
        try:
            review = review_with_llm(needle_text, question_text, answer_text)
        except Exception as e:
            review = f"ERROR: {e}"

        # Parse into response/score
        resp, score = parse_review(review)

        rows.append({
            "Needle Position": pos,
            "Needle-Haystack Similarity": haystack_sim,
            "Needle Question Similarity": q_sim,
            "Needle": needle_text,
            "Question": question_text,
            "Answer": answer_text,
            "Latency": latency,
            "Review": resp,
            "Score": score,
        })

        
        print(f"[{idx}/{len(data)}] → Review: {review} → waiting 6s...")
        time.sleep(2)

    # === Save results ===
    df = pd.DataFrame(rows, columns=[
        "Needle Position",
        "Needle-Haystack Similarity",
        "Needle Question Similarity",
        "Needle",
        "Question",
        "Answer",
        "Latency",
        "Review",
        "Score",
    ])

    df.to_csv(CSV_OUT, index=False, encoding="utf-8", quoting=1)
    df.to_excel(XLSX_OUT, index=False)

    print(f"\n✅ Done! Saved:\n- {CSV_OUT}\n- {XLSX_OUT}")


if __name__ == "__main__":
    main()
