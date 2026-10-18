"""
Layer 2 — Analyzer (Gemini & Groq edition)
Sends each article to Gemini (gemini-3.8-flash) or Groq (llama-3.3-70b) for bias/sentiment/framing analysis.
Outputs: analyzed.json  (articles.json fields + analysis fields merged)

Setup:
    Get a free Gemini API key: https://aistudio.google.com/app/apikey
    $env:GEMINI_API_KEY = "AIzaSy..."

    Or use Groq:
    $env:GROQ_API_KEY = "gsk_..."

Usage:
    python analyzer.py                  # analyze unanalyzed articles (up to default limit)
    python analyzer.py --limit 50       # analyze up to 50 articles
    python analyzer.py --all            # analyze all without limit
    python analyzer.py --reanalyze      # force re-analyze everything
"""

import json
import os
import time
import argparse
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
ARTICLES_FILE  = "articles.json"
ANALYZED_FILE  = "analyzed.json"
MAX_TEXT_CHARS = 3000
GEMINI_MODEL   = "gemini-3.8-flash"
GROQ_MODEL     = "llama-3.3-70b-versatile"
RETRY_LIMIT    = 3
DEFAULT_LIMIT  = 30

SYSTEM_PROMPT = """You are a professional media-bias analyst.
Your job is to read a news article and return a structured JSON object — nothing else.
No markdown fences, no explanation, no commentary outside the JSON.

The JSON must strictly follow this schema:
{
  "bias_label":      string,   // one of: "Left", "Left-Center", "Center", "Right-Center", "Right", "Unknown"
  "sentiment_score": float,    // -1.0 (very negative) to +1.0 (very positive)
  "framing":         string,   // one of: "Alarmist", "Neutral", "Optimistic", "Analytical", "Sensationalist", "Sympathetic", "Critical"
  "key_entities": [            // up to 5 most prominent people / orgs / policies
    {
      "name":      string,
      "portrayal": string      // one of: "Positive", "Neutral", "Negative"
    }
  ],
  "reasoning": string          // <= 60 words explaining your labels
}"""

USER_TEMPLATE = """Source outlet: {source}
Article title: {title}

Article text (may be truncated):
{text}

Return only the JSON object."""

ANALYSIS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "bias_label": {
            "type": "string",
            "enum": ["Left", "Left-Center", "Center", "Right-Center", "Right", "Unknown"]
        },
        "sentiment_score": {
            "type": "number",
            "description": "-1.0 (very negative) to +1.0 (very positive)"
        },
        "framing": {
            "type": "string",
            "enum": ["Alarmist", "Neutral", "Optimistic", "Analytical", "Sensationalist", "Sympathetic", "Critical"]
        },
        "key_entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "portrayal": {
                        "type": "string",
                        "enum": ["Positive", "Neutral", "Negative"]
                    }
                },
                "required": ["name", "portrayal"]
            }
        },
        "reasoning": {
            "type": "string",
            "description": "<= 60 words explaining your labels"
        }
    },
    "required": ["bias_label", "sentiment_score", "framing", "key_entities", "reasoning"]
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: str) -> list:
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def truncate(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    if not text:
        return ""
    return text[:max_chars] + ("…" if len(text) > max_chars else "")

def parse_analysis(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"    ⚠  JSON parse error: {e}", flush=True)
        return None

# ── Core analysis calls ───────────────────────────────────────────────────────

def analyze_article_gemini(client, article: dict) -> dict | None:
    from google.genai import types

    prompt = USER_TEMPLATE.format(
        source=article.get("source", "Unknown"),
        title=article.get("title", ""),
        text=truncate(article.get("text", "")),
    )

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                    max_output_tokens=512,
                    response_mime_type="application/json",
                    response_json_schema=ANALYSIS_JSON_SCHEMA,
                ),
            )
            raw = response.text or ""
            result = parse_analysis(raw)
            if result:
                return result
            print(f"    ✗ Bad JSON on attempt {attempt}, retrying…", flush=True)
        except Exception as e:
            err_str = str(e)
            if "API_KEY_INVALID" in err_str or "unregistered" in err_str.lower():
                raise SystemExit(
                    "\n✗ Authentication failed — GEMINI_API_KEY is invalid.\n"
                    "  Get your free key at: https://aistudio.google.com/app/apikey\n"
                    "  Then set: $env:GEMINI_API_KEY = 'AIzaSy...'"
                )
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                wait = 25 * attempt
                print(f"    ⏳ Rate limited — cooling down for {wait}s (attempt {attempt})...", flush=True)
                time.sleep(wait)
            else:
                print(f"    ✗ Gemini API error: {e} (attempt {attempt})", flush=True)
                time.sleep(5)

    return None


def analyze_article_groq(client, article: dict) -> dict | None:
    from groq import AuthenticationError, RateLimitError, APIError

    prompt = USER_TEMPLATE.format(
        source=article.get("source", "Unknown"),
        title=article.get("title", ""),
        text=truncate(article.get("text", "")),
    )

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=512,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            result = parse_analysis(raw)
            if result:
                return result
            print(f"    ✗ Bad JSON on attempt {attempt}, retrying…", flush=True)
        except AuthenticationError:
            raise SystemExit(
                "\n✗ Authentication failed — GROQ_API_KEY is invalid.\n"
                "  Get your free key at: https://console.groq.com\n"
                "  Then run: $env:GROQ_API_KEY = 'gsk_...'"
            )
        except RateLimitError:
            wait = 10 * attempt
            print(f"    ⏳ Rate limited — waiting {wait}s (attempt {attempt})", flush=True)
            time.sleep(wait)
        except APIError as e:
            print(f"    ✗ API error: {e} (attempt {attempt})", flush=True)
            time.sleep(10)

    return None

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reanalyze", action="store_true",
                        help="Re-analyze all articles even if already cached")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"Max articles to analyze in one run (default: {DEFAULT_LIMIT})")
    parser.add_argument("--all", action="store_true",
                        help="Analyze all pending articles without limit")
    args = parser.parse_args()

    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")

    if gemini_key:
        print(f"🤖 Using Google Gemini API ({GEMINI_MODEL})", flush=True)
        from google import genai
        client = genai.Client(api_key=gemini_key)
        analyze_fn = lambda a: analyze_article_gemini(client, a)
    elif groq_key:
        print(f"🤖 Using Groq API ({GROQ_MODEL})", flush=True)
        from groq import Groq
        client = Groq(api_key=groq_key)
        analyze_fn = lambda a: analyze_article_groq(client, a)
    else:
        raise SystemExit(
            "✗ No API key found.\n"
            "  Get a free Gemini API key at: https://aistudio.google.com/app/apikey\n"
            "  Then run: $env:GEMINI_API_KEY = 'AIzaSy...'\n\n"
            "  Or set GROQ_API_KEY: $env:GROQ_API_KEY = 'gsk_...'"
        )

    articles = load_json(ARTICLES_FILE)
    if not articles:
        print(f"✗ No articles found in {ARTICLES_FILE}. Run scraper.py first.", flush=True)
        return

    analyzed  = load_json(ANALYZED_FILE)
    done_urls: set[str] = set()
    if not args.reanalyze:
        done_urls = {a["url"] for a in analyzed if "bias_label" in a}

    to_process = [a for a in articles if a["url"] not in done_urls]
    print(f"📰 Articles total: {len(articles)}  |  "
          f"Already analyzed: {len(done_urls)}  |  "
          f"Pending analysis: {len(to_process)}", flush=True)

    if not to_process:
        print("✅ Nothing new to analyze.", flush=True)
        return

    if not args.all and len(to_process) > args.limit:
        print(f"⚡ Processing {args.limit} newest articles in this batch (rate-limit friendly).", flush=True)
        to_process = to_process[:args.limit]

    result_map: dict[str, dict] = {a["url"]: a for a in analyzed}

    for i, article in enumerate(to_process, 1):
        title_preview = article.get("title", "")[:60]
        print(f"[{i}/{len(to_process)}] {article.get('source','?')} — {title_preview}…", flush=True)

        analysis = analyze_fn(article)
        if analysis:
            merged = {**article, **analysis}
            result_map[article["url"]] = merged
            score = analysis.get("sentiment_score", 0)
            print(f"    ✓ {analysis.get('bias_label')} | "
                  f"sentiment={score:+.2f} | "
                  f"framing={analysis.get('framing')}", flush=True)
        else:
            result_map[article["url"]] = {**article, "analysis_error": True}
            print("    ✗ Analysis failed — stored with error flag", flush=True)

        save_json(ANALYZED_FILE, list(result_map.values()))

        # Sleep between requests to strictly respect the 15 RPM free tier limit
        sleep_sec = 4.2 if gemini_key else 0.5
        time.sleep(sleep_sec)

    final = load_json(ANALYZED_FILE)
    ok    = sum(1 for a in final if "bias_label" in a)
    err   = sum(1 for a in final if a.get("analysis_error"))
    print(f"\n✅ Done. {ok} analyzed, {err} errors → {ANALYZED_FILE}", flush=True)


if __name__ == "__main__":
    main()
