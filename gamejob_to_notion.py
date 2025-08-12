# file: gamejob_to_notion.py
# pip install requests beautifulsoup4 python-dotenv tenacity
#
# 예시:
#   (드라이런, 상세수정일 사용)
#   python gamejob_to_notion.py --dry-run --pages 1 --limit 10 --detail-mode always --detail-delay 0.5
#   (전체 업서트)
#   python gamejob_to_notion.py --pages 0 --delay 1.2 --detail-mode always --detail-delay 0.5 --notion-delay 0.3

import os, re, time, json, argparse
import datetime as dt
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# ---------------- Notion 설정 ----------------
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DB_ID = os.getenv("NOTION_DATABASE_ID")
NOTION_VERSION = "2022-06-28"
HEADERS_NOTION = {
    "Authorization": f"Bearer {NOTION_TOKEN or ''}",
    "Notion-Version": NOTION_VERSION,
    "Content-Type": "application/json",
}
TITLE_PROP_NAME = None  # DB의 title 속성명을 런타임에 자동 감지

# ---------------- 사이트/크롤러 기본값 ----------------
BASE = "https://www.gamejob.co.kr"
DEFAULT_LIST_URL = "https://www.gamejob.co.kr/Recruit/joblist?menucode=duty&duty=1"
AJAX_URL_TMPL = "/recruit/_GI_Job_List?Page={p}"
PAGE_PARAM_CANDIDATES = ["Page", "page", "thisPage", "pageIndex"]

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) GamejobCrawler/1.6"
})

# ---------------- 유틸 ----------------
def normalize_space(s): return re.sub(r"\s+", " ", s or "").strip()

def parse_posted(text):
    """한국어 상대표현·요일 괄호 포함 케이스까지 처리."""
    if not text: return None
    t = text.strip()
    t = re.sub(r"\([^)]*\)", "", t)  # 요일괄호 제거 예: 07/30(수)
    # N일 전
    m = re.search(r"(\d+)\s*일\s*전", t)
    if m:
        days = int(m.group(1))
        return (dt.date.today() - dt.timedelta(days=days)).isoformat()
    # YYYY[.-/년 ]MM[.-/월 ]DD
    m = re.search(r"(\d{4})[.\-/년\s]*(\d{1,2})[.\-/월\s]*(\d{1,2})", t)
    if m:
        y, mo, d = map(int, m.groups())
        try: return dt.date(y, mo, d).isoformat()
        except: return None
    # MM[.-/]DD
    m = re.search(r"(\d{1,2})[.\-/](\d{1,2})", t)
    if m:
        mo, d = map(int, m.groups())
        try: return dt.date(dt.date.today().year, mo, d).isoformat()
        except: return None
    if "오늘" in t: return dt.date.today().isoformat()
    if "어제" in t: return (dt.date.today() - dt.timedelta(days=1)).isoformat()
    return None

def merge_query(base_url, target_url):
    """list_url의 필터 쿼리(duty, menucode 등)를 페이지/조각 요청에도 유지."""
    bp, tp = urlparse(base_url), urlparse(target_url)
    bq, tq = parse_qs(bp.query), parse_qs(tp.query)
    bq.pop("Page", None)
    for k, v in bq.items():
        if k not in tq: tq[k] = v
    flat = {k: (v[0] if isinstance(v, list) else v) for k, v in tq.items()}
    return urlunparse((tp.scheme or bp.scheme, tp.netloc or bp.netloc, tp.path, "", urlencode(flat), ""))

# ---------------- HTTP ----------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_raw(url, headers=None):
    r = SESSION.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r

def fetch_list_html(list_url, page):
    if page == 1:
        return fetch_raw(list_url).text
    # AJAX 우선
    ajax_url = urljoin(BASE, AJAX_URL_TMPL.format(p=page))
    ajax_headers = {"Referer": list_url, "X-Requested-With": "XMLHttpRequest", "User-Agent": SESSION.headers["User-Agent"]}
    try:
        return fetch_raw(ajax_url, headers=ajax_headers).text
    except Exception:
        # 일반 쿼리 폴백
        parsed = urlparse(list_url); qs = parse_qs(urlparse(list_url).query)
        replaced = False
        for key in PAGE_PARAM_CANDIDATES:
            if key in qs:
                qs[key] = [str(page)]; replaced = True; break
        if not replaced: qs[PAGE_PARAM_CANDIDATES[0]] = [str(page)]
        url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{urlencode({k:v[0] for k,v in qs.items()})}"
        return fetch_raw(url).text

# ---------------- 페이지네이션 링크 ----------------
def find_pagination_url(base_url, html, target_page):
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.select(".pagination a[href]"):
        if (a.get_text() or "").strip() == str(target_page):
            return urljoin(base_url, a["href"])
    for a in soup.select(".pagination a[href]"):
        for attr, val in a.attrs.items():
            if attr.startswith("data") and str(val).strip() == str(target_page):
                return urljoin(base_url, a["href"])
    return None

# ---------------- 리스트 파싱 ----------------
def extract_jobs(html, base=BASE):
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select('#dev-gi-list table.tblList tbody tr')
    if not rows: rows = soup.select('table.tblList tbody tr')
    if not rows: rows = [tr for tr in soup.select('tr') if tr.select_one('div.tit a')]
    jobs = []
    for tr in rows:
        a = tr.select_one('div.tit a[href]'); company_el = tr.select_one('div.company strong')
        if not a or not company_el: continue
        title = normalize_space(a.get_text())
        url   = urljoin(base, a.get('href',''))
        company = normalize_space(company_el.get_text())
        info_spans = [normalize_space(s.get_text()) for s in tr.select('div.tit p.info span')]
        experience = next((t for t in info_spans if any(k in t for k in ['경력','신입','무관','연차','이상'])), '')
        location   = next((t for t in info_spans if any(k in t for k in ['서울','경기','인천','부산','대구','대전','광주','세종','울산','강원','충북','충남','전북','전남','경북','경남','제주','해외','재택','시','도','구','군'])), '')
        # 리스트상 날짜(등록/수정) - 상세에서 재정의 예정
        date_el = tr.select_one('td:nth-of-type(3) .date')
        mod_el  = tr.select_one('td:nth-of-type(3) .modifyDate')
        posted_text = (mod_el.get_text(strip=True) if mod_el else (date_el.get_text(strip=True) if date_el else ""))
        posted_iso  = parse_posted(posted_text)
        jobs.append({
            "title": title[:2000], "url": url, "company": company[:2000],
            "location": location[:2000], "experience": experience[:2000],
            "posted": posted_iso, "source": "게임잡",
        })
    return list({j["url"]: j for j in jobs}.values())

# ---------------- 상세 페이지에서 '최종 수정일' 뽑기 ----------------
def _find_date_near(text, anchors):
    """원본문자열에서 앵커(수정/등록 등) 근처 60자 창에 있는 날짜 스니펫 추출."""
    for anchor in anchors:
        for m in re.finditer(anchor, text):
            win = text[max(0, m.start()-20): m.end()+60]
            # 절대일자
            m1 = re.search(r"(\d{4})[.\-/년\s]*(\d{1,2})[.\-/월\s]*(\d{1,2})", win)
            if m1: return m1.group(0)
            m2 = re.search(r"(\d{1,2})[.\-/](\d{1,2})(?:\([^)]*\))?", win)
            if m2: return m2.group(0)
            m3 = re.search(r"(\d+)\s*일\s*전|오늘|어제", win)
            if m3: return m3.group(0)
    return None

def extract_detail_modified(html):
    """상세 페이지 전체 텍스트에서 '수정' 계열을 최우선, 없으면 '등록일'을 본다."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    # 우선순위: 최종수정/수정일/최근수정 → 등록일
    cand = _find_date_near(text, anchors=[r"최종\s*수정", r"수정일", r"최근\s*수정", r"업데이트"])
    if not cand:
        cand = _find_date_near(text, anchors=[r"등록일", r"게재일"])
    return parse_posted(cand) if cand else None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
def fetch_detail_html(url, referer=None):
    headers = {"Referer": referer or BASE, "User-Agent": SESSION.headers["User-Agent"]}
    return fetch_raw(url, headers=headers).text

def enrich_jobs_with_detail_dates(jobs, referer, mode="always", detail_delay=0.4, limit=None):
    """
    mode:
      - 'always'  : 상세 수정일로 항상 덮어쓰기(없으면 등록일/기존값 유지)
      - 'fallback': 리스트에 posted 없을 때만 상세에서 보충
      - 'off'     : 상세 미조회
    """
    if mode == "off": return jobs
    count = 0
    for j in jobs:
        if limit and count >= limit: break
        if mode == "fallback" and j.get("posted"): continue
        try:
            html = fetch_detail_html(j["url"], referer=referer)
            mod_iso = extract_detail_modified(html)
            if mod_iso:
                j["posted"] = mod_iso
        except Exception as e:
            print(f"[상세일자 실패] {j.get('title','(no title)')} -> {e}")
        count += 1
        if detail_delay > 0: time.sleep(detail_delay)
    return jobs

# ---------------- Notion 전용 요청 래퍼(재시도/백오프) ----------------
RETRY_STATUSES = {429, 500, 502, 503, 504}

@retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=15))
def notion_request(method: str, url: str, **kwargs) -> requests.Response:
    """
    Notion 전용 요청 래퍼: 타임아웃, 재시도(429/5xx 및 네트워크 오류), 백오프.
    """
    if "timeout" not in kwargs:
        kwargs["timeout"] = (10, 30)  # (connect, read)
    try:
        r = requests.request(method, url, **kwargs)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        # 네트워크 오류는 재시도 트리거
        raise e
    if r.status_code in RETRY_STATUSES:
        raise requests.HTTPError(f"Retryable status: {r.status_code}", response=r)
    r.raise_for_status()
    return r

# ---------------- Notion ----------------
def get_title_prop_name(db_id):
    r = notion_request("GET", f"https://api.notion.com/v1/databases/{db_id}", headers=HEADERS_NOTION)
    for name, prop in r.json().get("properties", {}).items():
        if prop.get("type") == "title": return name
    raise SystemExit("이 데이터베이스에서 'title' 타입 속성을 찾지 못했습니다.")

def notion_query_by_url(db_id, url):
    query = {"filter": {"property": "URL", "url": {"equals": url}}, "page_size": 1}
    r = notion_request("POST", f"https://api.notion.com/v1/databases/{db_id}/query",
                       headers=HEADERS_NOTION, json=query)
    data = r.json()
    return data["results"][0] if data.get("results") else None

def to_notion_properties(job):
    props = {
        TITLE_PROP_NAME: {"title": [{"text": {"content": job["title"]}}]},
        "Company":   {"rich_text": [{"text": {"content": job.get("company","")}}]} if job.get("company") else {"rich_text": []},
        "Location":  {"rich_text": [{"text": {"content": job.get("location","")}}]} if job.get("location") else {"rich_text": []},
        "Experience":{"rich_text": [{"text": {"content": job.get("experience","")}}]} if job.get("experience") else {"rich_text": []},
        "URL":       {"url": job["url"]},
        "Source":    {"select": {"name": job.get("source", "게임잡")}},
        "ScrapedAt": {"date": {"start": dt.datetime.now().astimezone().isoformat()}},
    }
    if job.get("posted"): props["Posted"] = {"date": {"start": job["posted"]}}
    return props

def notion_create(db_id, job):
    payload = {"parent": {"database_id": db_id}, "properties": to_notion_properties(job)}
    r = notion_request("POST", "https://api.notion.com/v1/pages",
                       headers=HEADERS_NOTION, json=payload)
    return r.json()

def notion_update(page_id, job):
    payload = {"properties": to_notion_properties(job)}
    r = notion_request("PATCH", f"https://api.notion.com/v1/pages/{page_id}",
                       headers=HEADERS_NOTION, json=payload)
    return r.json()

def upsert_to_notion(db_id, jobs, notion_delay: float = 0.2):
    created = updated = 0
    for j in jobs:
        try:
            existing = notion_query_by_url(db_id, j["url"])
            if existing:
                notion_update(existing["id"], j)
                updated += 1
            else:
                notion_create(db_id, j)
                created += 1
            if notion_delay > 0:
                time.sleep(notion_delay)
        except requests.HTTPError as e:
            body = e.response.text if getattr(e, "response", None) else str(e)
            print(f"[Notion Error] {j.get('title','(no title)')}\n{body}")
        except Exception as e:
            print(f"[Upsert Error] {j.get('title','(no title)')} -> {e}")
    return created, updated

# ---------------- 페이지 자동 감지 + 크롤링 ----------------
def get_max_page_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    nums = []
    for a in soup.select(".pagination a[href]"):
        t = (a.get_text() or "").strip()
        if t.isdigit(): nums.append(int(t))
    return max(nums) if nums else 1

def crawl(list_url, pages, delay):
    all_jobs = []
    # 1) 1페이지
    html = fetch_list_html(list_url, 1)
    with open("last_list.html", "w", encoding="utf-8") as f: f.write(html)
    # 2) 페이지 수 자동 감지
    max_pages = min(get_max_page_from_html(html), 200) if pages <= 0 else pages
    if pages <= 0: print(f"[안내] 페이지 자동 감지: 총 {max_pages}페이지로 수집합니다.")
    # 3) 1페이지
    page_jobs = extract_jobs(html); print(f"[페이지 1] 추출 {len(page_jobs)}건"); all_jobs.extend(page_jobs)
    prev_html = html
    if max_pages > 1: time.sleep(max(0.0, delay))
    # 4) 2..N
    for p in range(2, max_pages + 1):
        page_url = find_pagination_url(list_url, prev_html, p) if prev_html else None
        if page_url:
            page_url = merge_query(list_url, page_url)
            headers = {"Referer": list_url, "X-Requested-With": "XMLHttpRequest", "User-Agent": SESSION.headers["User-Agent"]}
            html = fetch_raw(page_url, headers=headers).text
        else:
            html = fetch_list_html(list_url, p)
        if p == 2:
            with open("last_list_p2.html", "w", encoding="utf-8") as f: f.write(html)
        page_jobs = extract_jobs(html); print(f"[페이지 {p}] 추출 {len(page_jobs)}건"); all_jobs.extend(page_jobs)
        prev_html = html
        if p < max_pages: time.sleep(max(0.0, delay))
    return list({j["url"]: j for j in all_jobs}.values())

# ---------------- 메인 ----------------
def apply_filters(jobs, keyword):
    if not keyword: return jobs
    key = keyword.lower(); out = []
    for j in jobs:
        hay = " ".join([j.get("title",""), j.get("company",""), j.get("location",""), j.get("experience","")]).lower()
        if key in hay: out.append(j)
    return out

def main():
    ap = argparse.ArgumentParser(description="게임잡 -> Notion 업서트 (상세페이지 수정일 사용 + Notion 재시도/딜레이)")
    ap.add_argument("--list-url", default=DEFAULT_LIST_URL, help="리스트 URL")
    ap.add_argument("--pages", type=int, default=2, help="가져올 페이지 수 (0 또는 음수=자동 감지)")
    ap.add_argument("--delay", type=float, default=1.0, help="리스트 페이지 요청 간 대기(초)")
    ap.add_argument("--detail-mode", choices=["always","fallback","off"], default="always",
                    help="상세페이지 수정일 사용 모드: always(항상), fallback(리스트에 없을 때만), off(미사용)")
    ap.add_argument("--detail-delay", type=float, default=0.4, help="상세 페이지 요청 간 대기(초)")
    ap.add_argument("--notion-delay", type=float, default=0.2, help="Notion 업서트 사이 대기(초)")
    ap.add_argument("--keyword", type=str, default=None, help="간이 키워드 필터")
    ap.add_argument("--dry-run", action="store_true", help="Notion 업로드 없이 파일 출력만")
    ap.add_argument("--limit", type=int, default=0, help="최대 N건만 사용(0=제한없음)")
    args = ap.parse_args()

    jobs = crawl(args.list_url, args.pages, args.delay)
    jobs = apply_filters(jobs, args.keyword)
    # 상세페이지에서 최종 수정일 반영
    jobs = enrich_jobs_with_detail_dates(jobs, referer=args.list_url, mode=args.detail_mode,
                                         detail_delay=args.detail_delay)
    if args.limit and args.limit > 0: jobs = jobs[:args.limit]
    print(f"\n총 {len(jobs)}건 (필터/중복 제거 후)\n")

    if args.dry_run:
        for j in jobs[:10]: print(json.dumps(j, ensure_ascii=False))
        with open("dryrun_jobs.json", "w", encoding="utf-8") as f: json.dump(jobs, f, ensure_ascii=False, indent=2)
        print("✅ 드라이런 완료: last_list.html / last_list_p2.html / dryrun_jobs.json 확인")
        return

    if not NOTION_TOKEN or not NOTION_DB_ID:
        raise SystemExit("'.env'의 NOTION_TOKEN / NOTION_DATABASE_ID 가 필요합니다.")

    global TITLE_PROP_NAME
    TITLE_PROP_NAME = get_title_prop_name(NOTION_DB_ID)
    created, updated = upsert_to_notion(NOTION_DB_ID, jobs, notion_delay=args.notion_delay)
    print(f"✅ Notion 업서트 완료: 생성 {created} / 갱신 {updated}")

if __name__ == "__main__":
    main()
