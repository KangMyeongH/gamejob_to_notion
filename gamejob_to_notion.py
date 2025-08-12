# file: gamejob_to_notion.py
# 요구 패키지:
#   pip install requests beautifulsoup4 python-dotenv tenacity
#
# 실행 예시:
#   (드라이런) 1페이지만 10건 미리보기
#     python gamejob_to_notion.py --dry-run --pages 1 --limit 10
#   (게임개발-클라이언트 전체, 페이지 자동 감지)
#     python gamejob_to_notion.py --pages 0 --delay 1.2
#   (사이트 전체 채용)
#     python gamejob_to_notion.py --list-url "https://www.gamejob.co.kr/Recruit/joblist?menucode=searchall" --pages 0 --delay 1.5

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

# 직종 > 게임 제작 > 게임개발(클라이언트)
DEFAULT_LIST_URL = "https://www.gamejob.co.kr/Recruit/joblist?menucode=duty&duty=1"

# 2페이지 이상일 때 우선 시도할 AJAX 엔드포인트
AJAX_URL_TMPL = "/recruit/_GI_Job_List?Page={p}"

# AJAX 실패 시 폴백할 페이지 파라미터 후보
PAGE_PARAM_CANDIDATES = ["Page", "page", "thisPage", "pageIndex"]

# HTTP 세션/UA
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) GamejobCrawler/1.3"
})

# ---------------- 유틸 ----------------
def normalize_space(s):
    return re.sub(r"\s+", " ", s or "").strip()

def parse_posted(text):
    """게시일 텍스트를 ISO(YYYY-MM-DD)로 추정 변환."""
    if not text:
        return None
    t = text.strip()

    m = re.search(r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})", t)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return dt.date(y, mo, d).isoformat()
        except Exception:
            return None

    m = re.search(r"(\d{1,2})[.\-/](\d{1,2})", t)  # mm-dd / mm/dd
    if m:
        mo, d = map(int, m.groups())
        try:
            return dt.date(dt.date.today().year, mo, d).isoformat()
        except Exception:
            return None

    if "오늘" in t:
        return dt.date.today().isoformat()
    if "어제" in t:
        return (dt.date.today() - dt.timedelta(days=1)).isoformat()
    return None

# base_url의 쿼리(duty, menucode 등)를 target_url에 병합해 필터 유지
def merge_query(base_url, target_url):
    """
    base_url의 query(duty, menucode 등)를 target_url의 query에 병합해서 반환.
    target_url의 Page 값은 유지, 중복키는 target_url 우선.
    """
    bp = urlparse(base_url)
    tp = urlparse(target_url)

    bq = parse_qs(bp.query)
    tq = parse_qs(tp.query)

    bq.pop("Page", None)  # Page는 target 쪽 유지
    for k, v in bq.items():
        if k not in tq:
            tq[k] = v

    # 단일값으로 정리
    flat = {k: (v[0] if isinstance(v, list) else v) for k, v in tq.items()}
    new_q = urlencode(flat)
    return urlunparse((tp.scheme or bp.scheme, tp.netloc or bp.netloc, tp.path, "", new_q, ""))

# ---------------- HTTP ----------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_raw(url, headers=None):
    r = SESSION.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r

def fetch_list_html(list_url, page):
    """1페이지: 원본 URL, 2+페이지: AJAX 우선 → 실패 시 일반 쿼리 폴백(필터 유지)."""
    if page == 1:
        return fetch_raw(list_url).text

    # AJAX 우선
    ajax_url = urljoin(BASE, AJAX_URL_TMPL.format(p=page))
    ajax_headers = {
        "Referer": list_url,
        "X-Requested-With": "XMLHttpRequest",
        "User-Agent": SESSION.headers["User-Agent"],
    }
    try:
        return fetch_raw(ajax_url, headers=ajax_headers).text
    except Exception:
        # 일반 쿼리 폴백 (list_url의 기존 쿼리 유지 + Page만 교체/추가)
        parsed = urlparse(list_url)
        qs = parse_qs(parsed.query)
        replaced = False
        for key in PAGE_PARAM_CANDIDATES:
            if key in qs:
                qs[key] = [str(page)]
                replaced = True
                break
        if not replaced:
            qs[PAGE_PARAM_CANDIDATES[0]] = [str(page)]
        query_str = urlencode({k: v[0] for k, v in qs.items()})
        fallback_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{query_str}" if query_str else list_url
        return fetch_raw(fallback_url).text

# ---------------- 페이지네이션 링크 따라가기 ----------------
def find_pagination_url(base_url, html, target_page):
    """HTML의 페이지네이션에서 target_page의 실제 href를 찾아 절대경로로 반환."""
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.select(".pagination a[href]"):
        txt = (a.get_text() or "").strip()
        if txt == str(target_page):
            return urljoin(base_url, a["href"])
    for a in soup.select(".pagination a[href]"):
        for attr, val in a.attrs.items():
            if attr.startswith("data") and str(val).strip() == str(target_page):
                return urljoin(base_url, a["href"])
    return None

# ---------------- 파싱 ----------------
def extract_jobs(html, base=BASE):
    """실제 리스트 구조에 맞춘 고정 셀렉터 + 폴백으로 추출."""
    soup = BeautifulSoup(html, "html.parser")

    # 1차: 일반 페이지 구조
    rows = soup.select('#dev-gi-list table.tblList tbody tr')
    # 2차: AJAX 조각(테이블 래퍼 없음) 대응
    if not rows:
        rows = soup.select('table.tblList tbody tr')
    # 3차: 최후 폴백 - 제목 a가 있는 tr만
    if not rows:
        rows = [tr for tr in soup.select('tr') if tr.select_one('div.tit a')]

    jobs = []
    for tr in rows:
        a = tr.select_one('div.tit a[href]')
        company_el = tr.select_one('div.company strong')
        if not a or not company_el:
            continue  # 광고/머릿글/비정상 행 방어

        title = normalize_space(a.get_text())
        href = a.get('href', '')
        url = urljoin(base, href)
        company = normalize_space(company_el.get_text())

        # 경력/지역 등: tit 영역의 info 스팬들
        info_spans = [normalize_space(s.get_text()) for s in tr.select('div.tit p.info span')]
        experience = next((t for t in info_spans if any(k in t for k in ['경력','신입','무관','연차','이상'])), '')
        location   = next((t for t in info_spans if any(k in t for k in [
            '서울','경기','인천','부산','대구','대전','광주','세종','울산','강원',
            '충북','충남','전북','전남','경북','경남','제주','해외','재택','시','도','구','군'
        ])), '')

        date_el = tr.select_one('td:nth-of-type(3) .date')
        mod_el  = tr.select_one('td:nth-of-type(3) .modifyDate')
        posted_text = " ".join(x for x in [
            date_el.get_text(strip=True) if date_el else "",
            mod_el.get_text(strip=True) if mod_el else ""
        ] if x).strip()
        posted_iso = parse_posted(posted_text)

        jobs.append({
            "title": title[:2000],
            "url": url,
            "company": company[:2000],
            "location": location[:2000],
            "experience": experience[:2000],
            "posted": posted_iso,   # 변환 실패 시 None → Notion 미설정
            "source": "게임잡",
        })

    # URL 기준 dedup
    uniq = {j["url"]: j for j in jobs}
    return list(uniq.values())

# ---------------- Notion ----------------
def get_title_prop_name(db_id):
    r = requests.get(f"https://api.notion.com/v1/databases/{db_id}",
                     headers=HEADERS_NOTION, timeout=20)
    r.raise_for_status()
    data = r.json()
    for name, prop in data.get("properties", {}).items():
        if prop.get("type") == "title":
            return name
    raise SystemExit("이 데이터베이스에서 'title' 타입 속성을 찾지 못했습니다. Notion DB의 제목 속성을 확인하세요.")

def notion_query_by_url(db_id, url):
    query = {"filter": {"property": "URL", "url": {"equals": url}}, "page_size": 1}
    r = requests.post(f"https://api.notion.com/v1/databases/{db_id}/query",
                      headers=HEADERS_NOTION, json=query, timeout=20)
    r.raise_for_status()
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
    if job.get("posted"):
        props["Posted"] = {"date": {"start": job["posted"]}}
    return props

def notion_create(db_id, job):
    payload = {"parent": {"database_id": db_id}, "properties": to_notion_properties(job)}
    r = requests.post("https://api.notion.com/v1/pages", headers=HEADERS_NOTION, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()

def notion_update(page_id, job):
    payload = {"properties": to_notion_properties(job)}
    r = requests.patch(f"https://api.notion.com/v1/pages/{page_id}", headers=HEADERS_NOTION, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()

def upsert_to_notion(db_id, jobs):
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
        if t.isdigit():
            nums.append(int(t))
    return max(nums) if nums else 1

def crawl(list_url, pages, delay):
    all_jobs = []

    # 1) 1페이지
    html = fetch_list_html(list_url, 1)
    with open("last_list.html", "w", encoding="utf-8") as f:
        f.write(html)

    # 2) 페이지 수 자동 감지(pages <= 0)
    if pages <= 0:
        max_pages = min(get_max_page_from_html(html), 200)  # 안전상한
        print(f"[안내] 페이지 자동 감지: 총 {max_pages}페이지로 수집합니다.")
    else:
        max_pages = pages

    # 3) 1페이지 추출
    page_jobs = extract_jobs(html)
    print(f"[페이지 1] 추출 {len(page_jobs)}건")
    all_jobs.extend(page_jobs)
    prev_html = html
    if max_pages > 1:
        time.sleep(max(0.0, delay))

    # 4) 2 ~ max_pages
    for p in range(2, max_pages + 1):
        # 페이지네이션 링크 우선
        page_url = find_pagination_url(list_url, prev_html, p) if prev_html else None
        if page_url:
            # 카테고리 필터 쿼리 병합 (duty, menucode 등)
            page_url = merge_query(list_url, page_url)
            ajax_headers = {
                "Referer": list_url,
                "X-Requested-With": "XMLHttpRequest",
                "User-Agent": SESSION.headers["User-Agent"],
            }
            html = fetch_raw(page_url, headers=ajax_headers).text
        else:
            # 실패 시 AJAX → 일반 쿼리 폴백
            html = fetch_list_html(list_url, p)

        if p == 2:
            with open("last_list_p2.html", "w", encoding="utf-8") as f:
                f.write(html)

        page_jobs = extract_jobs(html)
        print(f"[페이지 {p}] 추출 {len(page_jobs)}건")
        all_jobs.extend(page_jobs)

        prev_html = html
        if p < max_pages:
            time.sleep(max(0.0, delay))

    # dedup
    uniq = {j["url"]: j for j in all_jobs}
    return list(uniq.values())

# ---------------- 메인 ----------------
def apply_filters(jobs, keyword):
    if not keyword:
        return jobs
    key = keyword.lower()
    out = []
    for j in jobs:
        hay = " ".join([
            j.get("title",""),
            j.get("company",""),
            j.get("location",""),
            j.get("experience","")
        ]).lower()
        if key in hay:
            out.append(j)
    return out

def main():
    ap = argparse.ArgumentParser(description="게임잡 -> Notion 업서트 (필터 유지/페이지 자동 감지)")
    ap.add_argument("--list-url", default=DEFAULT_LIST_URL, help="리스트 URL(다른 카테고리/검색 URL로 교체 가능)")
    ap.add_argument("--pages", type=int, default=2, help="가져올 페이지 수 (0 또는 음수=자동 감지)")
    ap.add_argument("--delay", type=float, default=1.0, help="요청 간 대기(초)")
    ap.add_argument("--keyword", type=str, default=None, help="제목/회사/지역/경력에 포함된 키워드 필터")
    ap.add_argument("--dry-run", action="store_true", help="Notion 업로드 없이 콘솔/파일 출력만")
    ap.add_argument("--limit", type=int, default=0, help="최대 N건만 사용(0=제한없음)")
    args = ap.parse_args()

    jobs = crawl(args.list_url, args.pages, args.delay)
    jobs = apply_filters(jobs, args.keyword)
    if args.limit and args.limit > 0:
        jobs = jobs[:args.limit]

    print(f"\n총 {len(jobs)}건 (필터/중복 제거 후)\n")

    if args.dry_run:
        for j in jobs[:10]:
            print(json.dumps(j, ensure_ascii=False))
        with open("dryrun_jobs.json", "w", encoding="utf-8") as f:
            json.dump(jobs, f, ensure_ascii=False, indent=2)
        print("✅ 드라이런 완료: last_list.html / last_list_p2.html / dryrun_jobs.json 확인")
        return

    if not NOTION_TOKEN or not NOTION_DB_ID:
        raise SystemExit("'.env'의 NOTION_TOKEN / NOTION_DATABASE_ID 가 필요합니다.")

    # 업서트 직전: 제목 속성명 자동 감지
    global TITLE_PROP_NAME
    TITLE_PROP_NAME = get_title_prop_name(NOTION_DB_ID)

    created, updated = upsert_to_notion(NOTION_DB_ID, jobs)
    print(f"✅ Notion 업서트 완료: 생성 {created} / 갱신 {updated}")

if __name__ == "__main__":
    main()
