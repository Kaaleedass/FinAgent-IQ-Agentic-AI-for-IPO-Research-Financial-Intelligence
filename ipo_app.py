"""
Local IPO dashboard backend + LangChain Data Analyst Agent.
- Reads IPO metadata from the project CSV.
- Inventories downloaded PDFs under configured SEBI/prospectus folders.
- Provides a company details API and a GPT-4o mini LangChain agent.
- The agent can list documents, inspect a selected company's prospectus, and search PDF text.
Run from IPO_Project: python ipo_app.py
Open: http://127.0.0.1:5000
"""
import os, re, json, shutil, subprocess, tempfile, time
from pathlib import Path
from functools import lru_cache
from threading import Thread
from urllib.request import urlopen

import pandas as pd
import pymupdf
from flask import Flask, jsonify, request, send_from_directory
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

SCRIPT_DIR = Path(__file__).resolve().parent
# Fixed user-configured IPO metadata path.
CSV_FILE = Path(r"C:\Users\Kaale\Desktop\MY_Personal_project\IPO_Project\IPO_DATA\ipo_data.csv")
PROJECT_DIR = CSV_FILE.parent.parent
UPLOAD_DIR = PROJECT_DIR / "uploaded_prospectuses"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DASHBOARD_FILE = (SCRIPT_DIR / "ipo_dashboard_company_research.html") if (SCRIPT_DIR / "ipo_dashboard_company_research.html").exists() else (PROJECT_DIR / "ipo_dashboard_company_research.html")
PDF_DIRS = [
    PROJECT_DIR / "sebi_documents",
    PROJECT_DIR / "IPO_DATA" / "sebi_documents",
    PROJECT_DIR / "ipo_tracker_connected_bundle" / "sebi_documents",
    PROJECT_DIR / "NSE_IPO_Downloads",
    Path(r"C:\Users\Kaale\Desktop\MY_Personal_project\IPO_Project\sebi_documents"),
    Path(r"C:\Users\Kaale\Desktop\MY_Personal_project\IPO_Project\IPO_DATA\sebi_documents"),
    Path(r"C:\Users\Kaale\Desktop\MY_Personal_project\NSE_IPO_Downloads"),
    Path(r"C:\Users\Kaale\OneDrive\Desktop\MY_Personal_project\NSE_IPO_Downloads"),
    UPLOAD_DIR,
]
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)

@lru_cache(maxsize=1)
def _cached_all_pdfs():
    found = []
    seen = set()
    for folder in PDF_DIRS:
        if not folder.exists():
            continue
        try:
            for p in folder.rglob("*.pdf"):
                key = str(p.resolve()).lower()
                if key not in seen:
                    seen.add(key); found.append(p)
        except (PermissionError, OSError):
            pass
    return tuple(found)

def all_pdfs():
    # Cache the expensive recursive PDF inventory instead of rescanning every API request.
    return list(_cached_all_pdfs())

def norm(s):
    s = str(s or "").lower()
    s = re.sub(r"\b(limited|ltd|india|the|private|pvt|company|co)\b", " ", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()

def match_score(company, path):
    a, b = set(norm(company).split()), set(norm(path.stem).split())
    if not a or not b: return 0.0
    return len(a & b) / max(1, len(a))

def match_pdf(company):
    matches = []
    for p in all_pdfs():
        score = max(match_score(company, p), match_score(company, Path(p.parent.name + " " + p.stem)))
        # Exact company-specific upload folder takes precedence.
        safe_company = re.sub(r"[^A-Za-z0-9_-]+", "_", company).strip("_")[:80]
        if p.parent.name.lower() == safe_company.lower():
            score = 1.0
        matches.append((score, p))
    matches = [(s,p) for s,p in matches if s >= 0.45]
    matches.sort(key=lambda x:(x[0], x[1].stat().st_mtime), reverse=True)
    return matches[0] if matches else (0.0, None)

def extract_pdf(path, max_chars=50000):
    doc = pymupdf.open(path)
    pages=[]
    try:
        for i, page in enumerate(doc):
            text=page.get_text("text")
            if text.strip():
                pages.append(f"[PDF page {i+1}]\n{text}")
            if sum(map(len,pages)) >= max_chars: break
    finally: doc.close()
    return "\n\n".join(pages)[:max_chars]

def load_rows():
    if not CSV_FILE.exists(): return []
    df = pd.read_csv(CSV_FILE, dtype=object)
    df = df.astype(object).where(pd.notna(df), None)
    return df.to_dict(orient="records")

def company_row(name):
    rows=load_rows()
    wanted=norm(name)
    exact=[r for r in rows if norm(r.get("ipo_name"))==wanted]
    if exact: return exact[0]
    partial=[r for r in rows if wanted in norm(r.get("ipo_name")) or norm(r.get("ipo_name")) in wanted]
    return partial[0] if len(partial)==1 else None

@tool
def list_available_prospectus_pdfs() -> str:
    """List PDF prospectus/RHP/DRHP documents available in the configured local SEBI folders."""
    return json.dumps([{"file":p.name,"path":str(p)} for p in all_pdfs()], ensure_ascii=False)

@tool
def get_selected_company_prospectus(company_name: str) -> str:
    """Find and extract text from the local prospectus PDF that best matches the selected IPO company. Includes page numbers."""
    score, path=match_pdf(company_name)
    if not path:
        return f"No matching prospectus PDF found in configured SEBI folders for {company_name}. Do not invent document facts."
    try:
        text=extract_pdf(path)
    except Exception as e:
        return f"Matched file {path.name}, but PDF extraction failed: {e}"
    if not text.strip():
        return f"PDF {path.name} has no extractable text. It may be scanned and need OCR."
    return json.dumps({"company":company_name,"file":path.name,"path":str(path),"filename_match_score":round(score,2),"text":text}, ensure_ascii=False)

@tool
def search_selected_prospectus(company_name: str, query: str) -> str:
    """Search the selected company's local prospectus for a phrase/topic and return relevant excerpts with PDF page numbers."""
    score,path=match_pdf(company_name)
    if not path: return f"No matching local prospectus PDF found for {company_name}."
    try: text=extract_pdf(path, max_chars=120000)
    except Exception as e: return f"Could not read {path.name}: {e}"
    terms=[t.lower() for t in re.findall(r"[A-Za-z0-9]+", query) if len(t)>2]
    chunks=[]
    for block in re.split(r"(?=\[PDF page \d+\])", text):
        low=block.lower()
        if any(term in low for term in terms):
            chunks.append(block[:2400])
        if len(chunks)>=8: break
    if not chunks: return f"No matching text found for query '{query}' in {path.name}. Document found, but this query was not located in extracted text."
    return f"Document: {path.name}; filename match score {score:.2f}\\n" + "\n---\n".join(chunks)

@tool
def get_ipo_metadata(company_name: str) -> str:
    """Return the selected IPO's available metadata and existing valuation fields from ipo_data.csv."""
    row=company_row(company_name)
    if not row: return f"No unique company record found for {company_name} in {CSV_FILE}."
    return json.dumps(row, ensure_ascii=False, default=str)

def extract_financial_charts(company_name, pdf_path):
    """Use GPT-4o mini to extract only explicit prospectus metrics for charts; no inferred values."""
    if not API_KEY:
        return []
    try:
        text = extract_pdf(pdf_path, max_chars=24000)
        if not text.strip():
            return []
        llm = ChatOpenAI(model=MODEL, temperature=0, api_key=API_KEY)
        prompt = f"""Extract chart-ready financial/operating data for {company_name} from this prospectus text.
Return valid JSON only with schema:
{{"charts":[{{"title":"Revenue","unit":"₹ crore","source":"PDF page 9","values":[{{"label":"FY2024","value":123.4}}]}}]}}
Rules: only include values explicitly reported in the supplied text; preserve units and periods; convert ₹ million to ₹ crore only if clear and do the conversion correctly; do not invent or annualize. Use at most 8 useful charts from metrics actually found such as revenue, EBITDA, PAT, EPS, borrowings, customer count, ARPU, churn, margins, cash flow. For each chart give the source PDF page number if present in text. Skip metrics without numeric values. Values must be JSON numbers.
TEXT:
{text[:24000]}"""
        response = llm.invoke(prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        parsed = json.loads(content[content.find("{"):content.rfind("}")+1])
        charts = parsed.get("charts", [])
        return charts[:8] if isinstance(charts, list) else []
    except Exception as exc:
        print(f"Financial chart extraction failed for {company_name}: {exc}")
        return []


def extract_fixed_financial_charts(company_name, pdf_path):
    """Extract three-year Revenue, Profit After Tax and Total Assets series from PDF evidence."""
    if not API_KEY:
        return {"charts": [], "note": "OPENAI_API_KEY unavailable; financial chart extraction was not run."}
    try:
        text = extract_pdf(pdf_path, max_chars=65000)
        if not text.strip():
            return {"charts": [], "note": "No extractable PDF text. Scanned documents may require OCR."}
        llm = ChatOpenAI(model=MODEL, temperature=0, api_key=API_KEY)
        prompt = f"""You are extracting historical company financial data from an IPO prospectus.
Company: {company_name}
Return JSON only in this schema:
{{
 "charts":[
  {{"metric":"Revenue","unit":"₹ crore","values":[{{"year":"FY2024","value":123.4,"page":45}},{{"year":"FY2025","value":125.6,"page":46}},{{"year":"FY2026","value":130.2,"page":47}}]}},
  {{"metric":"Profit after tax","unit":"₹ crore","values":[]}},
  {{"metric":"Total assets","unit":"₹ crore","values":[]}}
 ],
 "note":"brief extraction note"
}}
Rules:
- Try to provide the last 3 completed financial years available in the supplied text for each metric.
- Revenue means revenue from operations, unless the document clearly identifies a different headline revenue measure; identify it in the note if so.
- Profit means profit after tax / profit for the year, not EBITDA or PBT.
- Total assets means the reported balance-sheet total assets, not net assets or net worth.
- Preserve actual fiscal year labels and units. If figures are ₹ million, convert to ₹ crore by dividing by 10 only when units clearly say ₹ million.
- Do not annualize interim/quarter figures and do not use projections as historical results.
- Never infer missing years/figures. If fewer than three years are reported, return only the available years.
- Each numeric value must be explicitly supported by the document text and include its PDF page number if available in the extracted text; use null if page is unavailable.
- Return all three chart objects even if some values arrays are empty.
TEXT:
{text[:65000]}"""
        response = llm.invoke(prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        parsed = json.loads(content[content.find("{"):content.rfind("}")+1])
        raw = parsed.get("charts", [])
        wanted = {"revenue", "profit after tax", "total assets"}
        charts = []
        for metric in ("Revenue", "Profit after tax", "Total assets"):
            item = next((x for x in raw if str(x.get("metric","")).strip().lower() == metric.lower()), None)
            vals = []
            for v in (item or {}).get("values", []):
                try:
                    if v.get("value") is None: continue
                    vals.append({"year": str(v.get("year") or "Period not identified"),
                                 "value": float(v["value"]),
                                 "page": v.get("page")})
                except (TypeError, ValueError, KeyError):
                    continue
            charts.append({"metric": metric, "unit": (item or {}).get("unit", "₹ crore"), "values": vals[-3:]})
        return {"charts": charts, "note": str(parsed.get("note", ""))}
    except Exception as exc:
        return {"charts": [], "note": f"Financial extraction failed: {exc}"}

def build_agent():
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing from the environment of the process running ipo_app.py.")
    llm=ChatOpenAI(model=MODEL, temperature=0.1, api_key=API_KEY)
    return create_agent(
        model=llm,
        tools=[list_available_prospectus_pdfs, get_selected_company_prospectus, search_selected_prospectus, get_ipo_metadata],
        system_prompt=(
            "You are an evidence-first Indian IPO research data analyst. Use the tools to inspect local IPO metadata "
            "and available SEBI/RHP/DRHP PDFs. Answer for the selected company only unless asked to compare. "
            "For financial claims, first use the selected company prospectus tools and cite the exact PDF page number. Distinguish PDF-reported facts from ipo_data.csv metadata and your calculations. At the end include Sources: PDF filename and page(s), plus metadata CSV only if used. If a figure cannot be located in extracted PDF text, say it was not verified in the PDF instead of presenting it as a PDF fact. Separate prospectus-reported facts, "
            "calculations, and interpretation. Never invent missing data, peer multiples, issue price, financials, "
            "or page references. If no PDF is found, clearly say 'No document found in configured SEBI folder' "
            "and answer only from available metadata. For valuation, state the formula and assumptions; GMP is "
            "unofficial sentiment and cannot prove intrinsic value or listing gain. Do not give guaranteed returns. "
            "Be concise but complete. If data is unavailable, say so explicitly."
        ),
    )

_AGENT = None
def agent():
    global _AGENT
    if _AGENT is None: _AGENT=build_agent()
    return _AGENT

@app.get("/")
def home():
    # Look for the dashboard beside this script first, then in the fixed project folder.
    candidates = [
        SCRIPT_DIR / "ipo_dashboard_company_research.html",
        SCRIPT_DIR / "ipo_dashboard_listing_gain_gmp_fix.html",
        SCRIPT_DIR / "ipo_dashboard_listing_gain_fast.html",
        PROJECT_DIR / "ipo_dashboard_company_research.html",
        PROJECT_DIR / "ipo_dashboard_listing_gain_gmp_fix.html",
        PROJECT_DIR / "ipo_dashboard_listing_gain_fast.html",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return send_from_directory(str(candidate.parent), candidate.name)
    return (
        "Dashboard HTML file not found. Expected one of these files:<br><pre>"
        + "\\n".join(str(p) for p in candidates)
        + "</pre>Place ipo_dashboard_company_research.html beside ipo_app.py and restart Flask.",
        500,
    )

@app.get("/api/health")
def api_health():
    return jsonify({"ok": True, "csv_path": str(CSV_FILE), "csv_exists": CSV_FILE.exists()})

@app.get("/api/data")
def api_data():
    """Fast CSV-only endpoint used by the dashboard at startup."""
    if not CSV_FILE.exists():
        return jsonify({"error": f"IPO data CSV not found: {CSV_FILE}", "csv_path": str(CSV_FILE)}), 404
    try:
        # DictReader is intentionally used here: dashboard startup should not scan PDFs or call OpenAI.
        import csv
        with CSV_FILE.open("r", encoding="utf-8-sig", newline="") as f:
            records = list(csv.DictReader(f))
        return jsonify({"records": records, "count": len(records), "csv_path": str(CSV_FILE),
                        "modified": CSV_FILE.stat().st_mtime})
    except Exception as exc:
        app.logger.exception("Could not read IPO CSV")
        return jsonify({"error": f"Could not read IPO CSV: {exc}", "csv_path": str(CSV_FILE)}), 500

@app.get("/api/companies")
def api_companies():
    rows=load_rows()
    pdfs=all_pdfs()
    companies=[]
    for r in rows:
        name=str(r.get("ipo_name") or "")
        score,path=match_pdf(name) if name else (0,None)
        companies.append({"ipo_name":name,"status":r.get("status"),"board":r.get("board"),"sector":r.get("sector"),
                          "pdf_found":bool(path),"pdf_name":path.name if path else None,"match_score":round(score,2) if path else None})
    return jsonify({"companies":companies,"pdf_count":len(pdfs),"pdf_folders":[str(p) for p in PDF_DIRS]})

@app.get("/api/company")
def api_company():
    name=request.args.get("name","").strip()
    if not name: return jsonify({"error":"Company name is required"}),400
    row=company_row(name)
    if not row: return jsonify({"error":"Company not found in ipo_data.csv"}),404
    score,path=match_pdf(name)
    out=dict(row)
    charts = extract_financial_charts(name, path) if path else []
    fixed = extract_fixed_financial_charts(name, path) if path else {"charts": [], "note": "No document found in configured SEBI folder."}
    out.update({"pdf_found":bool(path),"pdf_name":path.name if path else None,"pdf_path":str(path) if path else None,
                "match_score":round(score,2) if path else None,"extraction_status":"PDF found; financial metrics extracted where supported" if path else "No document found",
                "financial_charts":charts, "fixed_financial_charts":fixed.get("charts", []),
                "financial_chart_note":fixed.get("note", "")})
    return jsonify({"company":out})


@app.post("/api/upload-pdf")
def api_upload_pdf():
    company = request.form.get("company_name", "").strip()
    file = request.files.get("file")
    if not company:
        return jsonify({"error": "Select a company before uploading."}), 400
    if not file or not file.filename:
        return jsonify({"error": "Choose a PDF file."}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are accepted."}), 400
    safe_company = re.sub(r"[^A-Za-z0-9_-]+", "_", company).strip("_")[:80] or "company"
    safe_file = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(file.filename).name)[:160]
    dest_dir = UPLOAD_DIR / safe_company
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (safe_company + "__" + safe_file)
    file.save(dest)
    try:
        with pymupdf.open(dest) as doc:
            page_count = len(doc)
    except Exception as exc:
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        return jsonify({"error": f"Uploaded file could not be opened as a PDF: {exc}"}), 400
    return jsonify({"ok": True, "company_name": company, "file_name": dest.name,
                    "file_path": str(dest), "page_count": page_count,
                    "message": f"PDF uploaded and available to the analyst ({page_count} pages)."})


@app.post("/api/chat")
def api_chat():
    payload=request.get_json(silent=True) or {}
    company=payload.get("company_name","").strip()
    question=payload.get("question","").strip()
    if not company or not question: return jsonify({"error":"company_name and question are required"}),400
    try:
        result=agent().invoke({"messages":[{"role":"user","content":f"Selected IPO company: {company}\nUser question: {question}\nInspect local PDFs and metadata for this exact company first. If a company-specific uploaded PDF exists, use it before weaker filename matches. Search the document for facts needed, cite PDF page numbers, and state clearly if no PDF is available."}]})
        msgs=result.get("messages",[])
        answer=""
        for m in reversed(msgs):
            if getattr(m,"type","")=="ai" and getattr(m,"content",None):
                answer=m.content
                if isinstance(answer,list):
                    answer="\n".join(str(x.get("text","")) for x in answer if isinstance(x,dict))
                break
        score, source_path = match_pdf(company)
        source_info = ({"file": source_path.name, "match_score": round(score, 2)} if source_path else None)
        return jsonify({"answer":answer or "No answer returned by the agent.","model":MODEL,"company":company,
                        "source_document": source_info,
                        "source_note": "Agent tools read extracted PDF text and may also use ipo_data.csv metadata."})
    except Exception as e:
        return jsonify({"error":str(e)}),500

if __name__=="__main__":
    from werkzeug.serving import make_server

    host, port = "127.0.0.1", 5000
    url = f"http://{host}:{port}/"
    server = make_server(host, port, app, threaded=True)
    server_thread = Thread(target=server.serve_forever, name="ipo-flask-server", daemon=True)
    server_thread.start()

    # Launch an isolated Chromium app window so its process lifetime tracks the
    # dashboard window, rather than an existing browser session.
    browser_candidates = [
        shutil.which("msedge"), shutil.which("chrome"), shutil.which("chromium"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
    ]
    browser_exe = next((p for p in browser_candidates if p and Path(p).is_file()), None)
    if browser_exe:
        profile_dir = tempfile.mkdtemp(prefix="ipo-dashboard-browser-")
        browser = subprocess.Popen([
            browser_exe, f"--user-data-dir={profile_dir}", "--no-first-run", "--no-default-browser-check",
            f"--app={url}",
        ])
        try:
            browser.wait()
        finally:
            server.shutdown()
            server_thread.join(timeout=5)
            shutil.rmtree(profile_dir, ignore_errors=True)
    else:
        print("Chrome or Microsoft Edge was not found; opening the dashboard in the default browser.")
        import webbrowser
        # Give Flask a moment to begin accepting requests before opening the page.
        for _ in range(50):
            try:
                with urlopen(f"http://{host}:{port}/api/health", timeout=0.25):
                    break
            except Exception:
                time.sleep(0.1)
        webbrowser.open(url)
        try:
            server_thread.join()
        except KeyboardInterrupt:
            server.shutdown()
