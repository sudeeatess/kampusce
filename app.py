from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:  
    PYPDF_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:  
    OPENAI_SDK_AVAILABLE = False




logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("univerify")


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    val = os.getenv(name)
    return val if val not in (None, "") else default


@dataclass
class Config:
    llm_model: str = field(default_factory=lambda: _env("FOUNDRY_LLM_MODEL"))
    embedding_model: str = field(default_factory=lambda: _env("FOUNDRY_EMBEDDING_MODEL", ""))
    foundry_endpoint: str = field(default_factory=lambda: _env("FOUNDRY_ENDPOINT"))
    foundry_api_key: str = field(default_factory=lambda: _env("FOUNDRY_API_KEY", "not-needed"))

    database_path: str = field(default_factory=lambda: _env("DATABASE_PATH", "data/univerify.db"))
    documents_path: str = field(default_factory=lambda: _env("DOCUMENTS_PATH", "documents"))

    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "3")))
    similarity_threshold: float = field(default_factory=lambda: float(_env("SIMILARITY_THRESHOLD", "0.60")))
    temperature: float = field(default_factory=lambda: float(_env("TEMPERATURE", "0.2")))
    max_chunk_chars: int = field(default_factory=lambda: int(_env("MAX_CHUNK_CHARS", "900")))
    chunk_overlap_chars: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP_CHARS", "120")))

    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8000")))

    max_upload_mb: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_MB", "25")))
    allowed_extensions: tuple = field(
        default_factory=lambda: tuple(
            e.strip().lower() for e in _env("ALLOWED_EXTENSIONS", ".pdf,.txt,.md").split(",") if e.strip()
        )
    )

    def db_abs(self) -> Path:
        p = Path(self.database_path)
        return p if p.is_absolute() else BASE_DIR / p

    def docs_abs(self) -> Path:
        p = Path(self.documents_path)
        return p if p.is_absolute() else BASE_DIR / p


CFG = Config()
CFG.db_abs().parent.mkdir(parents=True, exist_ok=True)
CFG.docs_abs().mkdir(parents=True, exist_ok=True)


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    title        TEXT NOT NULL,
    file_type    TEXT NOT NULL,
    file_hash    TEXT NOT NULL,
    stored_path  TEXT NOT NULL,
    pages        INTEGER DEFAULT 0,
    chunk_count  INTEGER DEFAULT 0,
    status       TEXT DEFAULT 'processing',
    error        TEXT,
    uploaded_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    page_number  INTEGER,
    section      TEXT,
    article      TEXT,
    content      TEXT NOT NULL,
    embedding    TEXT,
    created_at   TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS queries (
    id           TEXT PRIMARY KEY,
    question     TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS answers (
    id                   TEXT PRIMARY KEY,
    query_id             TEXT NOT NULL,
    answer               TEXT NOT NULL,
    verification_status  TEXT NOT NULL,
    confidence           REAL NOT NULL,
    sources_json         TEXT NOT NULL,
    retrieved_chunks     INTEGER DEFAULT 0,
    top_similarity       REAL DEFAULT 0,
    llm_model            TEXT,
    embedding_model      TEXT,
    created_at           TEXT NOT NULL,
    FOREIGN KEY (query_id) REFERENCES queries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_answers_query ON answers(query_id);
"""


@contextmanager
def db_conn():
    conn = sqlite3.connect(str(CFG.db_abs()), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with db_conn() as conn:
        conn.executescript(SCHEMA)
    log.info("SQLite hazır: %s", CFG.db_abs())


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

class DocumentReadError(Exception):
    pass


@dataclass
class PageText:
    page_number: int
    text: str


def read_pdf(path: Path) -> list[PageText]:
    if not PYPDF_AVAILABLE:
        raise DocumentReadError("pypdf kütüphanesi kurulu değil. 'pip install pypdf' çalıştırın.")
    try:
        reader = PdfReader(str(path))
    except Exception as e:
        raise DocumentReadError(f"PDF açılamadı: {e}") from e

    pages: list[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append(PageText(page_number=i, text=text))

    total_chars = sum(len(p.text.strip()) for p in pages)
    if total_chars < 20:
        raise DocumentReadError(
            "Bu belge okunamadı. Dosyanın bozuk olmadığından veya taranmış/görüntü "
            "tabanlı bir PDF olmadığından emin olun (OCR desteklenmiyor)."
        )
    return pages


def read_txt(path: Path) -> list[PageText]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise DocumentReadError(f"Metin dosyası okunamadı: {e}") from e
    return [PageText(page_number=1, text=text)]


def read_markdown(path: Path) -> list[PageText]:
    return read_txt(path)


def extract_pages(path: Path, ext: str) -> list[PageText]:
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".txt":
        return read_txt(path)
    if ext == ".md":
        return read_markdown(path)
    raise DocumentReadError(f"Desteklenmeyen dosya türü: {ext}")


ARTICLE_PATTERN = re.compile(r"(madde\s+\d+[a-zçğıöşü]*)", re.IGNORECASE)
ARTICLE_START_PATTERN = re.compile(r"^\s*madde\s+\d+", re.IGNORECASE)
SECTION_PATTERN = re.compile(r"^(bölüm|kısım|başlık)\s+.+", re.IGNORECASE | re.MULTILINE)


@dataclass
class Chunk:
    chunk_index: int
    page_number: Optional[int]
    section: Optional[str]
    article: Optional[str]
    content: str


def _insert_article_breaks(text: str) -> str:
    """PDF metin çıkarımı çoğu zaman paragraflar arasındaki boş satırları
    kaybeder (tek bir akan metin bloğu döner). Bu, "Madde N" başlıklarının
    yanlışlıkla önceki maddeyle aynı chunk'a düşmesine yol açar. Bu yüzden
    her "Madde N" başlangıcından önce açık bir paragraf sınırı ekliyoruz."""
    return re.sub(r"[ \t\n]*(madde\s+\d)", r"\n\n\1", text, flags=re.IGNORECASE)


def _split_into_paragraphs(text: str) -> list[str]:
    text = _insert_article_breaks(text)
    raw = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in raw if p.strip()]
    if paragraphs:
        return paragraphs
    return [line.strip() for line in text.splitlines() if line.strip()]


def _current_article(paragraph: str, last_article: Optional[str]) -> Optional[str]:
    m = ARTICLE_PATTERN.search(paragraph)
    if m:
        return m.group(1).strip().title()
    return last_article


def _current_section(paragraph: str, last_section: Optional[str]) -> Optional[str]:
    m = SECTION_PATTERN.match(paragraph)
    if m:
        return paragraph.strip()[:80]
    return last_section


def chunk_pages(pages: list[PageText]) -> list[Chunk]:
    """Paragraf sınırlarına saygılı, madde/bölüm metadata'sı taşıyan chunking.

    Chunk'lar MAX_CHUNK_CHARS'a yaklaşana kadar paragrafları biriktirir;
    bir paragrafın ortasından bölme yapılmaz. Ardışık chunk'lar arasında
    küçük bir bağlam örtüşmesi (overlap) bırakılır.
    """
    chunks: list[Chunk] = []
    idx = 0
    last_article: Optional[str] = None
    last_section: Optional[str] = None

    for page in pages:
        paragraphs = _split_into_paragraphs(page.text)
        buffer: list[str] = []
        buffer_len = 0
        buffer_article = last_article
        buffer_section = last_section

        def flush(overlap_tail: str = ""):
            nonlocal idx, buffer, buffer_len
            if not buffer:
                return
            content = "\n\n".join(buffer).strip()
            if content:
                chunks.append(
                    Chunk(
                        chunk_index=idx,
                        page_number=page.page_number,
                        section=buffer_section,
                        article=buffer_article,
                        content=content,
                    )
                )
                idx += 1
            buffer = [overlap_tail] if overlap_tail else []
            buffer_len = len(overlap_tail)

        for para in paragraphs:
            starts_new_article = bool(ARTICLE_START_PATTERN.match(para))
            last_article = _current_article(para, last_article)
            last_section = _current_section(para, last_section)

            if starts_new_article and buffer:
                flush()
                buffer_article, buffer_section = last_article, last_section
            elif buffer_len + len(para) > CFG.max_chunk_chars and buffer:
                tail = buffer[-1][-CFG.chunk_overlap_chars:] if CFG.chunk_overlap_chars else ""
                buffer_article, buffer_section = last_article, last_section
                flush(overlap_tail=tail)

            buffer.append(para)
            buffer_len += len(para)
            buffer_article = last_article
            buffer_section = last_section

        flush()

    return chunks

DEFAULT_FOUNDRY_ENDPOINT = "http://localhost:5273/v1"


class FoundryUnavailable(Exception):
    pass


class FoundryClient:
    """Microsoft Foundry Local ile iletişim kuran istemci.

    Foundry Local, çalıştığında OpenAI uyumlu bir local REST endpoint açar.
    Bu istemci önce 'foundry service status' CLI komutuyla gerçek endpoint'i
    bulmaya çalışır (kurulu SDK sürümünden bağımsız, en dayanıklı yöntem);
    bulamazsa .env / varsayılan adrese düşer. Hiçbir cloud API (OpenAI,
    Anthropic, Gemini) çağrılmaz — sadece localhost.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._endpoint: Optional[str] = None
        self._client: Optional["OpenAI"] = None
        self._last_error: Optional[str] = None

   
    def _discover_endpoint_via_cli(self) -> Optional[str]:
        for cmd in (["foundry", "server", "status"], ["foundry", "service", "status"]):
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=5,
                    encoding="utf-8", errors="ignore",
                )
                output = (result.stdout or "") + (result.stderr or "")
                m = re.search(r"https?://[^\s'\"]+", output)
                if m:
                    url = m.group(0).rstrip("/")
                    if not url.endswith("/v1"):
                        url = url + "/v1"
                    return url
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                continue
        return None

    def endpoint(self) -> str:
        if self._endpoint:
            return self._endpoint
        if self.cfg.foundry_endpoint:
            self._endpoint = self.cfg.foundry_endpoint.rstrip("/")
        else:
            self._endpoint = self._discover_endpoint_via_cli() or DEFAULT_FOUNDRY_ENDPOINT
        return self._endpoint

    def client(self) -> "OpenAI":
        if not OPENAI_SDK_AVAILABLE:
            raise FoundryUnavailable("'openai' paketi kurulu değil. 'pip install openai' çalıştırın.")
        if self._client is None:
            self._client = OpenAI(
                base_url=self.endpoint(), api_key=self.cfg.foundry_api_key,
                max_retries=0, timeout=240.0,
            )
        return self._client

    def _list_models_raw(self) -> list[str]:
        """Raises if Foundry Local is unreachable — used for real health checks."""
        models = self.client().models.list()
        return [m.id for m in models.data]

    def list_models(self) -> list[str]:
        try:
            return self._list_models_raw()
        except Exception as e:
            self._last_error = str(e)
            return []

    def resolve_model(self, configured: str, keyword_hint: str) -> str:
        """Config'te model belirtilmemişse, Foundry Local'daki mevcut
        modeller arasından uygun olanı otomatik seçer. Kodun içine tek bir
        model ismi hard-code edilmez."""
        if configured:
            return configured
        available = self.list_models()
        if not available:
            raise FoundryUnavailable(
                "Foundry Local'da yüklü model bulunamadı. "
                "'foundry model run <model-adi>' komutuyla bir model başlatın."
            )
        for m in available:
            if keyword_hint.lower() in m.lower():
                return m
        return available[0]

    def health(self) -> dict:
        try:
            models = self._list_models_raw()
            return {"connected": True, "endpoint": self.endpoint(), "models": models, "error": None}
        except Exception as e:
            return {"connected": False, "endpoint": self.endpoint(), "models": [], "error": str(e)}
        
    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        try:
            resp = self.client().embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            self._last_error = str(e)
            raise FoundryUnavailable(
                f"Embedding modeline ulaşılamadı ({model}). Foundry Local'ın çalıştığından "
                f"ve modelin indirildiğinden emin olun. Detay: {e}"
            ) from e

    def chat(self, model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            resp = self.client().chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            self._last_error = str(e)
            raise FoundryUnavailable(
                f"AI servisine ulaşılamıyor. Lütfen Microsoft Foundry Local'ın çalıştığından "
                f"emin olun. Detay: {e}"
            ) from e


FOUNDRY = FoundryClient(CFG)

def embedding_to_text(vec: list[float]) -> str:
    return json.dumps(vec)


def embedding_from_text(text: str) -> np.ndarray:
    return np.asarray(json.loads(text), dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    norms[norms == 0] = 1e-9
    return (matrix @ query) / norms


def compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def safe_filename(filename: str) -> str:
    name = Path(filename).name 
    name = re.sub(r"[^A-Za-z0-9ÇĞİÖŞÜçğıöşü _.-]", "_", name)
    return name or "belge"


def ingest_document(stored_path: Path, original_filename: str, ext: str) -> dict:
    """PDF/TXT/MD -> extract -> chunk -> embed -> SQLite. Tam pipeline."""
    doc_id = str(uuid.uuid4())
    title = Path(original_filename).stem.replace("_", " ").strip() or original_filename
    file_hash = compute_file_hash(stored_path)

    with db_conn() as conn:
        existing = conn.execute("SELECT id, status FROM documents WHERE file_hash = ?", (file_hash,)).fetchone()
        if existing:
            if existing["status"] == "ready":
                raise ValueError("Bu belge (aynı içerikle) zaten yüklenmiş.")
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (existing["id"],))
            conn.execute("DELETE FROM documents WHERE id = ?", (existing["id"],))
        conn.execute(
            """INSERT INTO documents (id, filename, title, file_type, file_hash, stored_path,
                                        pages, chunk_count, status, uploaded_at)
               VALUES (?, ?, ?, ?, ?, ?, 0, 0, 'processing', ?)""",
            (doc_id, original_filename, title, ext, file_hash, str(stored_path), now_iso()),
        )

    try:
        log.info("Document okunuyor: %s", original_filename)
        pages = extract_pages(stored_path, ext)

        log.info("Chunk'lara ayrılıyor: %s", original_filename)
        chunks = chunk_pages(pages)
        if not chunks:
            raise DocumentReadError("Belgeden hiçbir metin çıkarılamadı.")

        log.info("Embedding oluşturuluyor (%d chunk): %s", len(chunks), original_filename)
        embedding_model = FOUNDRY.resolve_model(CFG.embedding_model, "embedding")
        batch_size = 16
        embeddings: list[list[float]] = []
        for i in range(0, len(chunks), batch_size):
            batch = [c.content for c in chunks[i:i + batch_size]]
            embeddings.extend(FOUNDRY.embed(batch, embedding_model))

        log.info("SQLite'a kaydediliyor: %s", original_filename)
        with db_conn() as conn:
            for c, emb in zip(chunks, embeddings):
                conn.execute(
                    """INSERT INTO chunks (id, document_id, chunk_index, page_number, section,
                                             article, content, embedding, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (str(uuid.uuid4()), doc_id, c.chunk_index, c.page_number, c.section,
                     c.article, c.content, embedding_to_text(emb), now_iso()),
                )
            conn.execute(
                "UPDATE documents SET pages=?, chunk_count=?, status='ready', error=NULL WHERE id=?",
                (len(pages), len(chunks), doc_id),
            )
        log.info("Belge hazır: %s (%d sayfa, %d chunk)", original_filename, len(pages), len(chunks))
        return {"id": doc_id, "pages": len(pages), "chunks": len(chunks)}

    except (DocumentReadError, FoundryUnavailable, Exception) as e:
        err_msg = str(e)
        with db_conn() as conn:
            conn.execute("UPDATE documents SET status='error', error=? WHERE id=?", (err_msg, doc_id))
        log.error("Belge işlenemedi (%s): %s", original_filename, err_msg)
        raise


def reindex_all_documents():
    """Tüm belgeleri mevcut chunking/embedding ayarlarıyla yeniden işler."""
    with db_conn() as conn:
        docs = conn.execute("SELECT * FROM documents").fetchall()

    results = []
    for doc in docs:
        stored_path = Path(doc["stored_path"])
        if not stored_path.exists():
            results.append({"id": doc["id"], "status": "missing_file"})
            continue
        with db_conn() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc["id"],))
            conn.execute("UPDATE documents SET status='processing', error=NULL WHERE id=?", (doc["id"],))
        try:
            pages = extract_pages(stored_path, doc["file_type"])
            chunks = chunk_pages(pages)
            embedding_model = FOUNDRY.resolve_model(CFG.embedding_model, "embedding")
            embeddings = []
            for i in range(0, len(chunks), 16):
                batch = [c.content for c in chunks[i:i + 16]]
                embeddings.extend(FOUNDRY.embed(batch, embedding_model))
            with db_conn() as conn:
                for c, emb in zip(chunks, embeddings):
                    conn.execute(
                        """INSERT INTO chunks (id, document_id, chunk_index, page_number, section,
                                                 article, content, embedding, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (str(uuid.uuid4()), doc["id"], c.chunk_index, c.page_number, c.section,
                         c.article, c.content, embedding_to_text(emb), now_iso()),
                    )
                conn.execute(
                    "UPDATE documents SET pages=?, chunk_count=?, status='ready' WHERE id=?",
                    (len(pages), len(chunks), doc["id"]),
                )
            results.append({"id": doc["id"], "status": "ready", "chunks": len(chunks)})
        except Exception as e:
            with db_conn() as conn:
                conn.execute("UPDATE documents SET status='error', error=? WHERE id=?", (str(e), doc["id"]))
            results.append({"id": doc["id"], "status": "error", "error": str(e)})
    return results


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_title: str
    page_number: Optional[int]
    section: Optional[str]
    article: Optional[str]
    content: str
    similarity: float


def retrieve(question: str) -> list[RetrievedChunk]:
    with db_conn() as conn:
        rows = conn.execute(
            """SELECT c.id, c.document_id, d.title as document_title, c.page_number,
                      c.section, c.article, c.content, c.embedding
               FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE d.status = 'ready'"""
        ).fetchall()

    if not rows:
        return []

    embedding_model = FOUNDRY.resolve_model(CFG.embedding_model, "embedding")
    query_vec = np.asarray(FOUNDRY.embed([question], embedding_model)[0], dtype=np.float32)

    matrix = np.stack([embedding_from_text(r["embedding"]) for r in rows])
    sims = cosine_similarity_matrix(query_vec, matrix)

    scored = []
    for row, sim in zip(rows, sims):
        scored.append(RetrievedChunk(
            chunk_id=row["id"], document_id=row["document_id"], document_title=row["document_title"],
            page_number=row["page_number"], section=row["section"], article=row["article"],
            content=row["content"], similarity=float(sim),
        ))
    scored.sort(key=lambda x: x.similarity, reverse=True)

    above_threshold = [c for c in scored if c.similarity >= CFG.similarity_threshold]
    return above_threshold[:CFG.top_k]

SYSTEM_PROMPT = """Sen Kampüsce adlı bir üniversite yönetmelik doğrulama asistanısın.

KURALLARIN (kesinlikle uy):
1. SADECE sana verilen SOURCE bloklarındaki bilgiyi kullan. Kendi genel bilgini KULLANMA.
2. Context dışında hiçbir bilgi uydurma. Tahminde bulunma.
3. Cevap kaynaklarda açık ve net şekilde varsa "verified" de.
4. Kaynaklarda konuyla ilgili bilgi var ama soruya net cevap vermeye yetmiyorsa "partial" de.
5. Hiçbir kaynak soruyla alakalı değilse "not_found" de ve answer alanına bunu belirt.
6. Kullanılan her kaynağı (Document adı, Article/Madde, Page) sources listesinde belirt.
7. Birden fazla kaynak varsa hepsini ayrı ayrı listele.
8. Kaynaklar birbiriyle çelişiyorsa bunu answer içinde açıkça belirt, gizleme.
9. Türkçe soruya Türkçe cevap ver. Cevabı kısa, net ve öğrenci dostu yaz.
10. Sadece geçerli JSON döndür. Başka hiçbir metin, açıklama veya markdown ekleme.

Çıktı formatı (yalnızca bu JSON, başka hiçbir şey yazma):
{
  "answer": "kısa ve net cevap metni",
  "verification_status": "verified" | "partial" | "not_found",
  "sources": [
    {"document": "belge adı", "article": "Madde X veya null", "page": sayfa_no_veya_null}
  ]
}"""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"SOURCE {i}\n"
            f"Document: {c.document_title}\n"
            f"Page: {c.page_number if c.page_number else 'N/A'}\n"
            f"Article: {c.article if c.article else 'N/A'}\n\n"
            f"Content:\n{c.content}\n"
        )
    return "\n---\n".join(parts)


def extract_json_block(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    while start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            return obj
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
    return None


GREETING_PATTERN = re.compile(
    r"^\s*(merhaba|selam|hey|hi|hello|nasılsın|günaydın|iyi günler)\s*[!.?]*\s*$",
    re.IGNORECASE,
)


def run_rag(question: str) -> dict:
    """Tam RAG pipeline. Test 5'e göre ('Merhaba') gereksiz retrieval yapmaz."""
    question = question.strip()

    if not question:
        return {
            "answer": "Lütfen bir soru yazın. Boş sorgu gönderilemez.",
            "verification_status": "not_found",
            "confidence": 0.0,
            "sources": [],
            "retrieved_chunks": 0,
            "top_similarity": 0.0,
            "is_validation_error": True,
        }

    if GREETING_PATTERN.match(question):
        return {
            "answer": "Merhaba! Üniversiten hakkındaki yönetmelik, staj, sınav, burs veya "
                      "akademik takvim gibi konularda sana yardımcı olabilirim. Ne öğrenmek istersin?",
            "verification_status": "verified",
            "confidence": 1.0,
            "sources": [],
            "retrieved_chunks": 0,
            "top_similarity": 0.0,
        }

    with db_conn() as conn:
        doc_count = conn.execute("SELECT COUNT(*) c FROM documents WHERE status='ready'").fetchone()["c"]
    if doc_count == 0:
        return {
            "answer": "Bilgi tabanında henüz hiç belge bulunmuyor. Lütfen önce "
                      "'Belgeler' sayfasından bir yönetmelik/yönerge yükleyin.",
            "verification_status": "not_found",
            "confidence": 0.0,
            "sources": [],
            "retrieved_chunks": 0,
            "top_similarity": 0.0,
        }

    chunks = retrieve(question)

    if not chunks:
        return {
            "answer": "Bu konu hakkında mevcut bilgi tabanımda yeterli bilgi bulunmamaktadır.",
            "verification_status": "not_found",
            "confidence": 0.0,
            "sources": [],
            "retrieved_chunks": 0,
            "top_similarity": 0.0,
        }

    top_similarity = chunks[0].similarity
    context = build_context_block(chunks)
    user_prompt = f"SORU: {question}\n\n{context}\n\nYalnızca yukarıdaki SOURCE'ları kullanarak JSON formatında cevap ver."

    llm_model = FOUNDRY.resolve_model(CFG.llm_model, "phi")
    raw = FOUNDRY.chat(llm_model, SYSTEM_PROMPT, user_prompt, CFG.temperature)
    parsed = extract_json_block(raw)

    if parsed is None:
        log.warning("LLM cevabı JSON olarak ayrıştırılamadı, güvenli fallback kullanılıyor.")
        return {
            "answer": raw.strip()[:800] if raw.strip() else "Cevap oluşturulamadı.",
            "verification_status": "partial",
            "confidence": round(top_similarity, 2),
            "sources": [
                {"document": c.document_title, "article": c.article, "page": c.page_number,
                 "relevance": round(c.similarity, 2)}
                for c in chunks
            ],
            "retrieved_chunks": len(chunks),
            "top_similarity": round(top_similarity, 2),
            "llm_model": llm_model,
            "embedding_model": CFG.embedding_model,
            "json_fallback": True,
        }

    status = parsed.get("verification_status", "partial")
    if status not in ("verified", "partial", "not_found"):
        status = "partial"

    sources_out = []
    for i, s in enumerate(parsed.get("sources", []) or []):
        rel = chunks[i].similarity if i < len(chunks) else top_similarity
        sources_out.append({
            "document": s.get("document") or (chunks[i].document_title if i < len(chunks) else "?"),
            "article": s.get("article"),
            "page": s.get("page"),
            "relevance": round(rel, 2),
        })
    if not sources_out and status != "not_found":
        sources_out = [
            {"document": c.document_title, "article": c.article, "page": c.page_number,
             "relevance": round(c.similarity, 2)}
            for c in chunks
        ]

    return {
        "answer": parsed.get("answer", "").strip() or "Cevap oluşturulamadı.",
        "verification_status": status,
        "confidence": round(top_similarity, 2),
        "sources": sources_out,
        "retrieved_chunks": len(chunks),
        "top_similarity": round(top_similarity, 2),
        "llm_model": llm_model,
        "embedding_model": CFG.embedding_model,
    }

app = FastAPI(title="Kampüsce", version="1.0.0")


class ChatRequest(BaseModel):
    question: str


class SettingsUpdate(BaseModel):
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    temperature: Optional[float] = None


def doc_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "filename": row["filename"],
        "title": row["title"],
        "file_type": row["file_type"],
        "pages": row["pages"],
        "chunk_count": row["chunk_count"],
        "status": row["status"],
        "error": row["error"],
        "uploaded_at": row["uploaded_at"],
    }


@app.on_event("startup")
def on_startup():
    init_db()
    health = FOUNDRY.health()
    if health["connected"]:
        log.info("Foundry Local bağlı: %s | modeller: %s", health["endpoint"], health["models"])
    else:
        log.warning("Foundry Local'a şu an ulaşılamıyor (%s). Uygulama açık kalacak, "
                     "ancak sohbet/embedding istekleri Foundry Local çalışana kadar hata verecek.",
                     health.get("error"))

@app.get("/api/health")
def api_health():
    health = FOUNDRY.health()
    with db_conn() as conn:
        doc_count = conn.execute("SELECT COUNT(*) c FROM documents WHERE status='ready'").fetchone()["c"]
    return {
        "status": "ok" if health["connected"] else "degraded",
        "foundry_local": health["connected"],
        "foundry_endpoint": health["endpoint"],
        "foundry_error": health["error"],
        "llm_model": CFG.llm_model or (health["models"][0] if health["models"] else None),
        "embedding_model": CFG.embedding_model,
        "available_models": health["models"],
        "database": CFG.db_abs().exists(),
        "documents": doc_count,
    }


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    question = (req.question or "").strip()
    query_id = str(uuid.uuid4())

    try:
        result = run_rag(question)
    except FoundryUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("RAG pipeline hatası")
        raise HTTPException(status_code=500, detail=f"Beklenmeyen bir hata oluştu: {e}")

    if not result.get("is_validation_error"):
        with db_conn() as conn:
            conn.execute("INSERT INTO queries (id, question, created_at) VALUES (?, ?, ?)",
                         (query_id, question, now_iso()))
            conn.execute(
                """INSERT INTO answers (id, query_id, answer, verification_status, confidence,
                                          sources_json, retrieved_chunks, top_similarity,
                                          llm_model, embedding_model, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), query_id, result["answer"], result["verification_status"],
                 result["confidence"], json.dumps(result.get("sources", []), ensure_ascii=False),
                 result.get("retrieved_chunks", 0), result.get("top_similarity", 0.0),
                 result.get("llm_model"), result.get("embedding_model"), now_iso()),
            )

    result["query_id"] = query_id
    return result


@app.get("/api/documents")
def api_list_documents():
    with db_conn() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
    return {"documents": [doc_to_dict(r) for r in rows]}


@app.post("/api/documents/upload")
async def api_upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in CFG.allowed_extensions:
        raise HTTPException(400, f"Desteklenmeyen dosya türü '{ext}'. İzin verilenler: {', '.join(CFG.allowed_extensions)}")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > CFG.max_upload_mb:
        raise HTTPException(400, f"Dosya çok büyük ({size_mb:.1f} MB). Limit: {CFG.max_upload_mb} MB.")

    safe_name = safe_filename(file.filename or "belge")
    unique_name = f"{uuid.uuid4().hex[:10]}_{safe_name}"
    stored_path = (CFG.docs_abs() / unique_name).resolve()
    if CFG.docs_abs().resolve() not in stored_path.parents:
        raise HTTPException(400, "Geçersiz dosya yolu.")

    stored_path.write_bytes(contents)

    try:
        info = ingest_document(stored_path, file.filename or safe_name, ext)
        return {"status": "ready", **info}
    except ValueError as e:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(400, str(e))
    except DocumentReadError as e:
        raise HTTPException(422, str(e))
    except FoundryUnavailable as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.exception("Belge yükleme hatası")
        raise HTTPException(500, f"Belge işlenirken beklenmeyen bir hata oluştu: {e}")


@app.delete("/api/documents/{doc_id}")
def api_delete_document(doc_id: str):
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Belge bulunamadı.")
        stored_path = Path(row["stored_path"])
        conn.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    if stored_path.exists():
        stored_path.unlink(missing_ok=True)
    return {"status": "deleted", "id": doc_id}


@app.post("/api/documents/reindex")
def api_reindex():
    try:
        results = reindex_all_documents()
        return {"status": "done", "results": results}
    except Exception as e:
        log.exception("Reindex hatası")
        raise HTTPException(500, f"Reindex sırasında hata oluştu: {e}")


@app.get("/api/history")
def api_history(limit: int = 100):
    with db_conn() as conn:
        rows = conn.execute(
            """SELECT q.id as query_id, q.question, q.created_at,
                      a.answer, a.verification_status, a.confidence, a.sources_json,
                      a.retrieved_chunks, a.top_similarity, a.llm_model, a.embedding_model
               FROM queries q LEFT JOIN answers a ON a.query_id = q.id
               ORDER BY q.created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    items = []
    for r in rows:
        items.append({
            "query_id": r["query_id"],
            "question": r["question"],
            "created_at": r["created_at"],
            "answer": r["answer"],
            "verification_status": r["verification_status"],
            "confidence": r["confidence"],
            "sources": json.loads(r["sources_json"]) if r["sources_json"] else [],
            "retrieved_chunks": r["retrieved_chunks"],
            "top_similarity": r["top_similarity"],
            "llm_model": r["llm_model"],
            "embedding_model": r["embedding_model"],
        })
    return {"history": items}


@app.get("/api/stats")
def api_stats():
    with db_conn() as conn:
        doc_count = conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
        ready_count = conn.execute("SELECT COUNT(*) c FROM documents WHERE status='ready'").fetchone()["c"]
        chunk_count = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
        query_count = conn.execute("SELECT COUNT(*) c FROM queries").fetchone()["c"]
        last_doc = conn.execute("SELECT uploaded_at FROM documents ORDER BY uploaded_at DESC LIMIT 1").fetchone()
        status_breakdown = conn.execute(
            "SELECT verification_status, COUNT(*) c FROM answers GROUP BY verification_status"
        ).fetchall()
    return {
        "documents": doc_count,
        "documents_ready": ready_count,
        "chunks": chunk_count,
        "queries": query_count,
        "last_update": last_doc["uploaded_at"] if last_doc else None,
        "status_breakdown": {r["verification_status"]: r["c"] for r in status_breakdown},
    }

@app.get("/api/settings")
def api_get_settings():
    return {
        "llm_model": CFG.llm_model or "(otomatik seçilecek)",
        "embedding_model": CFG.embedding_model,
        "top_k": CFG.top_k,
        "similarity_threshold": CFG.similarity_threshold,
        "temperature": CFG.temperature,
        "database_path": str(CFG.db_abs()),
        "documents_path": str(CFG.docs_abs()),
        "foundry_endpoint": FOUNDRY.endpoint(),
    }


@app.post("/api/settings")
def api_update_settings(update: SettingsUpdate):
    if update.top_k is not None:
        if not (1 <= update.top_k <= 10):
            raise HTTPException(400, "TOP_K 1 ile 10 arasında olmalıdır.")
        CFG.top_k = update.top_k
    if update.similarity_threshold is not None:
        if not (0.0 <= update.similarity_threshold <= 1.0):
            raise HTTPException(400, "Similarity threshold 0 ile 1 arasında olmalıdır.")
        CFG.similarity_threshold = update.similarity_threshold
    if update.temperature is not None:
        if not (0.0 <= update.temperature <= 1.0):
            raise HTTPException(400, "Temperature 0 ile 1 arasında olmalıdır.")
        CFG.temperature = update.temperature
    return api_get_settings()


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=FRONTEND_HTML)




FRONTEND_HTML = """<!DOCTYPE html>
<html lang="tr" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kampüsce AI — Sorunu sor, üniversitenin kaynağından öğren.</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  /* --- Kurumsal (lacivert / beyaz / altın) --- */
  --bg:#F3F5F9; --surface:#FFFFFF; --surface-2:#EEF1F7; --surface-hover:#E4E9F3;
  --ink:#111827; --ink-dim:#5B6472; --ink-faint:#8B93A1;
  --border:#E2E6EF; --border-strong:#CBD3E1;
  --sidebar-bg:#0F2A6B; --sidebar-ink:#DCE4F7; --sidebar-ink-dim:#8CA0D6; --sidebar-active:#173B8C;
  --navy:#0F2A6B; --navy-ink:#FFFFFF;
  --gold:#E7A93A; --gold-ink:#1B2340; --gold-soft:#FBEFD7;
  --verified:#1E8A4C; --verified-soft:#E4F5EA;
  --partial:#B07C0A; --partial-soft:#FBF0D7;
  --notfound:#D23C3C; --notfound-soft:#FBE7E7;
  --radius-s:6px; --radius-m:10px; --radius-l:14px;
  --shadow:0 1px 2px rgba(15,23,42,.04), 0 8px 20px -12px rgba(15,23,42,.14);
  --font-display:'Inter', sans-serif; --font-body:'Inter', sans-serif; --font-mono:'IBM Plex Mono', monospace;
}
[data-theme="dark"]{
  --bg:#0B1220; --surface:#121A2E; --surface-2:#16204A; --surface-hover:#1B2750;
  --ink:#EAF0FF; --ink-dim:#9AA6C7; --ink-faint:#6B7699;
  --border:#22304F; --border-strong:#324268;
  --sidebar-bg:#0A1F52; --sidebar-ink:#DCE4F7; --sidebar-ink-dim:#7C8FBF; --sidebar-active:#123068;
  --navy:#173B8C; --navy-ink:#FFFFFF;
  --gold:#E7A93A; --gold-ink:#1B2340; --gold-soft:#2A2210;
  --verified:#3CC17E; --verified-soft:#0F2A1D;
  --partial:#E3A83B; --partial-soft:#2B2210;
  --notfound:#F0666C; --notfound-soft:#2E1518;
  --shadow:0 1px 2px rgba(0,0,0,.3), 0 12px 28px -14px rgba(0,0,0,.55);
}
/* eski değişken adlarının geri kalan CSS/JS ile uyumlu kalması için eşliyoruz */
:root, [data-theme="dark"]{ --accent:var(--gold); --accent-2:var(--navy); --accent-soft:var(--gold-soft); }
*{box-sizing:border-box;}
html,body{margin:0;padding:0;height:100%;}
body{
  font-family:var(--font-body); background:var(--bg); color:var(--ink);
  -webkit-font-smoothing:antialiased; overflow:hidden;
}
button{font-family:inherit; cursor:pointer;}
input,textarea,select{font-family:inherit;}
::selection{background:var(--gold-soft);}
.mono{font-family:var(--font-mono);}

.app-shell{display:flex; height:100vh; width:100vw;}

/* ---------- SIDEBAR — kurumsal lacivert panel ---------- */
.sidebar{
  width:258px; background:var(--sidebar-bg); color:var(--sidebar-ink);
  display:flex; flex-direction:column; flex-shrink:0; transition:width .18s ease;
  border-right:1px solid rgba(255,255,255,.06);
}
.sidebar.collapsed{width:76px;}
.sidebar-head{display:flex; align-items:center; gap:10px; padding:22px 18px 18px 22px; border-bottom:3px solid var(--gold); margin-bottom:10px;}
.sidebar-head .seal-mark{width:32px;height:32px;flex-shrink:0;}
.sidebar-head .wordmark{font-family:var(--font-display); font-weight:800; font-size:18px; color:#fff; letter-spacing:.2px; white-space:nowrap;}
.sidebar.collapsed .wordmark, .sidebar.collapsed .word-sub{display:none;}
.sidebar-head .word-sub{font-size:9.5px; color:var(--gold); text-transform:uppercase; letter-spacing:1.6px; margin-top:2px; font-weight:700;}
.sidebar-collapse-btn{
  margin-left:auto; background:transparent; border:none; color:var(--sidebar-ink-dim);
  width:26px;height:26px; border-radius:7px; display:flex;align-items:center;justify-content:center;
}
.sidebar-collapse-btn:hover{background:rgba(255,255,255,.08); color:#fff;}


.nav{display:flex; flex-direction:column; gap:2px; padding:8px 12px; flex:1; overflow-y:auto;}
.nav-item{
  display:flex; align-items:center; gap:12px; padding:10px 12px; border-radius:8px;
  color:var(--sidebar-ink-dim); background:transparent; border:none; text-align:left; font-size:14px; font-weight:600;
  white-space:nowrap; overflow:hidden; border-left:3px solid transparent;
}
.nav-item svg{flex-shrink:0; width:18px;height:18px;}
.nav-item:hover{background:rgba(255,255,255,.06); color:#fff;}
.nav-item.active{background:rgba(255,255,255,.08); color:#fff; border-left:3px solid var(--gold);}
.nav-item.active svg{color:var(--gold);}
.sidebar.collapsed .nav-item span{display:none;}
.sidebar.collapsed .nav-item{justify-content:center;}

.sidebar-foot{padding:14px 18px 18px; border-top:1px solid rgba(255,255,255,.08);}
.status-pill{display:flex; align-items:center; gap:8px; font-size:12px; color:var(--sidebar-ink-dim); font-weight:600;}
.status-dot{width:7px;height:7px;border-radius:50%; background:var(--ink-faint); flex-shrink:0;}
.status-dot.on{background:#3CC17E; box-shadow:0 0 0 3px rgba(60,193,126,.18);}
.status-dot.off{background:#F0666C; box-shadow:0 0 0 3px rgba(240,102,108,.18);}
.sidebar.collapsed .status-pill span{display:none;}

/* ---------- MAIN ---------- */
.main{flex:1; display:flex; flex-direction:column; min-width:0; background:var(--bg);}
.topbar{
  display:flex; align-items:center; justify-content:space-between; padding:16px 28px;
  border-bottom:1px solid var(--border); flex-shrink:0; gap:16px; background:var(--surface);
}
.topbar h1{font-family:var(--font-display); font-size:19px; font-weight:800; margin:0; color:var(--navy);}
.topbar-right{display:flex; align-items:center; gap:10px;}
.icon-btn{
  width:36px;height:36px; border-radius:8px; border:1px solid var(--border); background:var(--surface);
  display:flex; align-items:center; justify-content:center; color:var(--ink-dim);
}
.icon-btn:hover{background:var(--surface-hover); color:var(--ink);}
.icon-btn svg{width:17px;height:17px;}

.view{display:none; flex:1; overflow-y:auto; padding:28px; }
.view.active{display:block;}
.container{max-width:880px; margin:0 auto;}

/* ---------- HERO / QUERY ---------- */
.hero{padding:34px 0 10px;}
.hero-eyebrow{
  font-family:var(--font-mono); font-size:11px; letter-spacing:1.5px; text-transform:uppercase;
  color:var(--navy); background:var(--gold-soft); padding:5px 10px; border-radius:100px;
  margin-bottom:14px; display:inline-flex; align-items:center; gap:8px; font-weight:700;
}
.hero h2{font-family:var(--font-display); font-size:30px; line-height:1.3; font-weight:800; margin:0 0 8px; color:var(--navy);}
[data-theme="dark"] .hero h2{color:var(--ink);}
.hero p.tagline{color:var(--ink-dim); font-size:15px; margin:0 0 26px; max-width:520px;}

.query-box{
  background:var(--surface); border:1.5px solid var(--border-strong); border-radius:var(--radius-m);
  padding:6px 6px 6px 18px; display:flex; align-items:flex-end; gap:10px; box-shadow:var(--shadow);
}
.query-box:focus-within{border-color:var(--navy);}
.query-box textarea{
  flex:1; border:none; outline:none; resize:none; background:transparent; color:var(--ink);
  font-size:15.5px; line-height:1.5; padding:12px 0; max-height:160px; min-height:26px;
}
.query-box textarea::placeholder{color:var(--ink-faint);}
.send-btn{
  background:var(--gold); color:var(--gold-ink); border:none; border-radius:8px; width:44px;height:44px;
  display:flex;align-items:center;justify-content:center; flex-shrink:0; transition:transform .1s ease, opacity .15s;
}
.send-btn:hover{filter:brightness(1.06);}
.send-btn:disabled{opacity:.4; cursor:default; filter:none;}
.send-btn svg{width:18px;height:18px;}

.examples{display:flex; flex-wrap:wrap; gap:8px; margin-top:16px;}
.example-chip{
  background:var(--surface); border:1px solid var(--border); color:var(--ink-dim); font-size:13px; font-weight:600;
  padding:8px 14px; border-radius:var(--radius-s); display:flex; align-items:center; gap:7px;
}
.example-chip:hover{border-color:var(--navy); color:var(--navy); background:var(--surface-2);}
[data-theme="dark"] .example-chip:hover{color:#fff;}

.status-card{
  margin-top:22px; background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-m);
  padding:16px; display:flex; align-items:center; gap:14px; box-shadow:var(--shadow);
}
.status-card .grid4{display:grid; grid-template-columns:repeat(4,1fr); gap:0; flex:1;}
.status-row{display:flex; align-items:center; gap:8px; font-size:12.5px; color:var(--ink-dim); padding:4px 14px; border-left:1px solid var(--border);}
.status-row:first-child{border-left:none;}

/* ---------- CONVERSATION / SEAL CARDS ---------- */
.thread{display:flex; flex-direction:column; gap:18px; margin-top:26px;}
.q-bubble{
  align-self:flex-end; background:var(--sidebar-active); color:#fff; padding:11px 18px; border-radius:16px 16px 4px 16px;
  max-width:75%; font-size:14.5px; line-height:1.5;
}
[data-theme="dark"] .q-bubble{background:var(--accent-soft); color:var(--ink);}

.answer-card{
  background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-l);
  overflow:hidden; box-shadow:var(--shadow); animation:riseIn .35s ease;
}
@keyframes riseIn{from{opacity:0; transform:translateY(6px);} to{opacity:1; transform:none;}}
.answer-head{
  display:flex; align-items:center; gap:12px; padding:16px 20px; border-bottom:1px solid var(--border);
}
.answer-head.verified{background:var(--verified-soft);}
.answer-head.partial{background:var(--partial-soft);}
.answer-head.not_found{background:var(--notfound-soft);}
.seal-icon{width:34px;height:34px; flex-shrink:0;}
.answer-head .badge-text{font-weight:700; font-size:13.5px; letter-spacing:.3px;}
.answer-head.verified .badge-text{color:var(--verified);}
.answer-head.partial .badge-text{color:var(--partial);}
.answer-head.not_found .badge-text{color:var(--notfound);}
.answer-head .badge-sub{font-size:12px; color:var(--ink-dim); margin-top:1px;}

.answer-body{padding:20px;}
.answer-text{font-size:15.5px; line-height:1.65; color:var(--ink); white-space:pre-wrap;}

.confidence-wrap{margin-top:18px;}
.confidence-label{display:flex; justify-content:space-between; font-size:12px; color:var(--ink-dim); margin-bottom:6px;}
.confidence-label .tip{cursor:help; border-bottom:1px dotted var(--ink-faint);}
.confidence-bar{height:7px; border-radius:100px; background:var(--surface-2); overflow:hidden;}
.confidence-fill{height:100%; border-radius:100px; transition:width .5s ease;}
.confidence-fill.verified{background:var(--verified);}
.confidence-fill.partial{background:var(--partial);}
.confidence-fill.not_found{background:var(--notfound);}

.sources-title{font-size:12px; text-transform:uppercase; letter-spacing:1px; color:var(--ink-faint); margin:22px 0 10px; font-family:var(--font-mono);}
.source-list{display:flex; flex-direction:column; gap:8px;}
.source-card{
  border:1px solid var(--border); border-radius:12px; padding:12px 14px; display:flex; align-items:center; gap:12px;
  background:var(--surface-2)/1;
}
.source-card .doc-icon{width:30px;height:30px; border-radius:8px; background:var(--accent-soft); color:var(--accent); display:flex;align-items:center;justify-content:center; flex-shrink:0;}
.source-card .doc-icon svg{width:15px;height:15px;}
.source-meta{flex:1; min-width:0;}
.source-meta .doc-name{font-size:13.5px; font-weight:600; color:var(--ink);}
.source-meta .doc-loc{font-size:11.5px; color:var(--ink-dim); font-family:var(--font-mono); margin-top:2px;}
.source-relevance{font-family:var(--font-mono); font-size:12.5px; color:var(--ink-dim); text-align:right; flex-shrink:0;}
.source-relevance b{color:var(--ink); font-size:14px;}

.tech-details{margin-top:16px;}
.tech-toggle{
  font-size:12px; color:var(--ink-dim); background:none; border:none; display:flex; align-items:center; gap:5px; padding:4px 0;
  font-family:var(--font-mono);
}
.tech-toggle svg{width:12px;height:12px; transition:transform .15s;}
.tech-toggle.open svg{transform:rotate(90deg);}
.tech-panel{display:none; margin-top:8px; background:var(--surface-2); border-radius:10px; padding:12px 14px; font-family:var(--font-mono); font-size:12px; color:var(--ink-dim); line-height:1.8;}
.tech-panel.open{display:block;}

/* ---------- EMPTY / LOADING ---------- */
.empty-state{text-align:center; padding:70px 20px; color:var(--ink-dim);}
.empty-state svg{width:46px;height:46px; margin-bottom:16px; color:var(--ink-faint);}
.empty-state h3{font-family:var(--font-display); font-size:19px; color:var(--ink); margin:0 0 8px;}
.empty-state p{font-size:14px; margin:0 0 18px;}

.loading-card{
  background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-l); padding:22px 24px;
  display:flex; flex-direction:column; gap:10px; box-shadow:var(--shadow);
}
.loading-step{display:flex; align-items:center; gap:10px; font-size:13.5px; color:var(--ink-faint); transition:color .2s;}
.loading-step.active{color:var(--ink);}
.loading-step .dot{width:16px;height:16px; border-radius:50%; border:2px solid var(--border); flex-shrink:0; position:relative;}
.loading-step.active .dot{border-color:var(--accent); border-top-color:transparent; animation:spin .7s linear infinite;}
.loading-step.done .dot{border-color:var(--verified); background:var(--verified);}
@keyframes spin{to{transform:rotate(360deg);}}

/* ---------- CARDS / BUTTONS (generic) ---------- */
.btn{
  display:inline-flex; align-items:center; gap:8px; padding:10px 18px; border-radius:var(--radius-s); font-size:13.5px; font-weight:700;
  border:1.5px solid var(--border-strong); background:var(--surface); color:var(--ink); letter-spacing:.1px;
}
.btn:hover{background:var(--surface-hover); border-color:var(--navy);}
.btn.primary{background:var(--navy); border-color:var(--navy); color:#fff;}
.btn.primary:hover{filter:brightness(1.15); border-color:var(--navy);}
.btn.danger-ghost{color:var(--notfound); border-color:transparent; background:transparent; padding:6px 10px; font-size:13px;}
.btn.danger-ghost:hover{background:var(--notfound-soft); border-color:transparent;}
.btn svg{width:15px;height:15px;}

/* ---------- DOCUMENTS VIEW ---------- */
.upload-zone{
  border:2px dashed var(--border-strong); border-radius:var(--radius-l); padding:44px 20px; text-align:center;
  background:var(--surface); transition:border-color .15s, background .15s; cursor:pointer;
}
.upload-zone:hover, .upload-zone.dragover{border-color:var(--accent); background:var(--accent-soft);}
.upload-zone svg{width:34px;height:34px; color:var(--accent); margin-bottom:12px;}
.upload-zone h3{font-family:var(--font-display); font-size:17px; margin:0 0 6px;}
.upload-zone p{font-size:13px; color:var(--ink-dim); margin:0 0 16px;}

.doc-grid{display:flex; flex-direction:column; gap:10px; margin-top:22px;}
.doc-row{
  display:flex; align-items:center; gap:14px; padding:14px 16px; background:var(--surface); border:1px solid var(--border);
  border-radius:var(--radius-m);
}
.doc-row .doc-icon-lg{width:38px;height:38px; border-radius:10px; background:var(--accent-soft); color:var(--accent); display:flex;align-items:center;justify-content:center; flex-shrink:0;}
.doc-row .doc-icon-lg svg{width:18px;height:18px;}
.doc-info{flex:1; min-width:0;}
.doc-info .name{font-weight:600; font-size:14.5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.doc-info .meta{font-size:12px; color:var(--ink-dim); margin-top:3px; font-family:var(--font-mono);}
.doc-badge{font-size:11px; font-weight:700; padding:4px 10px; border-radius:100px; text-transform:uppercase; letter-spacing:.4px; flex-shrink:0;}
.doc-badge.ready{background:var(--verified-soft); color:var(--verified);}
.doc-badge.processing{background:var(--partial-soft); color:var(--partial);}
.doc-badge.error{background:var(--notfound-soft); color:var(--notfound);}
.doc-actions{display:flex; gap:4px; flex-shrink:0;}

/* ---------- HISTORY ---------- */
.history-group{margin-bottom:22px;}
.history-group h4{font-size:12px; text-transform:uppercase; letter-spacing:1px; color:var(--ink-faint); font-family:var(--font-mono); margin:0 0 10px;}
.history-item{
  display:flex; align-items:center; gap:12px; padding:13px 16px; background:var(--surface); border:1px solid var(--border);
  border-radius:12px; margin-bottom:8px;
}
.history-item .seal-icon{width:24px;height:24px;flex-shrink:0;}
.history-item .q-text{flex:1; font-size:14px; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.history-item .q-time{font-size:11.5px; color:var(--ink-faint); font-family:var(--font-mono); flex-shrink:0;}

/* ---------- SETTINGS ---------- */
.settings-grid{display:flex; flex-direction:column; gap:14px;}
.settings-row{
  display:flex; align-items:center; justify-content:space-between; gap:20px; padding:16px 18px; background:var(--surface);
  border:1px solid var(--border); border-radius:var(--radius-m);
}
.settings-row .label{font-weight:600; font-size:14px;}
.settings-row .desc{font-size:12.5px; color:var(--ink-dim); margin-top:2px; max-width:400px;}
.settings-row .value{font-family:var(--font-mono); font-size:13px; color:var(--ink-dim); flex-shrink:0;}
.settings-row input[type=range]{width:140px;}
.settings-row input[type=number]{
  width:80px; padding:7px 10px; border-radius:8px; border:1px solid var(--border); background:var(--bg); color:var(--ink); font-family:var(--font-mono);
}

/* ---------- ABOUT ---------- */
.about-hero{text-align:center; padding:36px 20px;}
.about-hero .seal-mark-lg{width:64px;height:64px; margin:0 auto 18px;}
.about-hero h2{font-family:var(--font-display); font-size:26px; margin:0 0 10px;}
.about-hero .slogan{font-family:var(--font-display); font-style:italic; font-size:16px; color:var(--accent); margin-bottom:18px;}
.pipeline{display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-top:24px;}
.pipeline .step{background:var(--surface); border:1px solid var(--border); padding:9px 14px; border-radius:100px; font-size:12.5px; font-family:var(--font-mono); color:var(--ink-dim);}
.pipeline .arrow{color:var(--ink-faint); align-self:center;}

.toast{
  position:fixed; bottom:22px; left:50%; transform:translateX(-50%) translateY(20px); background:var(--ink); color:var(--bg);
  padding:12px 20px; border-radius:12px; font-size:13.5px; opacity:0; pointer-events:none; transition:all .25s ease; z-index:999;
  box-shadow:0 10px 30px rgba(0,0,0,.25);
}
.toast.show{opacity:1; transform:translateX(-50%) translateY(0);}

@media (max-width:820px){
  .sidebar{position:fixed; z-index:50; height:100vh;}
  .sidebar:not(.mobile-open){transform:translateX(-100%);}
  .main{width:100%;}
  .status-card .grid4{grid-template-columns:repeat(2,1fr); gap:6px;}
  .status-row{border-left:none;}
}
</style>
</head>
<body>
<div class="app-shell">

  <aside class="sidebar" id="sidebar">
    <div class="sidebar-head">
      <div class="seal-mark" id="sealMarkSlot"></div>
      <div>
        <div class="wordmark">Kampüsce</div>
        <div class="word-sub">Bilgi Doğrulama Sistemi</div>
      </div>
      <button class="sidebar-collapse-btn" onclick="toggleSidebar()" title="Daralt/Genişlet">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 18l-6-6 6-6"/></svg>
      </button>
    </div>
    <nav class="nav" id="navList"></nav>
    <div class="sidebar-foot">
      <div class="status-pill">
        <span class="status-dot" id="sidebarStatusDot"></span>
        <span id="sidebarStatusText">Bağlantı kontrol ediliyor…</span>
      </div>
    </div>
  </aside>

  <main class="main">
    <div class="topbar">
      <h1 id="viewTitle">Yeni Sorgu</h1>
      <div class="topbar-right">
        <button class="icon-btn" onclick="toggleTheme()" id="themeBtn" title="Tema"></button>
      </div>
    </div>

    <!-- QUERY / DASHBOARD VIEW -->
    <section class="view active" id="view-query">
      <div class="container">
        <div class="hero">
          <div class="hero-eyebrow">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><path d="M9 12l2 2 4-4"/><circle cx="12" cy="12" r="9"/></svg>
            Sorunu sor, üniversitenin kaynağından öğren.
          </div>
          <h2>Merhaba 👋<br>Üniversiten hakkında ne öğrenmek istiyorsun?</h2>
          <p class="tagline">Yönetmelikler, staj yönergeleri, sınav kuralları ve burs şartları arasından yalnızca kaynağı olan cevapları getiririm.</p>

          <div class="query-box">
            <textarea id="questionInput" rows="1" placeholder="Sorunu buraya yaz…" onkeydown="handleQuestionKey(event)" oninput="autoGrow(this)"></textarea>
            <button class="send-btn" id="sendBtn" onclick="submitQuestion()">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4 20-7z"/></svg>
            </button>
          </div>

          <div class="examples" id="exampleChips"></div>

          <div class="status-card" id="statusCard">
            <div class="grid4" id="statusGrid"></div>
          </div>
        </div>

        <div class="thread" id="thread"></div>
      </div>
    </section>

    <!-- DOCUMENTS VIEW -->
    <section class="view" id="view-documents">
      <div class="container">
        <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 16V4M12 4l-4 4M12 4l4 4"/><path d="M4 16v3a2 2 0 002 2h12a2 2 0 002-2v-3"/></svg>
          <h3>Belgeleri yükleyin</h3>
          <p>PDF · TXT · MD — Madde/sayfa bilgisi otomatik çıkarılır</p>
          <button class="btn primary" onclick="event.stopPropagation(); document.getElementById('fileInput').click()">Dosya Seç</button>
          <input type="file" id="fileInput" accept=".pdf,.txt,.md" multiple style="display:none" onchange="handleFiles(this.files)">
        </div>
        <div id="uploadProgressSlot"></div>
        <div class="doc-grid" id="docGrid"></div>
      </div>
    </section>

    <!-- KNOWLEDGE BASE VIEW -->
    <section class="view" id="view-kb">
      <div class="container">
        <div class="status-card" style="margin-top:0;">
          <div class="grid4" id="kbStatsGrid"></div>
        </div>
        <div class="sources-title" style="margin-top:26px;">Doğrulama Dağılımı</div>
        <div class="doc-grid" id="kbBreakdown"></div>
      </div>
    </section>

    <!-- HISTORY VIEW -->
    <section class="view" id="view-history">
      <div class="container" id="historyContainer"></div>
    </section>

    <!-- SETTINGS VIEW -->
    <section class="view" id="view-settings">
      <div class="container">
        <div class="settings-grid" id="settingsGrid"></div>
      </div>
    </section>

    <!-- ABOUT VIEW -->
    <section class="view" id="view-about">
      <div class="container">
        <div class="about-hero">
          <div class="seal-mark-lg" id="sealMarkLgSlot"></div>
          <h2>Kampüsce</h2>
          <div class="slogan">"Cevabı tahmin etmez. Kaynağını bulur."</div>
          <p style="color:var(--ink-dim); max-width:520px; margin:0 auto; font-size:14.5px; line-height:1.6;">
            Üniversite yönetmelikleri, staj yönergeleri, sınav kuralları ve burs şartları gibi resmi belgeler
            içinde kaybolmak yerine; sorunu sor, yalnızca kaynağı olan cevabı al. Tamamen offline çalışır —
            hiçbir soru veya belge internete gönderilmez.
          </p>
          <div class="pipeline">
            <span class="step">Soru</span><span class="arrow">→</span>
            <span class="step">Embedding</span><span class="arrow">→</span>
            <span class="step">Cosine Similarity</span><span class="arrow">→</span>
            <span class="step">Top-K Kaynak</span><span class="arrow">→</span>
            <span class="step">Foundry Local LLM</span><span class="arrow">→</span>
            <span class="step">Cevap + Kaynak + Durum</span>
          </div>
        </div>
      </div>
    </section>

  </main>
</div>

<div class="toast" id="toast"></div>

<script>
const API = "";

const ICONS = {
  query: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>',
  kb: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/></svg>',
  doc: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6"/></svg>',
  history: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v5h5"/><path d="M3.05 13A9 9 0 106 5.3L3 8"/><path d="M12 7v5l4 2"/></svg>',
  settings: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 11-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 11-2.83-2.83l.06-.06A1.65 1.65 0 004.6 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 112.83-2.83l.06.06A1.65 1.65 0 009 4.6a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 112.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>',
  info: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>',
  sun: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
  moon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>',
  trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18"/><path d="M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14z"/></svg>',
  chevron: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M9 6l6 6-6 6"/></svg>',
  spark: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3l1.9 5.6L19.5 10l-5.6 1.9L12 17.5l-1.9-5.6L4.5 10l5.6-1.9z"/></svg>',
  book: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/></svg>',
};

function sealSVG(status, size){
  size = size || 34;
  const colors = { verified:'var(--verified)', partial:'var(--partial)', not_found:'var(--notfound)' };
  const c = colors[status] || colors.partial;
  let icon = '';
  if(status === 'verified'){
    icon = `<path d="M15 24.5l6 6 12-13" fill="none" stroke="#fff" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/>`;
  } else if(status === 'partial'){
    icon = `<path d="M14 26.5c2.4-4.8 5.8-7.2 8.2-2.4s5.8 2.4 8.2-2.4" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round"/>`;
  } else {
    icon = `<path d="M17 17l12 12M29 17l-12 12" stroke="#fff" stroke-width="3" stroke-linecap="round"/>`;
  }
  return `<svg class="seal-icon" width="${size}" height="${size}" viewBox="0 0 48 48">
    <circle cx="24" cy="24" r="21" fill="${c}"/>
    <circle cx="24" cy="24" r="21" fill="none" stroke="#fff" stroke-width="1.5" opacity=".35"/>
    ${icon}
  </svg>`;
}

const STATUS_META = {
  verified: {label:'DOĞRULANDI', sub:'Kaynaklarda net karşılık bulundu', badgeClass:'ready'},
  partial: {label:'KISMEN DOĞRULANDI', sub:'İlgili bilgi var, kesin cevap için yeterli değil', badgeClass:'processing'},
  not_found: {label:'BİLGİ BULUNAMADI', sub:'Bilgi tabanında bu konuda içerik yok', badgeClass:'error'},
};

const NAV_ITEMS = [
  {id:'query', label:'Yeni Sorgu', icon:'query'},
  {id:'kb', label:'Bilgi Tabanı', icon:'kb'},
  {id:'documents', label:'Belgeler', icon:'doc'},
  {id:'history', label:'Arama Geçmişi', icon:'history'},
  {id:'settings', label:'Ayarlar', icon:'settings'},
  {id:'about', label:'Hakkında', icon:'info'},
];

const EXAMPLES = [
  'Staj şartları nelerdir?',
  'Mezuniyet için kaç kredi gerekiyor?',
  'Sınav mazeret sınavı nasıl yapılır?',
  'Burs başvuru şartları neler?',
  'Akademik takvimde kayıt tarihi ne?',
];

let state = { theme: localStorage.getItem('uv_theme') || 'light', collapsed:false, view:'query' };

function initTheme(){
  document.documentElement.setAttribute('data-theme', state.theme);
  document.getElementById('themeBtn').innerHTML = state.theme==='dark' ? ICONS.sun : ICONS.moon;
}
function toggleTheme(){
  state.theme = state.theme === 'dark' ? 'light' : 'dark';
  localStorage.setItem('uv_theme', state.theme);
  initTheme();
}
function toggleSidebar(){
  state.collapsed = !state.collapsed;
  document.getElementById('sidebar').classList.toggle('collapsed', state.collapsed);
}

function renderNav(){
  const el = document.getElementById('navList');
  el.innerHTML = NAV_ITEMS.map(item => `
    <button class="nav-item ${state.view===item.id?'active':''}" onclick="switchView('${item.id}')">
      ${ICONS[item.icon]}<span>${item.label}</span>
    </button>`).join('');
}

const VIEW_TITLES = {query:'Yeni Sorgu', kb:'Bilgi Tabanı', documents:'Belgeler', history:'Arama Geçmişi', settings:'Ayarlar', about:'Hakkında'};

function switchView(id){
  state.view = id;
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.getElementById('view-'+id).classList.add('active');
  document.getElementById('viewTitle').textContent = VIEW_TITLES[id];
  renderNav();
  if(id==='documents') loadDocuments();
  if(id==='kb') loadStats();
  if(id==='history') loadHistory();
  if(id==='settings') loadSettings();
}

function renderExamples(){
  document.getElementById('exampleChips').innerHTML = EXAMPLES.map(q => `
    <button class="example-chip" onclick="askExample(this)">${ICONS.spark}${q}</button>`).join('');
}
function askExample(btn){
  document.getElementById('questionInput').value = btn.textContent.trim();
  submitQuestion();
}

function autoGrow(el){ el.style.height='auto'; el.style.height = Math.min(el.scrollHeight,160)+'px'; }
function handleQuestionKey(e){
  if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); submitQuestion(); }
}

function showToast(msg){
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'), 3200);
}

function escapeHtml(s){
  return (s||'').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

const LOADING_STAGES = ['Belgeler taranıyor…', 'İlgili kaynaklar bulunuyor…', 'Cevap oluşturuluyor…'];

async function submitQuestion(){
  const input = document.getElementById('questionInput');
  const question = input.value.trim();
  if(!question) return;
  const sendBtn = document.getElementById('sendBtn');
  sendBtn.disabled = true;

  const thread = document.getElementById('thread');
  const qBubble = document.createElement('div');
  qBubble.className = 'q-bubble';
  qBubble.textContent = question;
  thread.insertBefore(qBubble, thread.firstChild);

  const loadingCard = document.createElement('div');
  loadingCard.className = 'loading-card';
  loadingCard.innerHTML = LOADING_STAGES.map((s,i)=>`<div class="loading-step ${i===0?'active':''}" data-i="${i}"><span class="dot"></span>${s}</div>`).join('');
  thread.insertBefore(loadingCard, qBubble.nextSibling);

  let stage = 0;
  const stageTimer = setInterval(()=>{
    stage = Math.min(stage+1, LOADING_STAGES.length-1);
    loadingCard.querySelectorAll('.loading-step').forEach((el,i)=>{
      el.classList.toggle('done', i<stage);
      el.classList.toggle('active', i===stage);
    });
  }, 900);

  input.value=''; autoGrow(input);

  try{
    const res = await fetch(API+'/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({question})});
    const data = await res.json();
    clearInterval(stageTimer);
    loadingCard.remove();
    if(!res.ok){ showToast(data.detail || 'Bir hata oluştu.'); qBubble.remove(); return; }
    thread.insertBefore(renderAnswerCard(data), qBubble.nextSibling);
  } catch(err){
    clearInterval(stageTimer);
    loadingCard.remove();
    showToast('Sunucuya ulaşılamadı. Uygulamanın çalıştığından emin olun.');
  } finally {
    sendBtn.disabled = false;
  }
}

function confidenceLabel(v){
  if(v >= 0.85) return 'Yüksek güven';
  if(v >= 0.6) return 'Orta güven';
  return 'Düşük güven';
}

function renderAnswerCard(data){
  const status = data.verification_status || 'not_found';
  const meta = STATUS_META[status];
  const pct = Math.round((data.confidence||0)*100);

  const sourcesHtml = (data.sources||[]).map(s => `
    <div class="source-card">
      <div class="doc-icon">${ICONS.doc}</div>
      <div class="source-meta">
        <div class="doc-name">${escapeHtml(s.document||'Bilinmeyen kaynak')}</div>
        <div class="doc-loc">${s.article ? escapeHtml(s.article) : 'Madde belirtilmemiş'}${s.page ? ' · Sayfa '+s.page : ''}</div>
      </div>
      <div class="source-relevance"><b>${Math.round((s.relevance||0)*100)}%</b><br>benzerlik</div>
    </div>`).join('');

  const card = document.createElement('div');
  card.className = 'answer-card';
  card.innerHTML = `
    <div class="answer-head ${status}">
      ${sealSVG(status)}
      <div>
        <div class="badge-text">${meta.label}</div>
        <div class="badge-sub">${meta.sub}</div>
      </div>
    </div>
    <div class="answer-body">
      <div class="answer-text">${escapeHtml(data.answer)}</div>
      <div class="confidence-wrap">
        <div class="confidence-label">
          <span class="tip" title="Bu değer, kullanılan kaynakların soruyla anlamsal benzerliğini temsil eden sistem skorudur; resmi bir doğruluk garantisi değildir.">Güven · ${confidenceLabel(data.confidence||0)}</span>
          <span class="mono">${pct}%</span>
        </div>
        <div class="confidence-bar"><div class="confidence-fill ${status}" style="width:${pct}%"></div></div>
      </div>
      ${sourcesHtml ? `<div class="sources-title">Kaynaklar</div><div class="source-list">${sourcesHtml}</div>` : ''}
      <div class="tech-details">
        <button class="tech-toggle" onclick="this.classList.toggle('open'); this.nextElementSibling.classList.toggle('open')">
          ${ICONS.chevron} Teknik Detaylar
        </button>
        <div class="tech-panel">
Retrieved chunks: ${data.retrieved_chunks ?? 0}
Top similarity: ${data.top_similarity ?? 0}
LLM model: ${data.llm_model || '—'}
Embedding model: ${data.embedding_model || '—'}
        </div>
      </div>
    </div>`;
  return card;
}

async function loadHealth(){
  try{
    const res = await fetch(API+'/api/health');
    const h = await res.json();
    const dot = document.getElementById('sidebarStatusDot');
    const text = document.getElementById('sidebarStatusText');
    dot.className = 'status-dot ' + (h.foundry_local ? 'on' : 'off');
    text.textContent = h.foundry_local ? 'Foundry Local bağlı' : 'Foundry Local bağlı değil';

    const rows = [
      {label:'Foundry Local', on: h.foundry_local},
      {label:'Embedding Modeli', on: h.foundry_local},
      {label:'SQLite', on: h.database},
      {label:'Bilgi Tabanı', on: h.documents > 0},
    ];
    document.getElementById('statusGrid').innerHTML = rows.map(r => `
      <div class="status-row"><span class="status-dot ${r.on?'on':'off'}"></span>${r.label}</div>`).join('');
    return h;
  } catch(e){
    document.getElementById('sidebarStatusText').textContent = 'Sunucuya ulaşılamıyor';
    document.getElementById('sidebarStatusDot').className = 'status-dot off';
  }
}

async function loadDocuments(){
  const grid = document.getElementById('docGrid');
  grid.innerHTML = '<div class="empty-state">Yükleniyor…</div>';
  const res = await fetch(API+'/api/documents');
  const data = await res.json();
  if(!data.documents.length){
    grid.innerHTML = `<div class="empty-state">${ICONS.book}<h3>Henüz bilgi tabanınız boş.</h3><p>İlk belgenizi yükleyerek Kampüsce'ı kullanmaya başlayın.</p></div>`;
    return;
  }
  grid.innerHTML = data.documents.map(d => `
    <div class="doc-row">
      <div class="doc-icon-lg">${ICONS.doc}</div>
      <div class="doc-info">
        <div class="name">${escapeHtml(d.title)}</div>
        <div class="meta">${d.pages} sayfa · ${d.chunk_count} chunk ${d.error ? '· '+escapeHtml(d.error) : ''}</div>
      </div>
      <span class="doc-badge ${d.status==='ready'?'ready':(d.status==='error'?'error':'processing')}">${d.status==='ready'?'Hazır':(d.status==='error'?'Hata':'İşleniyor')}</span>
      <div class="doc-actions">
        <button class="btn danger-ghost" onclick="deleteDocument('${d.id}')">${ICONS.trash} Sil</button>
      </div>
    </div>`).join('');
}

async function handleFiles(files){
  const slot = document.getElementById('uploadProgressSlot');
  for(const file of files){
    const card = document.createElement('div');
    card.className = 'loading-card'; card.style.marginTop='16px';
    card.innerHTML = `<div class="loading-step active"><span class="dot"></span>${escapeHtml(file.name)} işleniyor: Belge okunuyor → Chunk'lara ayrılıyor → Embedding oluşturuluyor → SQLite'a kaydediliyor…</div>`;
    slot.appendChild(card);
    const form = new FormData(); form.append('file', file);
    try{
      const res = await fetch(API+'/api/documents/upload', {method:'POST', body:form});
      const data = await res.json();
      if(!res.ok) throw new Error(data.detail || 'Yükleme başarısız.');
      card.innerHTML = `<div class="loading-step done"><span class="dot"></span>${escapeHtml(file.name)} — Hazır (${data.chunks} chunk)</div>`;
      showToast(file.name + ' başarıyla eklendi.');
    } catch(err){
      card.innerHTML = `<div class="loading-step" style="color:var(--notfound)"><span class="dot" style="border-color:var(--notfound)"></span>${escapeHtml(file.name)}: ${escapeHtml(err.message)}</div>`;
    }
  }
  document.getElementById('fileInput').value = '';
  loadDocuments(); loadHealth();
}

async function deleteDocument(id){
  if(!confirm('Bu belgeyi silmek istediğinize emin misiniz?')) return;
  await fetch(API+'/api/documents/'+id, {method:'DELETE'});
  showToast('Belge silindi.');
  loadDocuments();
}

async function loadStats(){
  const res = await fetch(API+'/api/stats');
  const s = await res.json();
  document.getElementById('kbStatsGrid').innerHTML = `
    <div class="status-row"><b class="mono">${s.documents}</b>&nbsp;Toplam Belge</div>
    <div class="status-row"><b class="mono">${s.chunks}</b>&nbsp;Toplam Chunk</div>
    <div class="status-row"><b class="mono">${s.queries}</b>&nbsp;Toplam Sorgu</div>
    <div class="status-row">${s.last_update ? new Date(s.last_update).toLocaleDateString('tr-TR') : '—'}&nbsp;Son Güncelleme</div>`;
  const breakdown = s.status_breakdown || {};
  const total = Object.values(breakdown).reduce((a,b)=>a+b,0) || 1;
  const kb = document.getElementById('kbBreakdown');
  if(!total || total===1 && Object.keys(breakdown).length===0){
    kb.innerHTML = `<div class="empty-state">${ICONS.history}<h3>Henüz sorgu geçmişiniz bulunmuyor.</h3></div>`;
    return;
  }
  kb.innerHTML = ['verified','partial','not_found'].map(k => {
    const meta = STATUS_META[k]; const c = breakdown[k]||0;
    return `<div class="doc-row">${sealSVG(k,30)}<div class="doc-info"><div class="name">${meta.label}</div></div><div class="mono" style="font-weight:700">${c}</div></div>`;
  }).join('');
}

function groupByDay(items){
  const today = new Date().toDateString();
  const yesterday = new Date(Date.now()-86400000).toDateString();
  const groups = {'Bugün':[], 'Dün':[], 'Daha Önce':[]};
  items.forEach(it => {
    const d = new Date(it.created_at).toDateString();
    if(d===today) groups['Bugün'].push(it);
    else if(d===yesterday) groups['Dün'].push(it);
    else groups['Daha Önce'].push(it);
  });
  return groups;
}

async function loadHistory(){
  const container = document.getElementById('historyContainer');
  const res = await fetch(API+'/api/history');
  const data = await res.json();
  if(!data.history.length){
    container.innerHTML = `<div class="empty-state">${ICONS.history}<h3>Henüz sorgu geçmişiniz bulunmuyor.</h3></div>`;
    return;
  }
  const groups = groupByDay(data.history);
  container.innerHTML = Object.entries(groups).filter(([,v])=>v.length).map(([label, items]) => `
    <div class="history-group">
      <h4>${label}</h4>
      ${items.map(it => `
        <div class="history-item" onclick='openHistoryItem(${JSON.stringify(it).replace(/'/g,"&#39;")})'>
          ${sealSVG(it.verification_status, 22)}
          <div class="q-text">${escapeHtml(it.question)}</div>
          <div class="q-time">${new Date(it.created_at).toLocaleTimeString('tr-TR',{hour:'2-digit',minute:'2-digit'})}</div>
        </div>`).join('')}
    </div>`).join('');
}

function openHistoryItem(it){
  switchView('query');
  const thread = document.getElementById('thread');
  const qBubble = document.createElement('div');
  qBubble.className='q-bubble'; qBubble.textContent = it.question;
  thread.insertBefore(qBubble, thread.firstChild);
  thread.insertBefore(renderAnswerCard(it), qBubble.nextSibling);
}

async function loadSettings(){
  const res = await fetch(API+'/api/settings');
  const s = await res.json();
  const grid = document.getElementById('settingsGrid');
  grid.innerHTML = `
    <div class="settings-row"><div><div class="label">LLM Modeli</div><div class="desc">Foundry Local üzerinden kullanılan sohbet modeli.</div></div><div class="value">${escapeHtml(s.llm_model)}</div></div>
    <div class="settings-row"><div><div class="label">Embedding Modeli</div><div class="desc">Chunk ve soru vektörleri bu modelle oluşturulur.</div></div><div class="value">${escapeHtml(s.embedding_model)}</div></div>
    <div class="settings-row"><div><div class="label">Top K</div><div class="desc">Her soru için getirilecek en alakalı kaynak sayısı.</div></div>
      <div><input type="number" min="1" max="10" value="${s.top_k}" onchange="updateSetting('top_k', parseInt(this.value))"></div></div>
    <div class="settings-row"><div><div class="label">Benzerlik Eşiği</div><div class="desc">Bu değerin altındaki kaynaklar ilgisiz sayılır (BİLGİ BULUNAMADI).</div></div>
      <div><input type="number" step="0.05" min="0" max="1" value="${s.similarity_threshold}" onchange="updateSetting('similarity_threshold', parseFloat(this.value))"></div></div>
    <div class="settings-row"><div><div class="label">Temperature</div><div class="desc">Düşük değer = daha tutarlı, daha az yaratıcı cevaplar.</div></div>
      <div><input type="number" step="0.05" min="0" max="1" value="${s.temperature}" onchange="updateSetting('temperature', parseFloat(this.value))"></div></div>
    <div class="settings-row"><div><div class="label">Veritabanı Yolu</div><div class="desc">SQLite dosyasının konumu.</div></div><div class="value">${escapeHtml(s.database_path)}</div></div>
    <div class="settings-row"><div><div class="label">Foundry Local Endpoint</div><div class="desc">Otomatik keşfedilen veya .env'de belirtilen adres.</div></div><div class="value">${escapeHtml(s.foundry_endpoint)}</div></div>
    <div class="settings-row"><div><div class="label">Bilgi Tabanını Yeniden İndeksle</div><div class="desc">Ayarları değiştirdikten sonra tüm belgeleri yeniden chunk'layıp embed eder.</div></div>
      <button class="btn primary" onclick="reindexAll()">Yeniden İndeksle</button></div>`;
}

async function updateSetting(key, value){
  const body = {}; body[key] = value;
  const res = await fetch(API+'/api/settings', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  if(res.ok){ showToast('Ayar güncellendi.'); } else { const d = await res.json(); showToast(d.detail || 'Güncellenemedi.'); }
}

async function reindexAll(){
  showToast('Yeniden indeksleniyor, bu işlem belge sayısına göre zaman alabilir…');
  const res = await fetch(API+'/api/documents/reindex', {method:'POST'});
  const data = await res.json();
  if(res.ok){ showToast('Bilgi tabanı yeniden indekslendi.'); } else { showToast(data.detail || 'Hata oluştu.'); }
}

// ---- init ----
document.getElementById('sealMarkSlot').innerHTML = sealSVG('verified', 30);
document.getElementById('sealMarkLgSlot').innerHTML = sealSVG('verified', 64);
initTheme();
renderNav();
renderExamples();
loadHealth();
setInterval(loadHealth, 20000);

const uploadZone = document.getElementById('uploadZone');
['dragover'].forEach(evt => uploadZone.addEventListener(evt, e => { e.preventDefault(); uploadZone.classList.add('dragover'); }));
['dragleave','drop'].forEach(evt => uploadZone.addEventListener(evt, e => { e.preventDefault(); uploadZone.classList.remove('dragover'); }));
uploadZone.addEventListener('drop', e => { if(e.dataTransfer.files.length) handleFiles(e.dataTransfer.files); });
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    init_db()
    print(f"\n  Kampüsce  →  http://{CFG.host}:{CFG.port}\n")
    uvicorn.run(app, host=CFG.host, port=CFG.port, log_level="info")
