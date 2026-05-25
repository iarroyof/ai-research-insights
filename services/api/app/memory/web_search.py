from __future__ import annotations

from html import unescape
import re
from typing import Any, Dict, List
from xml.etree import ElementTree
import httpx

from app.config import settings
from app.memory.privacy import redact_query


EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBTATOR3_SEARCH_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/search/"
LITSENSE2_SENTENCE_URL = "https://www.ncbi.nlm.nih.gov/research/litsense2-api/api/sentences/"
LITSENSE2_PASSAGE_URL = "https://www.ncbi.nlm.nih.gov/research/litsense2-api/api/passages/"


def _element_text(element: ElementTree.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _pubmed_results_from_xml(xml_text: str) -> List[Dict[str, str]]:
    if not xml_text.strip():
        return []
    root = ElementTree.fromstring(xml_text)
    results: List[Dict[str, str]] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _element_text(article.find("./MedlineCitation/PMID"))
        title = _element_text(article.find("./MedlineCitation/Article/ArticleTitle"))
        abstract = " ".join(
            text for text in (_element_text(node) for node in article.findall("./MedlineCitation/Article/Abstract/AbstractText")) if text
        )
        ids = {
            (node.attrib.get("IdType") or "").lower(): _element_text(node)
            for node in article.findall("./PubmedData/ArticleIdList/ArticleId")
        }
        pmcid = ids.get("pmc", "")
        if title and (abstract or pmid):
            results.append(
                {
                    "source": "pubmed",
                    "title": title,
                    "snippet": abstract or title,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                    "pmid": pmid,
                    "pmcid": pmcid,
                    "pmc_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else "",
                }
            )
    return results


def _pmc_results_from_xml(xml_text: str) -> List[Dict[str, str]]:
    if not xml_text.strip():
        return []
    root = ElementTree.fromstring(xml_text)
    results: List[Dict[str, str]] = []
    for article in root.findall(".//article"):
        meta = article.find("./front/article-meta")
        if meta is None:
            continue
        title = _element_text(meta.find("./title-group/article-title"))
        abstract = " ".join(text for text in (_element_text(node) for node in meta.findall("./abstract")) if text)
        pmcid = ""
        pmid = ""
        for article_id in meta.findall("./article-id"):
            id_type = (article_id.attrib.get("pub-id-type") or "").lower()
            if id_type == "pmc":
                pmcid = _element_text(article_id)
            elif id_type == "pmid":
                pmid = _element_text(article_id)
        if title and (abstract or pmcid):
            results.append(
                {
                    "source": "pmc",
                    "title": title,
                    "snippet": abstract or title,
                    "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else "",
                    "pmid": pmid,
                    "pmcid": pmcid,
                    "pmc_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else "",
                }
            )
    return results


async def _eutils_search_ids(client: httpx.AsyncClient, *, db: str, query: str, retmax: int) -> List[str]:
    resp = await client.get(
        f"{EUTILS_BASE_URL}/esearch.fcgi",
        params={"db": db, "term": query, "retmode": "json", "retmax": str(retmax)},
    )
    resp.raise_for_status()
    return [str(item) for item in (resp.json().get("esearchresult") or {}).get("idlist") or []]


async def _eutils_fetch_xml(client: httpx.AsyncClient, *, db: str, ids: List[str]) -> str:
    if not ids:
        return ""
    resp = await client.get(
        f"{EUTILS_BASE_URL}/efetch.fcgi",
        params={"db": db, "id": ",".join(ids), "retmode": "xml"},
    )
    resp.raise_for_status()
    return resp.text


async def pubmed_fetch_by_pmids(pmids: List[str]) -> Dict[str, Dict[str, str]]:
    """Fetch PubMed abstracts for known PMIDs without sending a new text query."""
    ids = [str(item).strip() for item in pmids if str(item).strip()]
    if not ids:
        return {}
    async with httpx.AsyncClient(timeout=12) as client:
        xml_text = await _eutils_fetch_xml(client, db="pubmed", ids=ids[:20])
    return {str(item.get("pmid") or ""): item for item in _pubmed_results_from_xml(xml_text)}


async def pubmed_pmc_search(query: str, k: int | None = None) -> Dict[str, Any]:
    """
    Privacy-filtered PubMed then PMC lookup through NCBI E-utilities.

    PubMed abstracts are primary. PMC tops up the sparse external context when
    the PubMed result set does not fill the requested result budget.
    """
    k = k or settings.memory.web_k
    redaction = redact_query(query)
    if not redaction["safe_for_web"]:
        return {"query": redaction["query"], "redacted": redaction["redacted"], "results": []}

    results: List[Dict[str, str]] = []
    async with httpx.AsyncClient(timeout=12) as client:
        pubmed_ids = await _eutils_search_ids(client, db="pubmed", query=redaction["query"], retmax=k)
        results.extend(_pubmed_results_from_xml(await _eutils_fetch_xml(client, db="pubmed", ids=pubmed_ids)))
        if len(results) < k:
            pmc_ids = await _eutils_search_ids(client, db="pmc", query=redaction["query"], retmax=k - len(results))
            results.extend(_pmc_results_from_xml(await _eutils_fetch_xml(client, db="pmc", ids=pmc_ids)))

    return {"query": redaction["query"], "redacted": redaction["redacted"], "results": results[:k]}


def _normalize_pubtator_text(text: str) -> str:
    normalized = unescape(text or "")
    normalized = re.sub(r"</?m>", "", normalized)
    normalized = normalized.replace("@@@", "")
    normalized = re.sub(r"@(?:GENE|DISEASE|CHEMICAL|VARIANT|SPECIES|CELLLINE)_[^\s]+", "", normalized)
    return " ".join(normalized.split())


async def pubtator3_search(query: str, k: int | None = None) -> Dict[str, Any]:
    """
    Privacy-filtered PubTator 3 search.

    PubTator result text contains entity/highlight markup. Normalize it before
    it becomes assistant context while preserving PubMed and PMC identifiers.
    """
    k = k or settings.memory.web_k
    redaction = redact_query(query)
    if not redaction["safe_for_web"]:
        return {"query": redaction["query"], "redacted": redaction["redacted"], "results": []}

    async with httpx.AsyncClient(timeout=12) as client:
        resp = await client.get(PUBTATOR3_SEARCH_URL, params={"text": redaction["query"], "page": "1"})
        resp.raise_for_status()
        data = resp.json()

    results: List[Dict[str, str]] = []
    for item in data.get("results") or []:
        if len(results) >= k:
            break
        pmid = str(item.get("pmid") or "")
        pmcid = str(item.get("pmcid") or "")
        title = _normalize_pubtator_text(str(item.get("title") or ""))
        snippet = _normalize_pubtator_text(str(item.get("text_hl") or item.get("title") or ""))
        if not title or not snippet:
            continue
        results.append(
            {
                "source": "pubtator3",
                "title": title,
                "snippet": snippet,
                "url": f"https://www.ncbi.nlm.nih.gov/research/pubtator3/publication/{pmid}" if pmid else "",
                "pmid": pmid,
                "pmcid": pmcid,
                "pmc_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else "",
                "doi": str(item.get("doi") or ""),
            }
        )

    return {"query": redaction["query"], "redacted": redaction["redacted"], "results": results[:k]}


def _litsense2_results(items: list[dict], *, source: str, label: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for item in items:
        text = " ".join(str(item.get("text") or "").split())
        if not text:
            continue
        pmid = str(item.get("pmid") or "")
        pmcid = str(item.get("pmcid") or "")
        provenance_id = f"PMID {pmid}" if pmid else f"PMCID {pmcid}" if pmcid else "NCBI result"
        results.append(
            {
                "source": source,
                "title": f"LitSense {label} | {provenance_id}",
                "snippet": text,
                "url": (
                    f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    if pmid
                    else f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
                    if pmcid
                    else ""
                ),
                "pmid": pmid,
                "pmcid": pmcid,
                "pmc_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else "",
                "score": item.get("score"),
                "section": str(item.get("section") or ""),
                "annotations": item.get("annotations") or [],
            }
        )
    return results


async def litsense2_search(query: str, k: int | None = None) -> Dict[str, Any]:
    """
    Privacy-filtered LitSense 2.0 sentence search with passage top-up.

    LitSense exposes reranked PubMed/PMC snippets. Sentence results are the
    strongest sparse grounding unit; longer passage results fill unused slots.
    """
    k = k or settings.memory.web_k
    redaction = redact_query(query)
    if not redaction["safe_for_web"]:
        return {"query": redaction["query"], "redacted": redaction["redacted"], "results": []}

    results: List[Dict[str, Any]] = []
    params = {"query": redaction["query"], "rerank": "true"}
    async with httpx.AsyncClient(timeout=12) as client:
        sentence_resp = await client.get(LITSENSE2_SENTENCE_URL, params=params)
        sentence_resp.raise_for_status()
        results.extend(_litsense2_results(sentence_resp.json() or [], source="litsense2_sentence", label="sentence"))
        if len(results) < k:
            passage_resp = await client.get(LITSENSE2_PASSAGE_URL, params=params)
            passage_resp.raise_for_status()
            results.extend(_litsense2_results(passage_resp.json() or [], source="litsense2_passage", label="passage"))

    return {"query": redaction["query"], "redacted": redaction["redacted"], "results": results[:k]}


async def duckduckgo_search(query: str, k: int | None = None) -> Dict[str, Any]:
    """
    Privacy-filtered DuckDuckGo Instant Answer lookup.

    This intentionally uses the public JSON endpoint and sends only a redacted,
    shortened query. It returns sparse grounding snippets, not raw web pages.
    """
    k = k or settings.memory.web_k
    redaction = redact_query(query)
    if not redaction["safe_for_web"]:
        return {"query": redaction["query"], "redacted": redaction["redacted"], "results": []}

    params = {
        "q": redaction["query"],
        "format": "json",
        "no_redirect": "1",
        "no_html": "1",
        "skip_disambig": "1",
    }
    async with httpx.AsyncClient(timeout=12) as client:
        resp = await client.get("https://api.duckduckgo.com/", params=params)
        resp.raise_for_status()
        data = resp.json()

    results: List[Dict[str, str]] = []
    if data.get("AbstractText"):
        results.append(
            {
                "title": data.get("Heading") or "DuckDuckGo abstract",
                "snippet": data.get("AbstractText") or "",
                "url": data.get("AbstractURL") or "",
            }
        )

    def add_related(items: list[dict]) -> None:
        for item in items:
            if len(results) >= k:
                return
            if "Topics" in item:
                add_related(item.get("Topics") or [])
                continue
            text = item.get("Text") or ""
            if text:
                results.append(
                    {
                        "title": item.get("FirstURL") or "DuckDuckGo result",
                        "snippet": text,
                        "url": item.get("FirstURL") or "",
                    }
                )

    add_related(data.get("RelatedTopics") or [])
    return {"query": redaction["query"], "redacted": redaction["redacted"], "results": results[:k]}
