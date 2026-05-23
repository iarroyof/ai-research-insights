# services/streamlit/app.py
from __future__ import annotations

import os
import json
import httpx
import streamlit as st
import base64
from pathlib import Path
import hmac


def check_password():
    """Simple password protection with branded login page."""
    def password_entered():
        if hmac.compare_digest(st.session_state["password"], st.secrets.passwords.get(st.session_state["username"], "")):
            st.session_state["authenticated"] = True
            del st.session_state["password"]
        else:
            st.session_state["authenticated"] = False

    if st.session_state.get("authenticated"):
        return True

    # Display branded login page
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Logo
        try:
            logo_path = Path(__file__).parent / "Sabia_RN.png"
            if logo_path.exists():
                img_bytes = logo_path.read_bytes()
                encoded = base64.b64encode(img_bytes).decode()
                st.markdown(
                    f'<div style="text-align: center; margin-bottom: 20px;">'
                    f'<img src="data:image/png;base64,{encoded}" style="max-width: 250px;">'
                    f'</div>',
                    unsafe_allow_html=True
                )
        except Exception:
            pass
        
        st.markdown("<h2 style='text-align: center;'>Sabia: Biomedical Knowledge AI Research Platform</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>Please log in to continue</p>", unsafe_allow_html=True)
        
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered, use_container_width=True)

        if "authenticated" in st.session_state and not st.session_state["authenticated"]:
            st.error("Invalid credentials")
        
        # Lab info
        st.markdown("---")
        try:
            lab_info_path = Path(__file__).parent / "lab_info.md"
            if lab_info_path.exists():
                st.markdown(lab_info_path.read_text())
        except Exception:
            pass
    
    return False


@st.cache_data
def get_base64_of_local_file(filepath: str) -> str:
    """Reads a local file, encodes it to Base64, and returns the data URI string."""
    try:
        img_bytes = Path(filepath).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return f"data:image/png;base64,{encoded}"
    except FileNotFoundError:
        st.error(f"Error: Logo file not found at '{filepath}'. Please check the path.")
        return ""


st.set_page_config(layout="wide", page_title="AI Research Insights – Bench")

if not check_password():
    st.stop()

# API Configuration
API = os.environ.get("API_URL", "http://api:8080")
PUBLIC_API = os.environ.get("PUBLIC_API_URL", "")

if not PUBLIC_API:
    PUBLIC_API = ""
    if API.startswith("http://api:"):
        PUBLIC_API = "http://localhost:18081"
    else:
        PUBLIC_API = API

# Custom CSS for better table display
st.markdown(
    """
    <style>
    .custom-logo-img {
        width: 12.5vw !important;
        min-width: 120px;
        max-width: 250px;
        height: auto !important;
        padding: 10px 0 !important;
        margin-top: 10px !important;
        object-fit: contain !important;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0;
    }
    .st-emotion-cache-12fmw1f {
        visibility: hidden;
    }
    /* Make data editor text wrap and be more readable */
    [data-testid="stDataEditor"] {
        font-size: 13px;
    }
    [data-testid="stDataEditor"] td {
        white-space: normal !important;
        word-wrap: break-word !important;
        max-width: 400px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

logo_base64_src = get_base64_of_local_file("Sabia_RN.png")
logo_col, title_col = st.columns([2, 5])

with logo_col:
    if logo_base64_src:
        st.markdown(
            f'<img src="{logo_base64_src}" class="custom-logo-img">',
            unsafe_allow_html=True
        )

with title_col:
    st.title("Sabia: Search → Select → Run")

TENANT = st.sidebar.text_input("Tenant", "default")

# Load and display lab info at bottom of sidebar
try:
    lab_info_path = Path(__file__).parent / "lab_info.md"
    if lab_info_path.exists():
        st.sidebar.markdown("---")
        st.sidebar.markdown(lab_info_path.read_text())
except Exception:
    pass  # Silently skip if file not found
API_KEY = os.environ.get("API_KEY", "dev")
headers = {"X-Tenant-Id": TENANT, "X-API-Key": API_KEY, "Content-Type": "application/json"}

# ------------------------------------------------------------------
# SEARCH SECTION
# ------------------------------------------------------------------
with st.expander("Search", expanded=True):
    q = st.text_input("Query", "Immunological therapy for lung carcinoma")
    k = st.number_input("Top K", min_value=1, max_value=100, value=25, step=1)

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Search", type="primary"):
            try:
                with httpx.Client(timeout=60) as c:
                    r = c.post(
                        f"{API}/search/",
                        headers=headers,
                        json={"query": q, "target": "all", "filters": {}, "k": k},
                    )
                    r.raise_for_status()
                    results = r.json().get("items", [])
                    # Sort by confidence descending
                    results.sort(key=lambda x: x.get("confidence", 0) or 0, reverse=True)
                    st.session_state["results"] = results
                    st.session_state["selected_papers"] = []
                    st.success(f"Found {len(results)} results")
            except Exception as e:
                st.error(f"Search failed: {e}")

    with col2:
        if "results" in st.session_state and st.session_state["results"]:
            st.caption(f"**{len(st.session_state['results'])} results** loaded (sorted by confidence)")

# ------------------------------------------------------------------
# RESULTS TABLE WITH CHECKBOX SELECTION
# ------------------------------------------------------------------
chosen: list[dict] = []
if "results" in st.session_state and st.session_state["results"]:
    st.subheader("Search Results")

    # Build rows with user-friendly column order
    # Hidden internally: pmid, page, paper_id, sent_id, score
    rows = []
    for hit in st.session_state["results"]:
        # Truncate text for display but keep full version for selection
        text_full = hit.get("text", "")
        title_full = hit.get("title", "")
        
        rows.append(
            {
                "Select": False,
                "PMCID": hit.get("pmcid") or "-",
                "Text": text_full,
                # Title removed - no real title in triplets index
                "Confidence": round(hit.get("confidence", 0), 4) if hit.get("confidence") else 0.0,
                "Subject": hit.get("subject") or "-",
                "Relation": hit.get("relation") or "-",
                "Object": hit.get("object") or "-",
                # Hidden fields (stored for selection payload)
                "_paper_id": hit.get("paper_id"),
                "_sent_id": hit.get("sent_id"),
                "_pmid": hit.get("pmid"),
                "_score": hit.get("score", 0),
            }
        )

    edited = st.data_editor(
        rows,
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        column_config={
            "Select": st.column_config.CheckboxColumn("✓", default=False, width="small"),
            "PMCID": st.column_config.TextColumn("PMCID", width="small"),
            "Text": st.column_config.TextColumn("Sentence Text", width="large"),
            # Title column removed
            "Confidence": st.column_config.NumberColumn("Conf", format="%.3f", width="small"),
            "Subject": st.column_config.TextColumn("Subject", width="medium"),
            "Relation": st.column_config.TextColumn("Relation", width="small"),
            "Object": st.column_config.TextColumn("Object", width="medium"),
            # Hide internal fields
            "_paper_id": None,
            "_sent_id": None,
            "_pmid": None,
            "_score": None,
        },
        column_order=["PMCID", "Subject", "Relation", "Object", "Select", "Text", "Confidence"],
        key="results_table",
    )

    # Build selection list with all relevant fields for API
    chosen = [
        {
            "paper_id": r["_paper_id"],
            "sent_id": r["_sent_id"],
            "text": r["Text"],
            "subject": r.get("Subject"),
            "relation": r.get("Relation"),
            "object": r.get("Object"),
            "pmid": r.get("_pmid"),
            "pmcid": r.get("PMCID"),
        }
        for r in edited
        if r.get("Select") and r.get("_sent_id")
    ]

    st.caption(f"**Selected: {len(chosen)}** items")

    if chosen:
        st.session_state["selected_papers"] = chosen

# ------------------------------------------------------------------
# MAIN ACTION TABS
# ------------------------------------------------------------------
tab_chat, tab_sum, tab_triplets = st.tabs([
    "💬 Chat",
    "📝 Summarize (Conditioned)",
    "🕸️ Triplets / Graph"
])

# ------------------------------------------------------------------
# CHAT TAB
# ------------------------------------------------------------------
with tab_chat:
    st.subheader("Live Chat")

    st.markdown("**Search past chat messages** (server-side substring match in this tenant)")
    hist_q = st.text_input("Find in chat history", placeholder="e.g., PD-1 dosage window")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Search history"):
            try:
                with httpx.Client(timeout=30) as c:
                    r = c.get(f"{API}/chat/history/search", headers=headers, params={"q": hist_q})
                    if r.status_code == 200:
                        st.session_state["history_hits"] = r.json().get("matches", [])
                    else:
                        st.session_state["history_hits"] = []
            except Exception as e:
                st.error(f"History search failed: {e}")

    with col2:
        if "history_hits" in st.session_state:
            st.write("Matches:")
            st.json(st.session_state["history_hits"])

    selected_items = st.session_state.get("selected_papers", [])
    if selected_items:
        st.info(f"📌 **{len(selected_items)} items pinned as context**")
        with st.expander("View pinned items"):
            for item in selected_items[:5]:
                st.caption(f"**{item.get('subject')}** - {item.get('relation')} - **{item.get('object')}**")
                st.text(item.get('text', '')[:200] + "...")
    else:
        st.info("No pinned context. Chat will auto-search local evidence when enabled.")

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    if "chat_session_id" not in st.session_state:
        st.session_state["chat_session_id"] = None

    chat_col, reset_col = st.columns([3, 1])
    with chat_col:
        if st.session_state["chat_session_id"]:
            st.caption(f"Session ID: `{st.session_state['chat_session_id']}`")
        else:
            st.caption("New chat session")
    with reset_col:
        if st.button("New chat", use_container_width=True):
            st.session_state["chat_messages"] = []
            st.session_state["chat_session_id"] = None
            st.session_state["memory_diagnostics"] = {}
            st.rerun()

    allow_extra = st.checkbox("Allow extra retrieval if pinned < 3", value=False)
    allow_auto_context = st.checkbox("Auto-search evidence when nothing is pinned", value=True)
    allow_web_search = st.checkbox("Use privacy-filtered DuckDuckGo context if local memory is sparse", value=False)
    expose_memory_debug = st.checkbox("Show memory debug events", value=False)

    if st.session_state["chat_session_id"]:
        with st.expander("Session diagnostics"):
            if "memory_diagnostics" not in st.session_state:
                st.session_state["memory_diagnostics"] = {}

            diag_targets = [
                ("Ideas", "ideas"),
                ("Action values", "action-values"),
                ("Evidence tables", "evidence-tables"),
                ("Search notes", "search-notes"),
            ]
            diag_cols = st.columns(len(diag_targets))
            for col, (label, endpoint) in zip(diag_cols, diag_targets):
                with col:
                    if st.button(label, key=f"diag_{endpoint}", use_container_width=True):
                        try:
                            with httpx.Client(timeout=30) as c:
                                r = c.get(
                                    f"{API}/chat/memory/{endpoint}",
                                    headers=headers,
                                    params={"session_id": st.session_state["chat_session_id"], "limit": 10},
                                )
                                r.raise_for_status()
                                st.session_state["memory_diagnostics"][endpoint] = r.json()
                        except Exception as e:
                            st.session_state["memory_diagnostics"][endpoint] = {"error": str(e)}

            for endpoint, payload in st.session_state["memory_diagnostics"].items():
                st.caption(endpoint)
                st.json(payload)

    for entry in st.session_state["chat_messages"]:
        with st.chat_message(entry.get("role", "assistant")):
            st.markdown(entry.get("content", ""))
            if entry.get("citations"):
                with st.expander("📚 Citations & Sources"):
                    st.json(entry["citations"])
            if entry.get("warnings"):
                with st.expander("Consistency warnings"):
                    st.json(entry["warnings"])
            if entry.get("debug"):
                with st.expander("Memory debug"):
                    st.json(entry["debug"])

    msg = st.chat_input("Ask a research question")

    if msg:
        st.session_state["chat_messages"].append({"role": "user", "content": msg})
        with st.chat_message("user"):
            st.markdown(msg)

        try:
            with st.chat_message("assistant"):
                answer_text = ""
                citations_data = None
                warnings_data = []
                debug_data = {}
                placeholder = st.empty()

                payload = {
                    "message": msg,
                    "items": selected_items,
                    "options": {
                        "allow_extra_retrieval": allow_extra,
                        "allow_auto_context": allow_auto_context,
                        "allow_web_search": allow_web_search,
                        "expose_memory_debug": expose_memory_debug,
                    },
                }
                if st.session_state["chat_session_id"]:
                    payload["session_id"] = st.session_state["chat_session_id"]

                with httpx.Client(timeout=None) as c:
                    with c.stream("POST", f"{API}/chat/", headers=headers, json=payload) as r:
                        r.raise_for_status()

                        for line in r.iter_lines():
                            if not line:
                                continue

                            if isinstance(line, bytes):
                                line = line.decode("utf-8", errors="ignore")

                            if line.startswith("data:"):
                                data_str = line[5:].strip()

                                if data_str == "[DONE]":
                                    break

                                try:
                                    data = json.loads(data_str)
                                    event_type = data.get("type")

                                    if event_type == "token":
                                        answer_text += data.get("data", "")
                                        placeholder.markdown(answer_text)
                                    elif event_type == "citations":
                                        citations_data = data.get("data", {})
                                    elif event_type in {"warning", "consistency_warning"}:
                                        warnings_data.append(data.get("data", {}))
                                    elif event_type in {"memory_debug", "reward", "evidence_table", "conversation_frame"}:
                                        debug_data[event_type] = data.get("data", {})
                                    elif event_type == "final":
                                        session_id = data.get("data", {}).get("session_id")
                                        if session_id:
                                            st.session_state["chat_session_id"] = session_id

                                except json.JSONDecodeError:
                                    answer_text += data_str
                                    placeholder.markdown(answer_text)

                            elif line.startswith("event:"):
                                continue

                if citations_data:
                    with st.expander("📚 Citations & Sources"):
                        st.json(citations_data)
                if warnings_data:
                    with st.expander("Consistency warnings"):
                        st.json(warnings_data)
                if debug_data:
                    with st.expander("Memory debug"):
                        st.json(debug_data)

                st.session_state["chat_messages"].append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "citations": citations_data,
                        "warnings": warnings_data,
                        "debug": debug_data,
                    }
                )

        except Exception as e:
            st.error(f"Chat failed: {e}")
            import traceback
            st.code(traceback.format_exc())

# ------------------------------------------------------------------
# SUMMARIZE TAB
# ------------------------------------------------------------------
with tab_sum:
    st.header("Summarize Selected Papers conditioned on your question")

    selected_items = st.session_state.get("selected_papers", [])
    if selected_items:
        st.info(f"📌 **{len(selected_items)} items selected**")
    else:
        st.warning("⚠️ No items selected. Please select items from search results.")

    question = st.text_area(
        "Question / focus",
        "What are the main immunotherapy strategies for lung carcinoma and their evidence?"
    )

    if st.button("Run Conditioned Summary", type="primary"):
        if not selected_items:
            st.error("Please select items from search results first")
        else:
            try:
                with httpx.Client(timeout=120) as c:
                    r = c.post(
                        f"{API}/papers/summarize_conditioned",
                        headers=headers,
                        json={"message": question, "items": selected_items, "options": {}},
                    )
                    r.raise_for_status()
                    res = r.json()

                    paragraphs = res.get("paragraphs", [])
                    if paragraphs:
                        st.write("### 📄 Summary")
                        for i, para in enumerate(paragraphs):
                            st.markdown(para.get("text", ""))
                            support = para.get("support", [])
                            if support:
                                with st.expander(f"Supporting evidence ({len(support)} sources)"):
                                    for s in support:
                                        title = s.get('title') or s.get('pmcid') or s.get('paper_id', 'Unknown')
                                        st.caption(f"**Paper:** {title}")

                                        sentence = s.get("sentence") or s.get("text", "")
                                        if sentence:
                                            st.text(sentence)

                                        svos = s.get("svos", [])
                                        if svos:
                                            st.caption("Extracted triplets:")
                                            for svo in svos[:3]:
                                                subj = svo.get('subject', '?')
                                                pred = svo.get('predicate', '?')
                                                obj = svo.get('object', '?')
                                                st.caption(f"  • {subj} - {pred} - {obj}")
                    else:
                        st.warning("No summary generated. The endpoint returned empty paragraphs.")
                        st.json(res)

            except Exception as e:
                st.error(f"Summary failed: {e}")
                import traceback
                st.code(traceback.format_exc())

# ------------------------------------------------------------------
# TRIPLETS / GRAPH TAB
# ------------------------------------------------------------------
with tab_triplets:
    st.subheader("Triplets & Graph Visualization")

    selected_items = st.session_state.get("selected_papers", [])
    if selected_items:
        st.info(f"📌 **{len(selected_items)} items selected**")
    else:
        st.warning("⚠️ No items selected. Please select items from search results.")

    # Graph filtering controls
    st.markdown("**Graph Filters** (applied via API)")
    
    col_conf, col_ebio, col_topk = st.columns(3)
    
    with col_conf:
        conf = st.slider("Min Confidence", 0.0, 1.0, 0.6, 0.05, 
                        help="Minimum confidence score for triplets")
    
    with col_ebio:
        ebio_min = st.slider("Min Biomedical Score", 0.0, 1.0, 0.3, 0.05,
                            help="Minimum EBio probability (biomedical relevance)")
    
    with col_topk:
        top_k = st.number_input("Max Triplets", min_value=10, max_value=200, value=50, step=10,
                               help="Maximum number of triplets to include in graph")

    colA, colB = st.columns(2)

    with colA:
        if st.button("Build/refresh graph from selection", type="primary"):
            if not selected_items:
                st.error("Please select items from search results first")
            else:
                try:
                    items_payload = [
                        {
                            "paper_id": item["paper_id"],
                            "sent_id": item["sent_id"],
                            "text": item.get("text"),
                            "subject": item.get("subject"),
                            "relation": item.get("relation"),
                            "object": item.get("object"),
                            "pmid": item.get("pmid"),
                            "pmcid": item.get("pmcid"),
                        }
                        for item in selected_items
                        if item.get("paper_id") and item.get("sent_id")
                    ]

                    if not items_payload:
                        st.error("No valid sentence IDs found in selection")
                    else:
                        with httpx.Client(timeout=120) as c:
                            r = c.post(
                                f"{API}/triplets/graph/build",
                                headers=headers,
                                json=items_payload,
                                params={
                                    "confidence_min": conf,
                                    "ebio_min": ebio_min,
                                    "top_k": top_k,
                                },
                            )
                            r.raise_for_status()
                            result = r.json()

                            triple_ids = result.get("triple_ids", [])
                            st.session_state["graph_triple_ids"] = triple_ids
                            st.success(f"✅ Graph built with {len(triple_ids)} triplets (filtered from papers)")

                            with st.expander("Graph Build Details"):
                                st.json(result.get("debug", result))

                except Exception as e:
                    st.error(f"Graph build failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    with colB:
        if st.button("Open graph viewer"):
            triple_ids = st.session_state.get("graph_triple_ids", [])
            if triple_ids:
                ids_str = ",".join(str(id) for id in triple_ids)

                if PUBLIC_API:
                    base_url = f"{PUBLIC_API}/triplets/graph/view"
                else:
                    base_url = "/triplets/graph/view"

                url = (
                    f"{base_url}"
                    f"?triple_ids={ids_str}"
                    f"&confidence_min={conf}"
                    f"&tenant={TENANT}"
                )
                st.markdown(f"[🔗 Open Graph in new tab]({url})")
            else:
                st.warning("No graph built yet. Please click 'Build/refresh graph' first.")

    if "graph_triple_ids" in st.session_state:
        st.caption(f"Current graph: **{len(st.session_state['graph_triple_ids'])} triplets**")
