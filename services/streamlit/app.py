# services/streamlit/app.py
from __future__ import annotations

import base64
import hmac
import json
import os
import re
from pathlib import Path

import httpx
import streamlit as st


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

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            logo_path = Path(__file__).parent / "Sabia_RN.png"
            if logo_path.exists():
                encoded = base64.b64encode(logo_path.read_bytes()).decode()
                st.markdown(
                    f'<div style="text-align: center; margin-bottom: 20px;">'
                    f'<img src="data:image/png;base64,{encoded}" style="max-width: 250px;">'
                    f'</div>',
                    unsafe_allow_html=True,
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
    try:
        img_bytes = Path(filepath).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return f"data:image/png;base64,{encoded}"
    except FileNotFoundError:
        return ""


st.set_page_config(layout="wide", page_title="Sabia Chat")

if not check_password():
    st.stop()

API = os.environ.get("API_URL", "http://api:8080")
PUBLIC_API = os.environ.get("PUBLIC_API_URL", "")
if not PUBLIC_API:
    PUBLIC_API = "http://localhost:18081" if API.startswith("http://api:") else API

st.markdown(
    """
    <style>
    .custom-logo-img {
        width: 11vw !important;
        min-width: 110px;
        max-width: 220px;
        height: auto !important;
        padding: 6px 0 !important;
        margin-top: 4px !important;
        object-fit: contain !important;
    }
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 0;
    }
    .st-emotion-cache-12fmw1f {
        visibility: hidden;
    }
    [data-testid="stDataEditor"] {
        font-size: 13px;
    }
    [data-testid="stDataEditor"] td {
        white-space: normal !important;
        word-wrap: break-word !important;
        max-width: 400px;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

logo_base64_src = get_base64_of_local_file("Sabia_RN.png")
logo_col, title_col = st.columns([1, 5])
with logo_col:
    if logo_base64_src:
        st.markdown(f'<img src="{logo_base64_src}" class="custom-logo-img">', unsafe_allow_html=True)
with title_col:
    st.title("Sabia Chatbot")
    st.caption("Grounded biomedical chat with optional search, memory, and graph workflows.")

TENANT = st.sidebar.text_input("Tenant", "default")
try:
    lab_info_path = Path(__file__).parent / "lab_info.md"
    if lab_info_path.exists():
        st.sidebar.markdown("---")
        st.sidebar.markdown(lab_info_path.read_text())
except Exception:
    pass

API_KEY = os.environ.get("API_KEY", "dev")
headers = {"X-Tenant-Id": TENANT, "X-API-Key": API_KEY, "Content-Type": "application/json"}


@st.cache_data(ttl=60)
def fetch_model_catalog(api: str, tenant: str, api_key: str) -> list[dict]:
    request_headers = {"X-Tenant-Id": tenant, "X-API-Key": api_key, "Content-Type": "application/json"}
    fallback = [
        {
            "provider": "nvidia",
            "model": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
            "api_format": "openai_chat",
            "available": False,
            "source": "fallback",
        },
        {
            "provider": "nvidia",
            "model": "nvidia/nemotron-3-ultra-550b-a55b",
            "api_format": "openai_chat",
            "available": False,
            "source": "fallback",
        },
        {
            "provider": "nvidia",
            "model": "nvidia/nemotron-3-super-120b-a12b",
            "api_format": "openai_chat",
            "available": False,
            "source": "fallback",
        },
        {
            "provider": "nvidia",
            "model": "nvidia/nemotron-3-nano-30b-a3b",
            "api_format": "openai_chat",
            "available": False,
            "source": "fallback",
        },
        {
            "provider": "nvidia",
            "model": "nvidia/llama-3.1-nemotron-70b-instruct",
            "api_format": "openai_chat",
            "available": False,
            "source": "fallback",
        },
        {
            "provider": "nvidia",
            "model": "meta/llama-3.1-8b-instruct",
            "api_format": "openai_chat",
            "available": False,
            "source": "fallback",
        },
    ]
    try:
        with httpx.Client(timeout=20) as c:
            r = c.get(f"{api}/chat/models", headers=request_headers)
            r.raise_for_status()
            models = r.json().get("models", [])
            return models or fallback
    except Exception:
        return fallback


def model_label(item: dict) -> str:
    status = "live" if item.get("available") else item.get("source", "preset")
    return f"{item.get('model')}  [{item.get('provider')}, {status}]"


def select_model(label: str, key: str, models: list[dict]) -> dict:
    default_idx = 0
    for idx, item in enumerate(models):
        if item.get("available") and "nemotron" in item.get("model", "").lower():
            default_idx = idx
            break
    selected = st.selectbox(label, models, index=default_idx, key=key, format_func=model_label)
    custom = st.text_input(f"Custom {label.lower()} id", key=f"{key}_custom", placeholder="optional provider model id")
    if custom.strip():
        return {
            "provider": selected.get("provider", "nvidia"),
            "model": custom.strip(),
            "api_format": selected.get("api_format", "openai_chat"),
        }
    return selected


def selected_model_payload(prefix: str, item: dict) -> dict:
    return {
        f"{prefix}_provider": item.get("provider"),
        f"{prefix}_model": item.get("model"),
        f"{prefix}_api_format": item.get("api_format", "openai_chat"),
    }


def extract_clarification_options(text: str) -> list:
    """Parse lettered option lists (a/b/c...) from model response text.

    SHARED CONTRACT: the same detection rule (lettered, >=2, consecutive from 'a')
    is mirrored in the backend intent router
    (services/api/app/memory/intent_router.py::_text_offers_lettered_options),
    which uses it to bias context-poor intent toward prior_context. Keep both in
    sync so the UI checkbox launch and backend routing agree on what a
    multi-option clarification is.
    """
    lines = text.split("\n")
    options = []
    opt_re = re.compile(r"^\s*\(?([a-z])\)?[.:\)]\s*(.+)", re.IGNORECASE)
    for line in lines:
        m = opt_re.match(line.rstrip())
        if m:
            letter = m.group(1).lower()
            desc = m.group(2).strip()
            if desc:
                options.append((letter, desc))
    if len(options) >= 2:
        letters = [o[0] for o in options]
        if (letters[0] == "a" and
                all(ord(letters[i]) == ord(letters[i - 1]) + 1
                    for i in range(1, min(len(letters), 5)))):
            return options[:5]
    return []


def render_chat():
    models = fetch_model_catalog(API, TENANT, API_KEY)

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    if "chat_session_id" not in st.session_state:
        st.session_state["chat_session_id"] = None
    if "clarif_submitted_turns" not in st.session_state:
        st.session_state["clarif_submitted_turns"] = set()
    if "turn_ratings" not in st.session_state:
        st.session_state["turn_ratings"] = {}
    injected_msg = st.session_state.pop("pending_message", None)

    top_left, top_right = st.columns([4, 1])
    with top_left:
        if st.session_state["chat_session_id"]:
            st.caption(f"Session ID: `{st.session_state['chat_session_id']}`")
        else:
            st.caption("New chat session")
    with top_right:
        if st.button("New chat", use_container_width=True):
            st.session_state["chat_messages"] = []
            st.session_state["chat_session_id"] = None
            st.session_state["memory_diagnostics"] = {}
            st.session_state["clarif_submitted_turns"] = set()
            st.session_state["turn_ratings"] = {}
            st.session_state.pop("pending_message", None)
            st.rerun()

    with st.expander("Models, context, retrieval, and diagnostics", expanded=False):
        model_col, context_col = st.columns(2)
        use_agent_routing = st.checkbox(
            "Use server-side per-agent model routing",
            value=False,
            help="Each stage uses its configured model: frame->nemotron-super-49b, context_manager->nemotron-3-super-120b (reasoning=medium), answer->nemotron-super-49b, reflection->nemotron-super-49b",
        )
        if not use_agent_routing:
            with model_col:
                chat_model = select_model("Chat model", "chat_model_choice", models)
            with context_col:
                use_same_model = st.checkbox("Use chat model for context manager and search agents", value=True)
                context_model = chat_model if use_same_model else select_model("Context/search model", "context_model_choice", models)
        else:
            st.caption("Per-agent routing active · frame: nemotron-super-49b · context: nemotron-3-super-120b (reasoning=med) · answer: nemotron-super-49b · reflection: nemotron-super-49b")
            chat_model = None
            context_model = None

        retrieval_col, debug_col = st.columns(2)
        with retrieval_col:
            allow_extra = st.checkbox("Allow extra retrieval if pinned < 3", value=False)
            allow_auto_context = st.checkbox("Auto-search evidence when nothing is pinned", value=True)
            allow_web_search = st.checkbox("Use privacy-filtered DuckDuckGo context if local memory is sparse", value=True)
        with debug_col:
            expose_memory_debug = st.checkbox("Show memory debug events", value=False)
            if models:
                live_count = sum(1 for item in models if item.get("available"))
                st.caption(f"{live_count} live model(s) discovered from provider catalog.")

        selected_items = st.session_state.get("selected_papers", [])
        if selected_items:
            st.info(f"{len(selected_items)} items pinned as context")
            for item in selected_items[:8]:
                st.caption(f"**{item.get('subject')}** - {item.get('relation')} - **{item.get('object')}**")
                st.text(item.get("text", "")[:240] + ("..." if len(item.get("text", "")) > 240 else ""))
        else:
            st.info("No pinned context. Chat can auto-search local evidence when enabled.")

        st.markdown("**Search past chat messages**")
        hist_q = st.text_input("Find in chat history", placeholder="e.g., PD-1 dosage window")
        if st.button("Search history"):
            try:
                with httpx.Client(timeout=30) as c:
                    r = c.get(f"{API}/chat/history/search", headers=headers, params={"q": hist_q})
                    st.session_state["history_hits"] = r.json().get("matches", []) if r.status_code == 200 else []
            except Exception as e:
                st.error(f"History search failed: {e}")
        if "history_hits" in st.session_state:
            st.json(st.session_state["history_hits"])

        if st.session_state["chat_session_id"]:
            st.markdown("**Session diagnostics**")
            if "memory_diagnostics" not in st.session_state:
                st.session_state["memory_diagnostics"] = {}
            diag_targets = [
                ("Ideas", "ideas"),
                ("Action values", "action-values"),
                ("Evidence tables", "evidence-tables"),
                ("Search notes", "search-notes"),
            ]
            diag_cols = st.columns(len(diag_targets))
            for col, (diag_label, endpoint) in zip(diag_cols, diag_targets):
                with col:
                    if st.button(diag_label, key=f"diag_{endpoint}", use_container_width=True):
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
                with st.expander("Citations & Sources"):
                    st.json(entry["citations"])
            if entry.get("warnings"):
                with st.expander("Consistency warnings"):
                    st.json(entry["warnings"])
            if entry.get("debug"):
                with st.expander("Memory debug"):
                    st.json(entry["debug"])

    # ── Human feedback: star rating + clarification checkboxes ──
    last_asst_idx = None
    for _i, _e in enumerate(st.session_state["chat_messages"]):
        if _e.get("role") == "assistant":
            last_asst_idx = _i
    if last_asst_idx is not None:
        _last = st.session_state["chat_messages"][last_asst_idx]
        _turn_id = last_asst_idx
        _ratings = st.session_state["turn_ratings"]
        if _turn_id not in _ratings:
            _r = st.feedback("stars", key=f"feedback_{_turn_id}")
            if _r is not None:
                _ratings[_turn_id] = _r + 1  # feedback() returns 0-4
                st.session_state["turn_ratings"] = _ratings
        else:
            _stars = '\u2b50' * _ratings[_turn_id]
            st.caption(f"Your rating: {_stars}")
        _clarif_opts = _last.get("clarification_options") or []
        _form_key = f"clarif_done_{_turn_id}"
        if _clarif_opts and _turn_id not in st.session_state["clarif_submitted_turns"]:
            st.divider()
            with st.form(f"clarif_form_{_turn_id}"):
                st.caption("Select the option(s) that match your intent (or type your own):")
                _selected_descs = []
                for _letter, _desc in _clarif_opts:
                    if st.checkbox(f"**{_letter.upper()})** {_desc}", key=f"ck_{_turn_id}_{_letter}"):
                        _selected_descs.append(_desc)
                _other_text = st.text_input("Other (describe your intent):", key=f"other_{_turn_id}")
                if st.form_submit_button("\u2192 Send selection"):
                    st.session_state["clarif_submitted_turns"].add(_turn_id)
                    _parts = _selected_descs + ([_other_text.strip()] if _other_text.strip() else [])
                    if _parts:
                        st.session_state["pending_message"] = "; ".join(_parts)
                    st.rerun()

    msg = injected_msg or st.chat_input("Ask Sabia")
    if not msg:
        return

    selected_items = st.session_state.get("selected_papers", [])
    st.session_state["chat_messages"].append({"role": "user", "content": msg})
    with st.chat_message("user"):
        st.markdown(msg)

    try:
        with st.chat_message("assistant"):
            answer_text = ""
            citations_data = None
            warnings_data = []
            debug_data = {}
            stream_error = None
            placeholder = st.empty()

            payload_options = {
                "allow_extra_retrieval": allow_extra,
                "allow_auto_context": allow_auto_context,
                "allow_web_search": allow_web_search,
                "expose_memory_debug": expose_memory_debug,
                **(selected_model_payload("chat", chat_model) if chat_model is not None else {}),
                **(selected_model_payload("context", context_model) if context_model is not None else {}),
            }
            payload = {"message": msg, "items": selected_items, "options": payload_options}
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
                        if line.startswith("event:"):
                            continue
                        if not line.startswith("data:"):
                            continue

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
                            elif event_type == "error":
                                stream_error = (data.get("data") or {}).get("message") or "The chat stream stopped before completion."
                                warnings_data.append(data.get("data", {}))
                                break
                            elif event_type in {"memory_debug", "reward", "evidence_table", "conversation_frame", "semantic_drift_trace"}:
                                debug_data[event_type] = data.get("data", {})
                            elif event_type == "final":
                                session_id = data.get("data", {}).get("session_id")
                                if session_id:
                                    st.session_state["chat_session_id"] = session_id
                        except json.JSONDecodeError:
                            answer_text += data_str
                            placeholder.markdown(answer_text)

            if stream_error:
                st.warning(stream_error)
            if citations_data:
                with st.expander("Citations & Sources"):
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
                    "clarification_options": extract_clarification_options(answer_text),
                }
            )
    except httpx.RemoteProtocolError:
        st.warning("The chat stream was interrupted before completion. This can happen if the API restarts during a response; please retry the message.")
        if answer_text:
            st.session_state["chat_messages"].append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "citations": citations_data,
                    "warnings": warnings_data + [{"message": "Stream interrupted before final event."}],
                    "debug": debug_data,
                    "clarification_options": extract_clarification_options(answer_text),
                }
            )
    except Exception as e:
        st.error(f"Chat failed: {e}")
        import traceback

        st.code(traceback.format_exc())


def render_search_select():
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
                        results.sort(key=lambda x: x.get("confidence", 0) or 0, reverse=True)
                        st.session_state["results"] = results
                        st.session_state["selected_papers"] = []
                        st.success(f"Found {len(results)} results")
                except Exception as e:
                    st.error(f"Search failed: {e}")

        with col2:
            if "results" in st.session_state and st.session_state["results"]:
                st.caption(f"**{len(st.session_state['results'])} results** loaded (sorted by confidence)")

    chosen: list[dict] = []
    if "results" not in st.session_state or not st.session_state["results"]:
        return

    st.subheader("Search Results")
    rows = []
    for hit in st.session_state["results"]:
        rows.append(
            {
                "Select": False,
                "PMCID": hit.get("pmcid") or "-",
                "Text": hit.get("text", ""),
                "Confidence": round(hit.get("confidence", 0), 4) if hit.get("confidence") else 0.0,
                "Subject": hit.get("subject") or "-",
                "Relation": hit.get("relation") or "-",
                "Object": hit.get("object") or "-",
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
            "Confidence": st.column_config.NumberColumn("Conf", format="%.3f", width="small"),
            "Subject": st.column_config.TextColumn("Subject", width="medium"),
            "Relation": st.column_config.TextColumn("Relation", width="small"),
            "Object": st.column_config.TextColumn("Object", width="medium"),
            "_paper_id": None,
            "_sent_id": None,
            "_pmid": None,
            "_score": None,
        },
        column_order=["PMCID", "Subject", "Relation", "Object", "Select", "Text", "Confidence"],
        key="results_table",
    )

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


def render_summary():
    st.header("Summarize Selected Papers conditioned on your question")
    selected_items = st.session_state.get("selected_papers", [])
    if selected_items:
        st.info(f"{len(selected_items)} items selected")
    else:
        st.warning("No items selected. Select items from Search / Select first.")

    question = st.text_area("Question / focus", "What are the main immunotherapy strategies for lung carcinoma and their evidence?")
    if not st.button("Run Conditioned Summary", type="primary"):
        return
    if not selected_items:
        st.error("Please select items from search results first")
        return

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
        if not paragraphs:
            st.warning("No summary generated. The endpoint returned empty paragraphs.")
            st.json(res)
            return
        st.write("### Summary")
        for para in paragraphs:
            st.markdown(para.get("text", ""))
            support = para.get("support", [])
            if support:
                with st.expander(f"Supporting evidence ({len(support)} sources)"):
                    for s in support:
                        title = s.get("title") or s.get("pmcid") or s.get("paper_id", "Unknown")
                        st.caption(f"**Paper:** {title}")
                        sentence = s.get("sentence") or s.get("text", "")
                        if sentence:
                            st.text(sentence)
                        svos = s.get("svos", [])
                        if svos:
                            st.caption("Extracted triplets:")
                            for svo in svos[:3]:
                                st.caption(f"  - {svo.get('subject', '?')} - {svo.get('predicate', '?')} - {svo.get('object', '?')}")
    except Exception as e:
        st.error(f"Summary failed: {e}")
        import traceback

        st.code(traceback.format_exc())


def render_triplets():
    st.subheader("Triplets & Graph Visualization")
    selected_items = st.session_state.get("selected_papers", [])
    if selected_items:
        st.info(f"{len(selected_items)} items selected")
    else:
        st.warning("No items selected. Select items from Search / Select first.")

    st.markdown("**Graph Filters**")
    col_conf, col_ebio, col_topk = st.columns(3)
    with col_conf:
        conf = st.slider("Min Confidence", 0.0, 1.0, 0.6, 0.05, help="Minimum confidence score for triplets")
    with col_ebio:
        ebio_min = st.slider("Min Biomedical Score", 0.0, 1.0, 0.3, 0.05, help="Minimum EBio probability")
    with col_topk:
        top_k = st.number_input("Max Triplets", min_value=10, max_value=200, value=50, step=10)

    col_a, col_b = st.columns(2)
    with col_a:
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
                                params={"confidence_min": conf, "ebio_min": ebio_min, "top_k": top_k},
                            )
                            r.raise_for_status()
                            result = r.json()
                        triple_ids = result.get("triple_ids", [])
                        st.session_state["graph_triple_ids"] = triple_ids
                        st.success(f"Graph built with {len(triple_ids)} triplets")
                        with st.expander("Graph Build Details"):
                            st.json(result.get("debug", result))
                except Exception as e:
                    st.error(f"Graph build failed: {e}")
                    import traceback

                    st.code(traceback.format_exc())

    with col_b:
        if st.button("Open graph viewer"):
            triple_ids = st.session_state.get("graph_triple_ids", [])
            if triple_ids:
                ids_str = ",".join(str(id) for id in triple_ids)
                base_url = f"{PUBLIC_API}/triplets/graph/view" if PUBLIC_API else "/triplets/graph/view"
                url = f"{base_url}?triple_ids={ids_str}&confidence_min={conf}&tenant={TENANT}"
                st.markdown(f"[Open Graph in new tab]({url})")
            else:
                st.warning("No graph built yet. Build or refresh the graph first.")

    if "graph_triple_ids" in st.session_state:
        st.caption(f"Current graph: **{len(st.session_state['graph_triple_ids'])} triplets**")


tab_chat, tab_search, tab_sum, tab_triplets = st.tabs(
    ["Chatbot", "Search / Select", "Summarize", "Triplets / Graph"]
)

with tab_chat:
    render_chat()

with tab_search:
    render_search_select()

with tab_sum:
    render_summary()

with tab_triplets:
    render_triplets()
