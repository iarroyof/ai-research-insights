# services/streamlit/app.py
from __future__ import annotations

import os
import json
import httpx
import streamlit as st

st.set_page_config(layout="wide", page_title="AI Research Insights – Bench")

API = os.environ.get("API_URL", "http://localhost:18080")
TENANT = st.sidebar.text_input("Tenant", "default")
API_KEY = st.sidebar.text_input("API Key", "dev")
headers = {"X-Tenant-Id": TENANT, "X-API-Key": API_KEY, "Content-Type": "application/json"}

st.title("Search → Select → Run")

# --- Search ---
with st.expander("Search"):
    q = st.text_input("Query", "Immunological therapy for lung carcinoma")
    k = st.number_input("Top K", min_value=1, max_value=100, value=25, step=1)
    if st.button("Search"):
        with httpx.Client(timeout=60) as c:
            r = c.post(
                f"{API}/search",
                headers=headers,
                json={"query": q, "target": "all", "filters": {}, "k": k},
            )
            r.raise_for_status()
            st.session_state["results"] = r.json().get("hits", [])

# Results table with checkbox selection
chosen: list[dict] = []
if "results" in st.session_state and st.session_state["results"]:
    # Add a selection column
    rows = []
    for hit in st.session_state["results"]:
        rows.append(
            {
                "Select": False,
                "paper_id": hit.get("paper_id"),
                "title": hit.get("title"),
                "pmid": hit.get("pmid"),
                "pmcid": hit.get("pmcid"),
                "page": hit.get("page"),
                "sent_id": hit.get("sent_id"),
                "score": hit.get("score"),
                "text": hit.get("text", ""),
            }
        )
    edited = st.data_editor(
        rows,
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn("Select", default=False)},
        key="results_table",
    )
    chosen = [
        {"paper_id": r["paper_id"], "sent_id": r["sent_id"]}
        for r in edited
        if r.get("Select") and r.get("paper_id") and r.get("sent_id") is not None
    ]
    st.caption(f"Selected: {len(chosen)}")

# --- Tabs for actions ---
tab_chat, tab_sum, tab_triplets = st.tabs(["Chat (Pinned Context)", "Summarize (Conditioned)", "Triplets / Graph"])

with tab_chat:
    st.subheader("Live Chat with Pinned Context")
    # Chat history retrieval query (new feature)
    st.markdown("**Search past chat messages** (server-side substring match in this tenant)")
    hist_q = st.text_input("Find in chat history", placeholder="e.g., PD-1 dosage window")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Search history"):
            with httpx.Client(timeout=30) as c:
                r = c.get(f"{API}/chat/history/search", headers=headers, params={"q": hist_q})
                if r.status_code == 200:
                    st.session_state["history_hits"] = r.json().get("matches", [])
                else:
                    st.session_state["history_hits"] = []
    with col2:
        if "history_hits" in st.session_state:
            st.write("Matches:")
            st.json(st.session_state["history_hits"])

    msg = st.text_area("Message", "Summarize the immune therapy evidence from these sentences.")
    allow_extra = st.checkbox("Allow extra retrieval if pinned < 3", value=False)
    if st.button("Start Chat"):
        # stream SSE tokens
        with httpx.Client(timeout=None) as c:
            r = c.post(
                f"{API}/chat",
                headers=headers,
                json={"message": msg, "items": chosen, "options": {"allow_extra_retrieval": allow_extra}},
                stream=True,
            )
            r.raise_for_status()
            st.write("### Answer (live)")
            buf = ""
            placeholder = st.empty()
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    continue
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    # We expect token deltas as plain text (see router)
                    buf += data
                    placeholder.markdown(buf)

with tab_sum:
    st.subheader("Summarize Selected Papers conditioned on your question")
    question = st.text_area("Question / focus", "What are the main immunotherapy strategies for lung carcinoma and their evidence?")
    if st.button("Run Conditioned Summary"):
        with httpx.Client(timeout=120) as c:
            r = c.post(
                f"{API}/papers/summarize_conditioned",
                headers=headers,
                json={"message": question, "items": chosen, "options": {}},
            )
            r.raise_for_status()
            res = r.json()
            st.write("### Summary")
            for para in res.get("paragraphs", []):
                st.markdown(para["text"])
                with st.expander("Supporting sentences (SVO + metadata)"):
                    st.json(para["support"])

with tab_triplets:
    st.subheader("Triplets & Graph")
    conf = st.slider("Confidence ≥", 0.0, 1.0, 0.6, 0.01)
    colA, colB = st.columns(2)
    with colA:
        if st.button("Build/refresh graph from selection"):
            # Build graph (extract if missing)
            with httpx.Client(timeout=120) as c:
                r = c.post(
                    f"{API}/triplets/graph/build",
                    headers=headers,
                    json=chosen,
                    params={"confidence_min": conf},
                )
                r.raise_for_status()
                st.session_state["graph_triple_ids"] = r.json().get("triple_ids", [])
                st.success(f"Triples: {len(st.session_state['graph_triple_ids'])}")
    with colB:
        if st.button("Open graph viewer"):
            ids = ",".join(st.session_state.get("graph_triple_ids", []))
            url = f"{API}/triplets/graph/view?triple_ids={ids}&confidence_min={conf}"
            st.markdown(f"[Open Graph in new tab]({url})")

