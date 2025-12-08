# services/streamlit/app.py
from __future__ import annotations

import os
import json
import httpx
import streamlit as st
import base64
from pathlib import Path

# 🎯 Helper function to convert a local file to a Base64 string
@st.cache_data
def get_base64_of_local_file(filepath: str) -> str:
    """Reads a local file, encodes it to Base64, and returns the data URI string."""
    try:
        # Read the file in binary mode
        img_bytes = Path(filepath).read_bytes()
        # Encode to base64 bytes, then decode to a string for insertion into HTML
        encoded = base64.b64encode(img_bytes).decode()
        # Assume the file is a PNG, adjust MIME type if necessary (e.g., 'image/jpeg')
        return f"data:image/png;base64,{encoded}"
    except FileNotFoundError:
        st.error(f"Error: Logo file not found at '{filepath}'. Please check the path.")
        return ""

st.set_page_config(layout="wide", page_title="AI Research Insights – Bench")

# API Configuration
API = os.environ.get("API_URL", "http://api:8080")

# Public API base (used in browser links)
# In production, this should be set via PUBLIC_API_URL env var
PUBLIC_API = os.environ.get("PUBLIC_API_URL", "")

# If PUBLIC_API not set, use empty string to force relative URLs
# This works when Streamlit is behind the same reverse proxy as the API
if not PUBLIC_API:
    PUBLIC_API = ""  # Will use relative paths
    # If we’re using the internal Docker hostname, map it to the host port
    if API.startswith("http://api:"):
        # Default: API container is exposed on localhost:18081
        PUBLIC_API = "http://localhost:18081"
    else:
        # If API is already a host URL, reuse it
        PUBLIC_API = API

# ... (Custom CSS remains here)
st.markdown(
    """
    <style>
    /* Update logo image CSS with proper height and padding */
    .custom-logo-img {
        width: 12.5vw !important; 
        min-width: 120px; 
        max-width: 250px; 
        height: auto !important;
        padding: 10px 0 !important; /* Add padding to prevent cropping */
        margin-top: 10px !important; /* Add space at the top */
        object-fit: contain !important; /* Ensure the entire logo is visible */
    }
    
    /* Container adjustments */
    .block-container {
        padding-top: 2rem !important; /* Increase top padding */
        padding-bottom: 0;
    }
    
    /* Hide the native Streamlit menu/hamburger button */
    .st-emotion-cache-12fmw1f { 
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Get the encoded logo string
logo_base64_src = get_base64_of_local_file("Sabia_RN.png") # ⬅️ Ensure this path is correct relative to app.py

# Create a column layout for the custom header (Logo | Title)
logo_col, title_col = st.columns([2, 5])

with logo_col:
    # Use HTML to display the image with the Base64 source
    if logo_base64_src:
        st.markdown(
            f'<img src="{logo_base64_src}" class="custom-logo-img">', # ⬅️ Updated src
            unsafe_allow_html=True
        )

with title_col:
    st.title("Sabia: Search → Select → Run")


# --- END OF LOGO/HEADER MODIFICATION ---

TENANT = st.sidebar.text_input("Tenant", "default")
API_KEY = st.sidebar.text_input("API Key", "dev")
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
                    st.session_state["results"] = r.json().get("items", [])
                    st.session_state["selected_papers"] = []  # Clear selections
                    st.success(f"Found {len(st.session_state['results'])} results")
            except Exception as e:
                st.error(f"Search failed: {e}")
    
    with col2:
        if "results" in st.session_state and st.session_state["results"]:
            st.caption(f"**{len(st.session_state['results'])} results** loaded")

# ------------------------------------------------------------------
# RESULTS TABLE WITH CHECKBOX SELECTION
# ------------------------------------------------------------------
chosen: list[dict] = []
if "results" in st.session_state and st.session_state["results"]:
    st.subheader("Search Results")
    
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
                "score": round(hit.get("score", 0), 2),
                "confidence": round(hit.get("confidence", 0), 4) if hit.get("confidence") else None,
                "subject": hit.get("subject"),
                "relation": hit.get("relation"),
                "object": hit.get("object"),
                "text": hit.get("text", ""),
            }
        )

    edited = st.data_editor(
        rows,
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        column_config={
            "Select": st.column_config.CheckboxColumn("Select", default=False),
            "confidence": st.column_config.NumberColumn("Conf", format="%.4f"),
            "score": st.column_config.NumberColumn("Score", format="%.2f"),
        },
        key="results_table",
    )
    
    # Build selection list with all relevant fields
    chosen = [
        {
            "paper_id": r["paper_id"],
            "sent_id": r["sent_id"],
            "text": r["text"],
            "subject": r.get("subject"),
            "relation": r.get("relation"),
            "object": r.get("object"),
            "pmid": r.get("pmid"),
            "pmcid": r.get("pmcid"),
        }
        for r in edited
        if r.get("Select") and r.get("sent_id")
    ]
    
    st.caption(f"**Selected: {len(chosen)}** items")
    
    # Store in session state for use across tabs
    if chosen:
        st.session_state["selected_papers"] = chosen

# ------------------------------------------------------------------
# MAIN ACTION TABS
# ------------------------------------------------------------------
tab_chat, tab_sum, tab_triplets = st.tabs([
    "💬 Chat (Pinned Context)", 
    "📝 Summarize (Conditioned)", 
    "🕸️ Triplets / Graph"
])

# ------------------------------------------------------------------
# CHAT TAB
# ------------------------------------------------------------------
with tab_chat:
    st.subheader("Live Chat with Pinned Context")

    # Chat history retrieval
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

    # Show selected context
    selected_items = st.session_state.get("selected_papers", [])
    if selected_items:
        st.info(f"📌 **{len(selected_items)} items pinned as context**")
        with st.expander("View pinned items"):
            for item in selected_items[:5]:  # Show first 5
                st.caption(f"**{item.get('subject')}** - {item.get('relation')} - **{item.get('object')}**")
                st.text(item.get('text', '')[:200] + "...")

    msg = st.text_area("Message", "Summarize the immune therapy evidence from these sentences.")
    allow_extra = st.checkbox("Allow extra retrieval if pinned < 3", value=False)

    if st.button("Start Chat", type="primary"):
        try:
            # Stream SSE response from API
            with httpx.Client(timeout=None) as c:
                with c.stream(
                    "POST",
                    f"{API}/chat/",
                    headers=headers,
                    json={
                        "message": msg, 
                        "items": selected_items, 
                        "options": {"allow_extra_retrieval": allow_extra}
                    },
                ) as r:
                    r.raise_for_status()
                    
                    st.write("### 🤖 Answer (streaming)")
                    
                    answer_text = ""
                    citations_data = None
                    placeholder = st.empty()
                    
                    for line in r.iter_lines():
                        if not line:
                            continue

                        # Normalize to string
                        if isinstance(line, bytes):
                            line = line.decode("utf-8", errors="ignore")

                        # Parse SSE format: "data: {json}"
                        if line.startswith("data:"):
                            data_str = line[5:].strip()
                            
                            if data_str == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                                
                                # Handle different event types
                                if data.get("type") == "token":
                                    # Streaming text token
                                    answer_text += data.get("data", "")
                                    placeholder.markdown(answer_text)
                                
                                elif data.get("type") == "citations":
                                    # Citations metadata
                                    citations_data = data.get("data", {})
                                
                                elif data.get("type") == "final":
                                    # Final event with session ID
                                    session_id = data.get("data", {}).get("session_id")
                                    if session_id:
                                        st.caption(f"Session ID: `{session_id}`")
                            
                            except json.JSONDecodeError:
                                # Not JSON, might be raw text
                                answer_text += data_str
                                placeholder.markdown(answer_text)
                        
                        elif line.startswith("event:"):
                            # Event type marker (ignore, we use data.type)
                            continue
                    
                    # Show citations if available
                    if citations_data:
                        with st.expander("📚 Citations & Sources"):
                            st.json(citations_data)

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
                # Regular JSON request (not streaming)
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
                                        # Use fallback hierarchy: title -> pmcid -> paper_id
                                        title = s.get('title') or s.get('pmcid') or s.get('paper_id', 'Unknown')
                                        st.caption(f"**Paper:** {title}")
                                        
                                        # Show sentence text if available
                                        sentence = s.get("sentence") or s.get("text", "")
                                        if sentence:
                                            st.text(sentence)
                                        
                                        # Show triplets if available
                                        svos = s.get("svos", [])
                                        if svos:
                                            st.caption("Extracted triplets:")
                                            for svo in svos[:3]:  # Show first 3
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
    
    conf = st.slider("Confidence ≥", 0.0, 1.0, 0.6, 0.01)
    
    colA, colB = st.columns(2)
    
    with colA:
        if st.button("Build/refresh graph from selection", type="primary"):
            if not selected_items:
                st.error("Please select items from search results first")
            else:
                try:
                    # Build payload in the shape expected by /triplets/graph/build:
                    # items: [{paper_id: str, sent_id: str}, ...]
                    items_payload = [
                        {
                            "paper_id": item["paper_id"],
                            "sent_id": item["sent_id"],
                            "text": item.get("text"),
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
                                params={"confidence_min": conf},
                            )
                            r.raise_for_status()
                            result = r.json()

                            triple_ids = result.get("triple_ids", [])
                            st.session_state["graph_triple_ids"] = triple_ids
                            st.success(f"✅ Graph built with {len(triple_ids)} triplets")

                            # Show some stats
                            if result:
                                with st.expander("Graph Details"):
                                    st.json(result)

                except Exception as e:
                    st.error(f"Graph build failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    with colB:
        if st.button("Open graph viewer"):
            triple_ids = st.session_state.get("graph_triple_ids", [])
            if triple_ids:
                ids_str = ",".join(str(id) for id in triple_ids)

                # Use PUBLIC_API if set, otherwise use relative path
                if PUBLIC_API:
                    base_url = f"{PUBLIC_API}/triplets/graph/view"
                else:
                    # Relative URL - works when behind same reverse proxy
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


    # Show current graph stats
    if "graph_triple_ids" in st.session_state:
        st.caption(f"Current graph: **{len(st.session_state['graph_triple_ids'])} triplets**")
