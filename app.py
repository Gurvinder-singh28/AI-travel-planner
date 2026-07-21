"""
Streamlit front-end for the LangGraph multi-agent Travel Planner.

Run with:
    streamlit run app.py
"""

import os
import time
from datetime import date

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Try to import the compiled graph. Kept optional so the UI still renders
# (with a clear setup notice) even if graph.py / deps aren't wired up yet.
# ---------------------------------------------------------------------------
GRAPH_IMPORT_ERROR = None
try:
    from graph import travel_planner_application
except Exception as exc:  # noqa: BLE001
    travel_planner_application = None
    GRAPH_IMPORT_ERROR = str(exc)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Styling — "boarding pass / travel dossier" visual identity
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

    :root {
        --ink: #16213E;
        --ink-soft: #2B3A5E;
        --paper: #F6F1E7;
        --paper-dim: #EDE6D6;
        --brass: #C99A2E;
        --teal: #3A6B67;
        --rust: #B54A3F;
    }

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 10%, #1c2b4d 0%, #16213E 45%, #101a33 100%);
    }

    section[data-testid="stSidebar"] {
        background: var(--ink-soft);
        border-right: 1px solid rgba(201,154,46,0.25);
    }
    section[data-testid="stSidebar"] * {
        color: #EDE6D6 !important;
    }
    section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] textarea {
        color: var(--ink) !important;
    }

    /* Hero ticket header */
    .ticket-hero {
        background: var(--paper);
        border-radius: 18px;
        padding: 34px 42px;
        position: relative;
        box-shadow: 0 20px 45px rgba(0,0,0,0.35);
        border: 1px solid rgba(0,0,0,0.06);
        margin-bottom: 8px;
    }
    .ticket-hero::before, .ticket-hero::after {
        content: "";
        position: absolute;
        width: 26px;
        height: 26px;
        background: #101a33;
        border-radius: 50%;
        top: 50%;
        transform: translateY(-50%);
    }
    .ticket-hero::before { left: -13px; }
    .ticket-hero::after { right: -13px; }

    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 3px;
        font-size: 12px;
        color: var(--teal);
        text-transform: uppercase;
        font-weight: 600;
    }
    .hero-title {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 40px;
        color: var(--ink);
        margin: 6px 0 2px 0;
        line-height: 1.05;
    }
    .hero-sub {
        font-family: 'IBM Plex Mono', monospace;
        color: #6b6250;
        font-size: 13.5px;
    }

    .perforation {
        border-top: 2px dashed rgba(43,58,94,0.25);
        margin: 26px 0 22px 0;
        position: relative;
    }

    .route-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'IBM Plex Mono', monospace;
        color: var(--ink);
    }
    .route-code {
        font-family: 'Fraunces', serif;
        font-size: 26px;
        font-weight: 700;
        color: var(--ink);
    }
    .route-label {
        font-size: 11px;
        letter-spacing: 2px;
        color: var(--teal);
        text-transform: uppercase;
    }
    .route-arrow {
        flex: 1;
        text-align: center;
        color: var(--brass);
        font-size: 20px;
    }

    /* Agent pipeline stops */
    .stop-track {
        display: flex;
        align-items: center;
        margin: 18px 0 6px 0;
    }
    .stop-dot {
        width: 14px; height: 14px; border-radius: 50%;
        border: 2px solid var(--brass);
        background: transparent;
        flex-shrink: 0;
    }
    .stop-dot.done { background: var(--brass); }
    .stop-dot.active { background: var(--teal); border-color: var(--teal); box-shadow: 0 0 0 4px rgba(58,107,103,0.25); }
    .stop-line { flex: 1; height: 2px; background: rgba(201,154,46,0.35); }
    .stop-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        color: #cfc8b7;
        text-align: center;
        width: 130px;
        flex-shrink: 0;
    }

    /* Dossier card for output */
    .dossier {
        background: var(--paper);
        border-radius: 14px;
        padding: 36px 40px;
        box-shadow: 0 20px 45px rgba(0,0,0,0.35);
        border: 1px solid rgba(0,0,0,0.06);
        color: var(--ink);
        position: relative;
    }
    .stamp {
        position: absolute;
        top: 22px;
        right: 30px;
        border: 2px solid var(--rust);
        color: var(--rust);
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 11px;
        letter-spacing: 2px;
        padding: 5px 10px;
        border-radius: 6px;
        transform: rotate(6deg);
        opacity: 0.85;
    }
    .dossier h2, .dossier h3 { font-family: 'Fraunces', serif; color: var(--ink); }
    .dossier p, .dossier li { color: #2c2c2c; }

    div.stButton > button {
        background: var(--brass);
        color: var(--ink);
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        letter-spacing: 1px;
        border: none;
        border-radius: 8px;
        padding: 0.6em 1.4em;
        width: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(201,154,46,0.35);
        color: var(--ink);
    }

    .error-strip {
        background: rgba(181,74,63,0.12);
        border-left: 4px solid var(--rust);
        padding: 12px 16px;
        border-radius: 6px;
        color: #f3d9d4;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar — optional structured hints (the real input is the query box below)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎫 AI Travel Planner")
    st.caption(
        "Type your trip request in plain English in the main panel. "
        "These fields are optional — use them only if you want quick presets "
        "to auto-fill the query box."
    )

    origin = st.text_input("Origin city (optional)", value="")
    destination = st.text_input("Destination (optional)", value="")
    days = st.slider("Trip length (days)", min_value=1, max_value=30, value=6)
    budget = st.number_input(
        "Budget (INR, optional)", min_value=0, step=5000, value=0, format="%d"
    )
    depart_date = st.date_input("Preferred departure (optional)", value=date.today())

    if st.button("↳ Fill query box from fields"):
        pieces = []
        if origin and destination:
            pieces.append(f"Plan a {days}-day trip from {origin} to {destination}")
        elif destination:
            pieces.append(f"Plan a {days}-day trip to {destination}")
        else:
            pieces.append(f"Plan a {days}-day trip")
        if budget:
            pieces.append(f"with a budget of {int(budget)} INR")
        pieces.append(f"departing around {depart_date.strftime('%d %b %Y')}")
        st.session_state["query_box"] = " ".join(pieces) + "."

    st.markdown("---")
    st.markdown("### 🔑 Environment")
    key_present = bool(os.getenv("MISTRAL_API_KEY"))
    st.write("MISTRAL_API_KEY:", "✅ found" if key_present else "❌ missing")
    if not key_present:
        st.caption("Add `MISTRAL_API_KEY` to your `.env` file to enable the pipeline.")


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="ticket-hero">
        <div class="eyebrow">Production Agentic AI · Multi-Agent Travel Planner Engine</div>
        <div class="hero-title">AI Travel Planner</div>
        <div class="hero-sub">Tell the agents where you want to go — they'll research, budget, and build the itinerary</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ---------------------------------------------------------------------------
# Query box — the actual user_query sent to the graph
# ---------------------------------------------------------------------------
if "query_box" not in st.session_state:
    st.session_state["query_box"] = ""

col_q, col_btn = st.columns([5, 1])
with col_q:
    user_query = st.text_area(
        "Ask the travel planner anything",
        key="query_box",
        placeholder='e.g. "Plan a 6-day trip from Delhi to Japan with a budget of 200000 INR."',
        height=90,
        label_visibility="collapsed",
    )
with col_btn:
    st.write("")
    st.write("")
    run_clicked = st.button("✈️  Plan Trip")

PIPELINE_STAGES = ["Intake", "Research", "Budgeting", "Itinerary", "Compile"]


def render_stage_track(active_index: int, total: int):
    """Render the agent pipeline as boarding stops. active_index=-1 means idle,
    active_index=total means fully complete."""
    cols = st.columns([1] + [0.3, 1] * (len(PIPELINE_STAGES) - 1))
    html_parts = ['<div class="stop-track">']
    for i, stage in enumerate(PIPELINE_STAGES):
        state = "done" if i < active_index else ("active" if i == active_index else "")
        html_parts.append(f'<div class="stop-dot {state}"></div>')
        if i < len(PIPELINE_STAGES) - 1:
            html_parts.append('<div class="stop-line"></div>')
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)
    label_cols = st.columns(len(PIPELINE_STAGES))
    for c, stage in zip(label_cols, PIPELINE_STAGES):
        c.markdown(f'<div class="stop-label">{stage}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
if run_clicked:
    if GRAPH_IMPORT_ERROR:
        st.markdown(
            f'<div class="error-strip">⚠️ Could not import <code>graph.py</code>: '
            f"{GRAPH_IMPORT_ERROR}</div>",
            unsafe_allow_html=True,
        )
    elif not key_present:
        st.markdown(
            '<div class="error-strip">⚠️ MISTRAL_API_KEY is missing — add it to your '
            ".env file before running the pipeline.</div>",
            unsafe_allow_html=True,
        )
    elif not user_query.strip():
        st.markdown(
            '<div class="error-strip">⚠️ Type a trip request in the query box first '
            "(e.g. \"Plan a 6-day trip from Delhi to Japan with a budget of 200000 INR.\").</div>",
            unsafe_allow_html=True,
        )
    else:
        query = user_query.strip()

        initial_payload = {
            "user_query": query,
            "conversation_history": [],
            "errors": [],
        }

        stage_slot = st.empty()
        status_slot = st.empty()

        with stage_slot:
            render_stage_track(0, len(PIPELINE_STAGES))
        status_slot.info("Orchestrating multi-agent execution pipeline…")

        result = None
        try:
            # Prefer streaming so we can reflect real node progress; fall back
            # to a plain invoke if the compiled graph doesn't support .stream().
            if hasattr(travel_planner_application, "stream"):
                last_state = {}
                step = 0
                for update in travel_planner_application.stream(initial_payload):
                    step += 1
                    last_state.update(
                        {k: v for node_out in update.values() for k, v in node_out.items()}
                    )
                    with stage_slot:
                        render_stage_track(
                            min(step, len(PIPELINE_STAGES) - 1), len(PIPELINE_STAGES)
                        )
                    time.sleep(0.05)
                with stage_slot:
                    render_stage_track(len(PIPELINE_STAGES), len(PIPELINE_STAGES))
                result = last_state or travel_planner_application.invoke(initial_payload)
            else:
                result = travel_planner_application.invoke(initial_payload)
                with stage_slot:
                    render_stage_track(len(PIPELINE_STAGES), len(PIPELINE_STAGES))

            status_slot.empty()

        except Exception as runtime_fault:  # noqa: BLE001
            status_slot.empty()
            st.markdown(
                f'<div class="error-strip">💥 Critical Application Orchestration '
                f"Core Fault: {str(runtime_fault)}</div>",
                unsafe_allow_html=True,
            )
            result = None

        if result is not None:
            st.write("")
            if result.get("errors"):
                st.markdown("#### ❌ Pipeline execution encountered handling exceptions")
                for err in result["errors"]:
                    st.markdown(
                        f'<div class="error-strip">{err}</div>', unsafe_allow_html=True
                    )
            else:
                final_answer = result.get(
                    "final_answer", "_No answer was produced by the pipeline._"
                )
                subtitle = query if len(query) <= 90 else query[:87] + "…"
                st.markdown(
                    f"""
                    <div class="dossier">
                        <div class="stamp">APPROVED</div>
                        <h2>Destination Travel Plan</h2>
                        <p style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:#6b6250;">
                            {subtitle}
                        </p>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(final_answer)
                st.markdown("</div>", unsafe_allow_html=True)

                file_slug = (destination or "trip").lower().replace(" ", "_") or "trip"
                st.download_button(
                    "⬇️  Download dossier as text",
                    data=str(final_answer),
                    file_name=f"travel_dossier_{file_slug}.txt",
                    mime="text/plain",
                )
else:
    st.markdown(
        """
        <div class="dossier" style="opacity:0.85;">
            <div class="stamp">PENDING</div>
            <h3>No dossier generated yet</h3>
            <p>Fill in your trip details on the left and press
            <strong>Generate Travel Dossier</strong> to send the brief through the
            research, budgeting, and itinerary agents.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div style="text-align:center;color:#7d7458;font-family:'IBM Plex Mono',monospace;
                font-size:11px;margin-top:28px;">
        LangGraph Multi-Agent Travel Planner · Compiled Graph Runtime
    </div>
    """,
    unsafe_allow_html=True,
)
