"""Shared, session-safe UI helpers for SmartEstate."""

import streamlit as st


def _sync_numeric(source_key, target_key):
    st.session_state[target_key] = st.session_state[source_key]


def range_control(
    label,
    key,
    min_value,
    max_value,
    default,
    step,
    slider_format,
    help_text=None,
    number_icon=None,
):
    """Render a synchronized slider and exact-value number input."""
    slider_key = f"{key}_slider"
    number_key = f"{key}_number"

    st.session_state.setdefault(slider_key, default)
    st.session_state.setdefault(number_key, st.session_state[slider_key])

    st.markdown(f"**{label}**")
    slider_column, number_column = st.columns(
        [3.2, 1],
        gap="small",
        vertical_alignment="center",
    )

    with slider_column:
        st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            step=step,
            format=slider_format,
            key=slider_key,
            help=help_text,
            label_visibility="collapsed",
            on_change=_sync_numeric,
            args=(slider_key, number_key),
            persist_state="session",
        )

    with number_column:
        st.number_input(
            f"{label} exact value",
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=number_key,
            icon=number_icon,
            label_visibility="collapsed",
            on_change=_sync_numeric,
            args=(number_key, slider_key),
            persist_state="session",
        )

    return st.session_state[slider_key]


def render_brand_header():
    st.html(
        """
        <section class="smart-brand" aria-label="SmartEstate">
            <div class="smart-brand__mark" aria-hidden="true">
                <span class="smart-brand__glyph">⌂</span>
                <span class="smart-brand__gain">↗</span>
            </div>
            <div>
                <h1>SmartEstate</h1>
                <p>SmartEstate – House Price Prediction &amp; GenAI RAG Assistant</p>
            </div>
        </section>
        """
    )


def render_footer():
    st.html(
        """
        <footer class="smart-footer">
            <span class="smart-footer__brand">SmartEstate</span>
            <span class="smart-footer__dot" aria-hidden="true"></span>
            <span>Historical Bengaluru housing data</span>
            <span class="smart-footer__dot" aria-hidden="true"></span>
            <span>Estimates are not current market valuations or financial advice.</span>
        </footer>
        """
    )


def apply_ui_theme(dark_mode):
    if dark_mode:
        colors = {
            "page": "#07111F",
            "surface": "#0E1B2B",
            "surface_alt": "#142438",
            "text": "#F4F7FB",
            "muted": "#9FB0C6",
            "border": "#263A52",
            "accent": "#56C7FF",
            "accent_2": "#63E6BE",
            "shadow": "rgba(0, 0, 0, 0.38)",
            "input": "#101F31",
        }
    else:
        colors = {
            "page": "#F5F8FC",
            "surface": "#FFFFFF",
            "surface_alt": "#EDF4FA",
            "text": "#142033",
            "muted": "#627086",
            "border": "#D8E3ED",
            "accent": "#0F5F9F",
            "accent_2": "#0F8A77",
            "shadow": "rgba(15, 76, 129, 0.12)",
            "input": "#F4F8FB",
        }

    dark_only_css = ""
    if dark_mode:
        dark_only_css = """
        [data-testid="stNumberInputIcon"],
        [data-testid="stNumberInputIcon"] [data-testid="stIconMaterial"] {
            color: #B9CCE0 !important;
            fill: currentColor !important;
            opacity: 1 !important;
        }

        [data-testid="stTextArea"] textarea,
        [data-testid="stTextArea"] textarea:focus {
            color: #F4F7FB !important;
            -webkit-text-fill-color: #F4F7FB !important;
            caret-color: #56C7FF !important;
        }

        [data-testid="stTextArea"] textarea::placeholder {
            color: #AFC0D4 !important;
            -webkit-text-fill-color: #AFC0D4 !important;
            opacity: 1 !important;
        }

        .st-key-edit_inputs button {
            background: #142438 !important;
            border-color: #3A5675 !important;
            color: #F4F7FB !important;
        }

        .st-key-edit_inputs button p,
        .st-key-edit_inputs button [data-testid="stIconMaterial"] {
            color: #F4F7FB !important;
        }

        .st-key-edit_inputs button:hover {
            background: #1A2E46 !important;
            border-color: #56C7FF !important;
        }
        """

    st.html(
        f"""
        <style>
        :root {{
            --smart-page: {colors['page']};
            --smart-surface: {colors['surface']};
            --smart-surface-alt: {colors['surface_alt']};
            --smart-text: {colors['text']};
            --smart-muted: {colors['muted']};
            --smart-border: {colors['border']};
            --smart-accent: {colors['accent']};
            --smart-accent-2: {colors['accent_2']};
            --smart-shadow: {colors['shadow']};
            --smart-input: {colors['input']};
        }}

        {dark_only_css}

        html {{ scroll-behavior: smooth; }}

        .stApp,
        [data-testid="stAppViewContainer"] {{
            background: var(--smart-page);
            color: var(--smart-text);
            transition: background-color .35s ease, color .35s ease;
        }}

        [data-testid="stHeader"] {{
            position: fixed !important;
            inset: 0 0 auto 0;
            z-index: 999;
            min-height: 3.75rem;
            background: color-mix(in srgb, var(--smart-page) 92%, transparent) !important;
            border-bottom: 1px solid color-mix(in srgb, var(--smart-border) 78%, transparent);
            box-shadow: 0 8px 24px color-mix(in srgb, var(--smart-shadow) 55%, transparent);
            backdrop-filter: blur(18px) saturate(1.15);
            -webkit-backdrop-filter: blur(18px) saturate(1.15);
        }}

        .stApp:has(button[aria-label="Close fullscreen"]) [data-testid="stHeader"] {{
            opacity: 0;
            visibility: hidden;
            pointer-events: none;
        }}

        .stApp:has(button[aria-label="Close fullscreen"]) [data-testid="stHeader"] * {{
            pointer-events: none !important;
        }}

        .stMainBlockContainer {{
            max-width: 1440px;
            padding-top: 6.5rem;
            padding-bottom: 1.5rem;
            color: var(--smart-text);
        }}

        .st-key-brand_shell {{ margin-bottom: 1.15rem; }}

        .stMainBlockContainer p,
        .stMainBlockContainer label,
        .stMainBlockContainer h1,
        .stMainBlockContainer h2,
        .stMainBlockContainer h3 {{
            color: var(--smart-text);
        }}

        .smart-brand {{
            display: flex;
            align-items: center;
            gap: .9rem;
        }}

        .smart-brand__mark {{
            display: grid;
            place-items: center;
            width: 3rem;
            height: 3rem;
            flex: 0 0 3rem;
            border-radius: 1rem;
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, var(--smart-accent), var(--smart-accent-2));
            box-shadow: 0 10px 25px color-mix(in srgb, var(--smart-accent) 34%, transparent);
            transition: transform .25s ease, box-shadow .25s ease;
            animation: smartIconEnter .55s cubic-bezier(.2,.8,.2,1);
        }}

        .smart-brand__mark::after {{
            content: "";
            position: absolute;
            inset: .3rem;
            border: 1px solid rgba(255,255,255,.2);
            border-radius: .72rem;
        }}

        .smart-brand__glyph {{
            color: #fff;
            font-size: 2.05rem;
            font-weight: 800;
            line-height: 1;
            transform: translateY(-.08rem);
        }}

        .smart-brand__gain {{
            position: absolute;
            top: .32rem;
            right: .38rem;
            z-index: 1;
            color: #fff;
            font-size: .8rem;
            font-weight: 900;
            line-height: 1;
        }}

        .smart-brand__mark:hover {{
            transform: translateY(-3px) rotate(-4deg) scale(1.05);
            box-shadow: 0 16px 32px color-mix(in srgb, var(--smart-accent) 42%, transparent);
        }}

        .smart-brand h1 {{
            margin: 0;
            font-size: clamp(1.75rem, 3vw, 2.5rem);
            line-height: 1;
            letter-spacing: -.045em;
            background: linear-gradient(100deg, var(--smart-accent), var(--smart-accent-2));
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent !important;
        }}

        .smart-brand p {{
            margin: .35rem 0 0;
            color: var(--smart-muted) !important;
            font-size: .93rem;
        }}

        .smart-footer {{
            display: flex;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
            gap: .55rem;
            margin-top: 1.25rem;
            padding: 1rem 1.25rem;
            border: 1px solid var(--smart-border);
            border-radius: 1rem;
            background: var(--smart-surface);
            color: var(--smart-muted);
            box-shadow: 0 8px 24px color-mix(in srgb, var(--smart-shadow) 65%, transparent);
            font-size: .86rem;
            text-align: center;
        }}

        .smart-footer__brand {{ color: var(--smart-text); font-weight: 750; }}

        .smart-footer__dot {{
            width: .28rem;
            height: .28rem;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--smart-accent), var(--smart-accent-2));
        }}

        .st-key-theme_control,
        .st-key-property_card,
        .st-key-live_card,
        .st-key-result_hero,
        .st-key-comparables_card,
        .st-key-ai_card,
        .st-key-empty_card {{
            background: var(--smart-surface);
            border-color: var(--smart-border) !important;
            box-shadow: 0 10px 32px var(--smart-shadow);
            transition: transform .25s ease, box-shadow .25s ease,
                        background-color .35s ease, border-color .35s ease;
            animation: smartFadeUp .55s cubic-bezier(.2,.8,.2,1) both;
        }}

        .st-key-property_card:hover,
        .st-key-live_card:hover,
        .st-key-result_hero:hover,
        .st-key-comparables_card:hover,
        .st-key-ai_card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 18px 44px var(--smart-shadow);
        }}

        .st-key-live_card {{ animation-delay: .08s; }}
        .st-key-ai_card {{ animation-delay: .18s; }}

        .st-key-comparables_card {{
            transform: none !important;
            animation: smartFadeOnly .55s ease both;
        }}

        .st-key-comparables_card:hover {{
            transform: none !important;
        }}

        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stNumberInputContainer"],
        [data-testid="stTextArea"] textarea {{
            background: var(--smart-input) !important;
            color: var(--smart-text) !important;
            border-color: var(--smart-border) !important;
            transition: border-color .2s ease, box-shadow .2s ease,
                        transform .2s ease;
        }}

        [data-testid="stSelectbox"] [role="group"],
        [data-testid="stSelectbox"] input,
        [data-testid="stSelectbox"] button {{
            background: var(--smart-input) !important;
            color: var(--smart-text) !important;
            border-color: var(--smart-border) !important;
        }}

        [data-testid="stNumberInputField"] {{
            color: var(--smart-text) !important;
        }}

        [data-testid="stTopNavLink"] p,
        [data-testid="stTopNavLink"] [data-testid="stIconMaterial"],
        [data-testid="stHeader"] button {{
            color: var(--smart-text) !important;
        }}

        [data-testid="stTopNavLink"][aria-current="page"] {{
            background: var(--smart-surface-alt) !important;
        }}

        .stMarkdownBadge {{
            color: var(--smart-text) !important;
            background: var(--smart-surface-alt) !important;
            border: 1px solid var(--smart-border);
        }}

        [data-testid="stSelectbox"] [data-baseweb="select"] > div:hover,
        [data-testid="stNumberInputContainer"]:hover,
        [data-testid="stTextArea"] textarea:hover {{
            border-color: var(--smart-accent) !important;
            box-shadow: 0 0 0 3px color-mix(in srgb, var(--smart-accent) 14%, transparent);
        }}

        [data-testid="stNumberInput"] button:hover {{
            color: var(--smart-accent) !important;
            transform: scale(1.08);
        }}

        [data-baseweb="slider"] [role="slider"] {{
            transition: transform .2s ease, box-shadow .2s ease;
        }}

        [data-baseweb="slider"] [role="slider"]:hover {{
            transform: scale(1.18);
            box-shadow: 0 0 0 6px color-mix(in srgb, var(--smart-accent) 18%, transparent);
        }}

        .stButton > button {{
            transition: transform .18s ease, box-shadow .18s ease,
                        filter .18s ease;
        }}

        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 24px color-mix(in srgb, var(--smart-accent) 24%, transparent);
            filter: saturate(1.12);
        }}

        .st-key-primary_cta button {{
            min-height: 3.2rem;
            font-weight: 760;
            background: linear-gradient(100deg, var(--smart-accent), var(--smart-accent-2));
            border: 0;
            color: white;
            background-size: 180% 180%;
            animation: smartGradient 5s ease infinite;
        }}

        .st-key-price_hero [data-testid="stMetricValue"] {{
            font-size: clamp(2.35rem, 5vw, 3.75rem);
            font-weight: 800;
            letter-spacing: -.045em;
        }}

        .st-key-result_price [data-testid="stMetricValue"] {{
            font-size: clamp(1.65rem, 3.2vw, 2.45rem);
            font-weight: 800;
            letter-spacing: -.035em;
        }}

        [data-testid="stMetric"] {{
            transition: transform .22s ease, border-color .22s ease;
        }}

        [data-testid="stMetric"]:hover {{
            transform: translateY(-3px);
            border-color: var(--smart-accent) !important;
        }}

        @keyframes smartFadeUp {{
            from {{ opacity: 0; transform: translateY(16px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        @keyframes smartFadeOnly {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}

        @keyframes smartIconEnter {{
            from {{ opacity: 0; transform: translateY(-8px) scale(.88); }}
            to {{ opacity: 1; transform: translateY(0) scale(1); }}
        }}

        @keyframes smartGradient {{
            0%, 100% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
        }}

        @media (max-width: 768px) {{
            .stMainBlockContainer {{
                padding: 6.25rem 1rem 1.5rem;
            }}
            .smart-brand {{ align-items: flex-start; }}
            .smart-brand__mark {{
                width: 2.65rem;
                height: 2.65rem;
                flex-basis: 2.65rem;
            }}
            .smart-brand p {{ font-size: .84rem; }}
            .smart-footer {{ align-items: flex-start; flex-direction: column; }}
            .smart-footer__dot {{ display: none; }}
        }}

        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{
                animation-duration: .01ms !important;
                animation-iteration-count: 1 !important;
                scroll-behavior: auto !important;
                transition-duration: .01ms !important;
            }}
        }}
        </style>
        """
    )
