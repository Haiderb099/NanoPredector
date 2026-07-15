import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski
from rdkit import RDLogger
from sklearn.ensemble import RandomForestRegressor
import os

RDLogger.DisableLog("rdApp.*")

# ══════════════════════════════════════════════════════════════════
#  STYLING
# ══════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Background */
    .stApp {
        background: #050d1a;
        color: #c8d8f0;
    }

    /* Main header */
    .main-header {
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
        border-bottom: 1px solid #0e2a4a;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-family: 'Space Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #00e5ff;
        letter-spacing: 0.04em;
        margin: 0;
    }
    .main-header p {
        color: #5a7fa8;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    /* Stat cards */
    .stat-row {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .stat-card {
        flex: 1;
        background: #070f1e;
        border: 1px solid #0e2a4a;
        border-radius: 8px;
        padding: 1rem 1.4rem;
        text-align: center;
    }
    .stat-card .value {
        font-family: 'Space Mono', monospace;
        font-size: 1.6rem;
        color: #00e5ff;
        font-weight: 700;
    }
    .stat-card .label {
        font-size: 0.75rem;
        color: #5a7fa8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.2rem;
    }

    /* Section headers */
    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: #00e5ff;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        border-left: 3px solid #00e5ff;
        padding-left: 0.8rem;
        margin: 2rem 0 1rem 0;
    }

    /* Result badges */
    .badge-safe   { background:#003322; color:#00e5a0; border:1px solid #00a060; border-radius:4px; padding:2px 8px; font-size:0.78rem; }
    .badge-warn   { background:#332500; color:#ffb700; border:1px solid #aa7000; border-radius:4px; padding:2px 8px; font-size:0.78rem; }
    .badge-danger { background:#330010; color:#ff4060; border:1px solid #aa0030; border-radius:4px; padding:2px 8px; font-size:0.78rem; }

    /* Streamlit widget overrides */
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background: #070f1e !important;
        border: 1px solid #0e2a4a !important;
        color: #c8d8f0 !important;
        border-radius: 6px !important;
    }
    .stButton > button {
        background: #00e5ff !important;
        color: #050d1a !important;
        font-family: 'Space Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.06em !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    /* Dataframe */
    .stDataFrame { border: 1px solid #0e2a4a !important; border-radius: 8px !important; }

    /* Divider */
    hr { border-color: #0e2a4a !important; }

    /* Info / success boxes */
    .stAlert { background: #070f1e !important; border: 1px solid #0e2a4a !important; border-radius: 8px !important; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #0e2a4a; }
    .stTabs [data-baseweb="tab"] { color: #5a7fa8; font-family: 'Space Mono', monospace; font-size: 0.78rem; letter-spacing: 0.08em; }
    .stTabs [aria-selected="true"] { color: #00e5ff !important; border-bottom: 2px solid #00e5ff !important; }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def smiles_to_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return np.array(fp)


def lipinski_profile(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {"status": "unknown", "violations": 0, "mw": None, "logp": None}
    mw   = round(Descriptors.MolWt(mol), 1)
    logp = round(Descriptors.MolLogP(mol), 2)
    hbd  = Lipinski.NumHDonors(mol)
    hba  = Lipinski.NumHAcceptors(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    status = {0: "safe", 1: "warn", 2: "danger"}.get(min(violations, 2), "danger")
    return {"status": status, "violations": violations, "mw": mw, "logp": logp, "hbd": hbd, "hba": hba}


@st.cache_data
def load_data(path: str = "pan_cancer_master.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.dropna(subset=["SMILES", "Binding_Affinity", "Target"])
    df["Binding_Affinity"] = pd.to_numeric(df["Binding_Affinity"], errors="coerce")
    df = df.dropna(subset=["Binding_Affinity"])
    return df


@st.cache_resource
def train_models(df: pd.DataFrame, targets: list) -> dict:
    models = {}
    for target in targets:
        subset = df[df["Target"] == target].copy()
        if len(subset) < 10:
            continue
        fps = []
        vals = []
        for _, row in subset.iterrows():
            fp = smiles_to_fingerprint(row["SMILES"])
            if fp is not None:
                fps.append(fp)
                vals.append(row["Binding_Affinity"])
        if len(fps) < 5:
            continue
        model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        model.fit(np.array(fps), np.array(vals))
        models[target] = model
    return models


# ══════════════════════════════════════════════════════════════════
#  AGRICULTURAL COMPOUND LIBRARY
# ══════════════════════════════════════════════════════════════════
COMPOUNDS = [
    {"name": "Mangiferin",         "source": "Mango Peels",         "smiles": "O=C1C2=C(C(=C(O)C=C2)O)OC3=C1C(O)=CC(=C3C4OC(CO)C(O)C(O)C4O)O"},
    {"name": "Luteolin",           "source": "Sugarcane Bagasse",   "smiles": "O=C1C=C(OC2=CC(O)=CC(O)=C12)C3=CC(O)=C(O)C=C3"},
    {"name": "Quercetin",          "source": "Onion Skins",         "smiles": "O=C1C=C(OC2=CC(O)=CC(O)=C12)C3=CC(O)=C(O)C=C3O"},
    {"name": "Ferulic Acid",       "source": "Rice Husk",           "smiles": "COC1=C(O)C=CC(=C1)C=CC(=O)O"},
    {"name": "p-Coumaric Acid",    "source": "Wheat Straw",         "smiles": "C1=CC(=CC=C1C=CC(=O)O)O"},
    {"name": "Juglone",            "source": "Walnut Shells",       "smiles": "C1=CC2=C(C(=C1)O)C(=O)C=CC2=O"},
    {"name": "Hesperidin",         "source": "Citrus Peels",        "smiles": "CC1OC(CC(O)C1O)OC2C(C(C(O2)CO)O)OC3=C(OC4=CC(=CC(=C4C3=O)O)O)C5=CC(=C(C=C5)OC)O"},
    {"name": "Gallic Acid",        "source": "Grape Seeds",         "smiles": "O=C(O)C1=CC(O)=C(O)C(O)=C1"},
    {"name": "Resveratrol",        "source": "Peanut Shells",       "smiles": "OC1=CC=C(C=C1)C=CC2=CC(O)=CC(O)=C2"},
    {"name": "Hydroxytyrosol",     "source": "Olive Pomace",        "smiles": "C1=CC(=C(C=C1CCO)O)O"},
    {"name": "Apigenin",           "source": "Parsley Stems",       "smiles": "O=C1C=C(OC2=CC(O)=CC(O)=C12)C3=CC=C(O)C=C3"},
    {"name": "Kaempferol",         "source": "Tea Waste",           "smiles": "O=C1C=C(OC2=CC(O)=CC(O)=C12)C3=CC=C(O)C=C3O"},
    {"name": "Chlorogenic Acid",   "source": "Coffee Pulp",         "smiles": "O=C(O[C@@H]1C[C@](O)(C[C@@H](O)[C@@H]1O)C(=O)O)/C=C/C2=CC=C(O)C(O)=C2"},
    {"name": "Curcumin",           "source": "Turmeric Root",       "smiles": "COC1=C(O)C=CC(=C1)/C=C/C(=O)CC(=O)/C=C/C2=CC(=C(O)C=C2)OC"},
    {"name": "EGCG",               "source": "Green Tea Extract",   "smiles": "O=C(OC1CC(OC2=CC(O)=CC(O)=C12)C3=CC(O)=C(O)C(O)=C3)C4=CC(O)=C(O)C(O)=C4"},
]


# ══════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="AgriCure AI · Pan-Cancer Platform",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()

    # ── Header ──────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🧬 AgriCure AI</h1>
        <p>Pan-Cancer Drug Discovery · Agricultural Waste Compound Screening</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load Data ────────────────────────────────────────────
    df = load_data()

    if df.empty:
        st.error("⚠️ Database not found. Make sure `pan_cancer_master.csv` is in the same folder as this file.")
        st.stop()

    targets    = sorted(df["Target"].unique().tolist())
    n_trials   = len(df)
    n_targets  = len(targets)
    n_compounds = len(COMPOUNDS)

    # ── Stat Cards ───────────────────────────────────────────
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card"><div class="value">{n_trials:,}</div><div class="label">Clinical Records</div></div>
        <div class="stat-card"><div class="value">{n_targets}</div><div class="label">Cancer Targets</div></div>
        <div class="stat-card"><div class="value">{n_compounds}</div><div class="label">Agri Compounds</div></div>
        <div class="stat-card"><div class="value">RF-150</div><div class="label">ML Model</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🔬  HIGH-THROUGHPUT SCREEN", "⚖️  SELECTIVITY ENGINE", "🧪  SINGLE COMPOUND"])

    # ════════════════════════════════════════════════════════
    #  TAB 1 – FULL MATRIX SCREEN
    # ════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-title">Select Cancer Targets</div>', unsafe_allow_html=True)
        selected_targets = st.multiselect(
            label="Choose which cancer targets to screen against:",
            options=targets,
            default=targets[:2] if len(targets) >= 2 else targets,
        )

        st.markdown('<div class="section-title">Safety Filter</div>', unsafe_allow_html=True)
        safety_threshold = st.radio(
            "Only show compounds that pass Lipinski's Rule of Five:",
            options=["Show All", "Exclude High Toxicity Risk", "Only Safe Compounds"],
            horizontal=True,
        )

        if st.button("▶  RUN PAN-CANCER SCREEN", key="btn_screen"):
            if not selected_targets:
                st.warning("Please select at least one cancer target.")
                st.stop()

            with st.spinner("Training models on clinical data..."):
                models = train_models(df, selected_targets)

            if not models:
                st.error("Could not train models — not enough data for the selected targets.")
                st.stop()

            with st.spinner("Screening compound library..."):
                rows = []
                for c in COMPOUNDS:
                    lipo = lipinski_profile(c["smiles"])

                    # Safety filter
                    if safety_threshold == "Exclude High Toxicity Risk" and lipo["status"] == "danger":
                        continue
                    if safety_threshold == "Only Safe Compounds" and lipo["status"] != "safe":
                        continue

                    fp = smiles_to_fingerprint(c["smiles"])
                    row = {
                        "Compound":  c["name"],
                        "Source":    c["source"],
                        "MW (Da)":   lipo["mw"],
                        "LogP":      lipo["logp"],
                        "Safety":    {"safe": "✅ Safe", "warn": "⚠️ Moderate", "danger": "❌ High Risk"}[lipo["status"]],
                    }
                    if fp is not None:
                        for t in selected_targets:
                            if t in models:
                                pred = models[t].predict([fp])[0]
                                row[t] = round(pred, 1)
                    rows.append(row)

            if not rows:
                st.warning("No compounds passed the selected safety filter.")
            else:
                result_df = pd.DataFrame(rows)

                # Sort by first selected target (ascending = best IC50)
                sort_col = selected_targets[0] if selected_targets[0] in result_df.columns else "Compound"
                result_df = result_df.sort_values(sort_col, ascending=True).reset_index(drop=True)

                st.success(f"✓ Screened {len(rows)} compounds across {len(models)} cancer targets.")
                st.dataframe(result_df, use_container_width=True, height=420)
                st.caption("IC50 values in nM — lower = more potent against that target.")

                # Best picks per target
                st.markdown('<div class="section-title">Top Candidate Per Target</div>', unsafe_allow_html=True)
                cols = st.columns(len(models))
                for i, (target, _) in enumerate(models.items()):
                    if target in result_df.columns:
                        best_row = result_df.loc[result_df[target].idxmin()]
                        cols[i].metric(
                            label=target,
                            value=best_row["Compound"],
                            delta=f"{best_row[target]:,.0f} nM",
                            delta_color="inverse",
                        )

    # ════════════════════════════════════════════════════════
    #  TAB 2 – SELECTIVITY ENGINE
    # ════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-title">Head-to-Head Selectivity Analysis</div>', unsafe_allow_html=True)
        st.write("Find compounds that are highly potent against one cancer while sparing another — the hallmark of precision medicine.")

        col_a, col_b = st.columns(2)
        with col_a:
            target_a = st.selectbox("🎯 Primary Target (attack)", targets, index=0)
        with col_b:
            remaining = [t for t in targets if t != target_a]
            target_b  = st.selectbox("🛡️ Off-Target (spare)", remaining if remaining else targets, index=0)

        min_ratio = st.slider("Minimum selectivity ratio (A must be X× more potent than B)", 1, 20, 3)

        if st.button("▶  RUN SELECTIVITY SCREEN", key="btn_select"):
            with st.spinner("Training models..."):
                models = train_models(df, [target_a, target_b])

            if target_a not in models or target_b not in models:
                st.error("Not enough clinical data for one or both selected targets.")
                st.stop()

            with st.spinner("Computing selectivity ratios..."):
                rows = []
                for c in COMPOUNDS:
                    fp = smiles_to_fingerprint(c["smiles"])
                    lipo = lipinski_profile(c["smiles"])
                    if fp is None:
                        continue
                    score_a = models[target_a].predict([fp])[0]
                    score_b = models[target_b].predict([fp])[0]
                    ratio   = round(score_b / score_a, 2) if score_a > 0 else None
                    rows.append({
                        "Compound":           c["name"],
                        "Source":             c["source"],
                        f"{target_a} IC50":   round(score_a, 1),
                        f"{target_b} IC50":   round(score_b, 1),
                        "Selectivity Ratio":  ratio,
                        "Safety":             {"safe": "✅", "warn": "⚠️", "danger": "❌"}[lipo["status"]],
                    })

            sel_df = pd.DataFrame(rows)
            sel_df = sel_df[sel_df["Selectivity Ratio"] >= min_ratio].sort_values(
                "Selectivity Ratio", ascending=False
            ).reset_index(drop=True)

            if sel_df.empty:
                st.info(f"No compounds achieved a selectivity ratio ≥ {min_ratio}×. Try lowering the threshold.")
            else:
                st.success(f"✓ Found {len(sel_df)} selectively active compounds.")
                st.dataframe(sel_df, use_container_width=True, height=360)
                st.caption(
                    f"Selectivity Ratio = {target_b} IC50 ÷ {target_a} IC50. "
                    "Higher ratio = more selective for the primary target."
                )

    # ════════════════════════════════════════════════════════
    #  TAB 3 – SINGLE COMPOUND
    # ════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-title">Predict a Custom Molecule</div>', unsafe_allow_html=True)
        st.write("Paste any SMILES string and test it against all available cancer targets instantly.")

        custom_smiles = st.text_input(
            "SMILES Code",
            value="COC1=C(O)C=CC(=C1)C=CC(=O)O",  # Ferulic acid default
            placeholder="e.g. COC1=C(O)C=CC(=C1)C=CC(=O)O",
        )
        custom_target = st.selectbox("Target Cancer", targets, key="custom_target")

        if st.button("▶  PREDICT", key="btn_predict"):
            fp = smiles_to_fingerprint(custom_smiles)
            if fp is None:
                st.error("Invalid SMILES string — could not parse the molecule.")
            else:
                lipo = lipinski_profile(custom_smiles)
                with st.spinner("Training model and predicting..."):
                    models = train_models(df, [custom_target])

                if custom_target not in models:
                    st.error("Not enough clinical data for this target.")
                else:
                    pred = models[custom_target].predict([fp])[0]

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Predicted IC50", f"{pred:,.0f} nM")
                    col2.metric("Mol. Weight", f"{lipo['mw']} Da")
                    col3.metric("LogP", lipo["logp"])
                    col4.metric("Rule Violations", lipo["violations"])

                    safety_map = {"safe": ("✅ Drug-Like", "normal"), "warn": ("⚠️ 1 Violation", "off"), "danger": ("❌ High Risk", "inverse")}
                    label, delta_col = safety_map[lipo["status"]]
                    st.info(f"**Lipinski Safety:** {label}  ·  HBD: {lipo['hbd']}  ·  HBA: {lipo['hba']}")

    # ── Footer ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#1e3a5a;font-size:0.75rem;font-family:Space Mono,monospace;'>"
        "AgriCure AI · Training data: ChEMBL · Model: Random Forest (150 trees) · Toxicity: Lipinski Ro5"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__" or True:
    main()