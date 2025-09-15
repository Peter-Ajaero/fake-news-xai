# --- paste everything from the next line down to the end of this cell ---

import os, re, json, time, typing as T
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="XAI Fake-News Detector", page_icon="📰", layout="centered")

# ---------- small utils ----------
def utc_now_iso():
    try:
        import datetime
        return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    except Exception:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

_word_re = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
def word_tokenize(text: str) -> T.List[str]:
    return _word_re.findall(text or "")

def render_colored_tokens(tokens: T.List[str], scores: np.ndarray) -> str:
    if len(tokens) == 0:
        return "<i>No tokens.</i>"
    s = np.array(scores, dtype=float).ravel()
    m = float(np.max(np.abs(s))) if s.size and np.max(np.abs(s)) > 0 else 1.0
    spans = []
    for tok, val in zip(tokens, s):
        a = float(abs(val) / m)
        a = min(max(a, 0.0), 1.0)
        bg = f"rgba(255, 0, 0, {0.15 + 0.55*a})" if val >= 0 else f"rgba(0, 120, 255, {0.15 + 0.55*a})"
        safe = tok.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        spacer = "" if re.match(r"^[^\w\s]$", tok) else " "
        spans.append(f'<span style="background:{bg}; padding:2px 4px; border-radius:4px; margin:1px;">{safe}</span>{spacer}')
    return '<div style="line-height:1.9; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto;">' + "".join(spans) + "</div>"

# ---------- artifact resolution (robust) ----------
def _unique_keep_order(xs: T.Iterable[str]) -> T.List[str]:
    out, seen = [], set()
    for x in xs:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _alt_roots(base: str) -> T.List[str]:
    """
    Given one base path, produce alternate mount candidates.
    - If path is /content/drive/.Encrypted/MyDrive/... -> add /content/drive_msc/MyDrive/...
    - If path is /content/drive_msc/MyDrive/...       -> add /content/drive/.Encrypted/MyDrive/...
    Keep only existing dirs, preserve order.
    """
    cands = [base]
    m = re.match(r"^/content/drive/\.Encrypted/MyDrive/(.*)$", base)
    if m:
        cands.append(f"/content/drive_msc/MyDrive/{m.group(1)}")
    m2 = re.match(r"^/content/drive_msc/MyDrive/(.*)$", base)
    if m2:
        cands.append(f"/content/drive/.Encrypted/MyDrive/{m2.group(1)}")
    cands = [c for c in _unique_keep_order(cands) if c and os.path.isdir(c)]
    return cands

def _has_hf_files(d: str) -> bool:
    if not d or not os.path.isdir(d): return False
    if not os.path.isfile(os.path.join(d, "config.json")): return False
    if any(os.path.exists(os.path.join(d, f)) for f in ("pytorch_model.bin", "model.safetensors")):
        return True
    # Accept config-only if tokenizer files exist (covers sharded checkpoints)
    return any(os.path.exists(os.path.join(d, f)) for f in ("tokenizer.json","tokenizer.model","vocab.json","merges.txt"))

def _pick_best_checkpoint(root: str) -> str:
    """Prefer best_checkpoint/, else newest checkpoint-*, else common fallbacks, else deep search."""
    cand = os.path.join(root, "best_checkpoint")
    if _has_hf_files(cand): return cand

    step_cands = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if os.path.isdir(full) and name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[-1])
            except Exception:
                continue
            if _has_hf_files(full):
                step_cands.append((step, full))
    if step_cands:
        step_cands.sort(key=lambda x: x[0])
        return step_cands[-1][1]

    for p in ("artifacts/model_best", "artifacts_final/model"):
        full = os.path.join(root, p)
        if _has_hf_files(full): return full

    best = ""
    best_score = -1
    for cur, _, _ in os.walk(root):
        if _has_hf_files(cur):
            scr = 0
            m = re.search(r"checkpoint-(\d+)", cur)
            if m:
                try: scr = int(m.group(1))
                except: scr = 0
            if cur.endswith("best_checkpoint"):
                scr += 10_000_000
            if scr > best_score:
                best_score = scr
                best = cur
    return best

def _find_meta(root: str) -> str:
    pref = os.path.join(root, "artifacts_final", "meta.json")
    if os.path.exists(pref): return pref
    for cur, _, files in os.walk(root):
        if "meta.json" in files:
            return os.path.join(cur, "meta.json")
    return ""

def find_artifacts_any(base: str) -> T.Tuple[str, str, str]:
    """Returns (resolved_root, best_dir, meta_path) or empty strings if not found."""
    for root in _alt_roots(base):
        best = _pick_best_checkpoint(root)
        meta = _find_meta(root)
        if best or meta:
            return root, best, meta
    return "", "", ""

def load_meta(meta_path: str) -> dict:
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

DEFAULT_ARTIF_DIR = os.environ.get(
    "ARTIF_DIR", "/content/drive/MyDrive/msc_fake_news_artifact_20250906_200554"
).strip()

ARTIF_DIR_INPUT = DEFAULT_ARTIF_DIR
RESOLVED_ROOT, BEST_DIR, META_PATH = find_artifacts_any(ARTIF_DIR_INPUT)
META = load_meta(META_PATH)
INFER_META = (META.get("inference") or {})
DEFAULT_T = float(INFER_META.get("temperature", 1.0))
DEFAULT_THR = float(INFER_META.get("decision_threshold", 0.50))

# optional isotonic calibrator
ISO = None
try:
    import pickle, glob
    if RESOLVED_ROOT:
        pref = os.path.join(RESOLVED_ROOT, "artifacts_final", "metrics", "iso.pkl")
        candidates = [pref] if os.path.exists(pref) else glob.glob(os.path.join(RESOLVED_ROOT, "**/iso.pkl"), recursive=True)
        if candidates:
            with open(candidates[0], "rb") as f:
                ISO = pickle.load(f)
except Exception:
    ISO = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource(show_spinner=False)
def _load_model(best_dir: str):
    if not best_dir or not os.path.isdir(best_dir):
        raise FileNotFoundError(
            "Model checkpoint folder not found. Set ARTIF_DIR to your artifact directory that contains "
            "best_checkpoint/ or checkpoint-*/ and artifacts_final/meta.json."
        )
    tok = AutoTokenizer.from_pretrained(best_dir, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(best_dir)
    mdl.to(DEVICE); mdl.eval()
    return tok, mdl

def get_model():
    return _load_model(BEST_DIR)

def predict_proba_fake(texts: T.List[str], temperature: float = 1.0) -> np.ndarray:
    tok, mdl = get_model()
    enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = mdl(**enc).logits
    if temperature and temperature != 1.0:
        logits = logits / float(temperature)
    probs = F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
    if ISO is not None:
        try:
            probs = ISO.predict(probs.reshape(-1,1)).clip(0,1).ravel()
        except Exception:
            pass
    return probs

def make_shap_explainer(temp_ref):
    import shap
    masker = shap.maskers.Text(lambda s: word_tokenize(s), mask_token="[MASK]", collapse_mask_token=" [MASK] ")
    def _predict(texts):
        return predict_proba_fake(list(texts), temperature=float(temp_ref()))
    return shap.Explainer(_predict, masker, algorithm="partition")

_SHAP = None

def shap_explain(text: str, temperature: float, max_evals: int = 100, max_tokens: int = 200):
    global _SHAP
    import shap
    if _SHAP is None:
        _SHAP = make_shap_explainer(lambda: temperature)
    toks = word_tokenize(text)
    if len(toks) > max_tokens:
        text = " ".join(toks[:max_tokens])
    exp = _SHAP([text], max_evals=max_evals)
    exp0 = exp[0]
    tokens = list(getattr(exp0, "data", toks))
    values = np.array(exp0.values, dtype=float).ravel()
    if len(values) != len(tokens):
        n = min(len(values), len(tokens))
        tokens, values = tokens[:n], values[:n]
    return tokens, values

def occlusion_explain(text: str, temperature: float, max_tokens: int = 200):
    toks = word_tokenize(text)
    if len(toks) > max_tokens:
        toks = toks[:max_tokens]
    try:
        tok, _ = get_model()
        mask_token = tok.mask_token or "[MASK]"
    except Exception:
        mask_token = "[MASK]"
    base = float(predict_proba_fake([" ".join(toks)], temperature=temperature)[0])
    scores = []
    for i in range(len(toks)):
        tmp = toks.copy(); tmp[i] = mask_token
        p = float(predict_proba_fake([" ".join(tmp)], temperature=temperature)[0])
        scores.append(base - p)
    return toks, np.array(scores, dtype=float)

# ---------- UI ----------
st.title("📰 XAI Fake-News Detector (LIAR)")
st.write("Enter a short political statement and click **Predict**. This demo uses your fine-tuned transformer and shows token-level explanations.")

with st.expander("ℹ️ About & Ethics", expanded=False):
    st.markdown(
        "**Disclaimer.** This tool is an *AI-assisted* classifier. It can be wrong.  \n"
        "Explanations show which tokens pushed the model toward **FAKE** (red) or **REAL** (blue).  \n"
        "Do **not** treat this as ground truth; always verify with human fact-checkers.\n\n"
        "**Bias & fairness.** The LIAR dataset may encode political/demographic biases.  \n"
        "We calibrated probabilities and audited some failure modes, but residual bias is possible."
    )

st.divider()

with st.sidebar:
    st.header("⚙️ Settings")
    art_in = st.text_input("Artifacts folder", value=DEFAULT_ARTIF_DIR)
    if art_in.strip() != DEFAULT_ARTIF_DIR:
        os.environ["ARTIF_DIR"] = art_in.strip()
        st.info("Artifact path updated. Reloading…")
        st.cache_resource.clear()
        st.rerun()
    st.caption(f"Resolved root: `{RESOLVED_ROOT or 'N/A'}`")
    st.caption(f"Checkpoint: `{BEST_DIR or 'N/A'}`")
    temperature = st.number_input("Temperature (calibration)", 0.1, 5.0, float(float(INFER_META.get('temperature', 1.0))), 0.05)
    threshold = st.slider("Decision threshold on p(FAKE)", 0.05, 0.95, float(float(INFER_META.get('decision_threshold', 0.5))), 0.01)
    use_shap = st.checkbox("Use SHAP explanations (slower)", value=True)
    max_evals = st.slider("SHAP max_evals", 20, 300, 100, 10)
    max_tokens = st.slider("Max tokens for explanation", 50, 300, 200, 10)

text = st.text_area("Statement", height=120, placeholder="e.g., 'The unemployment rate is the lowest in 50 years.'")
go = st.button("Predict", type="primary", use_container_width=True)

if go:
    if not text.strip():
        st.warning("Please enter a statement."); st.stop()
    try:
        _ = get_model()
    except FileNotFoundError as e:
        st.error(str(e)); st.stop()

    with st.spinner("Running model…"):
        p = float(predict_proba_fake([text], temperature=float(temperature))[0])
        label = "FAKE" if p >= float(threshold) else "REAL"

    c1, c2 = st.columns([1,1])
    with c1:
        st.metric("p(FAKE)", f"{p:.3f}")
        st.write(f"**Prediction:** {label}")
    with c2:
        st.caption(f"Temperature: {temperature:.2f} • Threshold: {threshold:.2f} • Device: { 'cuda' if torch.cuda.is_available() else 'cpu' }")

    st.divider()
    st.subheader("🧠 Explanation")

    html_block, err_msg = "", ""
    start = time.time()
    try:
        if use_shap:
            try:
                import shap
            except Exception:
                raise RuntimeError("SHAP not installed. Disable SHAP or install the package.")
            tokens, scores = shap_explain(text, temperature=float(temperature), max_evals=int(max_evals), max_tokens=int(max_tokens))
        else:
            tokens, scores = occlusion_explain(text, temperature=float(temperature), max_tokens=int(max_tokens))
        html_block = render_colored_tokens(tokens, scores)
    except Exception as ex:
        err_msg = f"SHAP failed: {ex.__class__.__name__}: {ex}"
        try:
            tokens, scores = occlusion_explain(text, temperature=float(temperature), max_tokens=int(max_tokens))
            html_block = render_colored_tokens(tokens, scores)
            err_msg += " → Fallback to occlusion succeeded."
        except Exception as ex2:
            err_msg += f" | Fallback failed: {ex2.__class__.__name__}: {ex2}"

    dur = time.time() - start
    if err_msg: st.warning(err_msg)
    st.caption(f"Explained in {dur:.2f}s • red=pushes FAKE • blue=pushes REAL")
    st.markdown(html_block, unsafe_allow_html=True)

    import pandas as pd
    k = min(15, len(tokens))
    idx = np.argsort(-np.abs(scores))[:k]
    df = pd.DataFrame({
        "token": [tokens[i] for i in idx],
        "contribution": [float(scores[i]) for i in idx],
        "polarity": ["FAKE+" if scores[i] > 0 else "REAL+" for i in idx],
    })
    st.dataframe(df, use_container_width=True)

st.divider()
st.caption(f"Root: `{RESOLVED_ROOT or 'N/A'}` • Checkpoint: `{BEST_DIR or 'N/A'}` • meta: `{META_PATH or 'N/A'}` • {utc_now_iso()} UTC")
