import re
from typing import Dict, Set

# --- Brand Caching Logic ---
_CACHED_BRANDS = set()

def build_brand_cache(products_list):
    global _CACHED_BRANDS
    _CACHED_BRANDS.clear()
    
    for p in products_list:
        fm = p.get("featuresMap", {})
        if not fm: continue
        
        for k, v in fm.items():
            if k.lower() == "brand" and isinstance(v, str):
                brand_token = v.lower().strip()
                brand_token = re.sub(r"[^a-z0-9]", "", brand_token)
                if brand_token:
                    _CACHED_BRANDS.add(brand_token)

    _CACHED_BRANDS.update([
        "samsung", "sony", "lg", "vizio", "panasonic", "sharp", 
        "toshiba", "philips", "insignia", "sanyo", "jvc", "rca", 
        "hisense", "haier", "magnavox", "tcl", "sansui", "hannspree", 
        "nec", "viewsonic", "sunbrite", "coby", "septre", "proscan"
    ])

def get_brand_from_product(product: Dict) -> str:
    kv = product.get("featuresMap", {})
    for k, v in kv.items():
        if k.lower() == "brand" and isinstance(v, str):
            b = re.sub(r"[^a-z0-9]", "", v.lower())
            if b: return b

    title = product.get("title", "").lower()
    all_values = " ".join(str(v).lower() for v in kv.values())
    search_text = title + " " + all_values
    tokens = set(re.split(r"[^a-z0-9]+", search_text))
    found = tokens.intersection(_CACHED_BRANDS)
    if found:
        return max(found, key=len)
    return ""

# --- IMPROVED INCH EXTRACTION LOGIC ---

def _is_valid_tv_size(val):
    """
    Sanity check. TVs are generally between 10 and 100 inches.
    This filters out 1080 (resolution), 120 (Hz), 6500 (Model ID).
    """
    return 13.0 <= val <= 99.0

def parse_inch_string(s):
    if not s: return None
    s = s.strip().lower()
    
    # 1. Handle fractions like "59-9/10" or "59 9/10"
    # Matches: number + space/dash + number + / + number
    frac_match = re.search(r'(\d+)[\-\s]+(\d+)/(\d+)', s)
    if frac_match:
        whole = float(frac_match.group(1))
        num = float(frac_match.group(2))
        den = float(frac_match.group(3))
        if den != 0:
            val = whole + (num / den)
            if _is_valid_tv_size(val): return val

    # 2. Handle standard decimal "59.9", "60", "60.0"
    # We look for the number, but we rely on the caller to ensure it was near an 'inch' unit
    # unless we are parsing a raw KV value that is known to be a size.
    simple_match = re.findall(r'(\d+(?:\.\d+)?)', s)
    for m in simple_match:
        try:
            val = float(m[0]) # m is a tuple due to group
            if _is_valid_tv_size(val): return val
        except: continue
            
    return None

def extract_diagonal_inches(product: Dict) -> float:
    """
    Robust extraction of diagonal screen size.
    """
    features = product.get("featuresMap", {})
    
    # Priority 1: High-confidence Keys
    # We look for keys containing 'display', 'screen', 'diagonal' AND 'size'
    target_keys = ["display size", "screen size", "diagonal", "size class"]
    
    for key, value in features.items():
        k_lower = key.lower()
        if any(t in k_lower for t in target_keys) and isinstance(value, str):
            val = parse_inch_string(value)
            if val: return val

    # Priority 2: Title Search (Stricter Regex)
    title = product.get("title", "").lower()
    
    # Regex explanation:
    # 1. (\d+(?:\.\d+)?)  -> Capture a number (int or float)
    # 2. \s*              -> Optional whitespace
    # 3. (?:...)          -> Non-capturing group for the unit
    # Units: " (quote), inch, inches, -inch, class, diag
    
    # We use findall to get all candidates, then filter by validity (13-99)
    # This prevents catching "1080p" (1080 > 99) or "120hz" (120 > 99)
    
    # Pattern A: Fraction pattern in title (rare but possible: "59-9/10\"")
    frac_match = re.search(r'(\d+)[\-\s]+(\d+)/(\d+)\s*(?:\"|”|inch|class|diag)', title)
    if frac_match:
        whole = float(frac_match.group(1))
        num = float(frac_match.group(2))
        den = float(frac_match.group(3))
        if den != 0:
            val = whole + (num / den)
            if _is_valid_tv_size(val): return val

    # Pattern B: Standard numbers with units
    # We explicitly look for the unit to avoid random model numbers
    candidates = re.findall(r'(\d+(?:\.\d+)?)\s*(?:\"|”|inch|in\b|class|diag)', title)
    
    for c in candidates:
        try:
            val = float(c)
            if _is_valid_tv_size(val):
                return val
        except: continue
            
    return None

# --- Existing Model Word Logic ---

_TITLE_MW_RE = re.compile(r"[a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*")
_KV_MW_RE = re.compile(r"(^\d+(\.\d+)?[a-zA-Z]+$)|(^\d+(\.\d+)?$)")

_UNIT_NORMALIZATIONS = {
    '"': "inch", "inch": "inch", "inches": "inch", "-inch": "inch", " inch": "inch",
    "hertz": "hz", "Hertz": "hz", "Hz": "hz", "HZ": "hz", " hz": "hz", "-hz": "hz", "hz": "hz",
}

_non_alnum_edges = re.compile(r"^[^0-9a-zA-Z]+|[^0-9a-zA-Z]+$")

def _normalize_token(tok: str) -> str:
    if not tok: return ""
    t = tok.strip().lower()
    t = t.replace('"', "inch")
    t = _non_alnum_edges.sub("", t)
    for key, val in _UNIT_NORMALIZATIONS.items():
        if key in t:
            t = t.replace(key, val)
    return t

def extract_title_model_words(title: str) -> Set[str]:
    if not isinstance(title, str): return set()
    title_l = title.lower()
    matches = set()
    for m in _TITLE_MW_RE.finditer(title_l):
        tok = _normalize_token(m.group(0))
        if tok: matches.add(tok)
    res_matches = re.findall(r'\b\d{3,4}[xX]\d{3,4}\b', title_l)
    for m in res_matches: matches.add(m.lower())
    return matches

def extract_kv_model_words(kv_dict: Dict, title_mw: Set[str] = None) -> Set[str]:
    out = set()
    if not isinstance(kv_dict, dict): return out
    title_mw = title_mw or set()
    for _, value in kv_dict.items():
        if not isinstance(value, str): continue
        tokens = re.split(r"[\s/,\-()]+", value)
        for tok in tokens:
            tok_norm = _normalize_token(tok)
            if not tok_norm: continue
            if _KV_MW_RE.match(tok_norm):
                num_match = re.match(r"\d+(\.\d+)?", tok_norm)
                if num_match: out.add(num_match.group(0))
            if tok_norm in title_mw: out.add(tok_norm)
    return out

def extract_model_words(product: Dict) -> Set[str]:
    if not isinstance(product, dict): return set()
    title = product.get("title", "")
    kv = product.get("featuresMap", {})
    title_mw = extract_title_model_words(title)
    kv_mw = extract_kv_model_words(kv, title_mw=title_mw)
    final_set = title_mw.union(kv_mw)
    
    brand = get_brand_from_product(product)
    if brand:
        final_set.add(brand)
    return final_set