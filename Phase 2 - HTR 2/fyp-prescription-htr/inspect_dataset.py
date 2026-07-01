import sys
import unicodedata
from collections import Counter
from pathlib import Path

# ---- EDIT THESE TWO LINES IF YOUR PATHS DIFFER ----------------------------------
CSV_PATH = Path("data/pharmacy_lk/labels.csv")
IMG_DIR  = Path("data/pharmacy_lk/images")
# ---------------------------------------------------------------------------------

def rule(t=""):
    print("\n" + "=" * 70)
    if t:
        print(t)
        print("=" * 70)

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas not installed — activate your venv first (source .venv/bin/activate)")

rule("0. PATHS")
print(f"CSV exists : {CSV_PATH.exists()}  -> {CSV_PATH.resolve()}")
print(f"IMG dir    : {IMG_DIR.exists()}  -> {IMG_DIR.resolve()}")
if not CSV_PATH.exists():
    sys.exit("labels.csv not found at the path above — edit CSV_PATH at the top of this script.")

# ---- Load -----------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

rule("1. COLUMNS & SHAPE")
print(f"rows: {len(df)}")
print(f"columns: {list(df.columns)}")
print("\nfirst 5 rows:")
print(df.head().to_string())

# Try to identify the image-filename and transcription columns automatically.
def looks_like_filenames(series):
    s = series.astype(str).head(20)
    return (s.str.contains(r"\.(?:jpg|jpeg|png|tif|tiff|bmp)$", case=False)).mean() > 0.5

img_candidates = [c for c in df.columns if looks_like_filenames(df[c])]
# The image column we want is the cropped-word filename, not the source prescription.
def img_score(c):
    cl = c.lower()
    return ("image" in cl or "word" in cl or "crop" in cl) - ("source" in cl or "prescription" in cl)
img_col = max(img_candidates, key=img_score) if img_candidates else df.columns[0]

txt_col = next((c for c in df.columns
                if c != img_col and any(k in c.lower()
                for k in ("trans", "label", "medicine", "text", "name", "gt"))), None)
if txt_col is None:
    txt_col = next((c for c in df.columns if c != img_col and not looks_like_filenames(df[c])),
                   df.columns[-1])

print(f"\n[auto-detected] image column = '{img_col}' | transcription column = '{txt_col}'")
print(f"all columns: {list(df.columns)}")
print("If those are wrong, tell me the correct column names and I'll set them explicitly.")

txt = df[txt_col].astype(str)

rule("2. MISSING / EMPTY VALUES")
print(f"missing transcription : {df[txt_col].isna().sum()}")
print(f"missing image filename: {df[img_col].isna().sum()}")
print(f"empty/whitespace labels: {(txt.str.strip() == '').sum()}")
print(f"duplicate image filenames: {df[img_col].duplicated().sum()}")

rule("3. LABEL STATISTICS")
print(f"unique transcriptions: {txt.nunique()}")
L = txt.str.len()
print(f"label length  -> min {L.min()} | median {int(L.median())} | max {L.max()}")
print(f"labels of length 1: {(L == 1).sum()}")
print(f"multi-word labels (contain space): {txt.str.contains(' ').sum()}")
print(f"labels with digits: {txt.str.contains(r'[0-9]').sum()}")
print(f"singletons (appear once): {(txt.value_counts() == 1).sum()}")

rule("4. CHARACTER SET (the important one)")
all_chars = "".join(txt)
charset = sorted(set(all_chars))
print(f"total distinct characters: {len(charset)}")
print("characters:", "".join(charset))
non_ascii = [c for c in charset if ord(c) > 127]
if non_ascii:
    print(f"\n*** NON-ASCII characters present ({len(non_ascii)}): ***")
    for c in non_ascii:
        try:
            name = unicodedata.name(c)
        except ValueError:
            name = "UNKNOWN"
        count = all_chars.count(c)
        print(f"   {c!r}  U+{ord(c):04X}  {name}  (appears {count}x)")
    print("   ^ these are likely homoglyph contamination (e.g. Cyrillic look-alikes)")
else:
    print("all characters are ASCII — good.")

rule("5. CASE-DUPLICATE CHECK")
lower_counts = Counter(txt.str.lower())
case_dupes = [(v, [t for t in txt.unique() if t.lower() == v])
              for v in lower_counts if len([t for t in txt.unique() if t.lower() == v]) > 1]
print(f"labels that collapse together when lowercased: {len(case_dupes)}")
for low, variants in case_dupes[:15]:
    print(f"   {variants}  -> '{low}'")
if len(case_dupes) > 15:
    print(f"   ... and {len(case_dupes) - 15} more")

rule("6. MOST / LEAST FREQUENT LABELS")
vc = txt.value_counts()
print("top 20 (check for non-medicine tokens like 'tablet','injection'):")
print(vc.head(20).to_string())
print("\nbottom 10:")
print(vc.tail(10).to_string())

rule("7. IMAGE FILES")
if IMG_DIR.exists():
    exts = Counter(p.suffix.lower() for p in IMG_DIR.iterdir() if p.is_file())
    n_files = sum(exts.values())
    print(f"files in image dir: {n_files}")
    print(f"extensions: {dict(exts)}")
    listed = set(df[img_col].astype(str))
    on_disk = set(p.name for p in IMG_DIR.iterdir() if p.is_file())
    missing = listed - on_disk
    orphan = on_disk - listed
    print(f"labelled images MISSING from disk: {len(missing)}")
    if missing:
        print("   examples:", list(missing)[:5])
    print(f"images on disk with NO label (orphans): {len(orphan)}")
    if orphan:
        print("   examples:", list(orphan)[:5])

    # Inspect a few image sizes / modes
    try:
        from PIL import Image
        print("\nsample image properties (first 5 found on disk):")
        for name in list(on_disk)[:5]:
            with Image.open(IMG_DIR / name) as im:
                print(f"   {name}: size={im.size} mode={im.mode}")
        widths, heights = [], []
        for name in list(on_disk)[:300]:
            with Image.open(IMG_DIR / name) as im:
                widths.append(im.size[0]); heights.append(im.size[1])
        if widths:
            print(f"\n(sample of {len(widths)} images) "
                  f"width  min/median/max = {min(widths)}/{sorted(widths)[len(widths)//2]}/{max(widths)} | "
                  f"height min/median/max = {min(heights)}/{sorted(heights)[len(heights)//2]}/{max(heights)}")
    except ImportError:
        print("PIL not installed — skip image property check.")
else:
    print("image directory not found — edit IMG_DIR at the top of this script.")

rule("DONE — copy everything above this line back to share.")
