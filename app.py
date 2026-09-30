
from pathlib import Path
 
import pandas as pd
import streamlit as st
 
st.set_page_config(page_title="旅行先レコメンドAI", page_icon="🧭", layout="wide")
 
# =========================
# データ読み込み
# =========================
 
DATA_PATH = Path(__file__).parent / "travel_recommendation_data_ml.csv"
 
 
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
 
    # 古いCSVで「観光データあり」列がなければ、来訪者数から判定する
    if "観光データあり" not in data.columns:
        data["観光データあり"] = data["年間観光来訪者数"].fillna(0) > 0
 
    # 観光データがない地域は、実績と差を空欄にする（0人扱いにしない）
    data.loc[~data["観光データあり"], ["観光人気度", "観光ポテンシャル差"]] = float("nan")
    return data
 
 
data = load_data()
 
# =========================
# 旅行タイプのプリセット
# =========================
 
TRAVEL_PATTERNS = {
    "海と自然でのんびり": {"sea": 1.0, "nature": 0.8, "ml_tourism": 0.6},
    "海沿い観光・リゾート": {"sea": 1.0, "sightseeing": 0.8, "ml_tourism": 0.7},
    "山と自然を楽しむ": {"mountain": 1.0, "nature": 1.0, "rural": 0.5, "ml_tourism": 0.6},
    "田舎でのんびり": {"rural": 1.0, "nature": 0.8, "car_trip": 0.6, "ml_tourism": 0.4},
    "都会で街歩き": {"urban": 1.0, "sightseeing": 0.8, "train_trip": 0.6, "ml_tourism": 0.5},
    "電車で行ける観光地": {"train_trip": 1.0, "sightseeing": 0.8, "urban": 0.4, "ml_tourism": 0.6},
}
 
# 重みの名前と、データの列名の対応
SCORE_COLUMNS = {
    "urban": "都会度",
    "nature": "自然度",
    "sea": "海度",
    "mountain": "山っぽさ",
    "rural": "田舎度",
    "train_trip": "電車旅向き",
    "car_trip": "車旅向き",
    "sightseeing": "観光・街歩き向き",
    "ml_tourism": "ML観光ポテンシャル",
}
 
DISPLAY_COLUMNS = [
    "地域", "おすすめスコア", "観光人気度", "ML観光ポテンシャル", "観光ポテンシャル差",
    "都会度", "自然度", "海度", "山っぽさ", "田舎度",
    "電車旅向き", "車旅向き", "観光・街歩き向き",
]
 
# =========================
# 推薦関数
# =========================
 
 
def recommend(data, weights, top_n=10):
    """重み付き平均でおすすめスコアを計算し、上位を返す"""
    total_weight = sum(weights.values())
    if total_weight == 0:
        return pd.DataFrame()
 
    result = data.copy()
    result["おすすめスコア"] = sum(
        w * result[SCORE_COLUMNS[key]] for key, w in weights.items()
    ) / total_weight
 
    return (
        result.sort_values("おすすめスコア", ascending=False)[DISPLAY_COLUMNS]
        .head(top_n)
        .round(1)
    )
 
 
def fmt(value):
    """数値を表示用の文字にする。欠損は「データなし」"""
    return "データなし" if pd.isna(value) else f"{value:.1f}"
 
 
def show_table(df):
    """1位から番号を振って表を表示する"""
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    st.dataframe(df, width="stretch")
 
 
# =========================
# サイドバー（共通の条件）
# =========================
 
st.sidebar.header("表示の条件")
 
top_n = st.sidebar.slider("表示する件数", min_value=5, max_value=30, value=10)
 
exclude_ku = st.sidebar.checkbox(
    "東京23区などの「区」を除外する",
    value=True,
    help="区は人口密度などが突出しているため、都会系の推薦が区だけで埋まるのを防ぎます。",
)
 
target_data = data[~data["地域"].str.endswith("区")] if exclude_ku else data
 
# =========================
# 画面
# =========================
 
st.title("旅行先レコメンドAI")
st.write(
    "全国の市区町村の統計データ・土地利用データ・観光来訪者数をもとに、"
    "旅行のタイプに合う地域を推薦します。"
)
 
tab_type, tab_custom, tab_hidden, tab_about = st.tabs(
    ["旅行タイプで探す", "条件を自分で決める", "穴場を探す", "このアプリについて"]
)
 
# ---------- 旅行タイプで探す ----------
with tab_type:
    pattern_name = st.selectbox("旅行タイプ", list(TRAVEL_PATTERNS.keys()))
    result = recommend(target_data, TRAVEL_PATTERNS[pattern_name], top_n=top_n)
 
    top = result.iloc[0]
    st.subheader(f"一番のおすすめ：{top['地域']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("おすすめスコア", fmt(top["おすすめスコア"]))
    c2.metric("観光人気度（実績）", fmt(top["観光人気度"]))
    c3.metric("観光ポテンシャル（予測）", fmt(top["ML観光ポテンシャル"]))
 
    st.subheader(f"「{pattern_name}」のおすすめ {top_n} 件")
    show_table(result)
 
# ---------- 条件を自分で決める ----------
with tab_custom:
    st.write("行きたい旅の条件を、0（気にしない）〜1（重視する）で選んでください。")
 
    left, right = st.columns(2)
    with left:
        urban = st.slider("都会で遊びたい", 0.0, 1.0, 0.0, 0.1)
        nature = st.slider("自然を楽しみたい", 0.0, 1.0, 0.0, 0.1)
        sea = st.slider("海に行きたい", 0.0, 1.0, 0.0, 0.1)
        mountain = st.slider("山に行きたい", 0.0, 1.0, 0.0, 0.1)
        rural = st.slider("田舎でのんびりしたい", 0.0, 1.0, 0.0, 0.1)
    with right:
        train_trip = st.slider("電車で行きたい", 0.0, 1.0, 0.0, 0.1)
        car_trip = st.slider("車で行きたい", 0.0, 1.0, 0.0, 0.1)
        sightseeing = st.slider("観光・街歩きをしたい", 0.0, 1.0, 0.0, 0.1)
        ml_tourism = st.slider("観光地としての魅力を重視する", 0.0, 1.0, 0.5, 0.1)
 
    custom_weights = {
        "urban": urban,
        "nature": nature,
        "sea": sea,
        "mountain": mountain,
        "rural": rural,
        "train_trip": train_trip,
        "car_trip": car_trip,
        "sightseeing": sightseeing,
        "ml_tourism": ml_tourism,
    }
    custom_result = recommend(target_data, custom_weights, top_n=top_n)
 
    if custom_result.empty:
        st.info("どれか1つ以上のスライダーを0より大きくすると、おすすめが表示されます。")
    else:
        st.subheader(f"あなたの条件でのおすすめ {top_n} 件")
        show_table(custom_result)
 
# ---------- 穴場を探す ----------
with tab_hidden:
    st.write(
        "地域の特徴から予測した観光ポテンシャルに比べて、実際の観光人気度が低い地域です。"
        "特徴の割にまだ人が少ない、穴場の候補と考えられます。"
    )
    hidden = (
        target_data[target_data["観光データあり"]]
        .sort_values("観光ポテンシャル差", ascending=False)
        [["地域", "観光ポテンシャル差", "ML観光ポテンシャル", "観光人気度",
          "自然度", "海度", "山っぽさ", "都会度"]]
        .head(top_n)
        .round(1)
    )
    show_table(hidden)
    st.caption("観光来訪者数のデータがない地域は、このランキングから除いています。")
 
# ---------- このアプリについて ----------
with tab_about:
    st.markdown(
        """
**指標の作り方**
 
森林や海の面積割合、人口密度、通勤手段、宿泊・飲食業の従業者数などを0〜100に換算し、
重みを付けて「都会度」「海度」などの指標を作っています。
 
**観光ポテンシャル（予測）と観光人気度（実績）**
 
- 観光ポテンシャル：地域の特徴から、観光来訪者数をランダムフォレストで予測した値
- 観光人気度：実際の年間観光来訪者数から作った値
- 観光ポテンシャル差：予測 − 実績。大きいほど「特徴の割に人が少ない」地域
 
予測モデルの精度は高くないため、予測値は地域同士を比べる目安として使っています。
観光来訪者数のデータがない地域は、観光人気度を「データなし」としています。
 
**使用データ**
 
- 市区町村の統計データ：e-Stat
- 土地利用データ：国土数値情報
- 観光来訪者数：観光庁「宿泊旅行統計調査」「訪日外国人消費動向調査」
"""
    )
 
