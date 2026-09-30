import streamlit as st
import pandas as pd

# =========================
# データ読み込み
# =========================

@st.cache_data
def load_data():
    data = pd.read_csv("travel_recommendation_data_ml.csv")
    return data

data = load_data()

# 東京23区など「区」を除外した版
data_no_ku = data[~data["地域"].str.endswith("区")].copy()

# =========================
# 旅行タイプのプリセット
# =========================

travel_patterns_ml = {
    "海と自然でのんびり": {
        "sea": 1.0,
        "nature": 0.8,
        "ml_tourism": 0.6
    },
    "海沿い観光・リゾート": {
        "sea": 1.0,
        "sightseeing": 0.8,
        "ml_tourism": 0.7
    },
    "山と自然を楽しむ": {
        "mountain": 1.0,
        "nature": 1.0,
        "rural": 0.5,
        "ml_tourism": 0.6
    },
    "田舎でのんびり": {
        "rural": 1.0,
        "nature": 0.8,
        "car_trip": 0.6,
        "ml_tourism": 0.4
    },
    "都会で街歩き": {
        "urban": 1.0,
        "sightseeing": 0.8,
        "train_trip": 0.6,
        "ml_tourism": 0.5
    },
    "電車で行ける観光地": {
        "train_trip": 1.0,
        "sightseeing": 0.8,
        "urban": 0.4,
        "ml_tourism": 0.6
    }
}

# =========================
# 推薦関数
# =========================

def recommend_area_ml(
    data,
    urban=0,
    nature=0,
    sea=0,
    mountain=0,
    rural=0,
    train_trip=0,
    car_trip=0,
    sightseeing=0,
    ml_tourism=0.5,
    top_n=10
):
    result = data.copy()

    total_weight = (
        urban + nature + sea + mountain + rural
        + train_trip + car_trip + sightseeing + ml_tourism
    )

    if total_weight == 0:
        return pd.DataFrame()

    result["おすすめスコア"] = (
        urban * result["都会度"]
        + nature * result["自然度"]
        + sea * result["海度"]
        + mountain * result["山っぽさ"]
        + rural * result["田舎度"]
        + train_trip * result["電車旅向き"]
        + car_trip * result["車旅向き"]
        + sightseeing * result["観光・街歩き向き"]
        + ml_tourism * result["ML観光ポテンシャル"]
    ) / total_weight

    display_cols = [
        "地域",
        "おすすめスコア",
        "観光人気度",
        "ML観光ポテンシャル",
        "観光ポテンシャル差",
        "都会度",
        "自然度",
        "海度",
        "山っぽさ",
        "田舎度",
        "電車旅向き",
        "車旅向き",
        "観光・街歩き向き"
    ]

    return result.sort_values("おすすめスコア", ascending=False)[display_cols].head(top_n)


def recommend_by_pattern_ml(data, pattern_name, top_n=10):
    params = travel_patterns_ml[pattern_name]
    return recommend_area_ml(data, top_n=top_n, **params)

# =========================
# 画面
# =========================

st.title("旅行先レコメンドAI")
st.write("市町村統計データ・土地利用メッシュ・観光来訪者数を用いて、旅行タイプに合う地域を推薦します。")

st.sidebar.header("条件設定")

pattern_name = st.sidebar.selectbox(
    "旅行タイプを選んでください",
    list(travel_patterns_ml.keys())
)

top_n = st.sidebar.slider(
    "表示する件数",
    min_value=5,
    max_value=30,
    value=10
)

exclude_ku = st.sidebar.checkbox(
    "区を除外する",
    value=True
)

if exclude_ku:
    target_data = data_no_ku
else:
    target_data = data

# =========================
# 推薦結果
# =========================

result = recommend_by_pattern_ml(
    target_data,
    pattern_name,
    top_n=top_n
)

st.subheader(f"おすすめ旅行タイプ：{pattern_name}")

st.dataframe(
    result,
    use_container_width=True
)

# =========================
# 上位1件の説明
# =========================

if len(result) > 0:
    top = result.iloc[0]

    st.subheader("一番おすすめの地域")

    st.markdown(f"""
    ### {top["地域"]}

    - おすすめスコア：{top["おすすめスコア"]:.2f}
    - 観光人気度：{top["観光人気度"]:.2f}
    - ML観光ポテンシャル：{top["ML観光ポテンシャル"]:.2f}
    - 海度：{top["海度"]:.2f}
    - 自然度：{top["自然度"]:.2f}
    - 山っぽさ：{top["山っぽさ"]:.2f}
    - 都会度：{top["都会度"]:.2f}
    """)

# =========================
# カスタム診断
# =========================

st.subheader("自分で条件を調整する")

urban = st.slider("都会で遊びたい", 0.0, 1.0, 0.0, 0.1)
nature = st.slider("自然を楽しみたい", 0.0, 1.0, 0.0, 0.1)
sea = st.slider("海に行きたい", 0.0, 1.0, 0.0, 0.1)
mountain = st.slider("山に行きたい", 0.0, 1.0, 0.0, 0.1)
rural = st.slider("田舎でのんびりしたい", 0.0, 1.0, 0.0, 0.1)
train_trip = st.slider("電車で行きたい", 0.0, 1.0, 0.0, 0.1)
car_trip = st.slider("車で行きたい", 0.0, 1.0, 0.0, 0.1)
sightseeing = st.slider("観光・街歩きをしたい", 0.0, 1.0, 0.0, 0.1)
ml_tourism = st.slider("観光ポテンシャルを重視する", 0.0, 1.0, 0.5, 0.1)

custom_result = recommend_area_ml(
    target_data,
    urban=urban,
    nature=nature,
    sea=sea,
    mountain=mountain,
    rural=rural,
    train_trip=train_trip,
    car_trip=car_trip,
    sightseeing=sightseeing,
    ml_tourism=ml_tourism,
    top_n=top_n
)

st.subheader("カスタム条件でのおすすめ地域")

st.dataframe(
    custom_result,
    use_container_width=True
)