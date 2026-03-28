import json
import os
import re
import textwrap
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import gdown
import gspread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from google.oauth2.service_account import Credentials
from matplotlib.backends.backend_pdf import PdfPages
from pydantic import BaseModel, Field


app = FastAPI(title="DMA HeySiri ETL Service")


# =========================
# HARDCODED CONFIG
# =========================
SERVICE_ACCOUNT_JSON_PATH = r"E:\AI_Sol\key\gen-lang-client-0013411653-b4c9c85cf6e5.json"

TIKTOK_SHEET_URL = "https://docs.google.com/spreadsheets/d/1EdFWhKlgTjHO82tsRsb7VQoHsgaWlfTJ4-XPdURYcc0/edit?usp=sharing"
OUTPUT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1EdFWhKlgTjHO82tsRsb7VQoHsgaWlfTJ4-XPdURYcc0/edit?gid=1703933461#gid=1703933461"

TIKTOK_RAW_WORKSHEETS = {
    "Photogenic": "PHOTOGENIC",
    "Phototime": "PHOTOTIME",
    "Photopalette": "PHOTOPALETTE",
}

TT_WORKSHEET_NAMES = {
    "Photogenic": "cleaned_TT_PHOTOGENIC",
    "Phototime": "cleaned_TT_PHOTOTIME",
    "Photopalette": "cleaned_TT_PHOTOPALETTE",
}

FB_WORKSHEET_NAMES = {
    "Photogenic": "cleaned_FB_PHOTOGENIC",
    "Phototime": "cleaned_FB_PHOTOTIME",
    "Photopalette": "cleaned_FB_PHOTOPALETTE",
}

FB_INPUT_LINKS = {
    "Photogenic": "https://drive.google.com/file/d/1_A6P5vKjtYrxBB5n3hXI5uOeD3EV2LsG/view?usp=sharing",
    "Phototime": "https://drive.google.com/file/d/1LeWrUOuevDxnnRW5hisUKab1yAMW7JEm/view?usp=sharing",
    "Photopalette": "https://drive.google.com/file/d/1rBGim6Xx3MsCkHWIrLwVT2BSEOI1F6bs/view?usp=sharing",
}

DOWNLOAD_DIR = Path(r"E:\AI_Sol\downloaded_json")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path(r"E:\AI_Sol\dma_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_ANALYSIS_DIR = OUTPUT_DIR / "video_analysis_reports"
VIDEO_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

PUBLIC_REPORTS_ROUTE = "/public-reports"
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
app.mount(PUBLIC_REPORTS_ROUTE, StaticFiles(directory=str(VIDEO_ANALYSIS_DIR)), name="public-reports")


# =========================
# GLOBAL STATE
# =========================
gc = None
tt_source_spreadsheet = None
output_spreadsheet = None

tt_raw: Optional[Dict[str, pd.DataFrame]] = None
tt_cleaned: Optional[Dict[str, pd.DataFrame]] = None
tt_all: Optional[pd.DataFrame] = None

fb_downloaded_files: Optional[Dict[str, Path]] = None
fb_cleaned: Optional[Dict[str, pd.DataFrame]] = None
fb_all: Optional[pd.DataFrame] = None

tt_pg_content: Optional[pd.DataFrame] = None
top_hashtags_pg: Optional[pd.DataFrame] = None
tt_brand_benchmark: Optional[pd.DataFrame] = None
fb_brand_summary: Optional[pd.DataFrame] = None
fb_content_mix: Optional[pd.DataFrame] = None
fb_community_signal: Optional[pd.DataFrame] = None
repurpose_summary: Optional[pd.DataFrame] = None
viral_summary: Optional[pd.DataFrame] = None


class ScrapedVideoItem(BaseModel):
    tiktok_username: str = ""
    video_id: str
    video_url: str = ""
    is_latest_video: Optional[bool] = None
    days_since_published: Optional[float] = None
    video_desc: str = ""
    hashtags: str = ""
    createTimeISO: Optional[str] = None
    publish_datetime: Optional[str] = None
    duration_seconds: Optional[float] = None
    views_count: Optional[float] = None
    likes_count_video: Optional[float] = None
    comments_count: Optional[float] = None
    shares_count: Optional[float] = None


class VideoBatchAnalysisRequest(BaseModel):
    videos: List[ScrapedVideoItem] = Field(default_factory=list)
    topic: Optional[str] = None
    top_n: int = Field(default=5, ge=1, le=30)
    export_pdf: bool = True


class PlotVideoItem(BaseModel):
    publish_datetime: str
    views_count: Optional[float] = None
    likes_count_video: Optional[float] = None
    comments_count: Optional[float] = None
    shares_count: Optional[float] = None


class PlotVideoRequest(BaseModel):
    videos: List[PlotVideoItem] = Field(default_factory=list)


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.where((pd.notna(b)) & (b != 0), a / b, np.nan)


def extract_hashtags(text):
    if not isinstance(text, str):
        return []
    return re.findall(r"#\w+", text.lower())


def count_hashtags_from_string(x):
    return str(x).count("#")


def standardize_brand_name(x):
    x = str(x).strip().lower()
    if "genic" in x:
        return "Photogenic"
    if "time" in x:
        return "Phototime"
    if "palette" in x:
        return "Photopalette"
    return x.title()


def classify_content(text):
    text = str(text).lower()
    if any(k in text for k in ["tutorial", "tip", "tips", "hướng dẫn", "how to", "cách "]):
        return "Tutorial"
    if "pov" in text:
        return "POV"
    if any(k in text for k in ["khai trương", "grand opening", "opening", "store", "chi nhánh", "cửa hàng"]):
        return "Check-in / Store"
    if any(k in text for k in ["photobooth", "frame", "concept", "room", "phòng", "background"]):
        return "Product / Concept"
    if any(k in text for k in ["event", "sự kiện", "voucher", "ưu đãi", "promotion", "sale", "discount", "giảm"]):
        return "Promotion / Event"
    return "Others"


def add_time_features_from_utc(df, utc_col, prefix="post"):
    df = df.copy()
    df[f"{prefix}_time_utc"] = pd.to_datetime(df[utc_col], errors="coerce", utc=True)
    df[f"{prefix}_time_vn"] = df[f"{prefix}_time_utc"].dt.tz_convert("Asia/Ho_Chi_Minh")
    df[f"{prefix}_date_vn"] = df[f"{prefix}_time_vn"].dt.date
    df[f"{prefix}_hour_vn"] = df[f"{prefix}_time_vn"].dt.hour
    df[f"{prefix}_dayofweek_vn"] = df[f"{prefix}_time_vn"].dt.day_name()
    df[f"{prefix}_month_vn"] = df[f"{prefix}_time_vn"].dt.month
    df[f"{prefix}_year_vn"] = df[f"{prefix}_time_vn"].dt.year
    return df


def iqr_outlier_flags(df, cols):
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            df[f"{col}_outlier_iqr"] = 0
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[f"{col}_outlier_iqr"] = ((series < lower) | (series > upper)).astype(int)
    return df


def extract_gdrive_file_id(url: str) -> str:
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    match = re.search(r"id=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    raise ValueError(f"Không tìm thấy file id trong URL: {url}")


def download_gdrive_file(url: str, output_path: str):
    file_id = extract_gdrive_file_id(url)
    download_url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(download_url, output_path, quiet=False)


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def ensure_google_clients():
    global gc, tt_source_spreadsheet, output_spreadsheet

    if gc is not None and tt_source_spreadsheet is not None and output_spreadsheet is not None:
        return

    if not Path(SERVICE_ACCOUNT_JSON_PATH).exists():
        raise FileNotFoundError(f"Không tìm thấy file service account: {SERVICE_ACCOUNT_JSON_PATH}")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON_PATH, scopes=scopes)
    gc = gspread.authorize(creds)
    tt_source_spreadsheet = gc.open_by_url(TIKTOK_SHEET_URL)
    output_spreadsheet = gc.open_by_url(OUTPUT_SHEET_URL)


def preprocess_tiktok_raw_dataframe(df_raw, brand):
    df = df_raw.copy()

    rename_map = {
        "video_id": "post_id",
        "video_url": "post_url",
        "video_desc": "text",
        "likes_count_video": "likes",
        "comments_count": "comments",
        "shares_count": "shares",
    }
    df = df.rename(columns=rename_map)

    df["brand"] = brand
    df["platform"] = "TikTok"

    expected_cols = [
        "tiktok_username", "post_id", "post_url", "is_latest_video", "days_since_published",
        "text", "hashtags", "createTimeISO", "publish_datetime", "duration_seconds",
        "views_count", "likes", "comments", "shares", "engagement_rate"
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    numeric_cols = [
        "days_since_published", "duration_seconds", "views_count",
        "likes", "comments", "shares", "engagement_rate"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["text"] = df["text"].fillna("").astype(str)
    df["hashtags"] = df["hashtags"].fillna("").astype(str)
    df["tiktok_username"] = df["tiktok_username"].fillna("").astype(str)

    df = add_time_features_from_utc(df, "createTimeISO", "post")

    df["text_length"] = df["text"].astype(str).str.len()
    df["hashtag_count"] = df["hashtags"].astype(str).apply(count_hashtags_from_string)
    df["engagement"] = df["likes"].fillna(0) + df["comments"].fillna(0) + df["shares"].fillna(0)
    df["er_by_views"] = safe_divide(df["engagement"], df["views_count"])
    df["share_velocity"] = safe_divide(df["shares"], df["views_count"])
    df["comment_intent_rate"] = safe_divide(df["comments"], df["views_count"])
    df["reach_proxy"] = df["views_count"]
    df["long_caption_flag"] = (df["text_length"] >= 300).astype(int)
    df["content_type"] = df["text"].apply(classify_content)
    df["brand"] = df["brand"].apply(standardize_brand_name)

    df = df.drop_duplicates(subset=["brand", "platform", "post_id"], keep="first")

    outlier_cols = [
        "views_count", "likes", "comments", "shares", "engagement",
        "er_by_views", "duration_seconds", "text_length", "hashtag_count"
    ]
    df = iqr_outlier_flags(df, outlier_cols)

    final_cols = [
        "brand", "platform", "tiktok_username", "post_id", "post_url",
        "post_time_utc", "post_time_vn", "post_date_vn", "post_hour_vn", "post_dayofweek_vn",
        "is_latest_video", "days_since_published",
        "text", "hashtags", "text_length", "hashtag_count",
        "duration_seconds", "views_count", "likes", "comments", "shares",
        "engagement_rate", "engagement", "er_by_views",
        "share_velocity", "comment_intent_rate",
        "long_caption_flag", "content_type"
    ]
    outlier_cols_created = [col for col in df.columns if col.endswith("_outlier_iqr")]
    final_cols += outlier_cols_created
    final_cols = [col for col in final_cols if col in df.columns]

    return df[final_cols].sort_values("post_time_vn", ascending=False).reset_index(drop=True)


def flatten_facebook_post(post, brand):
    user = post.get("user", {}) if isinstance(post.get("user"), dict) else {}
    media = post.get("media", []) if isinstance(post.get("media"), list) else []
    text_refs = post.get("textReferences", []) if isinstance(post.get("textReferences"), list) else []
    text = post.get("text", "")
    hashtags = extract_hashtags(text)

    return {
        "brand": brand,
        "platform": "Facebook",
        "post_id": post.get("postId"),
        "post_url": post.get("url"),
        "page_name": post.get("pageName"),
        "time_utc_raw": post.get("time"),
        "timestamp_unix": post.get("timestamp"),
        "text": text,
        "text_length": len(text) if isinstance(text, str) else 0,
        "hashtags": ", ".join(hashtags),
        "hashtag_count": len(hashtags),
        "has_hashtag": 1 if len(hashtags) > 0 else 0,
        "text_reference_count": len(text_refs) if isinstance(text_refs, (list, dict, str)) else 0,
        "likes": post.get("likes", 0),
        "comments": post.get("comments", 0),
        "shares": post.get("shares", 0),
        "views_count": post.get("viewsCount", np.nan),
        "top_reactions_count": post.get("topReactionsCount", 0),
        "reaction_like": post.get("reactionLikeCount", 0),
        "reaction_love": post.get("reactionLoveCount", 0),
        "reaction_care": post.get("reactionCareCount", 0),
        "reaction_haha": post.get("reactionHahaCount", 0),
        "reaction_wow": post.get("reactionWowCount", 0),
        "reaction_sad": post.get("reactionSadCount", 0),
        "reaction_angry": post.get("reactionAngryCount", 0),
        "is_video": int(bool(post.get("isVideo", False))),
        "media_count": len(media),
        "content_type": classify_content(text),
        "user_name": user.get("name"),
    }


def preprocess_facebook_posts(posts, brand):
    rows = [flatten_facebook_post(post, brand) for post in posts]
    df = pd.DataFrame(rows)

    numeric_cols = [
        "timestamp_unix", "text_length", "hashtag_count", "has_hashtag", "text_reference_count",
        "likes", "comments", "shares", "views_count", "top_reactions_count",
        "reaction_like", "reaction_love", "reaction_care", "reaction_haha",
        "reaction_wow", "reaction_sad", "reaction_angry", "is_video", "media_count"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    fill_zero_cols = [col for col in numeric_cols if col != "views_count"]
    existing = [col for col in fill_zero_cols if col in df.columns]
    df[existing] = df[existing].fillna(0)

    df = add_time_features_from_utc(df, "time_utc_raw", "post")
    df = df.drop_duplicates(subset=["brand", "platform", "post_id"], keep="first")

    df["total_reactions_detail"] = (
        df["reaction_like"] + df["reaction_love"] + df["reaction_care"] +
        df["reaction_haha"] + df["reaction_wow"] + df["reaction_sad"] + df["reaction_angry"]
    )

    df["engagement"] = df["likes"] + df["comments"] + df["shares"]
    df["reach_proxy"] = df["views_count"]
    df["er_by_views"] = safe_divide(df["engagement"], df["views_count"])
    df["share_velocity"] = safe_divide(df["shares"], df["views_count"])
    df["comment_intent_rate"] = safe_divide(df["comments"], df["views_count"])
    df["long_caption_flag"] = (df["text_length"] >= 300).astype(int)

    df["post_format"] = np.select(
        [
            df["is_video"] == 1,
            df["media_count"] > 1,
            df["media_count"] == 1,
        ],
        [
            "video",
            "album_photo",
            "single_photo",
        ],
        default="text_only"
    )

    outlier_cols = [
        "views_count", "likes", "comments", "shares", "engagement",
        "er_by_views", "text_length", "hashtag_count"
    ]
    df = iqr_outlier_flags(df, outlier_cols)

    final_cols = [
        "brand", "platform", "post_id", "post_url", "page_name",
        "post_time_utc", "post_time_vn", "post_date_vn", "post_hour_vn", "post_dayofweek_vn",
        "text", "text_length", "hashtags", "hashtag_count", "has_hashtag",
        "is_video", "media_count", "post_format", "content_type",
        "likes", "comments", "shares", "views_count",
        "engagement", "er_by_views", "share_velocity", "comment_intent_rate",
        "reaction_like", "reaction_love", "reaction_care", "reaction_haha",
        "reaction_wow", "reaction_sad", "reaction_angry",
        "total_reactions_detail", "long_caption_flag"
    ]
    outlier_cols_created = [col for col in df.columns if col.endswith("_outlier_iqr")]
    final_cols += outlier_cols_created
    final_cols = [col for col in final_cols if col in df.columns]

    return df[final_cols].sort_values("post_time_vn", ascending=False).reset_index(drop=True)


def get_or_create_worksheet(spreadsheet, title, rows=1000, cols=50):
    try:
        ws = spreadsheet.worksheet(title)
        ws.clear()
        return ws
    except gspread.WorksheetNotFound:
        return spreadsheet.add_worksheet(title=title, rows=rows, cols=cols)
    except gspread.exceptions.APIError as ex:
        status_code = getattr(getattr(ex, "response", None), "status_code", None)
        if status_code == 403:
            return None
        raise ex


def upload_dataframe_safe(spreadsheet, worksheet_title, dataframe):
    ws = get_or_create_worksheet(
        spreadsheet,
        title=worksheet_title,
        rows=max(len(dataframe) + 20, 1000),
        cols=max(len(dataframe.columns) + 5, 60),
    )
    if ws is None:
        return False
    try:
        set_with_dataframe(ws, dataframe, include_index=False, include_column_header=True)
        return True
    except gspread.exceptions.APIError as ex:
        status_code = getattr(getattr(ex, "response", None), "status_code", None)
        if status_code == 403:
            return False
        raise ex


def prepare_tiktok_data():
    global tt_raw, tt_cleaned, tt_all

    ensure_google_clients()

    loaded_raw = {}
    for brand, ws_name in TIKTOK_RAW_WORKSHEETS.items():
        ws = tt_source_spreadsheet.worksheet(ws_name)
        df_raw = get_as_dataframe(ws, evaluate_formulas=True, header=0).dropna(how="all")
        df_raw.columns = [str(col).strip() for col in df_raw.columns]
        loaded_raw[brand] = df_raw

    loaded_clean = {}
    for brand, raw_df in loaded_raw.items():
        loaded_clean[brand] = preprocess_tiktok_raw_dataframe(raw_df, brand)

    tt_raw = loaded_raw
    tt_cleaned = loaded_clean
    tt_all = pd.concat(tt_cleaned.values(), ignore_index=True)


def prepare_facebook_data():
    global fb_downloaded_files, fb_cleaned, fb_all

    downloaded = {}
    for brand, url in FB_INPUT_LINKS.items():
        output_path = DOWNLOAD_DIR / f"{brand.lower()}_facebook.json"
        download_gdrive_file(url, str(output_path))
        downloaded[brand] = output_path

    cleaned = {}
    for brand, path in downloaded.items():
        posts = load_json(path)
        cleaned[brand] = preprocess_facebook_posts(posts, brand)

    fb_downloaded_files = downloaded
    fb_cleaned = cleaned
    fb_all = pd.concat(fb_cleaned.values(), ignore_index=True)


def run_analytics_data():
    global tt_pg_content, top_hashtags_pg, tt_brand_benchmark
    global fb_brand_summary, fb_content_mix, fb_community_signal
    global repurpose_summary, viral_summary

    if tt_all is None:
        raise ValueError("Please run /prepare_tiktok first")
    if fb_all is None:
        raise ValueError("Please run /prepare_facebook first")

    tt = tt_all.copy()
    fb = fb_all.copy()
    tt_pg = tt[tt["brand"] == "Photogenic"].copy()
    fb_pg = fb[fb["brand"] == "Photogenic"].copy()

    tt_pg_content = (
        tt_pg.groupby("content_type", dropna=False)
        .agg(
            posts=("post_id", "count"),
            avg_views=("views_count", "mean"),
            median_views=("views_count", "median"),
            avg_engagement=("engagement", "mean"),
            avg_er=("er_by_views", "mean"),
            avg_shares=("shares", "mean"),
        )
        .sort_values(["avg_er", "avg_views"], ascending=False)
        .reset_index()
    )

    hashtag_series = (
        tt_pg["hashtags"]
        .fillna("")
        .str.lower()
        .str.findall(r"#\w+")
        .explode()
        .dropna()
    )
    top_hashtags_pg = hashtag_series.value_counts().head(20).reset_index()
    top_hashtags_pg.columns = ["hashtag", "frequency"]

    tt_brand_benchmark = (
        tt.groupby("brand")
        .agg(
            posts=("post_id", "count"),
            avg_views=("views_count", "mean"),
            median_views=("views_count", "median"),
            avg_engagement=("engagement", "mean"),
            avg_er=("er_by_views", "mean"),
            avg_duration=("duration_seconds", "mean"),
        )
        .sort_values("avg_er", ascending=False)
        .reset_index()
    )

    fb_brand_summary = (
        fb.groupby("brand")
        .agg(
            posts=("post_id", "count"),
            avg_views=("views_count", "mean"),
            median_views=("views_count", "median"),
            avg_engagement=("engagement", "mean"),
            median_engagement=("engagement", "median"),
            avg_er=("er_by_views", "mean"),
            video_ratio=("is_video", "mean"),
        )
        .sort_values("avg_er", ascending=False)
        .reset_index()
    )

    fb_content_mix = (
        fb.groupby(["brand", "content_type"])
        .agg(
            posts=("post_id", "count"),
            avg_views=("views_count", "mean"),
            avg_engagement=("engagement", "mean"),
            avg_er=("er_by_views", "mean"),
        )
        .reset_index()
        .sort_values(["brand", "avg_er"], ascending=[True, False])
    )

    fb_community_signal = (
        fb.groupby("brand")
        .agg(
            avg_comments=("comments", "mean"),
            avg_shares=("shares", "mean"),
            avg_comment_intent_rate=("comment_intent_rate", "mean"),
            avg_text_length=("text_length", "mean"),
            long_caption_ratio=("long_caption_flag", "mean"),
        )
        .sort_values("avg_comment_intent_rate", ascending=False)
        .reset_index()
    )

    repurpose_summary = pd.DataFrame(
        {
            "group": ["TikTok All", "Facebook Video Only", "Facebook All"],
            "posts": [
                tt_pg["post_id"].count(),
                fb_pg.loc[fb_pg["post_format"] == "video", "post_id"].count(),
                fb_pg["post_id"].count(),
            ],
            "avg_views": [
                tt_pg["views_count"].mean(),
                fb_pg.loc[fb_pg["post_format"] == "video", "views_count"].mean(),
                fb_pg["views_count"].mean(),
            ],
            "avg_engagement": [
                tt_pg["engagement"].mean(),
                fb_pg.loc[fb_pg["post_format"] == "video", "engagement"].mean(),
                fb_pg["engagement"].mean(),
            ],
            "avg_er": [
                tt_pg["er_by_views"].mean(),
                fb_pg.loc[fb_pg["post_format"] == "video", "er_by_views"].mean(),
                fb_pg["er_by_views"].mean(),
            ],
        }
    )

    viral_paradox = pd.concat(
        [
            tt[tt["brand"] == "Photogenic"][["platform", "post_id", "views_count", "engagement", "er_by_views"]],
            fb[fb["brand"] == "Photogenic"][["platform", "post_id", "views_count", "engagement", "er_by_views"]],
        ],
        ignore_index=True,
    )

    viral_paradox["view_bucket"] = viral_paradox.groupby("platform")["views_count"].transform(
        lambda values: pd.qcut(
            values.rank(method="first"),
            q=4,
            labels=["Low", "Mid-Low", "Mid-High", "High"],
        )
    )

    viral_summary = (
        viral_paradox.groupby(["platform", "view_bucket"])
        .agg(
            posts=("post_id", "count"),
            avg_views=("views_count", "mean"),
            avg_engagement=("engagement", "mean"),
            avg_er=("er_by_views", "mean"),
        )
        .reset_index()
    )


def export_excel_file() -> Path:
    if tt_all is None or fb_all is None:
        raise ValueError("Please run /prepare_tiktok and /prepare_facebook first")
    if any(table is None for table in [
        tt_brand_benchmark,
        tt_pg_content,
        top_hashtags_pg,
        fb_brand_summary,
        fb_content_mix,
        fb_community_signal,
        repurpose_summary,
        viral_summary,
    ]):
        raise ValueError("Please run /run_analytics first")

    summary_excel = OUTPUT_DIR / "DMA_HeySiri_summary_tables_v2.xlsx"

    tt_all_export = tt_all.copy()
    for col in ["post_time_utc", "post_time_vn"]:
        if col in tt_all_export.columns and pd.api.types.is_datetime64_any_dtype(tt_all_export[col]):
            tt_all_export[col] = tt_all_export[col].dt.tz_localize(None)

    fb_all_export = fb_all.copy()
    for col in ["post_time_utc", "post_time_vn"]:
        if col in fb_all_export.columns and pd.api.types.is_datetime64_any_dtype(fb_all_export[col]):
            fb_all_export[col] = fb_all_export[col].dt.tz_localize(None)

    with pd.ExcelWriter(summary_excel, engine="openpyxl") as writer:
        tt_all_export.to_excel(writer, sheet_name="TT_All_Cleaned", index=False)
        fb_all_export.to_excel(writer, sheet_name="FB_All_Cleaned", index=False)
        tt_brand_benchmark.to_excel(writer, sheet_name="TT_Benchmark", index=False)
        tt_pg_content.to_excel(writer, sheet_name="TT_PG_Content", index=False)
        top_hashtags_pg.to_excel(writer, sheet_name="TT_PG_Hashtags", index=False)
        fb_brand_summary.to_excel(writer, sheet_name="FB_Benchmark", index=False)
        fb_content_mix.to_excel(writer, sheet_name="FB_Content_Mix", index=False)
        fb_community_signal.to_excel(writer, sheet_name="FB_Community", index=False)
        repurpose_summary.to_excel(writer, sheet_name="Crossplatform_Repurpose", index=False)
        viral_summary.to_excel(writer, sheet_name="Viral_Paradox", index=False)

    return summary_excel


def _coerce_video_dataframe(videos: List[ScrapedVideoItem]) -> pd.DataFrame:
    records = [video.model_dump() for video in videos]
    df = pd.DataFrame(records)

    required_cols = [
        "tiktok_username",
        "video_id",
        "video_url",
        "video_desc",
        "hashtags",
        "createTimeISO",
        "publish_datetime",
        "duration_seconds",
        "views_count",
        "likes_count_video",
        "comments_count",
        "shares_count",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    numeric_cols = [
        "duration_seconds",
        "views_count",
        "likes_count_video",
        "comments_count",
        "shares_count",
        "days_since_published",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["video_desc"] = df["video_desc"].fillna("").astype(str)
    df["hashtags"] = df["hashtags"].fillna("").astype(str)
    df["tiktok_username"] = df["tiktok_username"].fillna("").astype(str)
    df["video_id"] = df["video_id"].fillna("").astype(str)
    df["video_url"] = df["video_url"].fillna("").astype(str)

    df["publish_ts"] = pd.to_datetime(df["createTimeISO"], errors="coerce", utc=True)
    df["publish_vn"] = df["publish_ts"].dt.tz_convert("Asia/Ho_Chi_Minh")

    now_utc = pd.Timestamp.now(tz="UTC")
    missing_days = df["days_since_published"].isna()
    df.loc[missing_days, "days_since_published"] = (
        (now_utc - df.loc[missing_days, "publish_ts"]).dt.total_seconds() / (24 * 3600)
    )

    df["days_since_published"] = df["days_since_published"].fillna(0).clip(lower=0)

    df["caption_length"] = df["video_desc"].str.len()
    df["hashtag_count"] = df["hashtags"].astype(str).apply(count_hashtags_from_string)
    df["engagement"] = (
        df["likes_count_video"].fillna(0)
        + df["comments_count"].fillna(0)
        + df["shares_count"].fillna(0)
    )
    df["engagement_rate"] = np.where(
        (df["views_count"].notna()) & (df["views_count"] > 0),
        df["engagement"] / df["views_count"],
        np.nan,
    )
    df["share_rate"] = np.where(
        (df["views_count"].notna()) & (df["views_count"] > 0),
        df["shares_count"] / df["views_count"],
        np.nan,
    )
    df["comment_rate"] = np.where(
        (df["views_count"].notna()) & (df["views_count"] > 0),
        df["comments_count"] / df["views_count"],
        np.nan,
    )

    return df


def _build_actionable_insights(df: pd.DataFrame, top_n: int) -> List[str]:
    insights: List[str] = []

    if df.empty:
        return ["Không có video hợp lệ để phân tích."]

    median_duration = float(df["duration_seconds"].median()) if df["duration_seconds"].notna().any() else 0.0
    high_er_threshold = df["engagement_rate"].quantile(0.75) if df["engagement_rate"].notna().any() else np.nan
    high_er_videos = df[df["engagement_rate"] >= high_er_threshold] if pd.notna(high_er_threshold) else pd.DataFrame()

    if pd.notna(high_er_threshold) and not high_er_videos.empty:
        rec_duration = float(high_er_videos["duration_seconds"].median()) if high_er_videos["duration_seconds"].notna().any() else median_duration
        insights.append(
            f"Tái sử dụng format video có ER cao: ưu tiên duration quanh {rec_duration:.0f}s (median nhóm top ER)."
        )

    top_hashtags = (
        df["hashtags"]
        .fillna("")
        .str.lower()
        .str.findall(r"#\\w+")
        .explode()
        .dropna()
        .value_counts()
        .head(top_n)
    )
    if not top_hashtags.empty:
        hashtag_text = ", ".join([f"{tag} ({cnt})" for tag, cnt in top_hashtags.items()])
        insights.append(f"Giữ lại hashtag core trong video mới: {hashtag_text}.")

    top_videos = df.sort_values(["engagement_rate", "views_count"], ascending=False).head(top_n)
    if not top_videos.empty:
        top_ids = ", ".join(top_videos["video_id"].astype(str).tolist())
        insights.append(
            f"Chọn {len(top_videos)} video làm source để remix hook/caption: {top_ids}."
        )

    long_caption_er = df.loc[df["caption_length"] >= 220, "engagement_rate"].mean()
    short_caption_er = df.loc[df["caption_length"] < 220, "engagement_rate"].mean()
    if pd.notna(long_caption_er) and pd.notna(short_caption_er):
        if long_caption_er > short_caption_er:
            insights.append("Caption dài đang cho ER tốt hơn; có thể test format kể chuyện + CTA rõ ràng.")
        else:
            insights.append("Caption ngắn đang hiệu quả hơn; ưu tiên hook ngắn + CTA trong 1-2 câu.")

    return insights


def _create_charts(df: pd.DataFrame, out_dir: Path) -> Dict[str, str]:
    chart_paths: Dict[str, str] = {}

    if df.empty:
        return chart_paths

    plt.figure(figsize=(8, 5))
    plt.hist(df["views_count"].dropna(), bins=15)
    plt.title("Distribution of Views")
    plt.xlabel("Views")
    plt.ylabel("Videos")
    views_hist_path = out_dir / "views_distribution.png"
    plt.tight_layout()
    plt.savefig(views_hist_path, dpi=160)
    plt.close()
    chart_paths["views_distribution"] = str(views_hist_path).replace("\\", "/")

    plt.figure(figsize=(8, 5))
    plt.scatter(df["duration_seconds"], df["engagement_rate"], alpha=0.7)
    plt.title("Duration vs Engagement Rate")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Engagement Rate")
    duration_er_path = out_dir / "duration_vs_er.png"
    plt.tight_layout()
    plt.savefig(duration_er_path, dpi=160)
    plt.close()
    chart_paths["duration_vs_engagement_rate"] = str(duration_er_path).replace("\\", "/")

    hashtag_avg = (
        df.groupby("hashtag_count", dropna=False)
        .agg(avg_views=("views_count", "mean"), videos=("video_id", "count"))
        .reset_index()
        .sort_values("hashtag_count")
    )
    plt.figure(figsize=(8, 5))
    plt.plot(hashtag_avg["hashtag_count"], hashtag_avg["avg_views"], marker="o")
    plt.title("Hashtag Count vs Average Views")
    plt.xlabel("Hashtag Count")
    plt.ylabel("Average Views")
    hashtag_line_path = out_dir / "hashtag_count_vs_avg_views.png"
    plt.tight_layout()
    plt.savefig(hashtag_line_path, dpi=160)
    plt.close()
    chart_paths["hashtag_count_vs_avg_views"] = str(hashtag_line_path).replace("\\", "/")

    return chart_paths


def _write_markdown_report(
    out_dir: Path,
    run_id: str,
    topic: Optional[str],
    summary: Dict[str, float],
    top_videos: pd.DataFrame,
    insights: List[str],
    chart_paths: Dict[str, str],
) -> Path:
    report_path = out_dir / f"analysis_{run_id}.md"

    lines = [
        "# TikTok Video Batch Analysis",
        "",
        f"- Run ID: `{run_id}`",
        f"- Topic: `{topic or 'N/A'}`",
        f"- Generated At (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## KPI Summary",
        f"- Videos analyzed: **{int(summary['videos_analyzed'])}**",
        f"- Avg views: **{summary['avg_views']:.2f}**",
        f"- Median views: **{summary['median_views']:.2f}**",
        f"- Avg engagement: **{summary['avg_engagement']:.2f}**",
        f"- Avg engagement rate: **{summary['avg_engagement_rate']:.4f}**",
        f"- Avg share rate: **{summary['avg_share_rate']:.4f}**",
        f"- Avg comment rate: **{summary['avg_comment_rate']:.4f}**",
        "",
        "## Actionable Insights",
    ]

    for idx, insight in enumerate(insights, start=1):
        lines.append(f"{idx}. {insight}")

    lines.extend([
        "",
        "## Top Videos (by engagement_rate then views)",
    ])

    if top_videos.empty:
        lines.append("No top videos available.")
    else:
        for _, row in top_videos.iterrows():
            lines.append(
                "- "
                f"`{row['video_id']}` | user: `{row['tiktok_username']}` | "
                f"views: {row['views_count']:.0f} | er: {row['engagement_rate']:.4f} | url: {row['video_url']}"
            )

    lines.extend([
        "",
        "## Charts",
    ])
    for chart_name, chart_path in chart_paths.items():
        lines.append(f"- {chart_name}: `{chart_path}`")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _write_pdf_report(
    out_dir: Path,
    run_id: str,
    topic: Optional[str],
    summary: Dict[str, float],
    insights: List[str],
    chart_paths: Dict[str, str],
) -> Optional[Path]:
    pdf_path = out_dir / f"analysis_{run_id}.pdf"

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")

        lines = [
            "TikTok Video Batch Analysis",
            "",
            f"Run ID: {run_id}",
            f"Topic: {topic or 'N/A'}",
            f"Generated At (UTC): {datetime.now(timezone.utc).isoformat()}",
            "",
            "KPI Summary:",
            f"- Videos analyzed: {int(summary['videos_analyzed'])}",
            f"- Avg views: {summary['avg_views']:.2f}",
            f"- Median views: {summary['median_views']:.2f}",
            f"- Avg engagement: {summary['avg_engagement']:.2f}",
            f"- Avg engagement rate: {summary['avg_engagement_rate']:.4f}",
            f"- Avg share rate: {summary['avg_share_rate']:.4f}",
            f"- Avg comment rate: {summary['avg_comment_rate']:.4f}",
            "",
            "Actionable Insights:",
        ]

        for idx, insight in enumerate(insights, start=1):
            wrapped = textwrap.wrap(f"{idx}. {insight}", width=95)
            lines.extend(wrapped)

        plt.text(0.03, 0.98, "\n".join(lines), va="top", fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

        for chart_path in chart_paths.values():
            fig = plt.figure(figsize=(11.69, 8.27))
            plt.axis("off")
            image = plt.imread(chart_path)
            plt.imshow(image)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    return pdf_path if pdf_path.exists() else None


def _safe_float(series: pd.Series, agg: str = "mean") -> float:
    if series is None or series.dropna().empty:
        return 0.0
    if agg == "median":
        return float(series.median())
    return float(series.mean())


def _to_public_url(local_path: str, request: Optional[Request] = None) -> str:
    normalized_path = str(local_path).replace("\\", "/")
    base_dir = str(VIDEO_ANALYSIS_DIR).replace("\\", "/")
    if not normalized_path.startswith(base_dir):
        return normalized_path

    relative_path = normalized_path[len(base_dir):].lstrip("/")

    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL}{PUBLIC_REPORTS_ROUTE}/{relative_path}"

    if request is not None:
        request_base = str(request.base_url).rstrip("/")
        return f"{request_base}{PUBLIC_REPORTS_ROUTE}/{relative_path}"

    return normalized_path


def _prepare_plot_dataframe(videos: List[PlotVideoItem]) -> pd.DataFrame:
    records = [video.model_dump() for video in videos]
    df = pd.DataFrame(records)

    if df.empty:
        return df

    for col in ["likes_count_video", "comments_count", "shares_count", "views_count"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["publish_datetime"] = pd.to_datetime(df["publish_datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["publish_datetime"]).copy()

    if df.empty:
        return df

    df["publish_datetime"] = df["publish_datetime"].dt.tz_convert("Asia/Ho_Chi_Minh")
    df["total_interactions"] = (
        df["likes_count_video"].fillna(0)
        + df["comments_count"].fillna(0)
        + df["shares_count"].fillna(0)
    )
    df["hour_of_day"] = df["publish_datetime"].dt.hour

    return df


def _plot_to_streaming_response() -> StreamingResponse:
    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=160)
    buffer.seek(0)
    plt.close()
    return StreamingResponse(buffer, media_type="image/png")


@app.post("/prepare_tiktok")
def prepare_tiktok():
    try:
        prepare_tiktok_data()
        return {
            "status": "success",
            "message": "Processed TikTok data",
            "rows": int(len(tt_all)),
        }
    except Exception as ex:
        return {"status": "error", "message": str(ex)}


@app.post("/prepare_facebook")
def prepare_facebook():
    try:
        prepare_facebook_data()
        return {
            "status": "success",
            "message": "Processed Facebook data",
            "rows": int(len(fb_all)),
        }
    except Exception as ex:
        return {"status": "error", "message": str(ex)}


@app.post("/upload_cleaned")
def upload_cleaned():
    global tt_cleaned, fb_cleaned

    if tt_cleaned is None and fb_cleaned is None:
        return {"status": "error", "message": "Please run /prepare_tiktok and /prepare_facebook first"}

    try:
        ensure_google_clients()

        failed_sheets = []

        if tt_cleaned is not None:
            for brand, df_brand in tt_cleaned.items():
                ws_name = TT_WORKSHEET_NAMES[brand]
                ok = upload_dataframe_safe(output_spreadsheet, ws_name, df_brand)
                if not ok:
                    failed_sheets.append(ws_name)

        if fb_cleaned is not None:
            for brand, df_brand in fb_cleaned.items():
                ws_name = FB_WORKSHEET_NAMES[brand]
                ok = upload_dataframe_safe(output_spreadsheet, ws_name, df_brand)
                if not ok:
                    failed_sheets.append(ws_name)

        return {
            "status": "success",
            "message": "Uploaded cleaned data",
            "failed_sheets": failed_sheets,
        }
    except Exception as ex:
        return {"status": "error", "message": str(ex)}


@app.post("/run_analytics")
def run_analytics():
    try:
        run_analytics_data()
        return {
            "status": "success",
            "message": "Analytics completed",
            "tables_created": 8,
        }
    except Exception as ex:
        return {"status": "error", "message": str(ex)}


@app.post("/export_excel")
def export_excel():
    try:
        file_path = export_excel_file()
        return {
            "status": "success",
            "file_path": str(file_path).replace("\\", "/"),
        }
    except Exception as ex:
        return {"status": "error", "message": str(ex)}


@app.post("/analyze_scraped_videos")
def analyze_scraped_videos(payload: VideoBatchAnalysisRequest, request: Request = None):
    try:
        if not payload.videos:
            return {
                "status": "error",
                "message": "videos is empty. Please send at least 1 normalized video item.",
            }

        df = _coerce_video_dataframe(payload.videos)
        df = df.drop_duplicates(subset=["video_id"], keep="first")

        if df.empty:
            return {
                "status": "error",
                "message": "No valid rows after normalization.",
            }

        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"_{uuid4().hex[:8]}"
        out_dir = VIDEO_ANALYSIS_DIR / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "videos_analyzed": float(len(df)),
            "avg_views": _safe_float(df["views_count"], "mean"),
            "median_views": _safe_float(df["views_count"], "median"),
            "avg_engagement": _safe_float(df["engagement"], "mean"),
            "avg_engagement_rate": _safe_float(df["engagement_rate"], "mean"),
            "avg_share_rate": _safe_float(df["share_rate"], "mean"),
            "avg_comment_rate": _safe_float(df["comment_rate"], "mean"),
        }

        top_videos = (
            df.sort_values(["engagement_rate", "views_count"], ascending=False)
            .head(payload.top_n)
            [["video_id", "tiktok_username", "video_url", "views_count", "engagement_rate", "duration_seconds", "hashtags"]]
            .reset_index(drop=True)
        )

        top_hashtags = (
            df["hashtags"]
            .fillna("")
            .str.lower()
            .str.findall(r"#\\w+")
            .explode()
            .dropna()
            .value_counts()
            .head(payload.top_n)
            .to_dict()
        )

        insights = _build_actionable_insights(df, payload.top_n)
        chart_paths = _create_charts(df, out_dir)

        markdown_path = _write_markdown_report(
            out_dir=out_dir,
            run_id=run_id,
            topic=payload.topic,
            summary=summary,
            top_videos=top_videos,
            insights=insights,
            chart_paths=chart_paths,
        )

        json_path = out_dir / f"analysis_{run_id}.json"
        json_payload = {
            "run_id": run_id,
            "topic": payload.topic,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "top_hashtags": top_hashtags,
            "top_videos": top_videos.to_dict(orient="records"),
            "insights": insights,
            "chart_paths": chart_paths,
            "artifacts": {
                "analysis_markdown": str(markdown_path).replace("\\", "/"),
            },
        }
        json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        pdf_path = None
        pdf_error = None
        if payload.export_pdf:
            try:
                pdf_path = _write_pdf_report(
                    out_dir=out_dir,
                    run_id=run_id,
                    topic=payload.topic,
                    summary=summary,
                    insights=insights,
                    chart_paths=chart_paths,
                )
            except Exception as pdf_ex:
                pdf_error = str(pdf_ex)

        result = {
            "status": "success",
            "message": "Video analysis completed",
            "run_id": run_id,
            "summary": summary,
            "top_hashtags": top_hashtags,
            "insights": insights,
            "files": {
                "json": str(json_path).replace("\\", "/"),
                "markdown": str(markdown_path).replace("\\", "/"),
                "charts": chart_paths,
            },
            "for_ai_agent": {
                "preferred_input": str(json_path).replace("\\", "/"),
                "alternate_input": str(markdown_path).replace("\\", "/"),
            },
        }

        if pdf_path is not None:
            result["files"]["pdf"] = str(pdf_path).replace("\\", "/")
        elif payload.export_pdf:
            result["pdf_notice"] = f"Không xuất được PDF, đã fallback sang JSON + Markdown. Lỗi: {pdf_error}"

        result["public_files"] = {
            "json": _to_public_url(result["files"]["json"], request),
            "markdown": _to_public_url(result["files"]["markdown"], request),
            "charts": {
                "views_distribution": _to_public_url(result["files"]["charts"]["views_distribution"], request),
                "duration_vs_engagement_rate": _to_public_url(result["files"]["charts"]["duration_vs_engagement_rate"], request),
                "hashtag_count_vs_avg_views": _to_public_url(result["files"]["charts"]["hashtag_count_vs_avg_views"], request),
            },
        }
        if "pdf" in result["files"]:
            result["public_files"]["pdf"] = _to_public_url(result["files"]["pdf"], request)

        result["for_ai_agent"]["preferred_input"] = result["public_files"]["json"]
        result["for_ai_agent"]["alternate_input"] = result["public_files"]["markdown"]

        return result

    except Exception as ex:
        return {"status": "error", "message": str(ex)}


@app.post("/plot/interactions-over-time")
def plot_interactions_over_time(payload: List[PlotVideoItem]):
    if not payload:
        return {"status": "error", "message": "videos is empty. Please send at least 1 item."}

    df = _prepare_plot_dataframe(payload)
    if df.empty:
        return {"status": "error", "message": "No valid rows after parsing publish_datetime."}

    data = df.sort_values("publish_datetime")

    plt.figure(figsize=(10, 5))
    plt.plot(data["publish_datetime"], data["total_interactions"], color="green")
    plt.title("Change in Total Interactions Over Time (Vietnam Time)")
    plt.xlabel("publish_datetime")
    plt.ylabel("total_interactions")
    plt.grid(True)

    return _plot_to_streaming_response()


@app.post("/plot/posting-hour-distribution")
def plot_posting_hour_distribution(payload: List[PlotVideoItem]):
    if not payload:
        return {"status": "error", "message": "videos is empty. Please send at least 1 item."}

    df = _prepare_plot_dataframe(payload)
    if df.empty:
        return {"status": "error", "message": "No valid rows after parsing publish_datetime."}

    plt.figure(figsize=(10, 5))
    plt.hist(df["hour_of_day"].dropna(), bins=24, range=(0, 24), color="teal")
    plt.title("Distribution of Posting Hours")
    plt.xlabel("hour_of_day")
    plt.ylabel("number_of_videos")
    plt.grid(True)

    return _plot_to_streaming_response()


@app.post("/plot/avg-interactions-by-hour")
def plot_avg_interactions_by_hour(payload: List[PlotVideoItem]):
    if not payload:
        return {"status": "error", "message": "videos is empty. Please send at least 1 item."}

    df = _prepare_plot_dataframe(payload)
    if df.empty:
        return {"status": "error", "message": "No valid rows after parsing publish_datetime."}

    grouped = (
        df.groupby("hour_of_day", dropna=False)
        .agg(avg_total_interactions=("total_interactions", "mean"))
        .reset_index()
        .sort_values("hour_of_day")
    )

    plt.figure(figsize=(10, 5))
    plt.plot(grouped["hour_of_day"], grouped["avg_total_interactions"], color="purple")
    plt.title("Average Total Interactions by Hour of Post (Vietnam Time)")
    plt.xlabel("hour_of_day")
    plt.ylabel("avg_total_interactions")
    plt.grid(True)

    return _plot_to_streaming_response()