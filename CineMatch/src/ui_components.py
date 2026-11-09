"""Reusable, product-facing Streamlit components for CineMatch.

The app deliberately keeps recommendation decisions in ``src.recommender``.  This
module only knows how to present a movie and wire its controls to callbacks supplied
by the page, which keeps the interface cohesive without coupling it to ranking logic.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import streamlit as st

from src.services.tmdb_client import TMDBClient


def apply_product_styles() -> None:
    """Install the CineMatch visual system once per Streamlit rerun."""
    st.markdown(
        """
        <style>
        :root {
          --cm-ink: #f8f7ff;
          --cm-muted: #b8b7c9;
          --cm-dim: #807f97;
          --cm-bg: #090912;
          --cm-surface: rgba(24, 23, 40, .78);
          --cm-surface-strong: #1d1b31;
          --cm-line: rgba(255, 255, 255, .10);
          --cm-accent: #ff5f78;
          --cm-accent-bright: #ff7990;
          --cm-violet: #a78bfa;
          --cm-shadow: 0 20px 55px rgba(0, 0, 0, .28);
        }
        html { scroll-behavior: smooth; }
        [data-testid="stMain"] { overflow-anchor: none; scroll-behavior: auto; }
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
          color: var(--cm-ink);
          background:
            radial-gradient(circle at 8% -8%, rgba(167, 139, 250, .20), transparent 28rem),
            radial-gradient(circle at 95% 10%, rgba(255, 95, 120, .16), transparent 27rem),
            linear-gradient(145deg, #090912 0%, #0f0d1b 48%, #11101e 100%);
          max-width: 100%;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { opacity: .68; }
        [data-testid="stDialog"] {
          backdrop-filter: blur(8px);
          -webkit-backdrop-filter: blur(8px);
          background: rgba(5, 5, 12, .48);
        }
        [data-testid="stDialog"] [role="dialog"] {
          box-sizing: border-box;
          width: 60vw;
          max-width: min(1000px, calc(100vw - 2rem));
          max-height: calc(100vh - 2rem);
          overflow-y: auto;
        }
        /* One centred, modal loading layer.  The native Streamlit spinner is
           intentionally hidden so it cannot render a second circle underneath
           the CineMatch indicator. */
        [data-testid="stSpinner"] {
          position: fixed !important;
          inset: 0 !important;
          z-index: 2147483000 !important;
          display: grid !important;
          grid-template-rows: 78px auto;
          align-content: center;
          justify-content: center;
          gap: 1.15rem;
          width: 100vw !important;
          height: 100dvh !important;
          margin: 0 !important;
          padding: 0 !important;
          background: rgba(7, 6, 16, .68) !important;
          backdrop-filter: blur(14px) saturate(120%);
          -webkit-backdrop-filter: blur(14px) saturate(120%);
          isolation: isolate;
        }
        [data-testid="stSpinner"] > * { display: none !important; }
        [data-testid="stSpinner"]::before {
          box-sizing: border-box;
          grid-row: 1;
          justify-self: center;
          width: 76px;
          height: 76px;
          border: 3px solid rgba(248,247,255,.13);
          border-top-color: #ff6b86;
          border-right-color: #b6a1ff;
          border-radius: 50%;
          box-shadow:
            0 0 0 12px rgba(18,16,34,.50),
            0 18px 50px rgba(0,0,0,.38),
            0 0 28px rgba(167,139,250,.32);
          content: "";
          animation: cm-loader-spin 1s linear infinite;
        }
        [data-testid="stSpinner"]::after {
          grid-row: 2;
          max-width: min(330px, calc(100vw - 3rem));
          padding: .72rem 1rem;
          border: 1px solid rgba(255,255,255,.13);
          border-radius: 999px;
          color: #f8f7ff;
          background: rgba(28,25,48,.78);
          box-shadow: 0 16px 42px rgba(0,0,0,.25);
          content: "CineMatch is preparing your experience";
          font-size: .88rem;
          font-weight: 700;
          letter-spacing: .01em;
          line-height: 1.35;
          text-align: center;
        }
        @keyframes cm-loader-spin { to { transform: rotate(360deg); } }
        @media (prefers-reduced-motion: reduce) {
          [data-testid="stSpinner"]::before { animation-duration: 2.6s; }
        }
        .block-container {
          box-sizing: border-box;
          width: 100%;
          min-width: 0;
          max-width: 1680px;
          padding-right: clamp(1rem, 3vw, 3rem);
          padding-left: clamp(1rem, 3vw, 3rem);
          padding-top: 1.1rem;
          padding-bottom: 4rem;
        }
        [data-testid="stHorizontalBlock"], [data-testid="stColumn"],
        [data-testid="stColumn"] > div, [data-testid="stVerticalBlock"] {
          min-width: 0 !important;
          max-width: 100%;
        }
        #MainMenu, footer { visibility: hidden; }
        h1, h2, h3, p { color: var(--cm-ink); }
        h2, h3 { letter-spacing: -.025em; }
        [data-testid="stCaptionContainer"] { color: var(--cm-muted); }
        [data-testid="stDivider"] { border-color: var(--cm-line); margin: 2.2rem 0 1.8rem; }
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stMultiSelect"] > div > div,
        [data-testid="stRadio"] label {
          color: var(--cm-ink);
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
          box-sizing: border-box;
          width: 100%;
          min-width: 0;
          min-height: 48px;
          border: 1px solid rgba(255, 255, 255, .14);
          border-radius: 14px;
          background: rgba(13, 12, 25, .78);
          box-shadow: none;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] {
          width: 100%;
          min-width: 0;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
          border-color: rgba(255, 121, 144, .78);
          box-shadow: 0 0 0 3px rgba(255, 95, 120, .16);
        }
        [data-testid="stRadio"] > div { gap: .55rem; }
        [data-testid="stRadio"] label {
          border: 1px solid rgba(255,255,255,.10);
          border-radius: 999px;
          padding: .34rem .7rem;
          background: rgba(255,255,255,.035);
          transition: border-color .18s ease, background .18s ease;
        }
        [data-testid="stRadio"] label:has(input:checked) {
          border-color: rgba(255, 121, 144, .7);
          background: rgba(255, 95, 120, .13);
        }
        .stButton, .stButton > button, .stDownloadButton, .stDownloadButton > button {
          box-sizing: border-box;
          min-width: 0 !important;
          max-width: 100% !important;
        }
        .stButton > button, .stDownloadButton > button {
          box-sizing: border-box;
          width: 100%;
          min-height: 42px;
          border: 1px solid rgba(255, 255, 255, .14);
          border-radius: 12px;
          color: var(--cm-ink);
          background: rgba(255, 255, 255, .055);
          font-weight: 650;
          white-space: normal;
          overflow-wrap: break-word;
          line-height: 1.25;
          transition: transform .18s ease, border-color .18s ease, background .18s ease, box-shadow .18s ease;
        }
        .stButton > button:hover:not(:disabled), .stDownloadButton > button:hover:not(:disabled) {
          transform: translateY(-1px);
          border-color: rgba(255, 255, 255, .28);
          background: rgba(255, 255, 255, .10);
          box-shadow: 0 9px 20px rgba(0, 0, 0, .18);
        }
        .stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
          outline: 3px solid rgba(167, 139, 250, .55);
          outline-offset: 2px;
        }
        .stButton > button[kind="primary"] {
          border-color: transparent;
          color: #fff;
          background: linear-gradient(110deg, var(--cm-accent), #d75dce 58%, var(--cm-violet));
          box-shadow: 0 12px 28px rgba(255, 95, 120, .20);
        }
        .stButton > button[kind="primary"]:hover:not(:disabled) {
          background: linear-gradient(110deg, var(--cm-accent-bright), #e76bdb 58%, #b29aff);
          box-shadow: 0 15px 34px rgba(255, 95, 120, .30);
        }
        .stButton > button:disabled { opacity: .48; cursor: not-allowed; }
        .st-key-recently-explored [data-testid="stHorizontalBlock"] {
          align-items: flex-start;
          flex-wrap: wrap;
          gap: .55rem;
        }
        .st-key-recently-explored [data-testid="stColumn"] {
          flex: 0 1 auto !important;
          width: auto !important;
        }
        .st-key-recently-explored [data-testid="stButton"] {
          width: fit-content;
          max-width: min(280px, 100%);
        }
        .st-key-recently-explored .stButton > button {
          width: fit-content !important;
          max-width: min(280px, 100%);
          min-height: 36px;
          padding: 0 .95rem;
          border-color: rgba(167,139,250,.28);
          border-radius: 999px;
          color: #e7e0ff;
          background: linear-gradient(110deg, rgba(167,139,250,.12), rgba(255,95,120,.08));
          font-size: .82rem;
          overflow: hidden;
        }
        .st-key-recently-explored .stButton > button p {
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .st-key-recently-explored .stButton > button:hover:not(:disabled) {
          border-color: rgba(190,171,255,.62);
          background: linear-gradient(110deg, rgba(167,139,250,.22), rgba(255,95,120,.14));
        }
        .cm-nav {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          justify-content: space-between;
          gap: 1rem;
          padding: .35rem 0 1.15rem;
        }
        .cm-wordmark {
          display: inline-flex;
          flex: 0 0 auto;
          align-items: center;
          gap: .58rem;
          color: #f8f7ff;
          font-size: 1rem;
          font-weight: 850;
          letter-spacing: .12em;
          text-decoration: none !important;
          white-space: nowrap;
        }
        .cm-wordmark::before {
          display: grid;
          width: 2rem;
          height: 2rem;
          place-items: center;
          border: 1px solid rgba(255,255,255,.2);
          border-radius: 9px;
          color: #fff;
          background: linear-gradient(145deg, var(--cm-accent), #9b71e8);
          box-shadow: 0 5px 16px rgba(255,95,120,.2);
          content: "C";
          font-size: .95rem;
          letter-spacing: 0;
        }
        .cm-wordmark span { color: #ff9aac; }
        .cm-nav-links {
          display: flex;
          flex: 0 1 auto;
          flex-wrap: wrap;
          justify-content: flex-end;
          gap: .55rem 1rem;
        }
        .cm-nav-links a {
          color: var(--cm-muted);
          font-size: .82rem;
          font-weight: 650;
          text-decoration: none !important;
          white-space: nowrap;
        }
        .cm-nav-links a:hover { color: #fff; }
        .cm-hero {
          position: relative;
          overflow: hidden;
          min-height: 354px;
          margin: .1rem 0 2.1rem;
          padding: clamp(1.7rem, 5vw, 4rem);
          border: 1px solid rgba(255,255,255,.12);
          border-radius: 28px;
          background:
            linear-gradient(103deg, rgba(15, 13, 30, .98) 0%, rgba(15,13,30,.91) 50%, rgba(32,17,40,.68) 100%),
            radial-gradient(circle at 82% 26%, rgba(255, 95, 120, .48), transparent 12rem),
            radial-gradient(circle at 75% 74%, rgba(167, 139, 250, .38), transparent 16rem);
          box-shadow: 0 27px 70px rgba(0,0,0,.27);
          isolation: isolate;
        }
        .cm-hero::after {
          position: absolute;
          z-index: -1;
          inset: auto -5rem -6rem auto;
          width: 26rem;
          height: 26rem;
          border: 1px solid rgba(255,255,255,.14);
          border-radius: 48% 52% 41% 59%;
          content: "";
          transform: rotate(-18deg);
          box-shadow: inset 0 0 0 34px rgba(255,255,255,.025), inset 0 0 0 70px rgba(255,255,255,.018);
        }
        .cm-kicker {
          display: inline-flex;
          align-items: center;
          gap: .45rem;
          margin-bottom: 1rem;
          color: #ffd6de;
          font-size: .72rem;
          font-weight: 800;
          letter-spacing: .13em;
          text-transform: uppercase;
        }
        .cm-kicker::before { width: .5rem; height: .5rem; border-radius: 99px; background: var(--cm-accent); content: ""; box-shadow: 0 0 15px var(--cm-accent); }
        .cm-hero h1 {
          max-width: 680px;
          margin: 0;
          font-size: clamp(2.45rem, 5.5vw, 5.1rem);
          line-height: .97;
          letter-spacing: -.07em;
        }
        .cm-hero h1 em { color: #f9aec0; font-family: Georgia, serif; font-weight: 500; }
        .cm-hero-copy { max-width: 565px; margin: 1.25rem 0 1.45rem; color: #d1cfde; font-size: clamp(1rem, 1.7vw, 1.13rem); line-height: 1.62; }
        .cm-hero-actions { display: flex; flex-wrap: wrap; gap: .7rem; }
        .cm-hero-actions a {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          min-height: 43px;
          padding: 0 1rem;
          border: 1px solid rgba(255,255,255,.14);
          border-radius: 12px;
          color: #fff;
          font-size: .88rem;
          font-weight: 750;
          text-decoration: none;
          background: rgba(255,255,255,.055);
          transition: transform .18s ease, background .18s ease;
        }
        .cm-hero-actions a:first-child { border-color: transparent; background: linear-gradient(105deg, var(--cm-accent), #d75dce); }
        .cm-hero-actions a:hover { transform: translateY(-1px); background: rgba(255,255,255,.12); }
        .cm-hero-actions a:first-child:hover { background: linear-gradient(105deg, var(--cm-accent-bright), #e76bdb); }
        .cm-hero-trust { display: flex; flex-wrap: wrap; gap: .8rem 1.2rem; margin-top: 1.65rem; color: #c5c1d4; font-size: .78rem; }
        .cm-hero-trust span::before { color: #a7f3d0; content: "✓ "; }
        .cm-anchor { display: block; scroll-margin-top: 1rem; }
        .cm-section-intro { display: flex; align-items: end; justify-content: space-between; gap: 1rem; margin-bottom: .85rem; }
        .cm-section-intro h2 { margin: 0; font-size: clamp(1.38rem, 2vw, 1.85rem); }
        .cm-section-intro p { max-width: 680px; margin: .25rem 0 0; color: var(--cm-muted); font-size: .91rem; line-height: 1.5; }
        .cm-label { color: var(--cm-accent-bright); font-size: .7rem; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; }
        .cm-search-panel {
          margin-bottom: 2.15rem;
          padding: 1.1rem 1.2rem 1.2rem;
          border: 1px solid rgba(255,255,255,.105);
          border-radius: 20px;
          background: rgba(20, 18, 35, .72);
          box-shadow: var(--cm-shadow);
        }
        .cm-search-panel h2 { margin: .15rem 0 .25rem; font-size: 1.45rem; }
        .cm-search-panel p { margin: 0 0 .9rem; color: var(--cm-muted); font-size: .9rem; }
        .cm-now-playing {
          display: inline-flex;
          gap: .42rem;
          align-items: center;
          margin-top: .15rem;
          padding: .35rem .65rem;
          border: 1px solid rgba(167,139,250,.26);
          border-radius: 999px;
          color: #dcd1ff;
          font-size: .78rem;
          background: rgba(167,139,250,.10);
        }
        .cm-now-playing strong { color: #fff; }
        .cm-card-title {
          width: 100%;
          height: 1.3em;
          margin: .73rem 0 .18rem;
          overflow: hidden;
          color: #fff;
          font-size: 1.05rem;
          line-height: 1.3;
          letter-spacing: -.02em;
          white-space: nowrap;
        }
        .cm-card-title > span {
          display: inline-block;
          width: max-content;
          white-space: nowrap;
        }
        @keyframes cm-card-title-marquee {
          0%, 12% { transform: translateX(0); }
          88%, 100% { transform: translateX(var(--cm-title-distance)); }
        }
        .cm-card-title--overflow > span {
          animation: cm-card-title-marquee var(--cm-title-duration) linear infinite alternate;
          animation-delay: .6s;
          will-change: transform;
        }
        @media (prefers-reduced-motion: reduce) {
          .cm-card-title--overflow > span { animation: none; }
        }
        .cm-card-subtitle { min-height: 1.35rem; color: var(--cm-muted); font-size: .78rem; }
        .cm-poster, .cm-title-poster {
          display: block;
          width: 100%;
          aspect-ratio: 2 / 3;
          border-radius: 12px;
          object-fit: cover;
          background: linear-gradient(145deg, #2d2947, #151421);
        }
        .cm-poster { box-shadow: 0 10px 25px rgba(0,0,0,.22); }
        .cm-title-poster {
          display: grid;
          align-content: end;
          gap: .55rem;
          box-sizing: border-box;
          padding: 1.1rem;
          border: 1px solid rgba(255,255,255,.12);
          color: #fff;
          background:
            linear-gradient(155deg, hsl(var(--cm-cover-hue) 54% 34%), rgba(13, 12, 26, .94) 72%),
            radial-gradient(circle at 80% 12%, hsl(calc(var(--cm-cover-hue) + 35) 76% 64% / .55), transparent 34%);
          box-shadow: 0 10px 25px rgba(0,0,0,.22);
          overflow: hidden;
        }
        .cm-title-poster::before { width: 2.4rem; height: .22rem; border-radius: 99px; background: rgba(255,255,255,.74); content: ""; }
        .cm-title-poster__eyebrow { color: rgba(255,255,255,.72); font-size: .62rem; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; }
        .cm-title-poster__title { font-size: clamp(1.2rem, 2.2vw, 2rem); font-weight: 800; letter-spacing: -.04em; line-height: 1.02; overflow-wrap: anywhere; }
        .cm-title-poster__meta { color: rgba(255,255,255,.78); font-size: .72rem; line-height: 1.4; }
        .cm-card-meta { display: flex; flex-wrap: wrap; gap: .35rem; margin: .5rem 0 .62rem; }
        .cm-pill {
          display: inline-flex;
          align-items: center;
          min-height: 22px;
          padding: 0 .5rem;
          border: 1px solid rgba(255,255,255,.10);
          border-radius: 999px;
          color: #cfccda;
          font-size: .7rem;
          font-weight: 650;
          background: rgba(255,255,255,.045);
        }
        .cm-pill--rating { border-color: rgba(255, 209, 102, .26); color: #ffe39d; background: rgba(255,209,102,.08); }
        .cm-reasons { display: grid; gap: .28rem; min-height: 2.1rem; margin: 0 0 .75rem; color: #d9d1fd; font-size: .73rem; }
        .cm-reason { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .cm-reason::before { color: #bbabff; content: "✦ "; }
        .cm-detail-stage { margin-bottom: 1rem; }
        .cm-detail-stage .cm-poster, .cm-detail-stage .cm-title-poster { max-width: 280px; }
        .cm-detail-title { margin: .15rem 0 .45rem; font-size: clamp(1.75rem, 3vw, 2.5rem); line-height: 1.02; }
        .cm-detail-tagline { margin: .15rem 0 .8rem; color: #e5dffd; font-family: Georgia, serif; font-size: 1.05rem; font-style: italic; line-height: 1.45; }
        .cm-detail-copy { color: #cecbd8; line-height: 1.65; }
        .cm-detail-list { display: grid; gap: .42rem; margin: .9rem 0; color: #c9c6d4; font-size: .9rem; }
        .cm-detail-list strong { color: #fff; }
        .cm-detail-stats {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          margin: 1rem 0 1.25rem;
          border-top: 1px solid rgba(255,255,255,.12);
          border-bottom: 1px solid rgba(255,255,255,.12);
          background: linear-gradient(100deg, rgba(167,139,250,.07), transparent 70%);
        }
        .cm-detail-stat {
          position: relative;
          display: grid;
          align-content: center;
          gap: .22rem;
          min-height: 82px;
          padding: .7rem 1rem;
        }
        .cm-detail-stat + .cm-detail-stat::before {
          position: absolute;
          top: 20%;
          bottom: 20%;
          left: 0;
          width: 1px;
          background: rgba(255,255,255,.13);
          content: "";
        }
        .cm-detail-stat__label {
          color: var(--cm-muted);
          font-size: .68rem;
          font-weight: 750;
          letter-spacing: .11em;
          text-transform: uppercase;
        }
        .cm-detail-stat__value {
          color: #fff;
          font-size: clamp(1.35rem, 2.2vw, 1.8rem);
          font-weight: 750;
          letter-spacing: -.045em;
          line-height: 1.1;
          font-variant-numeric: tabular-nums;
        }
        .cm-empty {
          padding: 1.25rem;
          border: 1px dashed rgba(255,255,255,.16);
          border-radius: 16px;
          color: var(--cm-muted);
          text-align: center;
          background: rgba(255,255,255,.025);
        }
        .cm-footer { margin-top: 2.8rem; color: var(--cm-dim); font-size: .75rem; line-height: 1.55; }
        .cm-footer a { color: #cbbfff; text-decoration: none; }
        @media (max-width: 760px) {
          .block-container {
            padding: .7rem 1rem 3rem;
          }
          [data-testid="stDialog"] [role="dialog"] {
            width: calc(100vw - 1rem);
            max-width: calc(100vw - 1rem);
            max-height: calc(100vh - 1rem);
          }
          .cm-nav { align-items: flex-start; }
          .cm-nav-links { justify-content: flex-start; gap: .45rem .8rem; }
          .cm-hero { min-height: 0; border-radius: 22px; }
          .cm-section-intro { align-items: start; flex-direction: column; }
          .cm-detail-stage .cm-poster, .cm-detail-stage .cm-title-poster { max-width: none; }
          /* Streamlit columns retain a desktop minimum width by default. Stack
             them at this breakpoint so cards, filters, and button groups never
             create a page-wide horizontal scroll area. */
          [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
          [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex: 1 1 100% !important;
            width: 100% !important;
          }
          .st-key-recently-explored [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex: 0 1 auto !important;
            width: auto !important;
          }
        }
        @media (max-width: 500px) {
          .cm-detail-stats { grid-template-columns: 1fr; }
          .cm-detail-stat + .cm-detail-stat::before {
            top: 0;
            right: 1rem;
            bottom: auto;
            left: 1rem;
            width: auto;
            height: 1px;
          }
        }
        @media (prefers-reduced-motion: reduce) {
          *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; animation-duration: .01ms !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metadata_list(value: object) -> list[str]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if isinstance(item, str) and item.strip()]
    return []


def display_name(value: object) -> str:
    return str(value).replace("_", " ").strip().title()


def movie_year(movie: Mapping[str, Any]) -> str:
    value = movie.get("release_date", "")
    year = str(value)[:4] if value else ""
    return year if year.isdigit() else "—"


def short_text(value: object, limit: int) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}…"


def numeric_value(movie: Mapping[str, Any], key: str) -> float | None:
    value = movie.get(key)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def poster_url(
    movie: Mapping[str, Any], tmdb: TMDBClient, *, include_remote_fallback: bool = False
) -> str | None:
    path = movie.get("poster_path")
    if (not isinstance(path, str) or not path.strip()) and include_remote_fallback:
        details = tmdb.get_movie_details(int(movie["id"]))
        path = details.get("poster_path") if details else None
    return tmdb.poster_url(path)


def _poster_markup(movie: Mapping[str, Any], tmdb: TMDBClient, *, fallback: bool) -> str:
    title = html.escape(str(movie.get("title", "Movie")))
    url = poster_url(movie, tmdb, include_remote_fallback=fallback)
    if url:
        return (
            f'<img class="cm-poster" src="{html.escape(url, quote=True)}" '
            f'alt="Poster for {title}" loading="lazy">'
        )
    # The TMDB 5000 CSV does not include poster paths. When no TMDB key is
    # configured (or a lookup fails), keep the rail visually useful with a
    # deterministic title cover instead of an empty "poster unavailable" box.
    movie_id = int(movie.get("id", 0))
    hue = 230 + (movie_id * 37) % 105
    genres = metadata_list(movie.get("genres_list"))
    meta = " · ".join([movie_year(movie), *(display_name(genre) for genre in genres[:2])])
    return (
        f'<div class="cm-title-poster" style="--cm-cover-hue: {hue}" role="img" '
        f'aria-label="Title cover for {title}">'
        '<span class="cm-title-poster__eyebrow">CineMatch selection</span>'
        f'<span class="cm-title-poster__title">{title}</span>'
        f'<span class="cm-title-poster__meta">{html.escape(meta)}</span>'
        "</div>"
    )


def _pills(movie: Mapping[str, Any], *, match_score: float | None = None) -> str:
    pills: list[str] = []
    rating = numeric_value(movie, "vote_average")
    if rating and rating > 0:
        pills.append(f'<span class="cm-pill cm-pill--rating">★ {rating:.1f}</span>')
    if match_score is not None:
        score = max(0, min(100, round(match_score * 100)))
        pills.append(f'<span class="cm-pill">{score}% match signal</span>')
    for genre in metadata_list(movie.get("genres_list"))[:2]:
        pills.append(f'<span class="cm-pill">{html.escape(display_name(genre))}</span>')
    return "".join(pills)


def render_movie_card(
    movie: Mapping[str, Any],
    tmdb: TMDBClient,
    *,
    card_key: str,
    on_seed: Callable[[int], None],
    on_details: Callable[[int, str], None],
    on_save: Callable[[int], None],
    saved: bool,
    origin: str,
    on_like: Callable[[int], None] | None = None,
    liked: bool = False,
    match_reasons: Iterable[str] = (),
    primary_label: str = "Find similar",
    show_match_score: bool = False,
    primary_key: str | None = None,
    details_key: str | None = None,
) -> None:
    """Render a consistent movie card with all product interactions available."""
    movie_id = int(movie["id"])
    title = html.escape(str(movie.get("title", "Untitled movie")))
    year = movie_year(movie)
    match_score = numeric_value(movie, "similarity") if show_match_score else None
    reasons = [str(reason) for reason in match_reasons if str(reason).strip()][:2]

    with st.container(border=True):
        st.markdown(_poster_markup(movie, tmdb, fallback=True), unsafe_allow_html=True)
        st.markdown(
            f'<h3 class="cm-card-title" title="{title}"><span>{title}</span></h3>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="cm-card-subtitle">{html.escape(year)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="cm-card-meta">{_pills(movie, match_score=match_score)}</div>', unsafe_allow_html=True)
        if reasons:
            reason_markup = "".join(
                f'<div class="cm-reason">{html.escape(reason)}</div>' for reason in reasons
            )
            st.markdown(f'<div class="cm-reasons">{reason_markup}</div>', unsafe_allow_html=True)

        primary, detail = st.columns(2, gap="small")
        primary.button(
            primary_label,
            key=primary_key or f"{card_key}-similar",
            type="primary",
            use_container_width=True,
            on_click=on_seed,
            args=(movie_id,),
        )
        detail.button(
            "Details",
            key=details_key or f"{card_key}-details",
            use_container_width=True,
            on_click=on_details,
            args=(movie_id, origin),
        )
        if on_like is None:
            st.button(
                "Saved" if saved else "Save to list",
                key=f"{card_key}-save",
                use_container_width=True,
                on_click=on_save,
                args=(movie_id,),
            )
        else:
            save_column, like_column = st.columns(2, gap="small")
            save_column.button(
                "Saved" if saved else "Save",
                key=f"{card_key}-save",
                use_container_width=True,
                on_click=on_save,
                args=(movie_id,),
            )
            like_column.button(
                "In your taste" if liked else "Like",
                key=f"{card_key}-like",
                use_container_width=True,
                on_click=on_like,
                args=(movie_id,),
            )


def render_movie_details(
    movie: Mapping[str, Any],
    tmdb: TMDBClient,
    *,
    on_seed: Callable[[int], None],
    on_save: Callable[[int], None],
    is_saved: Callable[[int], bool],
) -> None:
    """Render an expanded movie profile with correct, non-duplicated metadata."""
    movie_id = int(movie["id"])
    title = html.escape(str(movie.get("title", "Untitled movie")))
    overview = movie.get("overview")
    if not isinstance(overview, str) or not overview.strip():
        overview = ""

    with st.container(border=True):
        poster_column, info_column = st.columns((1, 2.4), gap="large")
        with poster_column:
            st.markdown(
                f'<div class="cm-detail-stage">{_poster_markup(movie, tmdb, fallback=True)}</div>',
                unsafe_allow_html=True,
            )
        with info_column:
            st.markdown('<div class="cm-label">Movie profile</div>', unsafe_allow_html=True)
            st.markdown(
                f'<h2 class="cm-detail-title">{title} <span>({html.escape(movie_year(movie))})</span></h2>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="cm-card-meta">{_pills(movie)}</div>', unsafe_allow_html=True)

            metadata: list[tuple[str, str]] = []
            runtime = movie.get("runtime")
            if runtime and str(runtime).lower() != "nan":
                metadata.append(("Runtime", f"{runtime} min"))
            language = movie.get("original_language")
            if isinstance(language, str) and language.strip():
                metadata.append(("Language", language.upper()))
            directors = metadata_list(movie.get("director_names"))
            if directors:
                metadata.append(("Director", ", ".join(display_name(name) for name in directors[:2])))
            cast = metadata_list(movie.get("cast_names"))
            if cast:
                metadata.append(("Cast", ", ".join(display_name(name) for name in cast[:4])))
            if metadata:
                markup = "".join(
                    f'<div><strong>{html.escape(label)}:</strong> {html.escape(value)}</div>'
                    for label, value in metadata
                )
                st.markdown(f'<div class="cm-detail-list">{markup}</div>', unsafe_allow_html=True)
            if isinstance(overview, str) and overview.strip():
                st.markdown(
                    f'<p class="cm-detail-copy">{html.escape(overview.strip())}</p>',
                    unsafe_allow_html=True,
                )

            metric_items: list[tuple[str, str]] = []
            votes = numeric_value(movie, "vote_count")
            popularity = numeric_value(movie, "popularity")
            if votes is not None:
                metric_items.append(("Audience votes", f"{int(votes):,}"))
            if popularity is not None:
                metric_items.append(("Popularity", f"{popularity:.0f}"))
            if metric_items:
                stats = "".join(
                    '<div class="cm-detail-stat" role="listitem">'
                    f'<span class="cm-detail-stat__label">{html.escape(label)}</span>'
                    f'<span class="cm-detail-stat__value">{html.escape(value)}</span>'
                    "</div>"
                    for label, value in metric_items
                )
                st.markdown(
                    f'<div class="cm-detail-stats" role="list" aria-label="Audience and popularity">'
                    f"{stats}</div>",
                    unsafe_allow_html=True,
                )

            seed_column, save_column = st.columns(2, gap="medium")
            seed_column.button(
                "Find similar movies",
                key="details-find-similar",
                type="primary",
                use_container_width=True,
                on_click=on_seed,
                args=(movie_id,),
            )
            save_column.button(
                "Remove from list" if is_saved(movie_id) else "Save to list",
                key="details-save",
                use_container_width=True,
                on_click=on_save,
                args=(movie_id,),
            )
