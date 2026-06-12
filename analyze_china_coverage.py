import argparse
import json
import os
import re
import string
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.util import bigrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


DATA_DIR = "data"
SECURITIZATION_RESULT_DIR = "result_securitization"
RESULT_DIR = SECURITIZATION_RESULT_DIR
WEB_DATA_DIR = os.path.join(SECURITIZATION_RESULT_DIR, "web_data")
BALANCED_RESULT_DIR = "result_securitization_balanced"
BALANCED_WEB_DATA_DIR = os.path.join(BALANCED_RESULT_DIR, "web_data")
BALANCED_RANDOM_SEED = 42
MIN_TOKENS_WITHOUT_KEYWORDS = 20
MIN_REPRESENTATIVE_TOKENS_WITHOUT_KEYWORDS = 40
EXCLUDE_LOW_VALUE_TYPES = True
EXCLUDE_REVIEWS_AND_LIFESTYLE = True
EXCLUDE_LETTERS_AND_CORRECTIONS = True
DECADES = {
    "1990s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2025),
}
PHRASES = {
    "united states": "united_states",
    "south china sea": "south_china_sea",
    "human rights": "human_rights",
    "hong kong": "hong_kong",
    "xi jinping": "xi_jinping",
    "jiang zemin": "jiang_zemin",
    "hu jintao": "hu_jintao",
    "world trade organization": "world_trade_organization",
    "supply chain": "supply_chain",
    "supply chains": "supply_chains",
    "national security": "national_security",
    "intellectual property": "intellectual_property",
    "strategic competition": "strategic_competition",
    "trade war": "trade_war",
    "artificial intelligence": "artificial_intelligence",
    "export controls": "export_controls",
    "foreign investment": "foreign_investment",
    "economic security": "economic_security",
    "rare earths": "rare_earths",
    "forced labor": "forced_labor",
    "made in china": "made_in_china",
    "belt and road": "belt_and_road",
    "taiwan strait": "taiwan_strait",
    "communist party": "communist_party",
    "one child": "one_child_policy",
    "coronavirus": "covid",
}
ECONOMIC_TERMS = {
    "trade", "market", "markets", "investment", "investor", "investors",
    "export", "exports", "import", "imports", "manufacturing", "factory",
    "factories", "growth", "economy", "economic", "business", "company",
    "companies", "currency", "bank", "finance", "globalization",
    "globalisation", "wto", "world_trade_organization", "liberalization",
    "reform", "opening", "development", "supply_chain", "supply_chains",
    "foreign_investment", "consumer", "labor", "labour", "jobs", "industry",
    "industries",
}
SECURITY_TERMS = {
    "security", "national_security", "threat", "threats", "military",
    "defense", "defence", "army", "navy", "missile", "nuclear", "spy",
    "espionage", "surveillance", "cyber", "cybersecurity", "war", "conflict",
    "tensions", "rivalry", "rival", "competition", "strategic",
    "strategic_competition", "sanctions", "export_controls", "blacklist",
    "taiwan", "taiwan_strait", "south_china_sea", "intelligence", "risk",
    "risks", "vulnerability", "coercion",
}
TECHNOLOGY_TERMS = {
    "technology", "tech", "semiconductor", "semiconductors", "chip", "chips",
    "ai", "artificial_intelligence", "huawei", "tiktok", "internet", "data",
    "software", "hardware", "satellite", "telecom", "telecommunications",
    "5g", "quantum", "electric", "battery", "batteries", "rare_earths",
    "innovation", "export_controls",
}
ENGAGEMENT_TERMS = {
    "cooperation", "partner", "partnership", "engagement", "integration",
    "dialogue", "talks", "summit", "agreement", "agreements", "diplomacy",
    "diplomatic", "relations", "ties", "visit", "visits", "reform",
    "opening", "globalization", "wto", "world_trade_organization",
}
CONFLICT_TERMS = {
    "rival", "rivalry", "competition", "competitor", "threat",
    "confrontation", "tensions", "dispute", "disputes", "trade_war",
    "sanctions", "conflict", "war", "containment", "decoupling", "tariff",
    "tariffs", "retaliation", "crackdown", "pressure",
}
HUMAN_RIGHTS_TERMS = {
    "human_rights", "rights", "democracy", "democratic", "dissident",
    "dissidents", "censorship", "protest", "protests", "tiananmen",
    "xinjiang", "uighur", "uyghur", "tibet", "crackdown", "prison",
    "forced_labor", "religion", "freedom",
}
OPPORTUNITY_TERMS = {
    "opportunity", "opportunities", "growth", "growing", "market", "markets",
    "investment", "investors", "business", "cooperation", "partnership",
    "partner", "opening", "reform", "development", "prosperity", "integration",
    "globalization", "engagement", "agreement", "trade", "exports", "consumer",
    "innovation", "modernization", "benefit", "benefits", "potential",
}
THREAT_TERMS = {
    "threat", "threats", "risk", "risks", "danger", "dangerous", "security",
    "national_security", "military", "spy", "espionage", "surveillance",
    "coercion", "crackdown", "conflict", "war", "missile", "cyber",
    "cybersecurity", "sanctions", "blacklist", "repression", "censorship",
    "authoritarian", "aggression", "aggressive", "warning", "fear", "concern",
    "concerns",
}
COMPETITION_TERMS = {
    "competition", "competitor", "competitors", "rival", "rivalry",
    "strategic_competition", "race", "trade_war", "tariff", "tariffs",
    "decoupling", "containment", "dispute", "disputes", "confrontation",
    "pressure", "retaliation", "dominance", "challenge", "challenging",
}
COOPERATION_TERMS = {
    "cooperation", "cooperative", "partner", "partnership", "engagement",
    "dialogue", "talks", "summit", "agreement", "diplomacy", "diplomatic",
    "relations", "ties", "integration", "wto", "world_trade_organization",
    "reform", "opening",
}
INTERPRETIVE_STOPWORDS = {
    "china",
    "chinese",
    "chinas",
    "says",
    "said",
    "article",
    "letter",
    "editorial",
    "oped",
    "photo",
    "pres",
    "new",
    "york",
    "times",
    "would",
    "could",
    "may",
    "might",
    "one",
    "two",
    "many",
    "years",
    "even",
    "also",
    "still",
    "like",
    "take",
    "takes",
    "make",
    "made",
    "become",
    "back",
    "first",
    "last",
    "least",
    "time",
    "day",
    "days",
    "week",
    "month",
    "months",
    "year",
    "another",
    "among",
    "around",
    "much",
    "often",
    "well",
    "go",
    "wont",
    "say",
    "column",
    "nicholas",
    "countrys",
    "recent",
    "major",
    "biggest",
    "use",
    "three",
}
THEME_DICTIONARY = {
    "economy_trade": {
        "trade",
        "tariff",
        "tariffs",
        "market",
        "markets",
        "economy",
        "economic",
        "business",
        "companies",
        "company",
        "investment",
        "exports",
        "bank",
        "billion",
        "currency",
        "manufacturing",
    },
    "diplomacy_us": {
        "us",
        "united",
        "states",
        "american",
        "clinton",
        "bush",
        "obama",
        "trump",
        "biden",
        "diplomacy",
        "diplomatic",
        "relations",
        "ties",
        "visit",
        "summit",
    },
    "human_rights": {
        "human_rights",
        "rights",
        "dissidents",
        "dissident",
        "prison",
        "censorship",
        "democracy",
        "protest",
        "tiananmen",
        "xinjiang",
        "uighur",
        "uyghur",
        "tibet",
        "religion",
        "crackdown",
    },
    "military_security": {
        "military",
        "security",
        "army",
        "navy",
        "missile",
        "nuclear",
        "spy",
        "war",
        "defense",
        "south_china_sea",
        "cyber",
        "surveillance",
        "threat",
    },
    "technology_surveillance": {
        "technology",
        "tech",
        "internet",
        "surveillance",
        "censorship",
        "huawei",
        "tiktok",
        "semiconductor",
        "chip",
        "chips",
        "satellite",
        "ai",
        "data",
    },
    "public_health_covid": {
        "covid",
        "pandemic",
        "virus",
        "outbreak",
        "health",
        "wuhan",
        "quarantine",
        "vaccine",
        "disease",
    },
    "domestic_politics": {
        "party",
        "communist_party",
        "government",
        "leader",
        "leaders",
        "president",
        "xi_jinping",
        "jiang_zemin",
        "hu_jintao",
        "beijing",
        "political",
        "power",
    },
    "territory_identity": {
        "taiwan",
        "hong_kong",
        "tibet",
        "xinjiang",
        "mainland",
        "island",
        "sovereignty",
        "territory",
    },
}


def ensure_nltk_resources():
    """Download the NLTK stopword list if it is not already installed."""
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")


def fix_mojibake(text):
    """Repair common UTF-8 text that was accidentally decoded as Latin-1."""
    if not isinstance(text, str):
        return ""

    try:
        return text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text


def apply_phrase_normalization(text):
    """Join important historical phrases before punctuation is removed."""
    text = fix_mojibake(text).lower()

    for phrase, replacement in PHRASES.items():
        text = re.sub(rf"\b{re.escape(phrase)}\b", replacement, text)

    return text


def decade_for_year(year):
    """Return the decade label for a publication year."""
    for decade, (start, end) in DECADES.items():
        if start <= year <= end:
            return decade
    return None


def load_csv_files(input_files):
    """Load one or more CSV files and gracefully handle old or new schemas."""
    frames = []
    expected_columns = [
        "year",
        "date",
        "decade",
        "sample_window",
        "headline",
        "abstract",
        "lead_paragraph",
        "snippet",
        "section",
        "news_desk",
        "type_of_material",
        "document_type",
        "word_count",
        "keywords",
        "url",
    ]

    for path in input_files:
        df = pd.read_csv(path)

        if "year" not in df.columns:
            raise ValueError(f"{path} is missing required column: year")

        for column in expected_columns:
            if column not in df.columns:
                df[column] = ""

        frames.append(df[expected_columns])

    combined = pd.concat(frames, ignore_index=True)
    combined["year"] = pd.to_numeric(combined["year"], errors="coerce")
    combined = combined.dropna(subset=["year"])
    combined["year"] = combined["year"].astype(int)
    combined["decade"] = combined["year"].apply(decade_for_year)
    combined = combined.dropna(subset=["decade"])

    return combined


def clean_and_tokenize(text, stop_words):
    """Lowercase text, remove punctuation/non-letters, remove stopwords, and tokenize."""
    text = apply_phrase_normalization(text)
    punctuation = string.punctuation.replace("_", "")
    text = text.translate(str.maketrans("", "", punctuation))
    text = re.sub(r"[^a-z_\s]", " ", text)
    tokens = [
        token
        for token in text.split()
        if token not in stop_words and len(token) > 1
    ]
    return tokens


def prepare_documents(df):
    """Combine available NYT text fields into one text field and create cleaned tokens."""
    ensure_nltk_resources()
    stop_words = set(stopwords.words("english"))

    # Remove source terms that can appear in metadata-like text.
    # Keep "china" because the Word2Vec step studies its semantic neighbors.
    custom_stopwords = {
        "nyt",
        "new",
        "york",
        "times",
    }
    stop_words.update(custom_stopwords)

    df = df.copy()
    text_columns_without_keywords = ["headline", "abstract", "lead_paragraph", "snippet"]
    text_columns = text_columns_without_keywords + ["keywords"]

    for column in text_columns:
        if column not in df.columns:
            df[column] = ""

        df[column] = df[column].fillna("").map(fix_mojibake)

    # Main text includes NYT keyword metadata because those tags can reveal
    # editorial categorization. The no-keywords field lets you rerun or inspect
    # claims without metadata mixed into article prose.
    df["text_without_keywords"] = df[text_columns_without_keywords].agg(" ".join, axis=1)
    df["text"] = df[text_columns].agg(" ".join, axis=1)
    df["tokens_without_keywords"] = df["text_without_keywords"].apply(
        lambda text: clean_and_tokenize(text, stop_words)
    )
    df["tokens"] = df["text"].apply(lambda text: clean_and_tokenize(text, stop_words))
    df["clean_text"] = df["tokens"].apply(lambda tokens: " ".join(tokens))
    df["analysis_tokens"] = df["tokens"].apply(
        lambda tokens: [token for token in tokens if token not in INTERPRETIVE_STOPWORDS]
    )
    df["analysis_clean_text"] = df["analysis_tokens"].apply(lambda tokens: " ".join(tokens))

    return df


def count_plain_words(text):
    """Count plain words in an un-tokenized field for quality checks."""
    return len(re.findall(r"[A-Za-z_]+", fix_mojibake(text).lower()))


def contains_any(text, patterns):
    """Return True if a normalized text field contains any low-value pattern."""
    text = fix_mojibake(text).lower()
    return any(pattern in text for pattern in patterns)


def filter_suspicious_articles(df):
    """Flag and separate records likely to distort dictionary-based analysis.

    The filter exists because very short, empty, duplicate, or metadata-heavy
    rows can produce artificially large per-1,000-word scores. Suspicious rows
    are not deleted silently: they remain available in audit outputs.
    """
    filtered = df.copy()
    filtered["text_token_count_without_keywords"] = filtered["tokens_without_keywords"].apply(len)
    filtered["text_token_count_with_keywords"] = filtered["tokens"].apply(len)

    url_values = filtered["url"].fillna("").astype(str).str.strip()
    headline_values = filtered["headline"].fillna("").astype(str).str.strip()
    date_values = filtered["date"].fillna("").astype(str).str.strip()

    duplicate_url = url_values.ne("") & url_values.duplicated(keep="first")
    duplicate_headline_date = (
        headline_values.ne("")
        & date_values.ne("")
        & filtered.duplicated(subset=["headline", "date"], keep="first")
    )

    letter_correction_patterns = [
        "letter",
        "correction",
        "corrections",
        "paid death notice",
        "to the editor",
    ]
    low_value_patterns = [
        "photo",
        "slideshow",
        "video",
        "brief",
        "news summary",
        "digest",
        "business digest",
        "world briefing",
        "news in brief",
        "chronology",
        "index",
    ]
    review_lifestyle_patterns = [
        "obituary",
        "review",
        "book review",
        "theater review",
        "movie review",
        "travel",
        "style",
        "sports",
        "food",
        "real estate",
        "crossword",
        "quiz",
    ]
    headline_patterns = [
        "business digest",
        "world briefing",
        "news summary",
        "corrections",
        "letter",
        "to the editor",
        "arts briefing",
        "sports briefing",
        "in brief",
        "digest",
    ]

    quality_reasons = []
    quality_scores = []

    for index, row in filtered.iterrows():
        reasons = []
        metadata_text = " ".join(
            str(row.get(column, ""))
            for column in ["type_of_material", "document_type", "section", "headline"]
        ).lower()
        headline = str(row.get("headline", "") or "")

        if row["text_token_count_without_keywords"] < MIN_TOKENS_WITHOUT_KEYWORDS:
            reasons.append("very_short_text_without_keywords")

        description_token_counts = [
            count_plain_words(row.get("abstract", "")),
            count_plain_words(row.get("snippet", "")),
            count_plain_words(row.get("lead_paragraph", "")),
        ]
        if headline.strip() and sum(description_token_counts) < 12:
            reasons.append("empty_or_nearly_empty_description")

        if EXCLUDE_LETTERS_AND_CORRECTIONS and contains_any(metadata_text, letter_correction_patterns):
            reasons.append("letter_or_correction")

        if EXCLUDE_LOW_VALUE_TYPES and contains_any(metadata_text, low_value_patterns):
            reasons.append("low_value_format_or_digest")

        if EXCLUDE_REVIEWS_AND_LIFESTYLE and contains_any(metadata_text, review_lifestyle_patterns):
            reasons.append("review_lifestyle_or_non_news_section")

        if contains_any(headline, headline_patterns):
            reasons.append("low_value_headline_pattern")

        if bool(duplicate_url.loc[index]):
            reasons.append("duplicate_url")

        if bool(duplicate_headline_date.loc[index]):
            reasons.append("duplicate_headline_date")

        if not str(row.get("url", "") or "").strip():
            reasons.append("missing_url")

        quality_reasons.append("; ".join(dict.fromkeys(reasons)))
        quality_scores.append(max(0, 100 - 20 * len(set(reasons))))

    filtered["quality_reasons"] = quality_reasons
    filtered["quality_score"] = quality_scores
    filtered["quality_flag"] = filtered["quality_reasons"].apply(
        lambda reasons: "suspicious" if reasons else "keep"
    )

    return filtered


def quality_filter_summaries(quality_df):
    """Create transparent summary tables for the quality filter."""
    total_before = len(quality_df)
    kept = quality_df[quality_df["quality_flag"] == "keep"]
    removed = quality_df[quality_df["quality_flag"] == "suspicious"]
    total_after = len(kept)
    total_removed = len(removed)
    summary = pd.DataFrame(
        [
            {
                "total_articles_before_filtering": total_before,
                "total_articles_after_filtering": total_after,
                "total_removed": total_removed,
                "percent_removed": total_removed / total_before * 100 if total_before else 0,
            }
        ]
    )

    reason_counts = Counter()
    for reasons in removed["quality_reasons"].fillna(""):
        for reason in str(reasons).split("; "):
            if reason:
                reason_counts[reason] += 1

    removed_by_reason = pd.DataFrame(
        [
            {"quality_reason": reason, "removed_articles": count}
            for reason, count in reason_counts.most_common()
        ]
    )
    removed_by_decade = (
        removed.groupby("decade").size().reset_index(name="removed_articles")
        if not removed.empty
        else pd.DataFrame(columns=["decade", "removed_articles"])
    )
    kept_by_decade = (
        kept.groupby("decade").size().reset_index(name="kept_articles")
        if not kept.empty
        else pd.DataFrame(columns=["decade", "kept_articles"])
    )

    return summary, removed_by_reason, removed_by_decade, kept_by_decade


def create_balanced_decade_corpus(clean_df):
    """Randomly sample the same number of clean articles from each decade.

    This robustness corpus reduces decade-size imbalance. It should be read as
    a comparison against the full clean corpus, not as a replacement for it.
    """
    decade_counts = clean_df["decade"].value_counts()

    if decade_counts.empty:
        return clean_df.copy(), pd.DataFrame()

    target_n = int(decade_counts.min())
    sampled_groups = []

    for decade in DECADES:
        group = clean_df[clean_df["decade"] == decade]

        if group.empty:
            continue

        sampled = group.sample(n=target_n, random_state=BALANCED_RANDOM_SEED)
        sampled_groups.append(sampled)

    balanced_df = pd.concat(sampled_groups, ignore_index=True)
    balanced_df = balanced_df.sort_values(["decade", "year", "date", "headline"])
    balanced_df["balanced_sample"] = True
    balanced_df["balanced_sample_target_n_per_decade"] = target_n
    balanced_df["balanced_random_seed"] = BALANCED_RANDOM_SEED

    summary = pd.DataFrame(
        [
            {
                "decade": decade,
                "original_clean_articles": int(decade_counts.get(decade, 0)),
                "balanced_articles": int((balanced_df["decade"] == decade).sum()),
                "target_articles_per_decade": target_n,
                "random_seed": BALANCED_RANDOM_SEED,
            }
            for decade in DECADES
            if decade_counts.get(decade, 0) > 0
        ]
    )

    return balanced_df, summary


def save_balanced_corpus_outputs(clean_df):
    """Run key analyses on an equal-sized decade corpus in a separate folder."""
    os.makedirs(BALANCED_RESULT_DIR, exist_ok=True)
    os.makedirs(BALANCED_WEB_DATA_DIR, exist_ok=True)

    balanced_df, balance_summary = create_balanced_decade_corpus(clean_df)

    if balanced_df.empty:
        return balance_summary

    balanced_df.to_csv(os.path.join(BALANCED_RESULT_DIR, "balanced_clean_articles.csv"), index=False)
    balance_summary.to_csv(os.path.join(BALANCED_RESULT_DIR, "balanced_sampling_summary.csv"), index=False)

    summary = decade_summary(balanced_df)
    country_mentions = country_mention_stats(balanced_df)
    words = top_words_by_decade(balanced_df, top_n=40)
    word_pairs = top_bigrams_by_decade(balanced_df, top_n=20)
    keywords = tfidf_keywords_by_decade(balanced_df, top_n=30)
    scored = add_framing_scores(balanced_df)
    framing_year = aggregate_framing(scored, ["year"])
    framing_decade = aggregate_framing(scored, ["decade"])
    cooccurrence_year = economic_security_cooccurrence(scored, "year")
    cooccurrence_decade = economic_security_cooccurrence(scored, "decade")
    shift_decade = engagement_competition_shift(scored, "decade")
    technology_decade = technology_bridge(scored, "decade")
    close_reading = representative_articles(scored)

    summary.to_csv(os.path.join(BALANCED_RESULT_DIR, "decade_summary.csv"), index=False)
    country_mentions.to_csv(os.path.join(BALANCED_RESULT_DIR, "country_mention_stats.csv"), index=False)
    words.to_csv(os.path.join(BALANCED_RESULT_DIR, "top_40_words_by_decade.csv"), index=False)
    word_pairs.to_csv(os.path.join(BALANCED_RESULT_DIR, "top_20_bigrams_by_decade.csv"), index=False)
    keywords.to_csv(os.path.join(BALANCED_RESULT_DIR, "tfidf_keywords_by_decade.csv"), index=False)
    scored.to_csv(os.path.join(BALANCED_RESULT_DIR, "framing_scores_by_article.csv"), index=False)
    framing_year.to_csv(os.path.join(BALANCED_RESULT_DIR, "framing_scores_by_year.csv"), index=False)
    framing_decade.to_csv(os.path.join(BALANCED_RESULT_DIR, "framing_scores_by_decade.csv"), index=False)
    cooccurrence_year.to_csv(
        os.path.join(BALANCED_RESULT_DIR, "economic_security_cooccurrence_by_year.csv"),
        index=False,
    )
    cooccurrence_decade.to_csv(
        os.path.join(BALANCED_RESULT_DIR, "economic_security_cooccurrence_by_decade.csv"),
        index=False,
    )
    shift_decade.to_csv(
        os.path.join(BALANCED_RESULT_DIR, "engagement_competition_shift_by_decade.csv"),
        index=False,
    )
    technology_decade.to_csv(os.path.join(BALANCED_RESULT_DIR, "technology_bridge_by_decade.csv"), index=False)
    close_reading.to_csv(
        os.path.join(BALANCED_RESULT_DIR, "representative_articles_for_close_reading.csv"),
        index=False,
    )

    models = train_embedding_models(balanced_df)
    selected_terms = [
        "china",
        "trade",
        "market",
        "investment",
        "technology",
        "security",
        "taiwan",
        "chip",
        "semiconductor",
        "competition",
    ]
    embedding_neighbors_for_terms(models, selected_terms).to_csv(
        os.path.join(BALANCED_RESULT_DIR, "embedding_neighbors_selected_terms.csv"),
        index=False,
    )
    semantic_pairs = semantic_similarity_pairs(models)
    semantic_pairs.to_csv(
        os.path.join(BALANCED_RESULT_DIR, "semantic_similarity_pairs_by_decade.csv"),
        index=False,
    )

    save_frequency_plot(words, os.path.join(BALANCED_RESULT_DIR, "frequency_comparison.png"))
    save_keyword_plot(keywords, os.path.join(BALANCED_RESULT_DIR, "keyword_comparison.png"))
    save_economic_security_decade_bar(
        cooccurrence_decade,
        os.path.join(BALANCED_RESULT_DIR, "economic_security_cooccurrence_by_decade.png"),
    )
    save_decade_framing_bar(
        framing_decade,
        os.path.join(BALANCED_RESULT_DIR, "decade_positive_negative_framing.png"),
    )
    save_semantic_similarity_plot(
        semantic_pairs,
        os.path.join(BALANCED_RESULT_DIR, "semantic_similarity_pairs_by_decade.png"),
    )

    export_json(framing_decade, os.path.join(BALANCED_WEB_DATA_DIR, "framing_scores_by_decade.json"))
    export_json(
        cooccurrence_decade,
        os.path.join(BALANCED_WEB_DATA_DIR, "economic_security_cooccurrence_by_decade.json"),
    )
    export_json(semantic_pairs, os.path.join(BALANCED_WEB_DATA_DIR, "semantic_similarity_pairs_by_decade.json"))
    export_json(keywords, os.path.join(BALANCED_WEB_DATA_DIR, "top_keywords_by_decade.json"))
    export_json(balance_summary, os.path.join(BALANCED_WEB_DATA_DIR, "balanced_sampling_summary.json"))

    return balance_summary


def top_words_by_decade(df, top_n=30):
    """Calculate frequent interpretive words in each decade."""
    rows = []

    for decade, group in df.groupby("decade"):
        counts = Counter(token for tokens in group["analysis_tokens"] for token in tokens)
        total_words = sum(len(tokens) for tokens in group["analysis_tokens"])

        for rank, (word, count) in enumerate(counts.most_common(top_n), start=1):
            rows.append(
                {
                    "decade": decade,
                    "rank": rank,
                    "word": word,
                    "count": count,
                    "per_1000_words": (count / total_words) * 1000 if total_words else 0,
                }
            )

    return pd.DataFrame(rows)


def top_bigrams_by_decade(df, top_n=20):
    """Calculate frequent two-word interpretive phrases in each decade."""
    rows = []

    for decade, group in df.groupby("decade"):
        decade_bigrams = Counter()
        total_bigrams = 0

        for tokens in group["analysis_tokens"]:
            pairs = [" ".join(pair) for pair in bigrams(tokens)]
            decade_bigrams.update(pairs)
            total_bigrams += len(pairs)

        for rank, (bigram, count) in enumerate(decade_bigrams.most_common(top_n), start=1):
            rows.append(
                {
                    "decade": decade,
                    "rank": rank,
                    "bigram": bigram,
                    "count": count,
                    "per_1000_bigrams": (count / total_bigrams) * 1000 if total_bigrams else 0,
                }
            )

    return pd.DataFrame(rows)


def tfidf_keywords_by_decade(df, top_n=30):
    """Find terms that distinguish each decade using TF-IDF over decade-level corpora."""
    decade_docs = (
        df.groupby("decade")["analysis_clean_text"]
        .apply(lambda texts: " ".join(texts))
        .reindex(DECADES.keys())
        .dropna()
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf = vectorizer.fit_transform(decade_docs.values)
    terms = np.array(vectorizer.get_feature_names_out())
    rows = []

    for row_index, decade in enumerate(decade_docs.index):
        scores = tfidf[row_index].toarray().ravel()
        top_indices = scores.argsort()[::-1][:top_n]

        for rank, term_index in enumerate(top_indices, start=1):
            rows.append(
                {
                    "decade": decade,
                    "rank": rank,
                    "keyword": terms[term_index],
                    "tfidf": scores[term_index],
                }
            )

    return pd.DataFrame(rows)


def decade_summary(df):
    """Summarize corpus size so comparisons can be normalized by decade."""
    rows = []

    for decade, group in df.groupby("decade"):
        rows.append(
            {
                "decade": decade,
                "articles": len(group),
                "raw_words": sum(len(tokens) for tokens in group["tokens"]),
                "interpretive_words": sum(len(tokens) for tokens in group["analysis_tokens"]),
            }
        )

    return pd.DataFrame(rows)


def country_mention_stats(df):
    """Count generic China mentions separately from interpretive topic words."""
    country_terms = {"china", "chinese", "chinas"}
    rows = []

    for decade, group in df.groupby("decade"):
        article_count = len(group)
        total_words = sum(len(tokens) for tokens in group["tokens"])
        counts = Counter(token for tokens in group["tokens"] for token in tokens)

        for term in ["china", "chinese", "chinas"]:
            count = counts.get(term, 0)
            rows.append(
                {
                    "decade": decade,
                    "term": term,
                    "count": count,
                    "per_article": count / article_count if article_count else 0,
                    "per_1000_words": (count / total_words) * 1000 if total_words else 0,
                }
            )

        combined_count = sum(counts.get(term, 0) for term in country_terms)
        rows.append(
            {
                "decade": decade,
                "term": "china_chinese_chinas_combined",
                "count": combined_count,
                "per_article": combined_count / article_count if article_count else 0,
                "per_1000_words": (combined_count / total_words) * 1000 if total_words else 0,
            }
        )

    return pd.DataFrame(rows)


def section_distribution(df):
    """Show which NYT sections contributed to each decade's China coverage."""
    rows = []

    for decade, group in df.groupby("decade"):
        article_count = len(group)
        counts = group["section"].fillna("Unknown").replace("", "Unknown").value_counts()

        for section, count in counts.items():
            rows.append(
                {
                    "decade": decade,
                    "section": section,
                    "articles": count,
                    "percent_articles": (count / article_count) * 100 if article_count else 0,
                }
            )

    return pd.DataFrame(rows)


def theme_framing_by_decade(df):
    """Measure broad coverage frames by decade using a transparent dictionary."""
    rows = []

    for decade, group in df.groupby("decade"):
        article_count = len(group)
        total_words = sum(len(tokens) for tokens in group["analysis_tokens"])

        for theme, terms in THEME_DICTIONARY.items():
            article_hits = 0
            mention_count = 0

            for tokens in group["analysis_tokens"]:
                token_counts = Counter(tokens)
                theme_mentions = sum(token_counts.get(term, 0) for term in terms)
                mention_count += theme_mentions

                if theme_mentions > 0:
                    article_hits += 1

            rows.append(
                {
                    "decade": decade,
                    "theme": theme,
                    "articles_with_theme": article_hits,
                    "percent_articles": (article_hits / article_count) * 100 if article_count else 0,
                    "term_mentions": mention_count,
                    "term_mentions_per_1000_words": (
                        (mention_count / total_words) * 1000 if total_words else 0
                    ),
                }
            )

    return pd.DataFrame(rows)


def count_terms(tokens, terms):
    """Count dictionary terms in one token list."""
    counts = Counter(tokens)
    return sum(counts.get(term, 0) for term in terms)


def add_framing_scores(df):
    """Add dictionary-based framing scores to each article.

    These scores are transparent textual indicators, not objective proof of
    sentiment or authorial intent. They should be used to locate patterns and
    then interpreted together with close reading of the articles.
    """
    scored = df.copy()
    token_lengths = scored["tokens"].apply(len).replace(0, 1)

    scored["opportunity_raw"] = scored["tokens"].apply(lambda tokens: count_terms(tokens, OPPORTUNITY_TERMS))
    scored["threat_raw"] = scored["tokens"].apply(lambda tokens: count_terms(tokens, THREAT_TERMS))
    scored["competition_raw"] = scored["tokens"].apply(lambda tokens: count_terms(tokens, COMPETITION_TERMS))
    scored["cooperation_raw"] = scored["tokens"].apply(lambda tokens: count_terms(tokens, COOPERATION_TERMS))
    scored["economic_raw"] = scored["tokens"].apply(lambda tokens: count_terms(tokens, ECONOMIC_TERMS))
    scored["security_raw"] = scored["tokens"].apply(lambda tokens: count_terms(tokens, SECURITY_TERMS))
    scored["technology_raw"] = scored["tokens"].apply(lambda tokens: count_terms(tokens, TECHNOLOGY_TERMS))
    scored["engagement_raw"] = scored["tokens"].apply(lambda tokens: count_terms(tokens, ENGAGEMENT_TERMS))
    scored["conflict_raw"] = scored["tokens"].apply(lambda tokens: count_terms(tokens, CONFLICT_TERMS))
    scored["human_rights_raw"] = scored["tokens"].apply(lambda tokens: count_terms(tokens, HUMAN_RIGHTS_TERMS))

    scored["opportunity_score"] = scored["opportunity_raw"] / token_lengths * 1000
    scored["threat_score"] = scored["threat_raw"] / token_lengths * 1000
    scored["competition_score"] = scored["competition_raw"] / token_lengths * 1000
    scored["cooperation_score"] = scored["cooperation_raw"] / token_lengths * 1000
    scored["economic_score"] = scored["economic_raw"] / token_lengths * 1000
    scored["security_score"] = scored["security_raw"] / token_lengths * 1000
    scored["technology_score"] = scored["technology_raw"] / token_lengths * 1000
    scored["conflict_score"] = scored["conflict_raw"] / token_lengths * 1000
    scored["human_rights_score"] = scored["human_rights_raw"] / token_lengths * 1000
    scored["positive_framing_score"] = scored["opportunity_score"] + scored["cooperation_score"]
    scored["negative_framing_score"] = scored["threat_score"] + scored["competition_score"]
    scored["net_framing_score"] = (
        scored["positive_framing_score"] - scored["negative_framing_score"]
    )
    scored["rivalry_score"] = scored["threat_score"] + scored["competition_score"]
    scored["engagement_score"] = scored["opportunity_score"] + scored["cooperation_score"]
    scored["rivalry_to_engagement_ratio"] = (
        scored["rivalry_score"] / (scored["engagement_score"] + 1)
    )
    scored["securitization_index"] = scored["security_raw"] / (scored["economic_raw"] + 1)
    scored["conflict_to_engagement_ratio"] = scored["conflict_raw"] / (scored["engagement_raw"] + 1)
    scored["has_economic"] = scored["economic_raw"] > 0
    scored["has_security"] = scored["security_raw"] > 0
    scored["has_technology"] = scored["technology_raw"] > 0
    scored["has_economic_security"] = scored["has_economic"] & scored["has_security"]
    scored["has_technology_economy"] = scored["has_technology"] & scored["has_economic"]
    scored["has_technology_security"] = scored["has_technology"] & scored["has_security"]
    scored["has_technology_economy_security"] = (
        scored["has_technology"] & scored["has_economic"] & scored["has_security"]
    )
    scored["token_count"] = scored["tokens"].apply(len)

    for column in ["section", "news_desk", "document_type", "type_of_material"]:
        if column not in scored.columns:
            scored[column] = "Unknown"

        scored[column] = scored[column].fillna("Unknown").replace("", "Unknown")

    return scored


def aggregate_framing(scored, group_columns):
    """Average article-level framing indicators for a year, decade, or metadata group."""
    score_columns = [
        "opportunity_score",
        "threat_score",
        "competition_score",
        "cooperation_score",
        "positive_framing_score",
        "negative_framing_score",
        "net_framing_score",
        "rivalry_score",
        "engagement_score",
        "rivalry_to_engagement_ratio",
        "economic_score",
        "security_score",
        "technology_score",
        "conflict_score",
        "human_rights_score",
        "securitization_index",
        "conflict_to_engagement_ratio",
    ]
    raw_columns = [
        "opportunity_raw",
        "threat_raw",
        "competition_raw",
        "cooperation_raw",
        "economic_raw",
        "security_raw",
        "technology_raw",
        "engagement_raw",
        "conflict_raw",
        "human_rights_raw",
    ]

    grouped = (
        scored.groupby(group_columns)
        .agg(
            articles=("url", "count"),
            total_tokens=("token_count", "sum"),
            **{column: (column, "mean") for column in score_columns},
            **{column: (column, "sum") for column in raw_columns},
        )
        .reset_index()
    )

    return grouped


def detect_framing_turning_points(yearly_scores):
    """Identify sharp year-to-year changes in transparent framing indicators."""
    data = yearly_scores.sort_values("year").copy()
    rows = []

    indicators = [
        ("threat_score", "threat framing rises sharply", "rise"),
        ("competition_score", "competition framing rises sharply", "rise"),
        ("opportunity_score", "opportunity framing drops sharply", "drop"),
    ]

    for column, description, direction in indicators:
        diff_column = f"{column}_change"
        data[diff_column] = data[column].diff()
        changes = data[diff_column].dropna()

        if changes.empty:
            continue

        threshold = changes.mean() + changes.std()

        if direction == "drop":
            threshold = changes.mean() - changes.std()
            matches = data[data[diff_column] <= threshold]
        else:
            matches = data[data[diff_column] >= threshold]

        for _, row in matches.iterrows():
            rows.append(
                {
                    "year": int(row["year"]),
                    "indicator": column,
                    "turning_point_type": description,
                    "change_from_previous_year": row[diff_column],
                    "score": row[column],
                }
            )

    data["net_change"] = data["net_framing_score"].diff()
    data["previous_net_change"] = data["net_change"].shift(1)
    direction_changes = data[
        (data["net_change"] * data["previous_net_change"] < 0)
        & data["net_change"].notna()
        & data["previous_net_change"].notna()
    ]

    for _, row in direction_changes.iterrows():
        rows.append(
            {
                "year": int(row["year"]),
                "indicator": "net_framing_score",
                "turning_point_type": "net framing score changes direction",
                "change_from_previous_year": row["net_change"],
                "score": row["net_framing_score"],
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "year",
                "indicator",
                "turning_point_type",
                "change_from_previous_year",
                "score",
            ]
        )

    return pd.DataFrame(rows).sort_values(["year", "indicator"])


def create_historical_events(output_path):
    """Create starter events for plot annotation, not as independent evidence."""
    events = pd.DataFrame(
        [
            {
                "year": 1997,
                "event": "Hong Kong handover",
                "category": "territory_identity",
                "note": "Annotation only; use article evidence for claims.",
            },
            {
                "year": 1999,
                "event": "U.S. bombing of Chinese embassy in Belgrade",
                "category": "diplomacy_security",
                "note": "Annotation only; use article evidence for claims.",
            },
            {
                "year": 2001,
                "event": "China joins WTO",
                "category": "economic_integration",
                "note": "Annotation only; use article evidence for claims.",
            },
            {
                "year": 2008,
                "event": "Global financial crisis",
                "category": "global_economy",
                "note": "Annotation only; use article evidence for claims.",
            },
            {
                "year": 2012,
                "event": "Xi Jinping becomes CCP leader",
                "category": "domestic_politics",
                "note": "Annotation only; use article evidence for claims.",
            },
            {
                "year": 2018,
                "event": "U.S.-China trade war begins",
                "category": "competition_trade",
                "note": "Annotation only; use article evidence for claims.",
            },
            {
                "year": 2020,
                "event": "COVID-19 pandemic",
                "category": "public_health",
                "note": "Annotation only; use article evidence for claims.",
            },
            {
                "year": 2022,
                "event": "CHIPS and Science Act",
                "category": "technology_security",
                "note": "Annotation only; use article evidence for claims.",
            },
            {
                "year": 2022,
                "event": "Major U.S. semiconductor export controls",
                "category": "technology_security",
                "note": "Annotation only; use article evidence for claims.",
            },
        ]
    )
    events.to_csv(output_path, index=False)
    return events


def economic_security_cooccurrence(scored, group_column):
    """Measure whether economic language increasingly appears with security language."""
    rows = []

    for group_value, group in scored.groupby(group_column):
        articles = len(group)
        economic_articles = int(group["has_economic"].sum())
        security_articles = int(group["has_security"].sum())
        overlap_articles = int(group["has_economic_security"].sum())
        economic_mentions = int(group["economic_raw"].sum())
        security_mentions = int(group["security_raw"].sum())

        rows.append(
            {
                group_column: group_value,
                "articles": articles,
                "percent_with_economic_terms": economic_articles / articles * 100 if articles else 0,
                "percent_with_security_terms": security_articles / articles * 100 if articles else 0,
                "percent_with_both_economic_and_security": (
                    overlap_articles / articles * 100 if articles else 0
                ),
                "among_economic_articles_percent_with_security": (
                    overlap_articles / economic_articles * 100 if economic_articles else 0
                ),
                "avg_economic_term_count_per_article": economic_mentions / articles if articles else 0,
                "avg_security_term_count_per_article": security_mentions / articles if articles else 0,
                "security_per_economic_ratio": security_mentions / (economic_mentions + 1),
                "securitization_index": security_mentions / (economic_mentions + 1),
                "economic_security_overlap_rate": (
                    overlap_articles / economic_articles if economic_articles else 0
                ),
                "conflict_vs_engagement_ratio": (
                    group["conflict_raw"].sum() / (group["engagement_raw"].sum() + 1)
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(group_column)


def engagement_competition_shift(scored, group_column):
    """Track movement from engagement/cooperation framing toward rivalry language."""
    rows = []

    for group_value, group in scored.groupby(group_column):
        articles = len(group)
        engagement_articles = int((group["engagement_raw"] > 0).sum())
        conflict_articles = int((group["conflict_raw"] > 0).sum())

        rows.append(
            {
                group_column: group_value,
                "articles": articles,
                "engagement_prevalence": engagement_articles / articles * 100 if articles else 0,
                "conflict_prevalence": conflict_articles / articles * 100 if articles else 0,
                "engagement_mentions": int(group["engagement_raw"].sum()),
                "conflict_mentions": int(group["conflict_raw"].sum()),
                "conflict_to_engagement_ratio": (
                    group["conflict_raw"].sum() / (group["engagement_raw"].sum() + 1)
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(group_column)


def technology_bridge(scored, group_column):
    """Measure whether technology connects economic and security discourse."""
    rows = []

    for group_value, group in scored.groupby(group_column):
        articles = len(group)
        technology_only = group[
            group["has_technology"] & ~group["has_economic"] & ~group["has_security"]
        ]
        technology_economy = group[group["has_technology_economy"]]
        technology_security = group[group["has_technology_security"]]
        technology_all = group[group["has_technology_economy_security"]]

        rows.append(
            {
                group_column: group_value,
                "articles": articles,
                "technology_only_articles": len(technology_only),
                "technology_economy_articles": len(technology_economy),
                "technology_security_articles": len(technology_security),
                "technology_economy_security_articles": len(technology_all),
                "percent_technology_only": len(technology_only) / articles * 100 if articles else 0,
                "percent_technology_economy": len(technology_economy) / articles * 100 if articles else 0,
                "percent_technology_security": len(technology_security) / articles * 100 if articles else 0,
                "percent_technology_economy_security": (
                    len(technology_all) / articles * 100 if articles else 0
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(group_column)


def metadata_distribution(df, group_column, metadata_column):
    """Summarize how NYT metadata categories change over time."""
    rows = []

    for group_value, group in df.groupby(group_column):
        total = len(group)
        values = group[metadata_column].fillna("Unknown").replace("", "Unknown").value_counts()

        for value, count in values.items():
            rows.append(
                {
                    group_column: group_value,
                    metadata_column: value,
                    "articles": int(count),
                    "percent_articles": count / total * 100 if total else 0,
                }
            )

    return pd.DataFrame(rows)


def top_nyt_keywords_by_decade(df, top_n=30):
    """Use NYT keyword metadata as a separate signal of editorial categorization."""
    rows = []

    for decade, group in df.groupby("decade"):
        counts = Counter()

        for value in group["keywords"].fillna(""):
            for keyword in re.split(r";|,", fix_mojibake(value).lower()):
                keyword = apply_phrase_normalization(keyword).strip()
                keyword = re.sub(r"\s+", " ", keyword)

                if keyword:
                    counts[keyword] += 1

        for rank, (keyword, count) in enumerate(counts.most_common(top_n), start=1):
            rows.append(
                {
                    "decade": decade,
                    "rank": rank,
                    "keyword": keyword,
                    "count": count,
                }
            )

    return pd.DataFrame(rows)


def embedding_neighbors_for_terms(models, terms, top_n=20):
    """Generate nearest embedding neighbors for historically important terms."""
    rows = []

    for decade, model in models.items():
        for term in terms:
            if term not in model["word_to_index"]:
                rows.append(
                    {
                        "decade": decade,
                        "term": term,
                        "rank": None,
                        "neighbor": "term_not_in_vocabulary",
                        "similarity": None,
                    }
                )
                continue

            term_index = model["word_to_index"][term]
            term_vector = model["vectors"][term_index]
            similarities = model["vectors"] @ term_vector
            similarities[term_index] = -np.inf

            for rank, neighbor_index in enumerate(similarities.argsort()[::-1][:top_n], start=1):
                rows.append(
                    {
                        "decade": decade,
                        "term": term,
                        "rank": rank,
                        "neighbor": model["vocab"][neighbor_index],
                        "similarity": similarities[neighbor_index],
                    }
                )

    return pd.DataFrame(rows)


def semantic_similarity_pairs(models):
    """Track whether economic words move closer to security words in embeddings."""
    pairs = [
        ("trade", "security"),
        ("trade", "competition"),
        ("market", "security"),
        ("investment", "security"),
        ("technology", "security"),
        ("semiconductor", "security"),
        ("chip", "security"),
        ("supply_chain", "security"),
        ("china", "security"),
        ("china", "trade"),
        ("china", "competition"),
    ]
    rows = []

    for decade, model in models.items():
        for term_a, term_b in pairs:
            if term_a not in model["word_to_index"] or term_b not in model["word_to_index"]:
                rows.append(
                    {
                        "decade": decade,
                        "term_a": term_a,
                        "term_b": term_b,
                        "similarity": None,
                        "note": "one_or_both_terms_not_in_vocabulary",
                    }
                )
                continue

            vector_a = model["vectors"][model["word_to_index"][term_a]]
            vector_b = model["vectors"][model["word_to_index"][term_b]]
            rows.append(
                {
                    "decade": decade,
                    "term_a": term_a,
                    "term_b": term_b,
                    "similarity": float(vector_a @ vector_b),
                    "note": "",
                }
            )

    return pd.DataFrame(rows)


def representative_articles(scored, top_n=10):
    """Select articles for close reading from each dictionary-frame profile."""
    scored = scored[
        (scored.get("quality_flag", "keep") == "keep")
        & (scored["text_token_count_without_keywords"] >= MIN_REPRESENTATIVE_TOKENS_WITHOUT_KEYWORDS)
        & (scored["headline"].fillna("").astype(str).str.strip() != "")
        & (
            (scored["abstract"].fillna("").astype(str).str.strip() != "")
            | (scored["snippet"].fillna("").astype(str).str.strip() != "")
            | (scored["lead_paragraph"].fillna("").astype(str).str.strip() != "")
        )
    ].copy()

    if scored.empty:
        return pd.DataFrame()

    base_columns = [
        "year",
        "date",
        "headline",
        "abstract",
        "keywords",
        "section",
        "news_desk",
        "url",
        "economic_score",
        "security_score",
        "technology_score",
        "opportunity_score",
        "threat_score",
        "competition_score",
        "cooperation_score",
        "net_framing_score",
        "securitization_index",
    ]
    selections = []
    profiles = [
        (
            "high_economic_low_security",
            scored.assign(selection_score=scored["economic_score"] - scored["security_score"]),
        ),
        (
            "high_security_low_economic",
            scored.assign(selection_score=scored["security_score"] - scored["economic_score"]),
        ),
        (
            "high_economy_security_overlap",
            scored.assign(selection_score=scored["economic_score"] + scored["security_score"]),
        ),
        (
            "high_technology_security_overlap",
            scored.assign(selection_score=scored["technology_score"] + scored["security_score"]),
        ),
    ]

    for profile, data in profiles:
        subset = data.sort_values("selection_score", ascending=False).head(top_n).copy()
        subset.insert(0, "profile", profile)
        selections.append(subset[["profile"] + base_columns])

    return pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()


def train_embedding_models(df, vector_size=100, window=5, max_vocab=2000):
    """Train one co-occurrence/SVD word embedding model per decade.

    This replaces gensim Word2Vec so the script does not require Microsoft C++
    Build Tools. The method builds a word co-occurrence matrix, converts it to
    positive PMI values, then reduces it with TruncatedSVD.
    """
    models = {}

    for decade, group in df.groupby("decade"):
        sentences = [tokens for tokens in group["tokens"] if tokens]
        word_counts = Counter(token for tokens in sentences for token in tokens)
        vocab = [word for word, _ in word_counts.most_common(max_vocab)]

        if len(vocab) < 2:
            continue

        word_to_index = {word: index for index, word in enumerate(vocab)}
        cooccurrence = np.zeros((len(vocab), len(vocab)), dtype=np.float64)

        # Count nearby words within a fixed context window.
        for tokens in sentences:
            indexed_tokens = [word_to_index[token] for token in tokens if token in word_to_index]

            for position, center_index in enumerate(indexed_tokens):
                start = max(0, position - window)
                end = min(len(indexed_tokens), position + window + 1)

                for context_position in range(start, end):
                    if context_position == position:
                        continue

                    context_index = indexed_tokens[context_position]
                    distance = abs(position - context_position)
                    cooccurrence[center_index, context_index] += 1.0 / distance

        if cooccurrence.sum() == 0:
            continue

        # Positive PMI emphasizes meaningful associations over raw frequency.
        total = cooccurrence.sum()
        row_sums = cooccurrence.sum(axis=1, keepdims=True)
        column_sums = cooccurrence.sum(axis=0, keepdims=True)
        expected = row_sums @ column_sums / total

        with np.errstate(divide="ignore", invalid="ignore"):
            ppmi = np.log((cooccurrence * total) / expected)

        ppmi[~np.isfinite(ppmi)] = 0
        ppmi = np.maximum(ppmi, 0)

        components = min(vector_size, ppmi.shape[0] - 1, ppmi.shape[1] - 1)

        if components < 2:
            continue

        svd = TruncatedSVD(n_components=components, random_state=42)
        vectors = normalize(svd.fit_transform(ppmi))

        models[decade] = {
            "vocab": vocab,
            "word_to_index": word_to_index,
            "vectors": vectors,
        }

    return models


def china_neighbors(models, top_n=20):
    """Return the nearest neighbors of the word 'china' for each decade model."""
    rows = []

    for decade, model in models.items():
        if "china" not in model["word_to_index"]:
            rows.append(
                {
                    "decade": decade,
                    "rank": None,
                    "neighbor": "china_not_in_vocabulary",
                    "similarity": None,
                }
            )
            continue

        china_index = model["word_to_index"]["china"]
        china_vector = model["vectors"][china_index]
        similarities = model["vectors"] @ china_vector
        similarities[china_index] = -np.inf
        top_indices = similarities.argsort()[::-1][:top_n]

        for rank, neighbor_index in enumerate(top_indices, start=1):
            rows.append(
                {
                    "decade": decade,
                    "rank": rank,
                    "neighbor": model["vocab"][neighbor_index],
                    "similarity": similarities[neighbor_index],
                }
            )

    return pd.DataFrame(rows)


def save_embedding_vectors(models):
    """Save each decade's learned word vectors as CSV files."""
    for decade, model in models.items():
        vectors = pd.DataFrame(model["vectors"])
        vectors.insert(0, "word", model["vocab"])
        vectors.to_csv(
            os.path.join(RESULT_DIR, f"embedding_vectors_{decade}.csv"),
            index=False,
        )


def save_frequency_plot(top_words, output_path):
    """Create a bar chart comparing high-frequency words across decades."""
    selected_words = (
        top_words.groupby("word")["per_1000_words"]
        .sum()
        .sort_values(ascending=False)
        .head(12)
        .index
    )
    plot_data = (
        top_words[top_words["word"].isin(selected_words)]
        .pivot_table(index="word", columns="decade", values="per_1000_words", fill_value=0)
        .reindex(selected_words)
    )

    fig, ax = plt.subplots(figsize=(13, 7))
    plot_data.plot(kind="bar", ax=ax)
    ax.set_title("Interpretive Word Frequency in NYT China Coverage by Decade")
    ax.set_xlabel("Word")
    ax.set_ylabel("Mentions per 1,000 interpretive words")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Decade")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_keyword_plot(tfidf_keywords, output_path):
    """Create a faceted bar chart of top TF-IDF keywords by decade."""
    decades = [decade for decade in DECADES if decade in tfidf_keywords["decade"].unique()]
    fig, axes = plt.subplots(len(decades), 1, figsize=(12, 3.2 * len(decades)))

    if len(decades) == 1:
        axes = [axes]

    for ax, decade in zip(axes, decades):
        data = tfidf_keywords[tfidf_keywords["decade"] == decade].head(10)
        data = data.sort_values("tfidf")
        ax.barh(data["keyword"], data["tfidf"], color="#3f7f99")
        ax.set_title(f"{decade} Distinctive TF-IDF Keywords")
        ax.set_xlabel("TF-IDF score")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_country_mentions_plot(country_mentions, output_path):
    """Plot normalized generic China mentions by decade."""
    data = country_mentions[
        country_mentions["term"] == "china_chinese_chinas_combined"
    ].set_index("decade")
    data = data.reindex(DECADES.keys())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(data.index, data["per_1000_words"], color="#5d7896")
    ax.set_title("Generic China Mentions by Decade")
    ax.set_xlabel("Decade")
    ax.set_ylabel("China/Chinese/China's mentions per 1,000 raw words")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_theme_framing_plot(theme_framing, output_path):
    """Create a heatmap showing dominant coverage frames by decade."""
    plot_data = theme_framing.pivot_table(
        index="theme",
        columns="decade",
        values="percent_articles",
        fill_value=0,
    )
    plot_data = plot_data.reindex(columns=[decade for decade in DECADES if decade in plot_data])

    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(plot_data.values, aspect="auto", cmap="YlGnBu")
    ax.set_title("Coverage Frames by Decade")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Theme")
    ax.set_xticks(np.arange(len(plot_data.columns)))
    ax.set_xticklabels(plot_data.columns)
    ax.set_yticks(np.arange(len(plot_data.index)))
    ax.set_yticklabels(plot_data.index)

    for row in range(plot_data.shape[0]):
        for column in range(plot_data.shape[1]):
            value = plot_data.iloc[row, column]
            ax.text(column, row, f"{value:.0f}%", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, label="Percent of articles with theme")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def word_cloud_positions(count):
    """Return deterministic positions for a readable matplotlib word cloud."""
    base_positions = [
        (0.50, 0.52), (0.30, 0.68), (0.70, 0.66), (0.28, 0.38), (0.72, 0.38),
        (0.50, 0.78), (0.50, 0.25), (0.15, 0.55), (0.85, 0.55), (0.18, 0.78),
        (0.82, 0.78), (0.18, 0.22), (0.82, 0.22), (0.39, 0.88), (0.62, 0.88),
        (0.38, 0.12), (0.62, 0.12), (0.08, 0.40), (0.92, 0.40), (0.50, 0.08),
        (0.12, 0.68), (0.88, 0.68), (0.12, 0.30), (0.88, 0.30), (0.35, 0.55),
        (0.65, 0.55), (0.35, 0.28), (0.65, 0.28), (0.35, 0.78), (0.65, 0.78),
        (0.25, 0.88), (0.75, 0.88), (0.25, 0.10), (0.75, 0.10), (0.08, 0.84),
        (0.92, 0.84), (0.08, 0.15), (0.92, 0.15), (0.50, 0.92), (0.50, 0.16),
    ]
    return base_positions[:count]


def save_word_cloud_plot(top_words, decade, output_path, top_n=40):
    """Create a word-cloud style plot for one decade using matplotlib only."""
    data = top_words[top_words["decade"] == decade].head(top_n).copy()

    if data.empty:
        return

    max_count = data["count"].max()
    min_count = data["count"].min()

    positions = word_cloud_positions(len(data))
    colors = ["#1f4e79", "#7f3f3f", "#3f6f4f", "#6f5a2f", "#5b4f8a"]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(f"{decade}: Top 40 Interpretive Words", fontsize=16, pad=16)

    for index, row in data.reset_index(drop=True).iterrows():
        if max_count == min_count:
            font_size = 28
        else:
            font_size = 9 + ((row["count"] - min_count) / (max_count - min_count)) * 30

        x_coord, y_coord = positions[index]
        label = row["word"].replace("_", " ")
        ax.text(
            x_coord,
            y_coord,
            label,
            fontsize=font_size,
            color=colors[index % len(colors)],
            ha="center",
            va="center",
            weight="bold" if index < 5 else "normal",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_decade_word_clouds(top_words):
    """Save one word-cloud style frequency figure per decade."""
    for decade in DECADES:
        save_word_cloud_plot(
            top_words,
            decade,
            os.path.join(RESULT_DIR, f"word_cloud_{decade}.png"),
            top_n=40,
        )


def orthogonal_align(source_matrix, target_matrix):
    """Find the rotation matrix that aligns a model vocabulary to the 1990s model."""
    matrix = source_matrix.T @ target_matrix
    left, _, right_t = np.linalg.svd(matrix)
    return left @ right_t


def aligned_china_vectors(models):
    """Align decade models to the 1990s space and return comparable 'china' vectors."""
    if "1990s" not in models or "china" not in models["1990s"]["word_to_index"]:
        return {}

    base_model = models["1990s"]
    base_vocab = set(base_model["vocab"])
    base_china_index = base_model["word_to_index"]["china"]
    vectors = {"1990s": base_model["vectors"][base_china_index]}

    for decade, model in models.items():
        if decade == "1990s" or "china" not in model["word_to_index"]:
            continue

        shared_words = sorted(base_vocab.intersection(model["vocab"]))

        if len(shared_words) < 2:
            continue

        source = normalize(
            np.vstack([model["vectors"][model["word_to_index"][word]] for word in shared_words])
        )
        target = normalize(
            np.vstack(
                [base_model["vectors"][base_model["word_to_index"][word]] for word in shared_words]
            )
        )
        rotation = orthogonal_align(source, target)
        china_index = model["word_to_index"]["china"]
        vectors[decade] = model["vectors"][china_index] @ rotation

    return vectors


def pca_2d(matrix):
    """Project vectors to two dimensions with a small NumPy PCA implementation."""
    centered = matrix - matrix.mean(axis=0)
    _, _, right_t = np.linalg.svd(centered, full_matrices=False)
    return centered @ right_t[:2].T


def save_semantic_shift_plot(models, output_path):
    """Plot the movement of the word 'china' across aligned decade embedding spaces."""
    vectors = aligned_china_vectors(models)

    if len(vectors) < 2:
        return

    decades = [decade for decade in DECADES if decade in vectors]
    matrix = np.vstack([vectors[decade] for decade in decades])
    coords = pca_2d(matrix)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(coords[:, 0], coords[:, 1], marker="o", linewidth=2, color="#8a4f2a")

    for decade, (x_coord, y_coord) in zip(decades, coords):
        ax.annotate(decade, (x_coord, y_coord), xytext=(7, 5), textcoords="offset points")

    ax.axhline(0, color="#dddddd", linewidth=1)
    ax.axvline(0, color="#dddddd", linewidth=1)
    ax.set_title("Semantic Shift of 'china' Across Decades")
    ax.set_xlabel("PCA dimension 1")
    ax.set_ylabel("PCA dimension 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_neighbor_shift_plot(neighbors, output_path):
    """Plot nearest-neighbor similarity scores for 'china' in each decade."""
    valid = neighbors.dropna(subset=["rank", "similarity"]).copy()
    valid = valid[valid["rank"] <= 10]

    if valid.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for ax, decade in zip(axes, DECADES):
        data = valid[valid["decade"] == decade].sort_values("similarity")
        ax.barh(data["neighbor"], data["similarity"], color="#6b8f3f")
        ax.set_title(f"{decade}: Embedding Neighbors of 'china'")
        ax.set_xlabel("Embedding cosine similarity")
        ax.set_xlim(0, 1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def annotate_events(ax, events, ymin=None, ymax=None):
    """Add light event markers for orientation only, not causal proof."""
    if events is None or events.empty:
        return

    for _, event in events.iterrows():
        year = int(event["year"])
        ax.axvline(year, color="#cccccc", linewidth=0.8, linestyle="--", alpha=0.7)

    if ymin is None or ymax is None:
        ymin, ymax = ax.get_ylim()

    for _, event in events.drop_duplicates("year").iterrows():
        ax.text(
            int(event["year"]),
            ymax,
            str(event["year"]),
            rotation=90,
            va="top",
            ha="right",
            fontsize=7,
            color="#666666",
        )


def save_opportunity_threat_plot(yearly_scores, events, output_path):
    """Plot opportunity and threat framing over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    data = yearly_scores.sort_values("year")
    ax.plot(data["year"], data["opportunity_score"], marker="o", label="Opportunity framing")
    ax.plot(data["year"], data["threat_score"], marker="o", label="Threat framing")
    ax.set_title("Opportunity vs Threat Framing Around China")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average mentions per 1,000 article tokens")
    ax.legend()
    annotate_events(ax, events)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_engagement_rivalry_plot(yearly_scores, events, output_path):
    """Plot engagement and rivalry framing over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    data = yearly_scores.sort_values("year")
    ax.plot(data["year"], data["engagement_score"], marker="o", label="Engagement score")
    ax.plot(data["year"], data["rivalry_score"], marker="o", label="Rivalry score")
    ax.set_title("Engagement vs Rivalry Framing Around China")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average mentions per 1,000 article tokens")
    ax.legend()
    annotate_events(ax, events)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_net_framing_plot(yearly_scores, events, output_path):
    """Plot the net balance between positive/engagement and negative/rivalry terms."""
    fig, ax = plt.subplots(figsize=(12, 6))
    data = yearly_scores.sort_values("year")
    ax.plot(data["year"], data["net_framing_score"], marker="o", color="#5f6f2f")
    ax.axhline(0, color="#999999", linewidth=1)
    ax.set_title("Net Framing Score Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Positive framing minus negative framing")
    annotate_events(ax, events)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_decade_framing_bar(decade_scores, output_path):
    """Compare positive/opportunity and negative/threat framing by decade."""
    data = decade_scores.set_index("decade").reindex(DECADES.keys())
    plot_data = data[["opportunity_score", "threat_score", "positive_framing_score", "negative_framing_score"]]

    fig, ax = plt.subplots(figsize=(11, 6))
    plot_data.plot(kind="bar", ax=ax)
    ax.set_title("Positive/Opportunity vs Negative/Threat Framing by Decade")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Average mentions per 1,000 article tokens")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_section_framing_heatmap(section_scores, output_path):
    """Heatmap of negative/rivalry framing by decade and NYT section."""
    if section_scores.empty:
        return

    top_sections = (
        section_scores.groupby("section")["articles"]
        .sum()
        .sort_values(ascending=False)
        .head(12)
        .index
    )
    data = section_scores[section_scores["section"].isin(top_sections)]
    plot_data = data.pivot_table(
        index="section",
        columns="decade",
        values="rivalry_score",
        fill_value=0,
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(plot_data.values, aspect="auto", cmap="YlGnBu")
    ax.set_title("Rivalry Framing by Decade and NYT Section")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Section")
    ax.set_xticks(np.arange(len(plot_data.columns)))
    ax.set_xticklabels(plot_data.columns)
    ax.set_yticks(np.arange(len(plot_data.index)))
    ax.set_yticklabels(plot_data.index)
    fig.colorbar(image, ax=ax, label="Rivalry score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_economic_security_line_plot(cooccurrence_year, events, output_path):
    """Plot yearly economic/security/both article shares."""
    data = cooccurrence_year.sort_values("year")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data["year"], data["percent_with_economic_terms"], marker="o", label="Economic")
    ax.plot(data["year"], data["percent_with_security_terms"], marker="o", label="Security")
    ax.plot(
        data["year"],
        data["percent_with_both_economic_and_security"],
        marker="o",
        label="Economic + security",
    )
    ax.set_title("Economic-Security Co-occurrence in China Coverage")
    ax.set_xlabel("Year")
    ax.set_ylabel("Percent of articles")
    ax.legend()
    annotate_events(ax, events)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_economic_security_decade_bar(cooccurrence_decade, output_path):
    """Plot economic/security overlap by decade."""
    data = cooccurrence_decade.set_index("decade").reindex(DECADES.keys())
    plot_data = data[
        [
            "percent_with_economic_terms",
            "percent_with_security_terms",
            "percent_with_both_economic_and_security",
        ]
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_data.plot(kind="bar", ax=ax)
    ax.set_title("Economic-Security Co-occurrence by Decade")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Percent of articles")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_securitization_index_plot(cooccurrence_year, events, output_path):
    """Plot security mentions relative to economic mentions."""
    data = cooccurrence_year.sort_values("year")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data["year"], data["securitization_index"], marker="o", color="#7f3f3f")
    ax.set_title("Securitization Index: Security Mentions / (Economic Mentions + 1)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Securitization index")
    annotate_events(ax, events)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_engagement_competition_shift_plot(shift_year, events, output_path):
    """Plot engagement prevalence against conflict/rivalry prevalence."""
    data = shift_year.sort_values("year")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data["year"], data["engagement_prevalence"], marker="o", label="Engagement")
    ax.plot(data["year"], data["conflict_prevalence"], marker="o", label="Conflict/rivalry")
    ax.set_title("Engagement-to-Competition Shift")
    ax.set_xlabel("Year")
    ax.set_ylabel("Percent of articles")
    ax.legend()
    annotate_events(ax, events)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_technology_bridge_plot(technology_year, events, output_path):
    """Plot technology's overlap with economy and security frames."""
    data = technology_year.sort_values("year")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data["year"], data["percent_technology_economy"], marker="o", label="Technology + economy")
    ax.plot(data["year"], data["percent_technology_security"], marker="o", label="Technology + security")
    ax.plot(
        data["year"],
        data["percent_technology_economy_security"],
        marker="o",
        label="Technology + economy + security",
    )
    ax.set_title("Technology as a Bridge Between Economy and Security")
    ax.set_xlabel("Year")
    ax.set_ylabel("Percent of articles")
    ax.legend()
    annotate_events(ax, events)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_semantic_similarity_plot(similarity_pairs, output_path):
    """Plot selected economic-security semantic similarity pairs by decade."""
    data = similarity_pairs.dropna(subset=["similarity"]).copy()

    if data.empty:
        return

    data["pair"] = data["term_a"] + "/" + data["term_b"]
    selected_pairs = [
        "trade/security",
        "market/security",
        "technology/security",
        "semiconductor/security",
        "china/competition",
    ]
    data = data[data["pair"].isin(selected_pairs)]

    fig, ax = plt.subplots(figsize=(12, 6))

    for pair, group in data.groupby("pair"):
        ordered = group.set_index("decade").reindex(DECADES.keys()).dropna(subset=["similarity"])
        ax.plot(ordered.index, ordered["similarity"], marker="o", label=pair)

    ax.set_title("Economic-Security Semantic Similarity by Decade")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Cosine similarity in decade embeddings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def export_json(df, output_path):
    """Save a dataframe as website-ready records JSON."""
    df.to_json(output_path, orient="records", indent=2, force_ascii=False)


def default_input_files():
    """Use every CSV in data/ whose filename starts with china_."""
    return sorted(
        os.path.join(DATA_DIR, filename)
        for filename in os.listdir(DATA_DIR)
        if filename.startswith("china_") and filename.endswith(".csv")
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze NYT China coverage with a securitization-focused pipeline."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=default_input_files(),
        help="CSV file(s) to analyze. Defaults to data/china_*.csv.",
    )
    args = parser.parse_args()

    os.makedirs(SECURITIZATION_RESULT_DIR, exist_ok=True)
    os.makedirs(WEB_DATA_DIR, exist_ok=True)

    # 1. Load CSV files.
    df = load_csv_files(args.input)

    # 2-3. Combine article text fields, preserve a no-keywords text field, and clean tokens.
    df = prepare_documents(df)
    quality_df = filter_suspicious_articles(df)
    clean_df = quality_df[quality_df["quality_flag"] == "keep"].copy()
    suspicious_df = quality_df[quality_df["quality_flag"] == "suspicious"].copy()
    (
        quality_summary,
        quality_removed_by_reason,
        quality_removed_by_decade,
        quality_kept_by_decade,
    ) = quality_filter_summaries(quality_df)

    quality_df.to_csv(
        os.path.join(SECURITIZATION_RESULT_DIR, "quality_audit_all_articles.csv"),
        index=False,
    )
    suspicious_df.to_csv(
        os.path.join(SECURITIZATION_RESULT_DIR, "filtered_suspicious_articles.csv"),
        index=False,
    )
    clean_df.to_csv(
        os.path.join(SECURITIZATION_RESULT_DIR, "filtered_clean_articles.csv"),
        index=False,
    )
    quality_summary.to_csv(
        os.path.join(SECURITIZATION_RESULT_DIR, "quality_filter_summary.csv"),
        index=False,
    )
    quality_removed_by_reason.to_csv(
        os.path.join(SECURITIZATION_RESULT_DIR, "quality_removed_by_reason.csv"),
        index=False,
    )
    quality_removed_by_decade.to_csv(
        os.path.join(SECURITIZATION_RESULT_DIR, "quality_removed_by_decade.csv"),
        index=False,
    )
    quality_kept_by_decade.to_csv(
        os.path.join(SECURITIZATION_RESULT_DIR, "quality_kept_by_decade.csv"),
        index=False,
    )

    # From this point forward, all analysis uses only records that passed the
    # conservative quality filter. The full audit file remains available for
    # transparency and reproducibility.
    df = clean_df
    df.to_csv(os.path.join(SECURITIZATION_RESULT_DIR, "cleaned_documents.csv"), index=False)

    # Basic descriptive outputs remain useful for checking what the corpus contains.
    summary = decade_summary(df)
    country_mentions = country_mention_stats(df)
    sections = section_distribution(df)
    theme_framing = theme_framing_by_decade(df)
    words = top_words_by_decade(df, top_n=40)
    word_pairs = top_bigrams_by_decade(df, top_n=20)
    keywords = tfidf_keywords_by_decade(df, top_n=30)

    summary.to_csv(os.path.join(SECURITIZATION_RESULT_DIR, "decade_summary.csv"), index=False)
    country_mentions.to_csv(os.path.join(SECURITIZATION_RESULT_DIR, "country_mention_stats.csv"), index=False)
    sections.to_csv(os.path.join(SECURITIZATION_RESULT_DIR, "section_distribution_by_decade.csv"), index=False)
    theme_framing.to_csv(os.path.join(SECURITIZATION_RESULT_DIR, "theme_framing_by_decade.csv"), index=False)
    words.to_csv(os.path.join(SECURITIZATION_RESULT_DIR, "top_40_words_by_decade.csv"), index=False)
    word_pairs.to_csv(os.path.join(SECURITIZATION_RESULT_DIR, "top_20_bigrams_by_decade.csv"), index=False)
    keywords.to_csv(os.path.join(SECURITIZATION_RESULT_DIR, "tfidf_keywords_by_decade.csv"), index=False)

    # Dictionary scores are not objective proof of emotion or sentiment. They are
    # transparent indicators for tracing changes in framing and for selecting
    # articles that deserve close reading.
    scored = add_framing_scores(df)
    article_columns = [
        "year",
        "date",
        "decade",
        "headline",
        "abstract",
        "keywords",
        "section",
        "news_desk",
        "document_type",
        "url",
        "quality_flag",
        "quality_score",
        "quality_reasons",
        "text_token_count_without_keywords",
        "text_token_count_with_keywords",
        "opportunity_score",
        "threat_score",
        "competition_score",
        "cooperation_score",
        "positive_framing_score",
        "negative_framing_score",
        "net_framing_score",
        "rivalry_score",
        "engagement_score",
        "rivalry_to_engagement_ratio",
        "economic_score",
        "security_score",
        "technology_score",
        "securitization_index",
    ]
    scored[article_columns].to_csv(
        os.path.join(SECURITIZATION_RESULT_DIR, "framing_scores_by_article.csv"),
        index=False,
    )

    framing_year = aggregate_framing(scored, ["year"])
    framing_decade = aggregate_framing(scored, ["decade"])
    framing_section = aggregate_framing(scored, ["section"])
    framing_news_desk = aggregate_framing(scored, ["news_desk"])
    framing_document_type = aggregate_framing(scored, ["document_type"])
    framing_decade_section = aggregate_framing(scored, ["decade", "section"])

    framing_year.to_csv(os.path.join(SECURITIZATION_RESULT_DIR, "framing_scores_by_year.csv"), index=False)
    framing_decade.to_csv(os.path.join(SECURITIZATION_RESULT_DIR, "framing_scores_by_decade.csv"), index=False)
    framing_section.to_csv(os.path.join(SECURITIZATION_RESULT_DIR, "framing_scores_by_section.csv"), index=False)
    framing_news_desk.to_csv(os.path.join(SECURITIZATION_RESULT_DIR, "framing_scores_by_news_desk.csv"), index=False)
    framing_document_type.to_csv(
        os.path.join(SECURITIZATION_RESULT_DIR, "framing_scores_by_document_type.csv"),
        index=False,
    )

    cooccurrence_year = economic_security_cooccurrence(scored, "year")
    cooccurrence_decade = economic_security_cooccurrence(scored, "decade")
    cooccurrence_year.to_csv(
        os.path.join(SECURITIZATION_RESULT_DIR, "economic_security_cooccurrence_by_year.csv"),
        index=False,
    )
    cooccurrence_decade.to_csv(
        os.path.join(SECURITIZATION_RESULT_DIR, "economic_security_cooccurrence_by_decade.csv"),
        index=False,
    )

    securitization_year = cooccurrence_year[
        [
            "year",
            "articles",
            "securitization_index",
            "economic_security_overlap_rate",
            "conflict_vs_engagement_ratio",
        ]
    ]
    securitization_decade = cooccurrence_decade[
        [
            "decade",
            "articles",
            "securitization_index",
            "economic_security_overlap_rate",
            "conflict_vs_engagement_ratio",
        ]
    ]
    securitization_year.to_csv(
        os.path.join(SECURITIZATION_RESULT_DIR, "securitization_index_by_year.csv"),
        index=False,
    )
    securitization_decade.to_csv(
        os.path.join(SECURITIZATION_RESULT_DIR, "securitization_index_by_decade.csv"),
        index=False,
    )

    shift_year = engagement_competition_shift(scored, "year")
    shift_decade = engagement_competition_shift(scored, "decade")
    shift_year.to_csv(
        os.path.join(SECURITIZATION_RESULT_DIR, "engagement_competition_shift_by_year.csv"),
        index=False,
    )
    shift_decade.to_csv(
        os.path.join(SECURITIZATION_RESULT_DIR, "engagement_competition_shift_by_decade.csv"),
        index=False,
    )

    technology_year = technology_bridge(scored, "year")
    technology_decade = technology_bridge(scored, "decade")
    technology_year.to_csv(
        os.path.join(SECURITIZATION_RESULT_DIR, "technology_bridge_by_year.csv"),
        index=False,
    )
    technology_decade.to_csv(
        os.path.join(SECURITIZATION_RESULT_DIR, "technology_bridge_by_decade.csv"),
        index=False,
    )

    for group_column in ["year", "decade"]:
        for metadata_column in ["section", "news_desk", "document_type", "type_of_material"]:
            metadata_distribution(df, group_column, metadata_column).to_csv(
                os.path.join(
                    SECURITIZATION_RESULT_DIR,
                    f"{metadata_column}_distribution_by_{group_column}.csv",
                ),
                index=False,
            )

    nyt_keywords = top_nyt_keywords_by_decade(df)
    nyt_keywords.to_csv(os.path.join(SECURITIZATION_RESULT_DIR, "top_nyt_keywords_by_decade.csv"), index=False)

    turning_points = detect_framing_turning_points(framing_year)
    turning_points.to_csv(os.path.join(SECURITIZATION_RESULT_DIR, "framing_turning_points.csv"), index=False)
    events = create_historical_events(os.path.join(SECURITIZATION_RESULT_DIR, "historical_events.csv"))

    close_reading = representative_articles(scored)
    close_reading.to_csv(
        os.path.join(SECURITIZATION_RESULT_DIR, "representative_articles_for_close_reading.csv"),
        index=False,
    )

    # Train separate decade embedding models for selected neighbor and similarity checks.
    models = train_embedding_models(df)
    neighbors = china_neighbors(models, top_n=20)
    neighbors.to_csv(os.path.join(SECURITIZATION_RESULT_DIR, "china_embedding_neighbors.csv"), index=False)
    selected_terms = [
        "china",
        "trade",
        "market",
        "investment",
        "technology",
        "security",
        "taiwan",
        "chip",
        "semiconductor",
        "competition",
    ]
    selected_neighbors = embedding_neighbors_for_terms(models, selected_terms)
    selected_neighbors.to_csv(
        os.path.join(SECURITIZATION_RESULT_DIR, "embedding_neighbors_selected_terms.csv"),
        index=False,
    )
    semantic_pairs = semantic_similarity_pairs(models)
    semantic_pairs.to_csv(
        os.path.join(SECURITIZATION_RESULT_DIR, "semantic_similarity_pairs_by_decade.csv"),
        index=False,
    )
    save_embedding_vectors(models)

    # Visualizations for both overview and argument-driven interpretation.
    save_frequency_plot(words, os.path.join(SECURITIZATION_RESULT_DIR, "frequency_comparison.png"))
    save_keyword_plot(keywords, os.path.join(SECURITIZATION_RESULT_DIR, "keyword_comparison.png"))
    save_country_mentions_plot(
        country_mentions,
        os.path.join(SECURITIZATION_RESULT_DIR, "country_mentions_by_decade.png"),
    )
    save_theme_framing_plot(
        theme_framing,
        os.path.join(SECURITIZATION_RESULT_DIR, "theme_framing_by_decade.png"),
    )
    save_decade_word_clouds(words)
    save_semantic_shift_plot(models, os.path.join(SECURITIZATION_RESULT_DIR, "semantic_shift_china.png"))
    save_neighbor_shift_plot(neighbors, os.path.join(SECURITIZATION_RESULT_DIR, "semantic_neighbors_china.png"))
    save_opportunity_threat_plot(
        framing_year,
        events,
        os.path.join(SECURITIZATION_RESULT_DIR, "yearly_opportunity_vs_threat.png"),
    )
    save_engagement_rivalry_plot(
        framing_year,
        events,
        os.path.join(SECURITIZATION_RESULT_DIR, "yearly_engagement_vs_rivalry.png"),
    )
    save_net_framing_plot(
        framing_year,
        events,
        os.path.join(SECURITIZATION_RESULT_DIR, "net_framing_score_over_time.png"),
    )
    save_decade_framing_bar(
        framing_decade,
        os.path.join(SECURITIZATION_RESULT_DIR, "decade_positive_negative_framing.png"),
    )
    save_section_framing_heatmap(
        framing_decade_section,
        os.path.join(SECURITIZATION_RESULT_DIR, "framing_scores_by_decade_section_heatmap.png"),
    )
    save_economic_security_line_plot(
        cooccurrence_year,
        events,
        os.path.join(SECURITIZATION_RESULT_DIR, "economic_security_cooccurrence_by_year.png"),
    )
    save_economic_security_decade_bar(
        cooccurrence_decade,
        os.path.join(SECURITIZATION_RESULT_DIR, "economic_security_cooccurrence_by_decade.png"),
    )
    save_securitization_index_plot(
        cooccurrence_year,
        events,
        os.path.join(SECURITIZATION_RESULT_DIR, "securitization_index_by_year.png"),
    )
    save_engagement_competition_shift_plot(
        shift_year,
        events,
        os.path.join(SECURITIZATION_RESULT_DIR, "engagement_competition_shift_by_year.png"),
    )
    save_technology_bridge_plot(
        technology_year,
        events,
        os.path.join(SECURITIZATION_RESULT_DIR, "technology_bridge_by_year.png"),
    )
    save_semantic_similarity_plot(
        semantic_pairs,
        os.path.join(SECURITIZATION_RESULT_DIR, "semantic_similarity_pairs_by_decade.png"),
    )

    # Website-ready exports: simple records JSON for future dashboard loading.
    export_json(framing_year, os.path.join(WEB_DATA_DIR, "framing_scores_by_year.json"))
    export_json(framing_decade, os.path.join(WEB_DATA_DIR, "framing_scores_by_decade.json"))
    export_json(
        cooccurrence_year,
        os.path.join(WEB_DATA_DIR, "economic_security_cooccurrence_by_year.json"),
    )
    export_json(
        semantic_pairs,
        os.path.join(WEB_DATA_DIR, "semantic_similarity_pairs_by_decade.json"),
    )
    export_json(keywords, os.path.join(WEB_DATA_DIR, "top_keywords_by_decade.json"))
    export_json(
        close_reading,
        os.path.join(WEB_DATA_DIR, "representative_articles_for_close_reading.json"),
    )
    export_json(events, os.path.join(WEB_DATA_DIR, "historical_events.json"))
    export_json(quality_summary, os.path.join(WEB_DATA_DIR, "quality_filter_summary.json"))
    export_json(
        suspicious_df.head(100),
        os.path.join(WEB_DATA_DIR, "filtered_suspicious_articles_sample.json"),
    )
    balance_summary = save_balanced_corpus_outputs(df)

    print("Analysis complete.")
    print(
        "Articles before filtering: "
        f"{int(quality_summary.loc[0, 'total_articles_before_filtering'])}"
    )
    print(
        "Articles kept: "
        f"{int(quality_summary.loc[0, 'total_articles_after_filtering'])}"
    )
    print(f"Articles removed: {int(quality_summary.loc[0, 'total_removed'])}")
    print("Top removal reasons:")
    for _, row in quality_removed_by_reason.head(10).iterrows():
        print(f"  {row['quality_reason']}: {row['removed_articles']}")
    print(f"Outputs saved in: {SECURITIZATION_RESULT_DIR}")
    print(f"Website-ready JSON saved in: {WEB_DATA_DIR}")
    if not balance_summary.empty:
        target_n = int(balance_summary["target_articles_per_decade"].iloc[0])
        print(
            "Balanced robustness corpus saved in: "
            f"{BALANCED_RESULT_DIR} ({target_n} articles per decade)"
        )
    print(
        "Reminder: framing and sentiment scores are transparent dictionary-based "
        "indicators, not objective proof of emotion or intent. Use them with close reading."
    )


if __name__ == "__main__":
    main()
