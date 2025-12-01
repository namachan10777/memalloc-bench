#!/usr/bin/env python3
"""
Box::new vs Slab ベンチマーク結果の分析・可視化
"""

from pathlib import Path

import altair as alt
import polars as pl

# Altairの行数制限を解除
alt.data_transformers.disable_max_rows()

# ns -> ms 変換係数
NS_TO_MS = 1e-6

# 表示ラベルのマッピング
PLATFORM_LABELS = {
    "mbp": "MacBook Pro (M1Max)",
    "pegasus": "Intel Xeon Platinum 8468",
}

PATTERN_LABELS = {
    "immediate": "Immediate",
    "lifo": "LIFO",
    "fifo": "FIFO",
    "random": "Random",
}


def apply_labels(df: pl.DataFrame) -> pl.DataFrame:
    """プラットフォームとパターンのラベルを変換"""
    return df.with_columns(
        [
            pl.col("platform").replace(PLATFORM_LABELS).alias("platform"),
            pl.col("pattern").replace(PATTERN_LABELS).alias("pattern"),
        ]
    )


def load_all_data(results_dir: str = "results") -> pl.DataFrame:
    """results/配下の全parquetファイルを読み込んで結合"""
    parquet_files = list(Path(results_dir).glob("benchmark_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No benchmark_*.parquet files found in {results_dir}")

    print(f"Found {len(parquet_files)} parquet files:")
    for f in parquet_files:
        print(f"  - {f}")

    dfs = [pl.read_parquet(f) for f in parquet_files]
    return pl.concat(dfs)


def compute_stats(df: pl.DataFrame) -> pl.DataFrame:
    """各条件ごとの統計量を計算 (ms単位)"""
    return df.group_by(["platform", "allocator", "pattern", "size_bytes"]).agg(
        # total (throughput)
        (pl.col("total_ns").mean() * NS_TO_MS).alias("total_mean_ms"),
        (pl.col("total_ns").std() * NS_TO_MS).alias("total_std_ms"),
        (pl.col("total_ns").median() * NS_TO_MS).alias("total_median_ms"),
        (pl.col("total_ns").quantile(0.25) * NS_TO_MS).alias("total_p25_ms"),
        (pl.col("total_ns").quantile(0.75) * NS_TO_MS).alias("total_p75_ms"),
        (pl.col("total_ns").min() * NS_TO_MS).alias("total_min_ms"),
        (pl.col("total_ns").max() * NS_TO_MS).alias("total_max_ms"),
        # latency (single operation)
        (pl.col("latency_ns").mean() * NS_TO_MS).alias("latency_mean_ms"),
        (pl.col("latency_ns").std() * NS_TO_MS).alias("latency_std_ms"),
        (pl.col("latency_ns").median() * NS_TO_MS).alias("latency_median_ms"),
        (pl.col("latency_ns").quantile(0.25) * NS_TO_MS).alias("latency_p25_ms"),
        (pl.col("latency_ns").quantile(0.75) * NS_TO_MS).alias("latency_p75_ms"),
        (pl.col("latency_ns").min() * NS_TO_MS).alias("latency_min_ms"),
        (pl.col("latency_ns").max() * NS_TO_MS).alias("latency_max_ms"),
    )


def create_line_chart_by_pattern(
    stats: pl.DataFrame, pattern: str, platform: str, metric: str = "total"
) -> alt.Chart:
    """パターン別・プラットフォーム別のサイズ vs 時間の折れ線グラフ"""
    data = stats.filter(
        (pl.col("pattern") == pattern) & (pl.col("platform") == platform)
    ).to_pandas()
    col = f"{metric}_median_ms"

    chart = (
        alt.Chart(data)
        .mark_line(point=False)
        .encode(
            x=alt.X("size_bytes:Q", scale=alt.Scale(type="log"), title="Size (bytes)"),
            y=alt.Y(f"{col}:Q", title=f"Median {metric.title()} Time (ms)"),
            color=alt.Color("allocator:N", title="Allocator"),
        )
        .properties(
            width=600, height=400, title=f"{platform} - Pattern: {pattern} ({metric})"
        )
        .configure_legend(labelLimit=600)
    )
    return chart


def create_all_patterns_chart(
    stats: pl.DataFrame, platform: str, metric: str = "total"
) -> alt.Chart:
    """全パターンのファセットチャート（プラットフォーム別）"""
    data = stats.filter(pl.col("platform") == platform).to_pandas()
    col = f"{metric}_median_ms"

    chart = (
        alt.Chart(data)
        .mark_line(point=False)
        .encode(
            x=alt.X("size_bytes:Q", scale=alt.Scale(type="log"), title="Size (bytes)"),
            y=alt.Y(
                f"{col}:Q",
                scale=alt.Scale(type="log"),
                title=f"Median {metric.title()} Time (ms, log)",
            ),
            color=alt.Color("allocator:N", title="Allocator"),
        )
        .properties(width=300, height=200, title=platform)
        .facet(facet=alt.Facet("pattern:N", title="Pattern"), columns=2)
        .resolve_scale(y="independent")
        .configure_legend(labelLimit=600)
    )
    return chart


def create_box_plot(
    df: pl.DataFrame, pattern: str, size: int, platform: str, metric: str = "total"
) -> alt.Chart:
    """特定条件でのボックスプロット（プラットフォーム別）"""
    col = f"{metric}_ns"
    data = (
        df.filter(
            (pl.col("pattern") == pattern)
            & (pl.col("size_bytes") == size)
            & (pl.col("platform") == platform)
        )
        .with_columns((pl.col(col) * NS_TO_MS).alias("time_ms"))
        .to_pandas()
    )

    chart = (
        alt.Chart(data)
        .mark_boxplot()
        .encode(
            x=alt.X("allocator:N", title="Allocator"),
            y=alt.Y("time_ms:Q", title=f"{metric.title()} Time (ms)"),
            color=alt.Color("allocator:N", title="Allocator"),
        )
        .properties(
            width=300,
            height=300,
            title=f"{platform} - Pattern: {pattern}, Size: {size} bytes ({metric})",
        )
        .configure_legend(labelLimit=600)
    )
    return chart


def create_heatmap(
    stats: pl.DataFrame, allocator: str, platform: str, metric: str = "total"
) -> alt.Chart:
    """アロケータ別のパターン×サイズのヒートマップ（プラットフォーム別）"""
    col = f"{metric}_median_ms"
    data = stats.filter(
        (pl.col("allocator") == allocator) & (pl.col("platform") == platform)
    ).to_pandas()

    chart = (
        alt.Chart(data)
        .mark_rect()
        .encode(
            x=alt.X("size_bytes:O", title="Size (bytes)"),
            y=alt.Y("pattern:N", title="Pattern"),
            color=alt.Color(
                f"{col}:Q", scale=alt.Scale(scheme="viridis"), title="Median (ms)"
            ),
        )
        .properties(
            width=500,
            height=150,
            title=f"{platform} - Allocator: {allocator} ({metric})",
        )
        .configure_legend(labelLimit=600)
    )
    return chart


def create_comparison_chart(
    stats: pl.DataFrame, platform: str, metric: str = "total"
) -> alt.Chart:
    """Box vs Slab warm の比較（速度比）（プラットフォーム別）"""
    col = f"{metric}_median_ms"
    platform_stats = stats.filter(pl.col("platform") == platform)

    box_data = (
        platform_stats.filter(pl.col("allocator") == "box")
        .select(["pattern", "size_bytes", col])
        .rename({col: "box_ms"})
    )

    slab_warm_data = (
        platform_stats.filter(pl.col("allocator") == "slab_warm")
        .select(["pattern", "size_bytes", col])
        .rename({col: "slab_warm_ms"})
    )

    comparison = box_data.join(slab_warm_data, on=["pattern", "size_bytes"])
    comparison = comparison.with_columns(
        (pl.col("box_ms") / pl.col("slab_warm_ms")).alias("ratio")
    )

    data = comparison.to_pandas()

    chart = (
        alt.Chart(data)
        .mark_line(point=False)
        .encode(
            x=alt.X("size_bytes:Q", scale=alt.Scale(type="log"), title="Size (bytes)"),
            y=alt.Y("ratio:Q", title="Box / Slab(warm) ratio"),
            color=alt.Color("pattern:N", title="Pattern"),
        )
        .properties(
            width=600,
            height=400,
            title=f"{platform} - Box vs Slab(warm) Speed Ratio ({metric}, >1 means Slab is faster)",
        )
    )

    # ratio=1 の参照線
    rule = alt.Chart().mark_rule(strokeDash=[5, 5], color="gray").encode(y=alt.datum(1))

    return (chart + rule).configure_legend(labelLimit=600)


def save_svg(chart: alt.Chart, path: str):
    """チャートをSVGで保存"""
    chart.save(path, format="svg")


# プラットフォームのファイル名用キー
PLATFORM_KEYS = {
    "MacBook Pro (M1Max)": "mbp",
    "Intel Xeon Platinum 8468": "pegasus",
}


def main():
    print("Loading data...")
    df = load_all_data()
    platforms = df["platform"].unique().to_list()
    print(f"Loaded {len(df)} records from {len(platforms)} platform(s): {platforms}")

    # ラベル変換
    df = apply_labels(df)
    platform_labels = list(PLATFORM_LABELS.values())

    print("Computing statistics...")
    stats = compute_stats(df)

    # 統計サマリーを表示
    print("\n=== Summary Statistics ===")
    summary = stats.sort(["platform", "pattern", "allocator", "size_bytes"])
    print(summary.head(30))

    # グラフ生成
    print("\nGenerating charts (SVG)...")

    for metric in ["total", "latency"]:
        print(f"\n--- {metric.upper()} metric ---")

        for platform_label in platform_labels:
            platform_key = PLATFORM_KEYS.get(platform_label, platform_label)
            print(f"\n  Platform: {platform_label} ({platform_key})")

            # 1. 全パターンのファセットチャート
            all_patterns = create_all_patterns_chart(
                stats, platform_label, metric=metric
            )
            save_svg(all_patterns, f"results/{platform_key}_all_patterns_{metric}.svg")
            print(f"    - results/{platform_key}_all_patterns_{metric}.svg")

            # 2. 各パターン別の詳細チャート
            for pattern_key, pattern_label in PATTERN_LABELS.items():
                chart = create_line_chart_by_pattern(
                    stats, pattern_label, platform_label, metric=metric
                )
                save_svg(
                    chart, f"results/{platform_key}_pattern_{pattern_key}_{metric}.svg"
                )
                print(
                    f"    - results/{platform_key}_pattern_{pattern_key}_{metric}.svg"
                )

            # 3. ヒートマップ
            for allocator in [
                "box",
                "slab_cold",
                "slab_warm",
                "bufpool_cold",
                "bufpool_warm",
            ]:
                heatmap = create_heatmap(
                    stats, allocator, platform_label, metric=metric
                )
                save_svg(
                    heatmap, f"results/{platform_key}_heatmap_{allocator}_{metric}.svg"
                )
                print(f"    - results/{platform_key}_heatmap_{allocator}_{metric}.svg")

            # 4. Box vs Slab比較
            comparison = create_comparison_chart(stats, platform_label, metric=metric)
            save_svg(
                comparison, f"results/{platform_key}_comparison_ratio_{metric}.svg"
            )
            print(f"    - results/{platform_key}_comparison_ratio_{metric}.svg")

            # 5. ボックスプロット（代表的な条件）
            for pattern_key, size in [
                ("immediate", 64),
                ("lifo", 256),
                ("random", 1024),
            ]:
                pattern_label = PATTERN_LABELS[pattern_key]
                boxplot = create_box_plot(
                    df, pattern_label, size, platform_label, metric=metric
                )
                save_svg(
                    boxplot,
                    f"results/{platform_key}_boxplot_{pattern_key}_{size}_{metric}.svg",
                )
                print(
                    f"    - results/{platform_key}_boxplot_{pattern_key}_{size}_{metric}.svg"
                )

    # 統計データもCSVで保存
    stats.write_csv("results/stats.csv")
    print("  - results/stats.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
