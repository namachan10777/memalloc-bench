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
    stats: pl.DataFrame, pattern: str, metric: str = "total"
) -> alt.Chart:
    """パターン別のサイズ vs 時間の折れ線グラフ（プラットフォーム別）"""
    data = stats.filter(pl.col("pattern") == pattern).to_pandas()
    col = f"{metric}_median_ms"

    chart = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X("size_bytes:Q", scale=alt.Scale(type="log"), title="Size (bytes)"),
            y=alt.Y(f"{col}:Q", title=f"Median {metric.title()} Time (ms)"),
            color=alt.Color("allocator:N", title="Allocator"),
            strokeDash=alt.StrokeDash("platform:N", title="Platform"),
        )
        .properties(width=600, height=400, title=f"Pattern: {pattern} ({metric})")
    )
    return chart


def create_all_patterns_chart(stats: pl.DataFrame, metric: str = "total") -> alt.Chart:
    """全パターンのファセットチャート"""
    data = stats.to_pandas()
    col = f"{metric}_median_ms"

    chart = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X("size_bytes:Q", scale=alt.Scale(type="log"), title="Size (bytes)"),
            y=alt.Y(
                f"{col}:Q",
                scale=alt.Scale(type="log"),
                title=f"Median {metric.title()} Time (ms, log)",
            ),
            color=alt.Color("allocator:N", title="Allocator"),
            strokeDash=alt.StrokeDash("platform:N", title="Platform"),
        )
        .properties(width=300, height=200)
        .facet(facet=alt.Facet("pattern:N", title="Pattern"), columns=3)
        .resolve_scale(y="independent")
    )
    return chart


def create_box_plot(
    df: pl.DataFrame, pattern: str, size: int, metric: str = "total"
) -> alt.Chart:
    """特定条件でのボックスプロット"""
    col = f"{metric}_ns"
    data = (
        df.filter((pl.col("pattern") == pattern) & (pl.col("size_bytes") == size))
        .with_columns((pl.col(col) * NS_TO_MS).alias("time_ms"))
        .to_pandas()
    )

    chart = (
        alt.Chart(data)
        .mark_boxplot()
        .encode(
            x=alt.X("allocator:N", title="Allocator"),
            y=alt.Y("time_ms:Q", title=f"{metric.title()} Time (ms)"),
            color=alt.Color("platform:N", title="Platform"),
            column=alt.Column("platform:N", title="Platform"),
        )
        .properties(
            width=200,
            height=300,
            title=f"Pattern: {pattern}, Size: {size} bytes ({metric})",
        )
    )
    return chart


def create_heatmap(
    stats: pl.DataFrame, allocator: str, metric: str = "total"
) -> alt.Chart:
    """アロケータ別のパターン×サイズのヒートマップ（プラットフォーム別）"""
    col = f"{metric}_median_ms"
    data = stats.filter(pl.col("allocator") == allocator).to_pandas()

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
        .properties(width=500, height=150)
        .facet(row=alt.Row("platform:N", title="Platform"))
        .properties(title=f"Allocator: {allocator} ({metric})")
    )
    return chart


def create_comparison_chart(stats: pl.DataFrame, metric: str = "total") -> alt.Chart:
    """Box vs Slab warm の比較（速度比）"""
    col = f"{metric}_median_ms"
    box_data = (
        stats.filter(pl.col("allocator") == "box")
        .select(["platform", "pattern", "size_bytes", col])
        .rename({col: "box_ms"})
    )

    slab_warm_data = (
        stats.filter(pl.col("allocator") == "slab_warm")
        .select(["platform", "pattern", "size_bytes", col])
        .rename({col: "slab_warm_ms"})
    )

    comparison = box_data.join(slab_warm_data, on=["platform", "pattern", "size_bytes"])
    comparison = comparison.with_columns(
        (pl.col("box_ms") / pl.col("slab_warm_ms")).alias("ratio")
    )

    data = comparison.to_pandas()

    chart = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X("size_bytes:Q", scale=alt.Scale(type="log"), title="Size (bytes)"),
            y=alt.Y("ratio:Q", title="Box / Slab(warm) ratio"),
            color=alt.Color("pattern:N", title="Pattern"),
            strokeDash=alt.StrokeDash("platform:N", title="Platform"),
        )
        .properties(
            width=600,
            height=400,
            title=f"Box vs Slab(warm) Speed Ratio ({metric}, >1 means Slab is faster)",
        )
    )

    # ratio=1 の参照線
    rule = alt.Chart().mark_rule(strokeDash=[5, 5], color="gray").encode(y=alt.datum(1))

    return chart + rule


def create_platform_comparison_chart(
    stats: pl.DataFrame, metric: str = "total"
) -> alt.Chart:
    """プラットフォーム間の比較チャート"""
    col = f"{metric}_median_ms"
    data = stats.to_pandas()

    chart = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X("size_bytes:Q", scale=alt.Scale(type="log"), title="Size (bytes)"),
            y=alt.Y(f"{col}:Q", title=f"Median {metric.title()} Time (ms)"),
            color=alt.Color("platform:N", title="Platform"),
            strokeDash=alt.StrokeDash("allocator:N", title="Allocator"),
        )
        .properties(width=300, height=200)
        .facet(facet=alt.Facet("pattern:N", title="Pattern"), columns=3)
        .resolve_scale(y="independent")
    )
    return chart


def save_svg(chart: alt.Chart, path: str):
    """チャートをSVGで保存"""
    chart.save(path, format="svg")


def main():
    print("Loading data...")
    df = load_all_data()
    platforms = df["platform"].unique().to_list()
    print(f"Loaded {len(df)} records from {len(platforms)} platform(s): {platforms}")

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

        # 1. 全パターンのファセットチャート
        all_patterns = create_all_patterns_chart(stats, metric=metric)
        save_svg(all_patterns, f"results/all_patterns_{metric}.svg")
        print(f"  - results/all_patterns_{metric}.svg")

        # 2. 各パターン別の詳細チャート
        for pattern in ["immediate", "lifo", "fifo", "random"]:
            chart = create_line_chart_by_pattern(stats, pattern, metric=metric)
            save_svg(chart, f"results/pattern_{pattern}_{metric}.svg")
            print(f"  - results/pattern_{pattern}_{metric}.svg")

        # 3. ヒートマップ
        for allocator in ["box", "slab_cold", "slab_warm"]:
            heatmap = create_heatmap(stats, allocator, metric=metric)
            save_svg(heatmap, f"results/heatmap_{allocator}_{metric}.svg")
            print(f"  - results/heatmap_{allocator}_{metric}.svg")

        # 4. Box vs Slab比較
        comparison = create_comparison_chart(stats, metric=metric)
        save_svg(comparison, f"results/comparison_ratio_{metric}.svg")
        print(f"  - results/comparison_ratio_{metric}.svg")

        # 5. プラットフォーム間比較
        platform_comparison = create_platform_comparison_chart(stats, metric=metric)
        save_svg(platform_comparison, f"results/platform_comparison_{metric}.svg")
        print(f"  - results/platform_comparison_{metric}.svg")

        # 6. ボックスプロット（代表的な条件）
        for pattern, size in [("immediate", 64), ("lifo", 256), ("random", 1024)]:
            boxplot = create_box_plot(df, pattern, size, metric=metric)
            save_svg(boxplot, f"results/boxplot_{pattern}_{size}_{metric}.svg")
            print(f"  - results/boxplot_{pattern}_{size}_{metric}.svg")

    # 統計データもCSVで保存
    stats.write_csv("results/stats.csv")
    print("  - results/stats.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
