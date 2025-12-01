# memalloc-bench

Rustの `Box::new` と `slab` crateのメモリアロケーション性能を比較するベンチマーク。

## 測定対象

### アロケータ
- **Box::new** - システムアロケータ経由のヒープ確保
- **Slab (cold)** - 毎回新規Slabを作成（事前確保なし）
- **Slab (warm)** - `with_capacity`で事前確保済み

### データサイズ
8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096 bytes

### アクセスパターン

#### Immediate
alloc→deallocを繰り返す。割り当てたらすぐ解放。

```
for i in 0..BATCH_SIZE:
    ptr = alloc()
    dealloc(ptr)
```

#### LIFO (Last In, First Out)
スタック的なパターン。最後に確保したものを最初に解放。

```
stack = []
for i in 0..BATCH_SIZE:
    stack.push(alloc())
while stack is not empty:
    dealloc(stack.pop())  # 逆順
```

#### FIFO (First In, First Out)
キュー的なパターン。最初に確保したものを最初に解放。

```
queue = []
for i in 0..BATCH_SIZE:
    queue.push(alloc())
for ptr in queue:  # 順番通り
    dealloc(ptr)
```

#### Random
ランダムにalloc/deallocを混ぜる。スロットを選んで空ならalloc、埋まっていればdealloc。

```
slots = [None] * BATCH_SIZE
for i in 0..(BATCH_SIZE * 2):
    idx = random(0, BATCH_SIZE)
    if slots[idx] is not None:
        dealloc(slots[idx])
        slots[idx] = None
    else:
        slots[idx] = alloc()
# 残りを解放
for ptr in slots:
    if ptr is not None:
        dealloc(ptr)
```

## 実行方法

### ベンチマーク実行

```bash
# ビルド
cargo build --release

# 実行（プラットフォーム名を指定）
./target/release/memalloc-bench <platform>

# 例
./target/release/memalloc-bench local
./target/release/memalloc-bench hpc-xeon-8280
./target/release/memalloc-bench aws-c5
```

結果は `results/benchmark_<platform>.parquet` に出力されます。
複数プラットフォームの結果を同じ `results/` に配置すると、分析時に自動で結合されます。

### 分析・グラフ生成

```bash
uv run python analysis.py
```

生成されるグラフ（`results/` 以下、total/latency両方）:
- `all_patterns_{total,latency}.svg` - 全パターンのファセットチャート
- `pattern_*_{total,latency}.svg` - パターン別の詳細チャート
- `heatmap_*_{total,latency}.svg` - アロケータ別のヒートマップ
- `comparison_ratio_{total,latency}.svg` - Box vs Slab速度比
- `platform_comparison_{total,latency}.svg` - プラットフォーム間比較
- `boxplot_*_{total,latency}.svg` - 代表的条件でのボックスプロット
- `stats.csv` - 統計サマリー

## 出力データ形式

Parquetスキーマ:

| カラム | 型 | 説明 |
|--------|------|------|
| platform | string | プラットフォーム名 |
| allocator | string | "box", "slab_cold", "slab_warm" |
| pattern | string | "immediate", "lifo", "fifo", "random" |
| size_bytes | u32 | データサイズ |
| iteration | u32 | 試行番号 |
| total_ns | u64 | INNER_LOOP(1000)回の合計時間 (ナノ秒) |
| latency_ns | u64 | 1回目のイテレーションのレイテンシ (ナノ秒) |

## Pythonでの読み込み例

```python
import polars as pl

df = pl.read_parquet("results/benchmark_local.parquet")

# サイズ別・アロケータ別の平均時間
df.group_by(["allocator", "pattern", "size_bytes"]).agg(
    pl.col("total_ns").mean().alias("total_mean_ns"),
    pl.col("total_ns").std().alias("total_std_ns"),
    pl.col("latency_ns").mean().alias("latency_mean_ns"),
)
```
