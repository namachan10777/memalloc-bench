use arrow::array::{ArrayRef, StringArray, UInt32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use quanta::Clock;
use rand::{Rng, SeedableRng};
use slab::Slab;
use std::env;
use std::fs::File;
use std::hint::black_box;
use std::mem::MaybeUninit;
use std::sync::Arc;

// 測定パラメータ
const ITERATIONS: u32 = 100;
const BATCH_SIZE: usize = 100;
const INNER_LOOP: usize = 1000; // 1回の測定で何回アロケーションするか

// データサイズ (bytes)
const SIZES: &[usize] = &[
    8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
];

// アクセスパターン
#[derive(Clone, Copy, Debug)]
enum Pattern {
    Immediate, // alloc→deallocをBATCH_SIZE回繰り返す
    Lifo,      // BATCH_SIZE個alloc → 逆順dealloc
    Fifo,      // BATCH_SIZE個alloc → 順番dealloc
    Random,    // ランダムにalloc/deallocを混ぜる
}

impl Pattern {
    fn as_str(&self) -> &'static str {
        match self {
            Pattern::Immediate => "immediate",
            Pattern::Lifo => "lifo",
            Pattern::Fifo => "fifo",
            Pattern::Random => "random",
        }
    }

    fn all() -> &'static [Pattern] {
        &[
            Pattern::Immediate,
            Pattern::Lifo,
            Pattern::Fifo,
            Pattern::Random,
        ]
    }
}

// アロケータ種別
#[derive(Clone, Copy, Debug)]
enum Allocator {
    Box,
    SlabCold,
    SlabWarm,
}

impl Allocator {
    fn as_str(&self) -> &'static str {
        match self {
            Allocator::Box => "box",
            Allocator::SlabCold => "slab_cold",
            Allocator::SlabWarm => "slab_warm",
        }
    }

    fn all() -> &'static [Allocator] {
        &[Allocator::Box, Allocator::SlabCold, Allocator::SlabWarm]
    }
}

// 測定結果
struct BenchResult {
    platform: String,
    allocator: String,
    pattern: String,
    size_bytes: u32,
    iteration: u32,
    total_ns: u64,   // INNER_LOOP回の合計時間
    latency_ns: u64, // 1回目のレイテンシ
}

// 静的サイズのデータ構造（マクロで各サイズを生成）
// MaybeUninitを使ってゼロクリアのコストを排除
macro_rules! define_data_types {
    ($($name:ident, $size:expr);* $(;)?) => {
        $(
            #[repr(align(8))]
            struct $name {
                _data: MaybeUninit<[u8; $size]>,
            }

            impl $name {
                #[inline(always)]
                fn new() -> Self {
                    Self { _data: MaybeUninit::uninit() }
                }
            }
        )*
    };
}

define_data_types! {
    Data8, 8;
    Data12, 12;
    Data16, 16;
    Data24, 24;
    Data32, 32;
    Data48, 48;
    Data64, 64;
    Data96, 96;
    Data128, 128;
    Data192, 192;
    Data256, 256;
    Data384, 384;
    Data512, 512;
    Data768, 768;
    Data1024, 1024;
    Data1536, 1536;
    Data2048, 2048;
    Data3072, 3072;
    Data4096, 4096;
}

// ベンチマーク結果 (total_ns, latency_ns)
struct BenchTiming {
    total_ns: u64,
    latency_ns: u64,
}

// ベンチマーク関数をマクロで生成

// Immediate: alloc→deallocをBATCH_SIZE回繰り返す
macro_rules! bench_immediate_box {
    ($clock:expr, $data_type:ty) => {{
        // 1回目のレイテンシを計測
        let lat_start = $clock.raw();
        for _ in 0..BATCH_SIZE {
            let b = Box::new(<$data_type>::new());
            drop(black_box(b));
        }
        let lat_end = $clock.raw();
        let latency_ns = $clock.delta(lat_start, lat_end).as_nanos() as u64;

        // 残りのループ
        let start = $clock.raw();
        for _ in 1..INNER_LOOP {
            for _ in 0..BATCH_SIZE {
                let b = Box::new(<$data_type>::new());
                drop(black_box(b));
            }
        }
        let end = $clock.raw();
        let rest_ns = $clock.delta(start, end).as_nanos() as u64;

        BenchTiming {
            total_ns: latency_ns + rest_ns,
            latency_ns,
        }
    }};
}

macro_rules! bench_immediate_slab_cold {
    ($clock:expr, $data_type:ty) => {{
        // 1回目のレイテンシを計測
        let lat_start = $clock.raw();
        for _ in 0..BATCH_SIZE {
            let mut slab: Slab<$data_type> = Slab::new();
            let key = slab.insert(<$data_type>::new());
            let _ = black_box(slab.remove(key));
        }
        let lat_end = $clock.raw();
        let latency_ns = $clock.delta(lat_start, lat_end).as_nanos() as u64;

        // 残りのループ
        let start = $clock.raw();
        for _ in 1..INNER_LOOP {
            for _ in 0..BATCH_SIZE {
                let mut slab: Slab<$data_type> = Slab::new();
                let key = slab.insert(<$data_type>::new());
                let _ = black_box(slab.remove(key));
            }
        }
        let end = $clock.raw();
        let rest_ns = $clock.delta(start, end).as_nanos() as u64;

        BenchTiming {
            total_ns: latency_ns + rest_ns,
            latency_ns,
        }
    }};
}

macro_rules! bench_immediate_slab_warm {
    ($clock:expr, $data_type:ty) => {{
        // 1回目のレイテンシを計測
        let lat_start = $clock.raw();
        {
            let mut slab: Slab<$data_type> = Slab::with_capacity(1);
            for _ in 0..BATCH_SIZE {
                let key = slab.insert(<$data_type>::new());
                let _ = black_box(slab.remove(key));
            }
        }
        let lat_end = $clock.raw();
        let latency_ns = $clock.delta(lat_start, lat_end).as_nanos() as u64;

        // 残りのループ
        let start = $clock.raw();
        for _ in 1..INNER_LOOP {
            let mut slab: Slab<$data_type> = Slab::with_capacity(1);
            for _ in 0..BATCH_SIZE {
                let key = slab.insert(<$data_type>::new());
                let _ = black_box(slab.remove(key));
            }
        }
        let end = $clock.raw();
        let rest_ns = $clock.delta(start, end).as_nanos() as u64;

        BenchTiming {
            total_ns: latency_ns + rest_ns,
            latency_ns,
        }
    }};
}

macro_rules! bench_lifo_box {
    ($clock:expr, $data_type:ty) => {{
        // 1回目のレイテンシを計測
        let lat_start = $clock.raw();
        {
            let mut boxes: Vec<Box<$data_type>> = Vec::with_capacity(BATCH_SIZE);
            for _ in 0..BATCH_SIZE {
                boxes.push(Box::new(<$data_type>::new()));
            }
            while let Some(b) = boxes.pop() {
                drop(black_box(b));
            }
        }
        let lat_end = $clock.raw();
        let latency_ns = $clock.delta(lat_start, lat_end).as_nanos() as u64;

        // 残りのループ
        let start = $clock.raw();
        for _ in 1..INNER_LOOP {
            let mut boxes: Vec<Box<$data_type>> = Vec::with_capacity(BATCH_SIZE);
            for _ in 0..BATCH_SIZE {
                boxes.push(Box::new(<$data_type>::new()));
            }
            while let Some(b) = boxes.pop() {
                drop(black_box(b));
            }
        }
        let end = $clock.raw();
        let rest_ns = $clock.delta(start, end).as_nanos() as u64;

        BenchTiming {
            total_ns: latency_ns + rest_ns,
            latency_ns,
        }
    }};
}

macro_rules! bench_lifo_slab_cold {
    ($clock:expr, $data_type:ty) => {{
        // 1回目のレイテンシを計測
        let lat_start = $clock.raw();
        {
            let mut slab: Slab<$data_type> = Slab::new();
            let mut keys: Vec<usize> = Vec::with_capacity(BATCH_SIZE);
            for _ in 0..BATCH_SIZE {
                keys.push(slab.insert(<$data_type>::new()));
            }
            while let Some(key) = keys.pop() {
                let _ = black_box(slab.remove(key));
            }
        }
        let lat_end = $clock.raw();
        let latency_ns = $clock.delta(lat_start, lat_end).as_nanos() as u64;

        // 残りのループ
        let start = $clock.raw();
        for _ in 1..INNER_LOOP {
            let mut slab: Slab<$data_type> = Slab::new();
            let mut keys: Vec<usize> = Vec::with_capacity(BATCH_SIZE);
            for _ in 0..BATCH_SIZE {
                keys.push(slab.insert(<$data_type>::new()));
            }
            while let Some(key) = keys.pop() {
                let _ = black_box(slab.remove(key));
            }
        }
        let end = $clock.raw();
        let rest_ns = $clock.delta(start, end).as_nanos() as u64;

        BenchTiming {
            total_ns: latency_ns + rest_ns,
            latency_ns,
        }
    }};
}

macro_rules! bench_lifo_slab_warm {
    ($clock:expr, $data_type:ty) => {{
        // 1回目のレイテンシを計測
        let lat_start = $clock.raw();
        {
            let mut slab: Slab<$data_type> = Slab::with_capacity(BATCH_SIZE);
            let mut keys: Vec<usize> = Vec::with_capacity(BATCH_SIZE);
            for _ in 0..BATCH_SIZE {
                keys.push(slab.insert(<$data_type>::new()));
            }
            while let Some(key) = keys.pop() {
                let _ = black_box(slab.remove(key));
            }
        }
        let lat_end = $clock.raw();
        let latency_ns = $clock.delta(lat_start, lat_end).as_nanos() as u64;

        // 残りのループ
        let start = $clock.raw();
        for _ in 1..INNER_LOOP {
            let mut slab: Slab<$data_type> = Slab::with_capacity(BATCH_SIZE);
            let mut keys: Vec<usize> = Vec::with_capacity(BATCH_SIZE);
            for _ in 0..BATCH_SIZE {
                keys.push(slab.insert(<$data_type>::new()));
            }
            while let Some(key) = keys.pop() {
                let _ = black_box(slab.remove(key));
            }
        }
        let end = $clock.raw();
        let rest_ns = $clock.delta(start, end).as_nanos() as u64;

        BenchTiming {
            total_ns: latency_ns + rest_ns,
            latency_ns,
        }
    }};
}

macro_rules! bench_fifo_box {
    ($clock:expr, $data_type:ty) => {{
        // 1回目のレイテンシを計測
        let lat_start = $clock.raw();
        {
            let mut boxes: Vec<Box<$data_type>> = Vec::with_capacity(BATCH_SIZE);
            for _ in 0..BATCH_SIZE {
                boxes.push(Box::new(<$data_type>::new()));
            }
            for b in boxes.into_iter() {
                drop(black_box(b));
            }
        }
        let lat_end = $clock.raw();
        let latency_ns = $clock.delta(lat_start, lat_end).as_nanos() as u64;

        // 残りのループ
        let start = $clock.raw();
        for _ in 1..INNER_LOOP {
            let mut boxes: Vec<Box<$data_type>> = Vec::with_capacity(BATCH_SIZE);
            for _ in 0..BATCH_SIZE {
                boxes.push(Box::new(<$data_type>::new()));
            }
            for b in boxes.into_iter() {
                drop(black_box(b));
            }
        }
        let end = $clock.raw();
        let rest_ns = $clock.delta(start, end).as_nanos() as u64;

        BenchTiming {
            total_ns: latency_ns + rest_ns,
            latency_ns,
        }
    }};
}

macro_rules! bench_fifo_slab_cold {
    ($clock:expr, $data_type:ty) => {{
        // 1回目のレイテンシを計測
        let lat_start = $clock.raw();
        {
            let mut slab: Slab<$data_type> = Slab::new();
            let mut keys: Vec<usize> = Vec::with_capacity(BATCH_SIZE);
            for _ in 0..BATCH_SIZE {
                keys.push(slab.insert(<$data_type>::new()));
            }
            for key in keys.into_iter() {
                let _ = black_box(slab.remove(key));
            }
        }
        let lat_end = $clock.raw();
        let latency_ns = $clock.delta(lat_start, lat_end).as_nanos() as u64;

        // 残りのループ
        let start = $clock.raw();
        for _ in 1..INNER_LOOP {
            let mut slab: Slab<$data_type> = Slab::new();
            let mut keys: Vec<usize> = Vec::with_capacity(BATCH_SIZE);
            for _ in 0..BATCH_SIZE {
                keys.push(slab.insert(<$data_type>::new()));
            }
            for key in keys.into_iter() {
                let _ = black_box(slab.remove(key));
            }
        }
        let end = $clock.raw();
        let rest_ns = $clock.delta(start, end).as_nanos() as u64;

        BenchTiming {
            total_ns: latency_ns + rest_ns,
            latency_ns,
        }
    }};
}

macro_rules! bench_fifo_slab_warm {
    ($clock:expr, $data_type:ty) => {{
        // 1回目のレイテンシを計測
        let lat_start = $clock.raw();
        {
            let mut slab: Slab<$data_type> = Slab::with_capacity(BATCH_SIZE);
            let mut keys: Vec<usize> = Vec::with_capacity(BATCH_SIZE);
            for _ in 0..BATCH_SIZE {
                keys.push(slab.insert(<$data_type>::new()));
            }
            for key in keys.into_iter() {
                let _ = black_box(slab.remove(key));
            }
        }
        let lat_end = $clock.raw();
        let latency_ns = $clock.delta(lat_start, lat_end).as_nanos() as u64;

        // 残りのループ
        let start = $clock.raw();
        for _ in 1..INNER_LOOP {
            let mut slab: Slab<$data_type> = Slab::with_capacity(BATCH_SIZE);
            let mut keys: Vec<usize> = Vec::with_capacity(BATCH_SIZE);
            for _ in 0..BATCH_SIZE {
                keys.push(slab.insert(<$data_type>::new()));
            }
            for key in keys.into_iter() {
                let _ = black_box(slab.remove(key));
            }
        }
        let end = $clock.raw();
        let rest_ns = $clock.delta(start, end).as_nanos() as u64;

        BenchTiming {
            total_ns: latency_ns + rest_ns,
            latency_ns,
        }
    }};
}

// Random: ランダムにalloc/deallocを混ぜる
// スロットをランダムに選んでalloc済みならdealloc、空ならalloc
macro_rules! bench_random_box {
    ($clock:expr, $data_type:ty, $rng:expr) => {{
        // 1回目のレイテンシを計測
        let lat_start = $clock.raw();
        {
            let mut slots: Vec<Option<Box<$data_type>>> = (0..BATCH_SIZE).map(|_| None).collect();
            for _ in 0..(BATCH_SIZE * 2) {
                let idx = $rng.gen_range(0..BATCH_SIZE);
                if slots[idx].is_some() {
                    drop(black_box(slots[idx].take()));
                } else {
                    slots[idx] = Some(Box::new(<$data_type>::new()));
                    black_box(&slots[idx]);
                }
            }
            // 残りを解放
            for slot in slots.into_iter().flatten() {
                drop(black_box(slot));
            }
        }
        let lat_end = $clock.raw();
        let latency_ns = $clock.delta(lat_start, lat_end).as_nanos() as u64;

        // 残りのループ
        let start = $clock.raw();
        for _ in 1..INNER_LOOP {
            let mut slots: Vec<Option<Box<$data_type>>> = (0..BATCH_SIZE).map(|_| None).collect();
            for _ in 0..(BATCH_SIZE * 2) {
                let idx = $rng.gen_range(0..BATCH_SIZE);
                if slots[idx].is_some() {
                    drop(black_box(slots[idx].take()));
                } else {
                    slots[idx] = Some(Box::new(<$data_type>::new()));
                    black_box(&slots[idx]);
                }
            }
            for slot in slots.into_iter().flatten() {
                drop(black_box(slot));
            }
        }
        let end = $clock.raw();
        let rest_ns = $clock.delta(start, end).as_nanos() as u64;

        BenchTiming {
            total_ns: latency_ns + rest_ns,
            latency_ns,
        }
    }};
}

macro_rules! bench_random_slab_cold {
    ($clock:expr, $data_type:ty, $rng:expr) => {{
        // 1回目のレイテンシを計測
        let lat_start = $clock.raw();
        {
            let mut slab: Slab<$data_type> = Slab::new();
            let mut slots: Vec<Option<usize>> = (0..BATCH_SIZE).map(|_| None).collect();
            for _ in 0..(BATCH_SIZE * 2) {
                let idx = $rng.gen_range(0..BATCH_SIZE);
                if let Some(key) = slots[idx].take() {
                    let _ = black_box(slab.remove(key));
                } else {
                    let key = slab.insert(<$data_type>::new());
                    slots[idx] = Some(key);
                    black_box(key);
                }
            }
            // 残りを解放
            for key in slots.into_iter().flatten() {
                let _ = black_box(slab.remove(key));
            }
        }
        let lat_end = $clock.raw();
        let latency_ns = $clock.delta(lat_start, lat_end).as_nanos() as u64;

        // 残りのループ
        let start = $clock.raw();
        for _ in 1..INNER_LOOP {
            let mut slab: Slab<$data_type> = Slab::new();
            let mut slots: Vec<Option<usize>> = (0..BATCH_SIZE).map(|_| None).collect();
            for _ in 0..(BATCH_SIZE * 2) {
                let idx = $rng.gen_range(0..BATCH_SIZE);
                if let Some(key) = slots[idx].take() {
                    let _ = black_box(slab.remove(key));
                } else {
                    let key = slab.insert(<$data_type>::new());
                    slots[idx] = Some(key);
                    black_box(key);
                }
            }
            for key in slots.into_iter().flatten() {
                let _ = black_box(slab.remove(key));
            }
        }
        let end = $clock.raw();
        let rest_ns = $clock.delta(start, end).as_nanos() as u64;

        BenchTiming {
            total_ns: latency_ns + rest_ns,
            latency_ns,
        }
    }};
}

macro_rules! bench_random_slab_warm {
    ($clock:expr, $data_type:ty, $rng:expr) => {{
        // 1回目のレイテンシを計測
        let lat_start = $clock.raw();
        {
            let mut slab: Slab<$data_type> = Slab::with_capacity(BATCH_SIZE);
            let mut slots: Vec<Option<usize>> = (0..BATCH_SIZE).map(|_| None).collect();
            for _ in 0..(BATCH_SIZE * 2) {
                let idx = $rng.gen_range(0..BATCH_SIZE);
                if let Some(key) = slots[idx].take() {
                    let _ = black_box(slab.remove(key));
                } else {
                    let key = slab.insert(<$data_type>::new());
                    slots[idx] = Some(key);
                    black_box(key);
                }
            }
            // 残りを解放
            for key in slots.into_iter().flatten() {
                let _ = black_box(slab.remove(key));
            }
        }
        let lat_end = $clock.raw();
        let latency_ns = $clock.delta(lat_start, lat_end).as_nanos() as u64;

        // 残りのループ
        let start = $clock.raw();
        for _ in 1..INNER_LOOP {
            let mut slab: Slab<$data_type> = Slab::with_capacity(BATCH_SIZE);
            let mut slots: Vec<Option<usize>> = (0..BATCH_SIZE).map(|_| None).collect();
            for _ in 0..(BATCH_SIZE * 2) {
                let idx = $rng.gen_range(0..BATCH_SIZE);
                if let Some(key) = slots[idx].take() {
                    let _ = black_box(slab.remove(key));
                } else {
                    let key = slab.insert(<$data_type>::new());
                    slots[idx] = Some(key);
                    black_box(key);
                }
            }
            for key in slots.into_iter().flatten() {
                let _ = black_box(slab.remove(key));
            }
        }
        let end = $clock.raw();
        let rest_ns = $clock.delta(start, end).as_nanos() as u64;

        BenchTiming {
            total_ns: latency_ns + rest_ns,
            latency_ns,
        }
    }};
}

// サイズに応じたベンチマーク実行
macro_rules! run_bench_for_size {
    ($clock:expr, $allocator:expr, $pattern:expr, $size:expr, $rng:expr, $($sz:expr => $data_type:ty),* $(,)?) => {
        match $size {
            $(
                $sz => match ($allocator, $pattern) {
                    (Allocator::Box, Pattern::Immediate) => bench_immediate_box!($clock, $data_type),
                    (Allocator::SlabCold, Pattern::Immediate) => bench_immediate_slab_cold!($clock, $data_type),
                    (Allocator::SlabWarm, Pattern::Immediate) => bench_immediate_slab_warm!($clock, $data_type),
                    (Allocator::Box, Pattern::Lifo) => bench_lifo_box!($clock, $data_type),
                    (Allocator::SlabCold, Pattern::Lifo) => bench_lifo_slab_cold!($clock, $data_type),
                    (Allocator::SlabWarm, Pattern::Lifo) => bench_lifo_slab_warm!($clock, $data_type),
                    (Allocator::Box, Pattern::Fifo) => bench_fifo_box!($clock, $data_type),
                    (Allocator::SlabCold, Pattern::Fifo) => bench_fifo_slab_cold!($clock, $data_type),
                    (Allocator::SlabWarm, Pattern::Fifo) => bench_fifo_slab_warm!($clock, $data_type),
                    (Allocator::Box, Pattern::Random) => bench_random_box!($clock, $data_type, $rng),
                    (Allocator::SlabCold, Pattern::Random) => bench_random_slab_cold!($clock, $data_type, $rng),
                    (Allocator::SlabWarm, Pattern::Random) => bench_random_slab_warm!($clock, $data_type, $rng),
                },
            )*
            _ => panic!("Unsupported size: {}", $size),
        }
    };
}

fn run_benchmark(
    clock: &Clock,
    allocator: Allocator,
    pattern: Pattern,
    size: usize,
    rng: &mut rand::rngs::StdRng,
) -> BenchTiming {
    run_bench_for_size!(
        clock, allocator, pattern, size, rng,
        8 => Data8,
        12 => Data12,
        16 => Data16,
        24 => Data24,
        32 => Data32,
        48 => Data48,
        64 => Data64,
        96 => Data96,
        128 => Data128,
        192 => Data192,
        256 => Data256,
        384 => Data384,
        512 => Data512,
        768 => Data768,
        1024 => Data1024,
        1536 => Data1536,
        2048 => Data2048,
        3072 => Data3072,
        4096 => Data4096,
    )
}

fn warmup(clock: &Clock) {
    // CPU/タイマーのウォームアップ
    for _ in 0..10000 {
        let _ = black_box(clock.raw());
        let b = Box::new(Data64::new());
        drop(black_box(b));
    }
}

fn write_parquet(results: &[BenchResult], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let schema = Schema::new(vec![
        Field::new("platform", DataType::Utf8, false),
        Field::new("allocator", DataType::Utf8, false),
        Field::new("pattern", DataType::Utf8, false),
        Field::new("size_bytes", DataType::UInt32, false),
        Field::new("iteration", DataType::UInt32, false),
        Field::new("total_ns", DataType::UInt64, false),
        Field::new("latency_ns", DataType::UInt64, false),
    ]);

    let platforms: Vec<&str> = results.iter().map(|r| r.platform.as_str()).collect();
    let allocators: Vec<&str> = results.iter().map(|r| r.allocator.as_str()).collect();
    let patterns: Vec<&str> = results.iter().map(|r| r.pattern.as_str()).collect();
    let sizes: Vec<u32> = results.iter().map(|r| r.size_bytes).collect();
    let iterations: Vec<u32> = results.iter().map(|r| r.iteration).collect();
    let total: Vec<u64> = results.iter().map(|r| r.total_ns).collect();
    let latency: Vec<u64> = results.iter().map(|r| r.latency_ns).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(platforms)) as ArrayRef,
            Arc::new(StringArray::from(allocators)) as ArrayRef,
            Arc::new(StringArray::from(patterns)) as ArrayRef,
            Arc::new(UInt32Array::from(sizes)) as ArrayRef,
            Arc::new(UInt32Array::from(iterations)) as ArrayRef,
            Arc::new(UInt64Array::from(total)) as ArrayRef,
            Arc::new(UInt64Array::from(latency)) as ArrayRef,
        ],
    )?;

    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}

fn print_usage(program: &str) {
    eprintln!("Usage: {} <platform>", program);
    eprintln!();
    eprintln!("Arguments:");
    eprintln!("  <platform>  Platform name (e.g., 'local', 'hpc-cluster', 'aws-c5')");
    eprintln!();
    eprintln!("Example:");
    eprintln!("  {} local", program);
    eprintln!("  {} hpc-xeon-8280", program);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    let platform = &args[1];

    if platform == "-h" || platform == "--help" {
        print_usage(&args[0]);
        return Ok(());
    }

    println!("Platform: {}", platform);
    println!("Inner loop: {} iterations per measurement", INNER_LOOP);

    let clock = Clock::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    println!("Warming up...");
    warmup(&clock);

    let total = Allocator::all().len() * Pattern::all().len() * SIZES.len();
    let mut current = 0;

    let mut results = Vec::with_capacity(total * ITERATIONS as usize);

    for &allocator in Allocator::all() {
        for &pattern in Pattern::all() {
            for &size in SIZES {
                current += 1;
                println!(
                    "[{}/{}] {} / {} / {} bytes",
                    current,
                    total,
                    allocator.as_str(),
                    pattern.as_str(),
                    size
                );

                for iteration in 0..ITERATIONS {
                    let timing = run_benchmark(&clock, allocator, pattern, size, &mut rng);
                    results.push(BenchResult {
                        platform: platform.clone(),
                        allocator: allocator.as_str().to_string(),
                        pattern: pattern.as_str().to_string(),
                        size_bytes: size as u32,
                        iteration,
                        total_ns: timing.total_ns,
                        latency_ns: timing.latency_ns,
                    });
                }
            }
        }
    }

    std::fs::create_dir_all("results")?;
    let output_path = format!("results/benchmark_{}.parquet", platform);
    println!("Writing results to {}...", output_path);
    write_parquet(&results, &output_path)?;
    println!("Done! {} records written.", results.len());

    Ok(())
}
