use std::{
    cell::UnsafeCell,
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

pub trait Buffer {
    unsafe fn ptr(&self) -> *mut u8;
    unsafe fn size(&self) -> usize;
    fn reset(&mut self) {}
}

impl<const N: usize> Buffer for [u8; N] {
    unsafe fn ptr(&self) -> *mut u8 {
        self.as_ptr() as *mut u8
    }

    unsafe fn size(&self) -> usize {
        N
    }

    fn reset(&mut self) {}
}

impl Buffer for [u8] {
    unsafe fn ptr(&self) -> *mut u8 {
        self.as_ptr() as *mut u8
    }

    unsafe fn size(&self) -> usize {
        self.len()
    }

    fn reset(&mut self) {}
}

impl Buffer for Vec<u8> {
    unsafe fn ptr(&self) -> *mut u8 {
        self.as_ptr() as *mut _
    }

    unsafe fn size(&self) -> usize {
        self.len()
    }

    fn reset(&mut self) {
        self.clear();
    }
}

impl Buffer for Box<[u8]> {
    unsafe fn ptr(&self) -> *mut u8 {
        self.as_ptr() as *mut _
    }

    unsafe fn size(&self) -> usize {
        self.len() as _
    }

    fn reset(&mut self) {}
}

pub struct Resize<B> {
    buf: B,
    len: usize,
}

impl<B: Buffer> Buffer for Resize<B> {
    unsafe fn ptr(&self) -> *mut u8 {
        unsafe { self.buf.ptr() }
    }

    unsafe fn size(&self) -> usize {
        self.len
    }

    fn reset(&mut self) {
        self.buf.reset();
    }
}

impl<B: Buffer> Resize<B> {
    pub fn new(buf: B) -> Self {
        Self {
            len: unsafe { buf.size() },
            buf,
        }
    }

    pub fn new_with_size(buf: B, len: usize) -> Self {
        Self {
            len: unsafe { buf.size().min(len) },
            buf,
        }
    }

    pub fn resize(&mut self, len: usize) {
        self.len = self.len.min(len);
    }
}

pub trait BufferAllocator {
    type Buffer;
    type Error;

    fn allocate(&self) -> Result<Self::Buffer, Self::Error>;
}

/// エントリ: バッファを常に保持し、次の空きインデックスも持つ
struct Entry<T> {
    buffer: T,
    /// 空きの場合、次の空きスロットのインデックス（usize::MAXで終端）
    next_free: usize,
}

const FREE_LIST_END: usize = usize::MAX;

/// 内部プール状態（UnsafeCellで包む - シングルスレッド前提）
struct PoolInner<A: BufferAllocator> {
    allocator: A,
    entries: Vec<Entry<A::Buffer>>,
    /// フリーリストの先頭（usize::MAXで空）
    free_head: usize,
}

pub struct BufferPool<A: BufferAllocator> {
    inner: UnsafeCell<PoolInner<A>>,
}

pub struct Lease<'a, A: BufferAllocator> {
    /// プールへの参照（ライフタイムで生存を保証）
    pool: &'a BufferPool<A>,
    /// このリースが持っているエントリのインデックス
    index: usize,
    /// PhantomDataで不変性を保証
    _marker: PhantomData<&'a mut A::Buffer>,
}

impl<A: BufferAllocator> BufferPool<A> {
    pub fn new(allocator: A) -> Self {
        BufferPool {
            inner: UnsafeCell::new(PoolInner {
                allocator,
                entries: Vec::new(),
                free_head: FREE_LIST_END,
            }),
        }
    }

    #[inline]
    pub fn lease(&self) -> Result<Lease<'_, A>, A::Error> {
        // SAFETY: シングルスレッド前提、&self経由でのみアクセス
        let inner = unsafe { &mut *self.inner.get() };

        let index = if inner.free_head != FREE_LIST_END {
            // 空きスロットがある - フリーリストから取得
            let idx = inner.free_head;
            inner.free_head = inner.entries[idx].next_free;
            inner.entries[idx].next_free = FREE_LIST_END; // 使用中マーク
            idx
        } else {
            // 新規アロケーション
            let buf = inner.allocator.allocate()?;
            let idx = inner.entries.len();
            inner.entries.push(Entry {
                buffer: buf,
                next_free: FREE_LIST_END,
            });
            idx
        };

        Ok(Lease {
            pool: self,
            index,
            _marker: PhantomData,
        })
    }
}

impl<A: BufferAllocator> Drop for Lease<'_, A> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: ライフタイムで生存保証、シングルスレッド前提
        let inner = unsafe { &mut *self.pool.inner.get() };

        // フリーリストに追加（バッファは保持したまま）
        inner.entries[self.index].next_free = inner.free_head;
        inner.free_head = self.index;
    }
}

impl<A: BufferAllocator> Deref for Lease<'_, A> {
    type Target = A::Buffer;

    #[inline]
    fn deref(&self) -> &Self::Target {
        let inner = unsafe { &*self.pool.inner.get() };
        &inner.entries[self.index].buffer
    }
}

impl<A: BufferAllocator> DerefMut for Lease<'_, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        let inner = unsafe { &mut *self.pool.inner.get() };
        &mut inner.entries[self.index].buffer
    }
}

impl<A: BufferAllocator> Buffer for Lease<'_, A>
where
    A::Buffer: Buffer,
{
    unsafe fn ptr(&self) -> *mut u8 {
        unsafe { (**self).ptr() }
    }

    fn reset(&mut self) {
        (**self).reset();
    }

    unsafe fn size(&self) -> usize {
        unsafe { (**self).size() }
    }
}
