use std::{
    cell::RefCell,
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
    rc::{Rc, Weak},
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

struct BufferPoolImpl<A: BufferAllocator> {
    allocator: A,
    stack: RefCell<Vec<A::Buffer>>,
}

pub struct Lease<A: BufferAllocator> {
    buf: ManuallyDrop<A::Buffer>,
    pool: Weak<BufferPoolImpl<A>>,
}

pub struct BufferPool<A: BufferAllocator> {
    pool: Rc<BufferPoolImpl<A>>,
}

impl<A: BufferAllocator> BufferPool<A> {
    pub fn new(allocator: A) -> Self {
        BufferPool {
            pool: Rc::new(BufferPoolImpl {
                allocator,
                stack: RefCell::new(Vec::new()),
            }),
        }
    }

    pub fn lease(&self) -> Result<Lease<A>, A::Error> {
        if let Some(buf) = self.pool.stack.borrow_mut().pop() {
            Ok(Lease {
                buf: ManuallyDrop::new(buf),
                pool: Rc::downgrade(&self.pool),
            })
        } else {
            let buf = self.pool.allocator.allocate()?;
            Ok(Lease {
                buf: ManuallyDrop::new(buf),
                pool: Rc::downgrade(&self.pool),
            })
        }
    }
}

impl<A: BufferAllocator> Drop for Lease<A> {
    fn drop(&mut self) {
        unsafe {
            let buf = ManuallyDrop::take(&mut self.buf);
            if let Some(pool) = self.pool.upgrade() {
                pool.stack.borrow_mut().push(buf);
            } else {
                drop(buf);
            }
        }
    }
}

impl<A: BufferAllocator> Buffer for Lease<A>
where
    A::Buffer: Buffer,
{
    unsafe fn ptr(&self) -> *mut u8 {
        unsafe { self.buf.ptr() }
    }

    fn reset(&mut self) {
        self.buf.reset();
    }

    unsafe fn size(&self) -> usize {
        unsafe { self.buf.size() }
    }
}

impl<A: BufferAllocator> Deref for Lease<A> {
    type Target = A::Buffer;

    fn deref(&self) -> &Self::Target {
        &*self.buf
    }
}

impl<A: BufferAllocator> DerefMut for Lease<A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.buf
    }
}
