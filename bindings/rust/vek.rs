/*
 * vek - Rust FFI bindings
 * Zero-dependency extern "C" declarations for the vek C library
 *
 * Usage:
 *   1. Build the C library: `cd vek && make`
 *   2. Link against libvek.a / libvek.so / vek.dll
 *   3. Include this file or copy the declarations
 *
 * No crate dependencies required.
 */

use std::ffi::c_char;
use std::os::raw::c_int;

#[link(name = "vek")]
extern "C" {
    /// Initialize the dispatch table (optional, called lazily on first use)
    pub fn vek_init() -> c_int;

    /// Get the name of the active SIMD backend
    /// Returns: "scalar", "sse2", "avx2", "avx512", "neon", etc.
    pub fn vek_backend_name() -> *const c_char;

    /// Dot product: sum(a[i] * b[i])
    pub fn vek_dot_f32(a: *const f32, b: *const f32, n: usize) -> f32;

    /// Squared L2 distance: sum((a[i] - b[i])^2)
    pub fn vek_l2sq_f32(a: *const f32, b: *const f32, n: usize) -> f32;

    /// Cosine similarity: (a·b) / (||a|| * ||b||)
    /// Returns 0 if either vector has zero norm.
    pub fn vek_cosine_f32(a: *const f32, b: *const f32, n: usize) -> f32;
}

/// Safe wrapper around the raw FFI calls
pub struct Vek;

impl Vek {
    /// Initialize the library (optional)
    pub fn init() -> Result<(), &'static str> {
        let ret = unsafe { vek_init() };
        if ret == 0 {
            Ok(())
        } else {
            Err("vek_init failed")
        }
    }

    /// Get the active backend name
    pub fn backend_name() -> &'static str {
        let ptr = unsafe { vek_backend_name() };
        if ptr.is_null() {
            "unknown"
        } else {
            unsafe { std::ffi::CStr::from_ptr(ptr) }
                .to_str()
                .unwrap_or("invalid_utf8")
        }
    }

    /// Dot product
    pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vector length mismatch");
        unsafe { vek_dot_f32(a.as_ptr(), b.as_ptr(), a.len()) }
    }

    /// Squared L2 distance
    pub fn l2sq_f32(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vector length mismatch");
        unsafe { vek_l2sq_f32(a.as_ptr(), b.as_ptr(), a.len()) }
    }

    /// Cosine similarity
    pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vector length mismatch");
        unsafe { vek_cosine_f32(a.as_ptr(), b.as_ptr(), a.len()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [0.5f32, 1.5, 2.5, 3.5];

        // Should work even without C library linked (will fail at link time if not)
        // This test is mainly for compile-checking the FFI signatures
        let _ = Vek::dot_f32(&a, &b);
        let _ = Vek::l2sq_f32(&a, &b);
        let _ = Vek::cosine_f32(&a, &b);
    }
}