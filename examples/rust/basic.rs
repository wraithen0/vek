// vek - Rust usage example
// Requires: vek-sys crate (bindgen-generated) or manual extern "C" bindings

use std::os::raw::{c_float, c_int, c_uint, c_int8_t, c_uint8_t, c_int64_t, c_uint64_t};

// Manual FFI bindings (or use vek-sys crate)
extern "C" {
    fn vek_init() -> c_int;
    fn vek_backend_name() -> *const std::os::raw::c_char;
    
    // f32
    fn vek_dot_f32(a: *const c_float, b: *const c_float, n: usize) -> c_float;
    fn vek_l2sq_f32(a: *const c_float, b: *const c_float, n: usize) -> c_float;
    fn vek_cosine_f32(a: *const c_float, b: *const c_float, n: usize) -> c_float;
    
    // int8
    fn vek_dot_i8(a: *const c_int8_t, b: *const c_int8_t, n: usize) -> c_int;
    fn vek_l2sq_i8(a: *const c_int8_t, b: *const c_int8_t, n: usize) -> c_int;
    fn vek_cosine_i8(a: *const c_int8_t, b: *const c_int8_t, n: usize) -> c_float;
    
    // uint8
    fn vek_dot_u8(a: *const c_uint8_t, b: *const c_uint8_t, n: usize) -> c_uint;
    fn vek_l2sq_u8(a: *const c_uint8_t, b: *const c_uint8_t, n: usize) -> c_uint;
    fn vek_cosine_u8(a: *const c_uint8_t, b: *const c_uint8_t, n: usize) -> c_float;
    
    // binary (1-bit)
    fn vek_dot_b1(a: *const c_uint64_t, b: *const c_uint64_t, n: usize) -> c_int;
    fn vek_hamming_b1(a: *const c_uint64_t, b: *const c_uint64_t, n: usize) -> c_int;
    fn vek_cosine_b1(a: *const c_uint64_t, b: *const c_uint64_t, n: usize) -> c_float;
}

fn c_str_to_string(ptr: *const std::os::raw::c_char) -> String {
    unsafe { std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned() }
}

fn main() {
    unsafe {
        // Initialize
        let _ = vek_init();
        let backend = c_str_to_string(vek_backend_name());
        println!("Active backend: {}", backend);
        println!();

        // f32 example
        let a_f32 = [1.0f32, 2.0, 3.0, 4.0];
        let b_f32 = [0.5f32, 1.5, 2.5, 3.5];
        
        let dot_f32 = vek_dot_f32(a_f32.as_ptr(), b_f32.as_ptr(), a_f32.len());
        let l2sq_f32 = vek_l2sq_f32(a_f32.as_ptr(), b_f32.as_ptr(), a_f32.len());
        let cos_f32 = vek_cosine_f32(a_f32.as_ptr(), b_f32.as_ptr(), a_f32.len());
        
        println!("f32: a={:?}, b={:?}", a_f32, b_f32);
        println!("  dot={:.4}, l2sq={:.4}, cos={:.4}", dot_f32, l2sq_f32, cos_f32);
        println!();

        // int8 example
        let a_i8 = [1i8, 2, 3, 4, 5, 6, 7, 8];
        let b_i8 = [2i8, 3, 4, 5, 6, 7, 8, 9];
        
        let dot_i8 = vek_dot_i8(a_i8.as_ptr(), b_i8.as_ptr(), a_i8.len());
        let l2sq_i8 = vek_l2sq_i8(a_i8.as_ptr(), b_i8.as_ptr(), a_i8.len());
        let cos_i8 = vek_cosine_i8(a_i8.as_ptr(), b_i8.as_ptr(), a_i8.len());
        
        println!("i8: a={:?}, b={:?}", a_i8, b_i8);
        println!("  dot={}, l2sq={}, cos={:.4}", dot_i8, l2sq_i8, cos_i8);
        println!();

        // uint8 example
        let a_u8 = [10u8, 20, 30, 40, 50, 60, 70, 80];
        let b_u8 = [15u8, 25, 35, 45, 55, 65, 75, 85];
        
        let dot_u8 = vek_dot_u8(a_u8.as_ptr(), b_u8.as_ptr(), a_u8.len());
        let l2sq_u8 = vek_l2sq_u8(a_u8.as_ptr(), b_u8.as_ptr(), a_u8.len());
        let cos_u8 = vek_cosine_u8(a_u8.as_ptr(), b_u8.as_ptr(), a_u8.len());
        
        println!("u8: a={:?}, b={:?}", a_u8, b_u8);
        println!("  dot={}, l2sq={}, cos={:.4}", dot_u8, l2sq_u8, cos_u8);
        println!();

        // Binary (1-bit) example
        let a_b1 = [0xAAAAAAAAAAAAAAAAu64, 0x5555555555555555u64];
        let b_b1 = [0xAAAAAAAAAAAAAAAAu64, 0x5555555555555555u64];
        
        let dot_b1 = vek_dot_b1(a_b1.as_ptr(), b_b1.as_ptr(), 128);
        let ham_b1 = vek_hamming_b1(a_b1.as_ptr(), b_b1.as_ptr(), 128);
        let cos_b1 = vek_cosine_b1(a_b1.as_ptr(), b_b1.as_ptr(), 128);
        
        println!("b1 (128 bits):");
        println!("  dot={} (matching 1-bits)", dot_b1);
        println!("  hamming={} (differing bits)", ham_b1);
        println!("  cos={:.4}", cos_b1);
    }
}