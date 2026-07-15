#!/usr/bin/env python3
"""
vek - Python usage example via ctypes
"""

import ctypes
import sys
import os

# Load library
lib_path = os.path.join(os.path.dirname(__file__), '..', '..', 'libvek.so')
lib = ctypes.CDLL(lib_path)

# Define function signatures
lib.vek_init.argtypes = []
lib.vek_init.restype = ctypes.c_int

lib.vek_backend_name.argtypes = []
lib.vek_backend_name.restype = ctypes.c_char_p

# f32
lib.vek_dot_f32.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
lib.vek_dot_f32.restype = ctypes.c_float

lib.vek_l2sq_f32.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
lib.vek_l2sq_f32.restype = ctypes.c_float

lib.vek_cosine_f32.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
lib.vek_cosine_f32.restype = ctypes.c_float

# int8
lib.vek_dot_i8.argtypes = [ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int8), ctypes.c_size_t]
lib.vek_dot_i8.restype = ctypes.c_int32

lib.vek_l2sq_i8.argtypes = [ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int8), ctypes.c_size_t]
lib.vek_l2sq_i8.restype = ctypes.c_int32

lib.vek_cosine_i8.argtypes = [ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int8), ctypes.c_size_t]
lib.vek_cosine_i8.restype = ctypes.c_float

# uint8
lib.vek_dot_u8.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
lib.vek_dot_u8.restype = ctypes.c_uint32

lib.vek_l2sq_u8.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
lib.vek_l2sq_u8.restype = ctypes.c_uint32

lib.vek_cosine_u8.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
lib.vek_cosine_u8.restype = ctypes.c_float

# binary (1-bit)
lib.vek_dot_b1.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
lib.vek_dot_b1.restype = ctypes.c_int32

lib.vek_hamming_b1.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
lib.vek_hamming_b1.restype = ctypes.c_int32

lib.vek_cosine_b1.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
lib.vek_cosine_b1.restype = ctypes.c_float


def main():
    # Initialize
    lib.vek_init()
    backend = lib.vek_backend_name().decode('utf-8')
    print(f"Active backend: {backend}\n")

    # f32 example
    a_f32 = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
    b_f32 = (ctypes.c_float * 4)(0.5, 1.5, 2.5, 3.5)
    
    dot_f32 = lib.vek_dot_f32(a_f32, b_f32, 4)
    l2sq_f32 = lib.vek_l2sq_f32(a_f32, b_f32, 4)
    cos_f32 = lib.vek_cosine_f32(a_f32, b_f32, 4)
    
    print("f32: a=[1, 2, 3, 4], b=[0.5, 1.5, 2.5, 3.5]")
    print(f"  dot={dot_f32:.4f}, l2sq={l2sq_f32:.4f}, cos={cos_f32:.4f}\n")

    # int8 example
    a_i8 = (ctypes.c_int8 * 8)(1, 2, 3, 4, 5, 6, 7, 8)
    b_i8 = (ctypes.c_int8 * 8)(2, 3, 4, 5, 6, 7, 8, 9)
    
    dot_i8 = lib.vek_dot_i8(a_i8, b_i8, 8)
    l2sq_i8 = lib.vek_l2sq_i8(a_i8, b_i8, 8)
    cos_i8 = lib.vek_cosine_i8(a_i8, b_i8, 8)
    
    print(f"i8: a=[1..8], b=[2..9]")
    print(f"  dot={dot_i8}, l2sq={l2sq_i8}, cos={cos_i8:.4f}\n")

    # uint8 example
    a_u8 = (ctypes.c_uint8 * 8)(10, 20, 30, 40, 50, 60, 70, 80)
    b_u8 = (ctypes.c_uint8 * 8)(15, 25, 35, 45, 55, 65, 75, 85)
    
    dot_u8 = lib.vek_dot_u8(a_u8, b_u8, 8)
    l2sq_u8 = lib.vek_l2sq_u8(a_u8, b_u8, 8)
    cos_u8 = lib.vek_cosine_u8(a_u8, b_u8, 8)
    
    print(f"u8: a=[10..80], b=[15..85]")
    print(f"  dot={dot_u8}, l2sq={l2sq_u8}, cos={cos_u8:.4f}\n")

    # Binary (1-bit) example
    a_b1 = (ctypes.c_uint64 * 2)(0xAAAAAAAAAAAAAAAA, 0x5555555555555555)
    b_b1 = (ctypes.c_uint64 * 2)(0xAAAAAAAAAAAAAAAA, 0x5555555555555555)
    
    dot_b1 = lib.vek_dot_b1(a_b1, b_b1, 128)
    ham_b1 = lib.vek_hamming_b1(a_b1, b_b1, 128)
    cos_b1 = lib.vek_cosine_b1(a_b1, b_b1, 128)
    
    print("b1 (128 bits, identical):")
    print(f"  dot={dot_b1} (matching 1-bits)")
    print(f"  hamming={ham_b1} (differing bits)")
    print(f"  cos={cos_b1:.4f}\n")

    # Binary opposite
    c_b1 = (ctypes.c_uint64 * 2)(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)
    d_b1 = (ctypes.c_uint64 * 2)(0x0000000000000000, 0x0000000000000000)
    
    dot_b1_opp = lib.vek_dot_b1(c_b1, d_b1, 128)
    ham_b1_opp = lib.vek_hamming_b1(c_b1, d_b1, 128)
    cos_b1_opp = lib.vek_cosine_b1(c_b1, d_b1, 128)
    
    print("b1 (128 bits, all 1s vs all 0s):")
    print(f"  dot={dot_b1_opp}")
    print(f"  hamming={ham_b1_opp}")
    print(f"  cos={cos_b1_opp:.4f} (zero norm -> 0)")

    print("\nDone.")


if __name__ == '__main__':
    main()