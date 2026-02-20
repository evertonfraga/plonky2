/// SVE2-vectorized Poseidon2 S-box for aarch64 (Graviton3/4).
///
/// SVE2 provides native 64-bit multiply via svmul_u64 (low 64 bits) and
/// svmulh_u64 (high 64 bits), avoiding the vmull_u32 partial product approach
/// that was slower than scalar on Graviton.
///
/// Goldilocks multiply: a * b mod p, p = 2^64 - 2^32 + 1
/// Using svmul_u64 + svmulh_u64 + reduction.
///
/// On Graviton3/4 (Neoverse-V1/V2), SVE vector length = 256 bits = 4 × u64.
/// WIDTH=12: 3 iterations of 4 elements each.

use core::arch::aarch64::*;
use crate::field::goldilocks_field::GoldilocksField;
use crate::hash::poseidon2::config::WIDTH;

const P: u64 = 0xFFFFFFFF00000001;
const EPSILON: u64 = 0xFFFFFFFF;

/// Goldilocks reduction: given (lo, hi) = a * b split into 64-bit halves,
/// compute a * b mod p.
#[inline(always)]
unsafe fn reduce_mul(lo: svuint64_t, hi: svuint64_t, pg: svbool_t) -> svuint64_t {
    // hi = x_hi; lo = x_lo
    // x_hi_hi = x_hi >> 32
    // x_hi_lo = x_hi & EPSILON
    // result = x_lo - x_hi_hi + x_hi_lo * EPSILON (with carry handling)
    let eps = svdup_n_u64(EPSILON);
    let hi_hi = svlsr_n_u64_x(pg, hi, 32);
    let hi_lo = svand_u64_x(pg, hi, eps);
    // t0 = lo - hi_hi (may underflow)
    let t0 = svsub_u64_x(pg, lo, hi_hi);
    // carry = lo < hi_hi (underflow occurred)
    let carry = svcmplt_u64(pg, lo, hi_hi);
    // if carry: t0 -= EPSILON (equivalent to adding p - EPSILON = 2^64 - 2^32)
    let t0 = svsub_u64_m(carry, t0, eps);
    // t1 = hi_lo * EPSILON
    let t1 = svmul_u64_x(pg, hi_lo, eps);
    // result = t0 + t1 (non-canonical, may exceed p)
    let result = svadd_u64_x(pg, t0, t1);
    // Canonicalize: if result >= p, subtract p
    // Since inputs are < p, result < 2p, so one conditional subtract suffices
    let overflow = svcmpge_u64(pg, result, svdup_n_u64(P));
    svsub_u64_m(overflow, result, svdup_n_u64(P))
}

/// Compute x^2 mod p for a vector of Goldilocks elements.
#[inline(always)]
unsafe fn sq_gl(x: svuint64_t, pg: svbool_t) -> svuint64_t {
    let lo = svmul_u64_x(pg, x, x);
    let hi = svmulh_u64_x(pg, x, x);
    reduce_mul(lo, hi, pg)
}

/// Compute x * y mod p for vectors of Goldilocks elements.
#[inline(always)]
unsafe fn mul_gl(x: svuint64_t, y: svuint64_t, pg: svbool_t) -> svuint64_t {
    let lo = svmul_u64_x(pg, x, y);
    let hi = svmulh_u64_x(pg, x, y);
    reduce_mul(lo, hi, pg)
}

/// Compute x^7 mod p (Poseidon2 S-box) for a vector of Goldilocks elements.
#[inline(always)]
unsafe fn sbox_vec(x: svuint64_t, pg: svbool_t) -> svuint64_t {
    let x2 = sq_gl(x, pg);
    let x4 = sq_gl(x2, pg);
    let x3 = mul_gl(x, x2, pg);
    mul_gl(x3, x4, pg)
}

pub unsafe fn sbox_layer_sve2(state: &mut [GoldilocksField; WIDTH]) {
    // SVE vector length on Graviton3/4 = 256 bits = 4 × u64
    // Process 4 elements at a time, 3 iterations for WIDTH=12
    let pg = svptrue_b64(); // all-true predicate for 64-bit elements

    // We use svld1_u64 with a pointer — process in chunks of svcntd() elements
    // svcntd() = number of 64-bit elements per SVE vector = 4 on Graviton3/4
    let vl = svcntd() as usize; // should be 4

    let mut i = 0;
    while i + vl <= WIDTH {
        let v = svld1_u64(pg, state[i..].as_ptr() as *const u64);
        let v7 = sbox_vec(v, pg);
        svst1_u64(pg, state[i..].as_mut_ptr() as *mut u64, v7);
        i += vl;
    }
    // Handle remaining elements (if WIDTH is not a multiple of vl)
    if i < WIDTH {
        let rem = WIDTH - i;
        let pg_rem = svwhilelt_b64_u64(0, rem as u64);
        let v = svld1_u64(pg_rem, state[i..].as_ptr() as *const u64);
        let v7 = sbox_vec(v, pg_rem);
        svst1_u64(pg_rem, state[i..].as_mut_ptr() as *mut u64, v7);
    }
}
