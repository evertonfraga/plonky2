/// NEON-vectorized Poseidon2 operations for aarch64 (Graviton3/4).
///
/// Provides:
/// - external_linear_layer_neon: vectorized MDS matrix multiply
/// - sbox_layer_neon: vectorized x^7 S-box using vmull_u32 partial products
///
/// The scalar sbox_layer_full in poseidon_goldilocks_neon.rs has a comment
/// "insane latency on M1" — but Graviton3/4 (Neoverse-V1/V2) have different
/// microarchitecture. This implementation uses NEON vmull_u32 for 2-wide
/// Goldilocks multiply, processing pairs of state elements simultaneously.

use core::arch::aarch64::*;
use core::mem::transmute;

use crate::field::goldilocks_field::GoldilocksField;
use crate::hash::poseidon2::config::WIDTH;

const EPSILON: u64 = 0xffffffff;

// ── Vectorized Goldilocks multiply (2-wide NEON) ──────────────────────────────

#[inline(always)]
unsafe fn hi32(x: uint64x2_t) -> uint64x2_t { vshrq_n_u64::<32>(x) }

#[inline(always)]
unsafe fn mul_lo32(x: uint64x2_t, y: uint64x2_t) -> uint64x2_t {
    vmull_u32(vmovn_u64(x), vmovn_u64(y))
}

#[inline(always)]
unsafe fn mul64_64(x: uint64x2_t, y: uint64x2_t) -> (uint64x2_t, uint64x2_t) {
    let x_hi = hi32(x); let y_hi = hi32(y);
    let mul_ll = mul_lo32(x, y);
    let mul_lh = mul_lo32(x, y_hi);
    let mul_hl = mul_lo32(x_hi, y);
    let mul_hh = mul_lo32(x_hi, y_hi);
    let eps = vdupq_n_u64(EPSILON);
    let t0 = vaddq_u64(mul_hl, vshrq_n_u64::<32>(mul_ll));
    let t1 = vaddq_u64(mul_lh, vandq_u64(t0, eps));
    let res_hi = vaddq_u64(vaddq_u64(mul_hh, vshrq_n_u64::<32>(t0)), vshrq_n_u64::<32>(t1));
    let res_lo = vorrq_u64(vandq_u64(mul_ll, eps), vshlq_n_u64::<32>(t1));
    (res_hi, res_lo)
}

#[inline(always)]
unsafe fn square64(x: uint64x2_t) -> (uint64x2_t, uint64x2_t) {
    let x_hi = hi32(x);
    let mul_ll = mul_lo32(x, x);
    let mul_lh = mul_lo32(x, x_hi);
    let mul_hh = mul_lo32(x_hi, x_hi);
    let t0 = vaddq_u64(mul_lh, vshrq_n_u64::<33>(mul_ll));
    let res_hi = vaddq_u64(mul_hh, vshrq_n_u64::<31>(t0));
    let res_lo = vaddq_u64(mul_ll, vshlq_n_u64::<33>(mul_lh));
    (res_hi, res_lo)
}

#[inline(always)]
unsafe fn add_small(x: uint64x2_t, y: uint64x2_t) -> uint64x2_t {
    let s = vaddq_u64(x, y);
    vaddq_u64(s, vandq_u64(vcltq_u64(s, x), vdupq_n_u64(EPSILON)))
}

#[inline(always)]
unsafe fn sub_small(x: uint64x2_t, y: uint64x2_t) -> uint64x2_t {
    let d = vsubq_u64(x, y);
    vsubq_u64(d, vandq_u64(vcltq_u64(x, y), vdupq_n_u64(EPSILON)))
}

#[inline(always)]
unsafe fn reduce128(hi: uint64x2_t, lo: uint64x2_t) -> uint64x2_t {
    let hi_hi = vshrq_n_u64::<32>(hi);
    let lo1 = sub_small(lo, hi_hi);
    let t1 = vmull_u32(vmovn_u64(vandq_u64(hi, vdupq_n_u64(EPSILON))), vmov_n_u32(EPSILON as u32));
    add_small(lo1, t1)
}

#[inline(always)]
pub unsafe fn mul_gl(x: uint64x2_t, y: uint64x2_t) -> uint64x2_t {
    let (hi, lo) = mul64_64(x, y); reduce128(hi, lo)
}

#[inline(always)]
pub unsafe fn sq_gl(x: uint64x2_t) -> uint64x2_t {
    let (hi, lo) = square64(x); reduce128(hi, lo)
}

/// Vectorized x^7 for 2 Goldilocks elements simultaneously.
/// x^7 = x^3 * x^4 = (x * x^2) * (x^2)^2
#[inline(always)]
unsafe fn sbox2(x: uint64x2_t) -> uint64x2_t {
    let x2 = sq_gl(x);
    let x4 = sq_gl(x2);
    let x3 = mul_gl(x, x2);
    mul_gl(x3, x4)
}

/// Apply S-box (x^7) to all 12 state elements using NEON (6 pairs of 2).
/// Uses vectorized Goldilocks multiply via vmull_u32 partial products.
#[inline]
pub unsafe fn sbox_layer_neon(state: &mut [u64; WIDTH]) {
    for i in (0..WIDTH).step_by(2) {
        let v = vld1q_u64(state[i..].as_ptr());
        vst1q_u64(state[i..].as_mut_ptr(), sbox2(v));
    }
}

// ── Vectorized external_linear_layer (MDS) ───────────────────────────────────

#[inline(always)]
fn add_carry(a: u64, b: u64) -> (u64, u64) {
    let (s, c) = a.overflowing_add(b);
    (s, c as u64)
}

#[inline(always)]
unsafe fn reduce_lo_hi(lo: uint64x2_t, hi: uint64x2_t) -> uint64x2_t {
    let t1 = vsubq_u64(vshlq_n_u64::<32>(hi), hi);
    let sum = vaddq_u64(lo, t1);
    let adj = vandq_u64(vcltq_u64(sum, lo), vdupq_n_u64(EPSILON));
    vaddq_u64(sum, adj)
}

/// NEON implementation of Poseidon2 external_linear_layer.
#[inline]
pub unsafe fn external_linear_layer_neon(state: &mut [GoldilocksField; WIDTH]) {
    let s: &mut [u64; WIDTH] = transmute(state);

    // Step 1: Apply M4 to each group of 4
    for g in 0..3 {
        let b = g * 4;
        let (x0, x1, x2, x3) = (s[b], s[b + 1], s[b + 2], s[b + 3]);

        let (t01, t01_h) = add_carry(x0, x1);
        let (t23, t23_h) = add_carry(x2, x3);
        let (t0123, c) = add_carry(t01, t23);
        let t0123_h = t01_h + t23_h + c;

        let (tmp, c) = add_carry(t0123, t01);
        let h = t0123_h + t01_h + c;
        let (y0, c) = add_carry(tmp, x1);
        let y0_h = h + c;

        let (tmp, c) = add_carry(t0123, x1);
        let h = t0123_h + c;
        let (y1, c) = add_carry(tmp, x2.wrapping_add(x2));
        let y1_h = h + (x2 >> 63) + c;

        let (tmp, c) = add_carry(t0123, t23);
        let h = t0123_h + t23_h + c;
        let (y2, c) = add_carry(tmp, x3);
        let y2_h = h + c;

        let (tmp, c) = add_carry(t0123, x3);
        let h = t0123_h + c;
        let (y3, c) = add_carry(tmp, x0.wrapping_add(x0));
        let y3_h = h + (x0 >> 63) + c;

        vst1q_u64(s[b..].as_mut_ptr(), reduce_lo_hi(
            vcombine_u64(vcreate_u64(y0), vcreate_u64(y1)),
            vcombine_u64(vcreate_u64(y0_h), vcreate_u64(y1_h)),
        ));
        vst1q_u64(s[b + 2..].as_mut_ptr(), reduce_lo_hi(
            vcombine_u64(vcreate_u64(y2), vcreate_u64(y3)),
            vcombine_u64(vcreate_u64(y2_h), vcreate_u64(y3_h)),
        ));
    }

    // Step 2: Circulant outer matrix
    #[inline(always)]
    unsafe fn sum3(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
        let t = vaddq_u64(a, b);
        let c1 = vshrq_n_u64::<63>(vcltq_u64(t, a));
        let r = vaddq_u64(t, c);
        let c2 = vshrq_n_u64::<63>(vcltq_u64(r, t));
        reduce_lo_hi(r, vaddq_u64(c1, c2))
    }

    let sum01 = sum3(vld1q_u64(s[0..].as_ptr()), vld1q_u64(s[4..].as_ptr()), vld1q_u64(s[8..].as_ptr()));
    let sum23 = sum3(vld1q_u64(s[2..].as_ptr()), vld1q_u64(s[6..].as_ptr()), vld1q_u64(s[10..].as_ptr()));

    #[inline(always)]
    unsafe fn add_mod(cur: uint64x2_t, sum: uint64x2_t) -> uint64x2_t {
        let t = vaddq_u64(cur, sum);
        let adj = vandq_u64(vcltq_u64(t, cur), vdupq_n_u64(EPSILON));
        vaddq_u64(t, adj)
    }

    for g in 0..3 {
        let b = g * 4;
        vst1q_u64(s[b..].as_mut_ptr(),     add_mod(vld1q_u64(s[b..].as_ptr()),     sum01));
        vst1q_u64(s[b + 2..].as_mut_ptr(), add_mod(vld1q_u64(s[b + 2..].as_ptr()), sum23));
    }
}
