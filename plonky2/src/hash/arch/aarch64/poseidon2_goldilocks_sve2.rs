/// SVE2-vectorized Poseidon2 S-box for aarch64 (Graviton3/4).
///
/// SVE2 provides svmul_u64 (low 64 bits) and svmulh_u64 (high 64 bits),
/// enabling native 64-bit Goldilocks multiply without the vmull_u32 partial
/// product overhead. Graviton3/4 SVE vector length = 256 bits = 4 × u64.
/// WIDTH=12: 3 iterations of 4 elements each.

use core::arch::aarch64::*;
use crate::field::goldilocks_field::GoldilocksField;
use crate::hash::poseidon2::config::WIDTH;

const P: u64 = 0xFFFFFFFF00000001;
const EPSILON: u64 = 0xFFFFFFFF;

#[inline(always)]
unsafe fn reduce_mul(lo: svuint64_t, hi: svuint64_t, pg: svbool_t) -> svuint64_t {
    let eps = svdup_n_u64(EPSILON);
    let hi_hi = svlsr_n_u64_x(pg, hi, 32);
    let hi_lo = svand_u64_x(pg, hi, eps);
    let t0 = svsub_u64_x(pg, lo, hi_hi);
    let carry = svcmplt_u64(pg, lo, hi_hi);
    let t0 = svsub_u64_m(carry, t0, eps);
    let t1 = svmul_u64_x(pg, hi_lo, eps);
    let result = svadd_u64_x(pg, t0, t1);
    let overflow = svcmpge_u64(pg, result, svdup_n_u64(P));
    svsub_u64_m(overflow, result, svdup_n_u64(P))
}

#[inline(always)]
unsafe fn gl_sq(x: svuint64_t, pg: svbool_t) -> svuint64_t {
    reduce_mul(svmul_u64_x(pg, x, x), svmulh_u64_x(pg, x, x), pg)
}

#[inline(always)]
unsafe fn gl_mul(x: svuint64_t, y: svuint64_t, pg: svbool_t) -> svuint64_t {
    reduce_mul(svmul_u64_x(pg, x, y), svmulh_u64_x(pg, x, y), pg)
}

pub unsafe fn sbox_layer_sve2(state: &mut [GoldilocksField; WIDTH]) {
    let pg = svptrue_b64();
    let vl = svcntd() as usize; // 4 on Graviton3/4 (256-bit SVE)
    let mut i = 0;
    while i + vl <= WIDTH {
        let v = svld1_u64(pg, state[i..].as_ptr() as *const u64);
        let v2 = gl_sq(v, pg);
        let v4 = gl_sq(v2, pg);
        let v3 = gl_mul(v, v2, pg);
        let v7 = gl_mul(v3, v4, pg);
        svst1_u64(pg, state[i..].as_mut_ptr() as *mut u64, v7);
        i += vl;
    }
    // Remaining elements (if WIDTH not multiple of vl)
    if i < WIDTH {
        let pg_rem = svwhilelt_b64_u64(0, (WIDTH - i) as u64);
        let v = svld1_u64(pg_rem, state[i..].as_ptr() as *const u64);
        let v2 = gl_sq(v, pg_rem);
        let v4 = gl_sq(v2, pg_rem);
        let v7 = gl_mul(gl_mul(v, v2, pg_rem), v4, pg_rem);
        svst1_u64(pg_rem, state[i..].as_mut_ptr() as *mut u64, v7);
    }
}
