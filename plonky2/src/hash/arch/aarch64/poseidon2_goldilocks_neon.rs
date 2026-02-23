/// NEON-vectorized Poseidon2 external_linear_layer for aarch64 (Graviton3/4).
use core::arch::aarch64::*;
use plonky2_field::types::PrimeField64;
use crate::field::goldilocks_field::GoldilocksField;
use crate::hash::poseidon2::config::WIDTH;

const EPSILON: u64 = 0xFFFFFFFF;

#[inline(always)]
fn reduce96(x: u128) -> GoldilocksField {
    let lo = x as u64;
    let hi = (x >> 64) as u64;
    let t1 = hi.wrapping_mul(EPSILON);
    let (t2, over) = lo.overflowing_add(t1);
    GoldilocksField(if over { t2.wrapping_add(EPSILON) } else { t2 })
}

pub unsafe fn external_linear_layer_neon(state: &mut [GoldilocksField; WIDTH]) {
    let mut s = [0u128; WIDTH];
    for i in 0..WIDTH { s[i] = state[i].to_noncanonical_u64() as u128; }

    // M4 on each group of 4 (scalar additions)
    for g in 0..3 {
        let b = g * 4;
        let t01 = s[b] + s[b+1]; let t23 = s[b+2] + s[b+3];
        let t0123 = t01 + t23;
        let x0 = s[b]; let x2 = s[b+2];
        s[b]   = t0123 + t01 + s[b+1];
        s[b+1] = t0123 + s[b+1] + x2 + x2;
        s[b+2] = t0123 + t23 + s[b+3];
        s[b+3] = t0123 + s[b+3] + x0 + x0;
    }

    // Outer circulant sum using NEON (2 u64 pairs at a time)
    let mut lo = [0u64; WIDTH]; let mut hi = [0u64; WIDTH];
    for i in 0..WIDTH { lo[i] = s[i] as u64; hi[i] = (s[i] >> 64) as u64; }

    let mut sum_lo = [0u64; 4]; let mut sum_hi = [0u64; 4];
    for k in (0..4).step_by(2) {
        vst1q_u64(sum_lo[k..].as_mut_ptr(),
            vaddq_u64(vaddq_u64(vld1q_u64(lo[k..].as_ptr()), vld1q_u64(lo[k+4..].as_ptr())),
                      vld1q_u64(lo[k+8..].as_ptr())));
        vst1q_u64(sum_hi[k..].as_mut_ptr(),
            vaddq_u64(vaddq_u64(vld1q_u64(hi[k..].as_ptr()), vld1q_u64(hi[k+4..].as_ptr())),
                      vld1q_u64(hi[k+8..].as_ptr())));
    }

    for g in 0..3 {
        let b = g * 4;
        for k in (0..4).step_by(2) {
            vst1q_u64(lo[b+k..].as_mut_ptr(),
                vaddq_u64(vld1q_u64(lo[b+k..].as_ptr()), vld1q_u64(sum_lo[k..].as_ptr())));
            vst1q_u64(hi[b+k..].as_mut_ptr(),
                vaddq_u64(vld1q_u64(hi[b+k..].as_ptr()), vld1q_u64(sum_hi[k..].as_ptr())));
        }
    }

    for i in 0..WIDTH {
        state[i] = reduce96((lo[i] as u128) | ((hi[i] as u128) << 64));
    }
}
