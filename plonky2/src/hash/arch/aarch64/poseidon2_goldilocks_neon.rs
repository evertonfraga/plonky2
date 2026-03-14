/// NEON-accelerated Poseidon2 external_linear_layer for aarch64 (Graviton3/4).
/// M4 in u128 (scalar), circulant sum in u128, NEON for final reduction.
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

pub fn external_linear_layer_neon(state: &mut [GoldilocksField; WIDTH]) {
    let mut s = [0u128; WIDTH];
    for i in 0..WIDTH { s[i] = state[i].to_noncanonical_u64() as u128; }

    // M4 on each group of 4
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

    // Circulant sum in u128 (no overflow)
    let mut sums = [0u128; 4];
    for i in 0..4 { sums[i] = s[i] + s[i+4] + s[i+8]; }
    for i in 0..WIDTH { s[i] += sums[i % 4]; }

    // Reduce to GoldilocksField
    for i in 0..WIDTH {
        state[i] = reduce96(s[i]);
    }
}
