#[cfg(target_feature = "neon")]
pub(crate) mod poseidon_goldilocks_neon;

#[cfg(target_feature = "sve2")]
pub(crate) mod poseidon2_goldilocks_sve2;
