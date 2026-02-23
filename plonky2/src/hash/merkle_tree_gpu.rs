// GPU-accelerated Merkle tree: batch-hash all leaves on GPU, build tree on CPU.
use std::mem::MaybeUninit;
use plonky2_field::types::PrimeField64;
use plonky2_maybe_rayon::*;
use crate::hash::hash_types::{RichField, NUM_HASH_OUT_ELTS};
use crate::hash::merkle_tree::fill_subtree_with_hashes;
use crate::plonk::config::Hasher;

const GPU_LEAF_THRESHOLD: usize = 512;

extern "C" {
    fn poseidon2_two_to_one_gpu(inputs: *const u64, outputs: *mut u64, batch: u32) -> i32;
    fn poseidon2_hash_no_pad_gpu(inputs: *const u64, outputs: *mut u64, batch: u32, leaf_len: u32) -> i32;
}

static GPU_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
fn gpu_ok() -> bool {
    *GPU_AVAILABLE.get_or_init(|| {
        let inp = [0u64; 8];
        let mut out = [0u64; 4];
        unsafe { poseidon2_two_to_one_gpu(inp.as_ptr(), out.as_mut_ptr(), 1) == 0 }
    })
}

pub fn fill_digests_buf_gpu<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[Vec<F>],
    cap_height: usize,
) {
    let n = leaves.len();
    let leaf_len = leaves.first().map(|l| l.len()).unwrap_or(0);
    let uniform = leaf_len > 0 && leaves.iter().all(|l| l.len() == leaf_len);

    if n < GPU_LEAF_THRESHOLD || !uniform || !gpu_ok() {
        crate::hash::merkle_tree::fill_digests_buf::<F, H>(digests_buf, cap_buf, leaves, cap_height);
        return;
    }

    // GPU: batch-hash all leaves
    let inputs: Vec<u64> = leaves.iter()
        .flat_map(|leaf| leaf.iter().map(|x| x.to_noncanonical_u64()))
        .collect();
    let mut raw = vec![0u64; n * NUM_HASH_OUT_ELTS];
    let ok = unsafe {
        poseidon2_hash_no_pad_gpu(inputs.as_ptr(), raw.as_mut_ptr(), n as u32, leaf_len as u32)
    };
    if ok != 0 {
        crate::hash::merkle_tree::fill_digests_buf::<F, H>(digests_buf, cap_buf, leaves, cap_height);
        return;
    }

    // Convert raw GPU output to H::Hash.
    // GPU outputs non-canonical u64 values; canonicalize via to_canonical_u64.
    let leaf_hashes: Vec<H::Hash> = (0..n).map(|i| {
        let mut bytes = [0u8; 32];
        for j in 0..NUM_HASH_OUT_ELTS {
            // Canonicalize: reduce mod p
            let v = raw[i * NUM_HASH_OUT_ELTS + j];
            let canonical = F::from_noncanonical_u64(v).to_canonical_u64();
            bytes[j*8..(j+1)*8].copy_from_slice(&canonical.to_le_bytes());
        }
        H::Hash::from_bytes(&bytes)
    }).collect();

    // CPU: build tree with pre-computed leaf hashes
    if digests_buf.is_empty() {
        cap_buf.par_iter_mut().zip(leaf_hashes.par_iter()).for_each(|(c, h)| { c.write(*h); });
        return;
    }

    let subtree_digests_len = digests_buf.len() >> cap_height;
    let subtree_leaves_len = n >> cap_height;
    let digests_chunks = digests_buf.par_chunks_exact_mut(subtree_digests_len);
    let leaves_chunks = leaf_hashes.par_chunks_exact(subtree_leaves_len);
    digests_chunks.zip(cap_buf).zip(leaves_chunks).for_each(
        |((subtree_digests, subtree_cap), subtree_leaf_hashes)| {
            subtree_cap.write(fill_subtree_with_hashes::<F, H>(subtree_digests, subtree_leaf_hashes));
        },
    );
}
