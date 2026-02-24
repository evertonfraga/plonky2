// GPU-accelerated Merkle tree: batch-hash all leaves on GPU, build tree on CPU.
// Gives ~50% speedup on hashing (leaf hashing ≈ half of all Poseidon2 calls).
use std::mem::MaybeUninit;
use plonky2_field::types::PrimeField64;
use plonky2_maybe_rayon::*;
use crate::hash::hash_types::{RichField, NUM_HASH_OUT_ELTS};
use crate::hash::merkle_tree::fill_subtree_with_hashes;
use crate::plonk::config::{GenericHashOut, Hasher};

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

/// Convert raw GPU u64 output to H::Hash
fn raw_to_hash<F: RichField, H: Hasher<F>>(raw: &[u64], i: usize) -> H::Hash {
    let mut bytes = [0u8; 32];
    for j in 0..NUM_HASH_OUT_ELTS {
        let canonical = F::from_noncanonical_u64(raw[i * NUM_HASH_OUT_ELTS + j]).to_canonical_u64();
        bytes[j*8..(j+1)*8].copy_from_slice(&canonical.to_le_bytes());
    }
    H::Hash::from_bytes(&bytes)
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

    let hash_size = H::HASH_SIZE;

    // Level 0: hash all leaves on GPU
    let mut current: Vec<u64> = if leaf_len * 8 <= hash_size {
        // Noop path
        let mut v = vec![0u64; n * NUM_HASH_OUT_ELTS];
        for (i, leaf) in leaves.iter().enumerate() {
            for (j, x) in leaf.iter().enumerate() {
                v[i * NUM_HASH_OUT_ELTS + j] = x.to_canonical_u64();
            }
        }
        v
    } else {
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
        raw
    };

    // Build all levels on GPU using two_to_one_batch
    // levels[0] = leaf hashes, levels[k] = hashes at height k
    let depth = n.trailing_zeros() as usize;
    let cap_depth = depth - cap_height;
    let mut all_levels: Vec<Vec<u64>> = vec![current.clone()];

    for _ in 0..cap_depth {
        let cur_len = current.len() / NUM_HASH_OUT_ELTS;
        let next_len = cur_len / 2;
        // Interleave pairs: [left[4], right[4]] for each pair
        let mut pairs = vec![0u64; next_len * 8];
        for i in 0..next_len {
            pairs[i*8..i*8+4].copy_from_slice(&current[i*2*4..i*2*4+4]);
            pairs[i*8+4..i*8+8].copy_from_slice(&current[(i*2+1)*4..(i*2+1)*4+4]);
        }
        let mut next = vec![0u64; next_len * NUM_HASH_OUT_ELTS];
        let ok = unsafe {
            poseidon2_two_to_one_gpu(pairs.as_ptr(), next.as_mut_ptr(), next_len as u32)
        };
        if ok != 0 {
            crate::hash::merkle_tree::fill_digests_buf::<F, H>(digests_buf, cap_buf, leaves, cap_height);
            return;
        }
        all_levels.push(next.clone());
        current = next;
    }

    // Fill cap_buf from the top level
    let cap_raw = &all_levels[cap_depth];
    for (i, cap_entry) in cap_buf.iter_mut().enumerate() {
        cap_entry.write(raw_to_hash::<F, H>(cap_raw, i));
    }

    if digests_buf.is_empty() {
        return;
    }

    // Fill digests_buf using the level data.
    // The plonky2 digests_buf layout is a specific DFS traversal.
    // We fill it by reconstructing the tree structure from level data.
    let leaf_hashes: Vec<H::Hash> = (0..n).map(|i| raw_to_hash::<F, H>(&all_levels[0], i)).collect();
    let subtree_digests_len = digests_buf.len() >> cap_height;
    let subtree_leaves_len = n >> cap_height;
    let digests_chunks = digests_buf.par_chunks_exact_mut(subtree_digests_len);
    let leaves_chunks = leaf_hashes.par_chunks_exact(subtree_leaves_len);
    digests_chunks.zip(cap_buf).zip(leaves_chunks).for_each(
        |((subtree_digests, subtree_cap), subtree_leaf_hashes)| {
            // Use CPU fill_subtree_with_hashes for the layout — two_to_one calls
            // are now redundant (we already computed them on GPU) but this ensures
            // correct digests_buf layout. The cap was already set above.
            let root = fill_subtree_with_hashes::<F, H>(subtree_digests, subtree_leaf_hashes);
            subtree_cap.write(root);
        },
    );
}
