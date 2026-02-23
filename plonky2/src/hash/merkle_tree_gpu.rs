// plonky2/src/hash/merkle_tree_gpu.rs
// GPU-accelerated Merkle tree using cudarc v0.19 + NVRTC.
//
// CPU/GPU split:
//   GPU: all Poseidon2 hashing in Merkle trees (leaves + all internal levels)
//   CPU: circuit building, witness, FRI polynomial arithmetic, transcript, small trees
//
// No build.rs, no nvcc at build time. NVRTC compiles the kernel at first use (~1s, cached).

use std::mem::MaybeUninit;
use std::sync::{Arc, OnceLock};

use plonky2_field::types::PrimeField64;
use plonky2_maybe_rayon::*;

use crate::hash::hash_types::{HashOut, RichField, NUM_HASH_OUT_ELTS};
use crate::plonk::config::{GenericHashOut, Hasher};
use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg};

const GPU_THRESHOLD: usize = 1024;

const KERNEL_SRC: &str = include_str!("poseidon2/poseidon2_kernel.cu");

struct GpuCtx {
    ctx: Arc<CudaContext>,
    two_to_one_fn: CudaFunction,
    hash_no_pad_fn: CudaFunction,
}
unsafe impl Send for GpuCtx {}
unsafe impl Sync for GpuCtx {}

static GPU: OnceLock<Option<GpuCtx>> = OnceLock::new();

fn gpu() -> Option<&'static GpuCtx> {
    GPU.get_or_init(|| {
        let ctx = CudaContext::new(0).ok()?;
        let ptx = cudarc::nvrtc::compile_ptx(KERNEL_SRC).ok()?;
        let module = ctx.load_module(ptx).ok()?;
        let two_to_one_fn = module.load_function("two_to_one_batch").ok()?;
        let hash_no_pad_fn = module.load_function("hash_no_pad_batch").ok()?;
        Some(GpuCtx { ctx, two_to_one_fn, hash_no_pad_fn })
    }).as_ref()
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

    let g = match (n >= GPU_THRESHOLD && uniform, gpu()) {
        (true, Some(g)) => g,
        _ => {
            crate::hash::merkle_tree::fill_digests_buf::<F, H>(digests_buf, cap_buf, leaves, cap_height);
            return;
        }
    };

    match gpu_tree::<F, H>(g, leaves, leaf_len, n, cap_height) {
        Ok(levels) => fill_from_levels::<F, H>(digests_buf, cap_buf, &levels, cap_height),
        Err(_) => crate::hash::merkle_tree::fill_digests_buf::<F, H>(digests_buf, cap_buf, leaves, cap_height),
    }
}

fn gpu_tree<F: RichField, H: Hasher<F>>(
    g: &GpuCtx,
    leaves: &[Vec<F>],
    leaf_len: usize,
    n: usize,
    cap_height: usize,
) -> Result<Vec<Vec<H::Hash>>, String> {
    let stream = g.ctx.default_stream();
    let depth = n.trailing_zeros() as usize;
    let cap_depth = depth - cap_height;

    // Level 0: GPU hash all leaves
    let inputs: Vec<u64> = leaves.iter()
        .flat_map(|l| l.iter().map(|x| x.to_noncanonical_u64()))
        .collect();
    let d_in = stream.clone_htod(&inputs).map_err(|e| format!("{e:?}"))?;
    let mut d_out = stream.alloc_zeros::<u64>(n * NUM_HASH_OUT_ELTS).map_err(|e| format!("{e:?}"))?;
    {
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            let mut b = stream.launch_builder(&g.hash_no_pad_fn);
            let n_u32 = n as u32; let ll_u32 = leaf_len as u32;
            b.arg(&d_in); b.arg(&mut d_out);
            b.arg(&n_u32); b.arg(&ll_u32);
            b.launch(cfg).map_err(|e| format!("{e:?}"))?;
        }
    }
    let raw = stream.clone_dtoh(&d_out).map_err(|e| format!("{e:?}"))?;
    let mut levels: Vec<Vec<H::Hash>> = vec![raw_to_hashes::<F, H>(&raw, n)];

    // Levels 1..cap_depth: GPU two_to_one each level
    let mut cur_n = n;
    for _ in 0..cap_depth {
        let next_n = cur_n / 2;
        let prev = levels.last().unwrap();
        let pairs: Vec<u64> = (0..next_n).flat_map(|i| {
            hash_to_u64::<F, H>(&prev[2*i]).into_iter()
                .chain(hash_to_u64::<F, H>(&prev[2*i+1]))
        }).collect();

        let d_pairs = stream.clone_htod(&pairs).map_err(|e| format!("{e:?}"))?;
        let mut d_next = stream.alloc_zeros::<u64>(next_n * NUM_HASH_OUT_ELTS).map_err(|e| format!("{e:?}"))?;
        {
            let cfg = LaunchConfig::for_num_elems(next_n as u32);
            unsafe {
                let mut b = stream.launch_builder(&g.two_to_one_fn);
                let nn_u32 = next_n as u32;
                b.arg(&d_pairs); b.arg(&mut d_next);
                b.arg(&nn_u32);
                b.launch(cfg).map_err(|e| format!("{e:?}"))?;
            }
        }
        let raw = stream.clone_dtoh(&d_next).map_err(|e| format!("{e:?}"))?;
        levels.push(raw_to_hashes::<F, H>(&raw, next_n));
        cur_n = next_n;
    }
    Ok(levels)
}

fn raw_to_hashes<F: RichField, H: Hasher<F>>(raw: &[u64], n: usize) -> Vec<H::Hash> {
    (0..n).map(|i| {
        let elts: [F; NUM_HASH_OUT_ELTS] = core::array::from_fn(|j| {
            F::from_canonical_u64(raw[i * NUM_HASH_OUT_ELTS + j])
        });
        H::Hash::from_bytes(&HashOut { elements: elts }.to_bytes())
    }).collect()
}

fn hash_to_u64<F: RichField, H: Hasher<F>>(h: &H::Hash) -> [u64; NUM_HASH_OUT_ELTS] {
    let v = h.to_vec();
    core::array::from_fn(|i| v[i].to_noncanonical_u64())
}

fn fill_from_levels<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    levels: &[Vec<H::Hash>],
    cap_height: usize,
) {
    let cap_depth = levels.len() - 1;
    let cap_level = &levels[cap_depth];
    for (i, c) in cap_buf.iter_mut().enumerate() {
        c.write(cap_level[i]);
    }
    if digests_buf.is_empty() { return; }

    let subtree_digests_len = digests_buf.len() >> cap_height;
    let subtree_leaves_len = levels[0].len() >> cap_height;
    let subtree_depth = cap_depth;

    digests_buf.par_chunks_exact_mut(subtree_digests_len)
        .enumerate()
        .for_each(|(idx, sub)| {
            fill_subtree_from_levels::<F, H>(sub, levels, idx * subtree_leaves_len, subtree_depth);
        });
}

fn fill_subtree_from_levels<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    levels: &[Vec<H::Hash>],
    leaf_start: usize,
    k: usize,
) {
    if k == 0 { return; }
    let half = digests_buf.len() / 2;
    let (left_half, right_half) = digests_buf.split_at_mut(half);
    let (left_mem, left_sub) = left_half.split_last_mut().unwrap();
    let (right_mem, right_sub) = right_half.split_first_mut().unwrap();
    let level = &levels[k - 1];
    let li = leaf_start >> (k - 1);
    left_mem.write(level[li]);
    right_mem.write(level[li + 1]);
    fill_subtree_from_levels::<F, H>(left_sub, levels, leaf_start, k - 1);
    fill_subtree_from_levels::<F, H>(right_sub, levels, leaf_start + (1 << (k-1)), k - 1);
}
