// plonky2/src/hash/merkle_tree_gpu.rs
// GPU: leaf hashing only (one batch, one transfer). CPU: internal nodes (rayon).
// Threshold: 64 leaves. Uses cudarc v0.19 + NVRTC (no build.rs, no nvcc at build time).

use std::mem::MaybeUninit;
use std::sync::{Arc, OnceLock};

use plonky2_field::types::PrimeField64;
use plonky2_maybe_rayon::*;

use crate::hash::hash_types::{HashOut, RichField, NUM_HASH_OUT_ELTS};
use crate::hash::merkle_tree::fill_subtree_with_hashes;
use crate::plonk::config::{GenericHashOut, Hasher};
use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg};

const GPU_THRESHOLD: usize = 64;
const KERNEL_SRC: &str = include_str!("poseidon2/poseidon2_kernel.cu");

struct GpuCtx { ctx: Arc<CudaContext>, hash_no_pad_fn: CudaFunction }
unsafe impl Send for GpuCtx {}
unsafe impl Sync for GpuCtx {}

static GPU: OnceLock<Option<GpuCtx>> = OnceLock::new();

fn gpu() -> Option<&'static GpuCtx> {
    GPU.get_or_init(|| {
        let ctx = CudaContext::new(0).ok()?;
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            KERNEL_SRC,
            cudarc::nvrtc::CompileOptions {
                options: vec!["--device-int128".to_string()],
                ..Default::default()
            },
        ).ok()?;
        let module = ctx.load_module(ptx).ok()?;
        let hash_no_pad_fn = module.load_function("hash_no_pad_batch").ok()?;
        Some(GpuCtx { ctx, hash_no_pad_fn })
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
        _ => { crate::hash::merkle_tree::fill_digests_buf::<F, H>(digests_buf, cap_buf, leaves, cap_height); return; }
    };

    let leaf_hashes = match gpu_hash_leaves::<F, H>(g, leaves, leaf_len, n) {
        Ok(h) => h,
        Err(_) => { crate::hash::merkle_tree::fill_digests_buf::<F, H>(digests_buf, cap_buf, leaves, cap_height); return; }
    };

    if digests_buf.is_empty() {
        cap_buf.par_iter_mut().zip(leaf_hashes.par_iter()).for_each(|(c, h)| { c.write(*h); });
        return;
    }
    let subtree_digests_len = digests_buf.len() >> cap_height;
    let subtree_leaves_len = n >> cap_height;
    digests_buf.par_chunks_exact_mut(subtree_digests_len)
        .zip(cap_buf.par_iter_mut())
        .zip(leaf_hashes.par_chunks_exact(subtree_leaves_len))
        .for_each(|((sub_digests, sub_cap), sub_hashes)| {
            sub_cap.write(fill_subtree_with_hashes::<F, H>(sub_digests, sub_hashes));
        });
}

fn gpu_hash_leaves<F: RichField, H: Hasher<F>>(
    g: &GpuCtx, leaves: &[Vec<F>], leaf_len: usize, n: usize,
) -> Result<Vec<H::Hash>, String> {
    let stream = g.ctx.default_stream();
    let inputs: Vec<u64> = leaves.iter()
        .flat_map(|l| l.iter().map(|x| x.to_noncanonical_u64()))
        .collect();
    let d_in = stream.clone_htod(&inputs).map_err(|e| format!("{e:?}"))?;
    let mut d_out = stream.alloc_zeros::<u64>(n * NUM_HASH_OUT_ELTS).map_err(|e| format!("{e:?}"))?;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let n_u32 = n as u32; let ll_u32 = leaf_len as u32;
    unsafe {
        let mut b = stream.launch_builder(&g.hash_no_pad_fn);
        b.arg(&d_in); b.arg(&mut d_out); b.arg(&n_u32); b.arg(&ll_u32);
        b.launch(cfg).map_err(|e| format!("{e:?}"))?;
    }
    let raw = stream.clone_dtoh(&d_out).map_err(|e| format!("{e:?}"))?;
    Ok((0..n).map(|i| {
        let elts: [F; NUM_HASH_OUT_ELTS] = core::array::from_fn(|j| F::from_canonical_u64(raw[i * NUM_HASH_OUT_ELTS + j]));
        H::Hash::from_bytes(&HashOut { elements: elts }.to_bytes())
    }).collect())
}
