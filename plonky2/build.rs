fn main() {
    if std::env::var("CARGO_FEATURE_GPU").is_ok() {
        compile_cuda();
    }
}

fn compile_cuda() {
    use std::process::Command;
    use std::path::PathBuf;
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let cu_src = PathBuf::from(&manifest_dir).join("src/hash/poseidon2/poseidon2_gpu.cu");
    let lib_out = PathBuf::from(&out_dir).join("libposeidon2_gpu.a");
    let sm = detect_sm().unwrap_or_else(|| "sm_86".to_string());
    println!("cargo:warning=Compiling CUDA kernel for {}", sm);
    let status = Command::new("nvcc")
        .args(["-O3", &format!("-arch={}", sm), "--compiler-options", "-fPIC",
               "-lib", cu_src.to_str().unwrap(), "-o", lib_out.to_str().unwrap()])
        .status()
        .expect("nvcc not found — install CUDA toolkit or build without --features gpu");
    assert!(status.success(), "nvcc compilation failed");
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=poseidon2_gpu");
    // Find CUDA lib path dynamically
    let cuda_lib = std::process::Command::new("sh")
        .args(["-c", "find /usr/local/cuda*/targets/*/lib -name 'libcudart.so' 2>/dev/null | head -1 | xargs dirname"])
        .output().ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "/usr/local/cuda/lib64".to_string());
    println!("cargo:rustc-link-search=native={}", cuda_lib);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rerun-if-changed=src/hash/poseidon2/poseidon2_gpu.cu");
}

fn detect_sm() -> Option<String> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output().ok()?;
    let cap = String::from_utf8(out.stdout).ok()?;
    Some(format!("sm_{}", cap.trim().replace('.', "")))
}
