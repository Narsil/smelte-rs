//! This build script emits the openblas linking directive if requested
#[cfg(not(feature = "matrixmultiply"))]
#[derive(PartialEq, Eq)]
enum Library {
    Static,
    Dynamic,
}

pub const SEQUENTIAL: bool = false;
pub const THREADED: bool = !SEQUENTIAL;

pub const LD_DIR: &str = if cfg!(windows) {
    "PATH"
} else if cfg!(target_os = "linux") {
    "LD_LIBRARY_PATH"
} else if cfg!(target_os = "macos") {
    "DYLD_LIBRARY_PATH"
} else {
    ""
};

pub const DEFAULT_ONEAPI_ROOT: &str = if cfg!(windows) {
    "C:/Program Files (x86)/Intel/oneAPI/"
} else {
    "/opt/intel/oneapi/"
};

pub const MKL_CORE: &str = "mkl_core";
pub const MKL_THREAD: &str = if SEQUENTIAL {
    "mkl_sequential"
} else {
    "mkl_intel_thread"
};
pub const THREADING_LIB: &str = if cfg!(windows) { "libiomp5md" } else { "iomp5" };
pub const MKL_INTERFACE: &str = if cfg!(target_pointer_width = "32") {
    "mkl_intel_ilp64"
} else {
    "mkl_intel_lp64"
};

#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
pub const UNSUPPORTED_OS_ERROR: _ = "Target OS is not supported. Please contact me";

#[cfg(all(target_os = "windows", target_pointer_width = "32"))]
pub const LINK_DIRS: &[&str] = &[
    "compiler/latest/windows/compiler/lib/ia32_win",
    "mkl/latest/lib/ia32",
];

#[cfg(all(target_os = "windows", target_pointer_width = "64"))]
pub const LINK_DIRS: &[&str] = &[
    "compiler/latest/windows/compiler/lib/intel64_win",
    "mkl/latest/lib/intel64",
];

#[cfg(all(target_os = "windows", target_pointer_width = "32"))]
pub const SHARED_LIB_DIRS: &[&str] = &[
    "compiler/latest/windows/redist/ia32_win/compiler",
    "mkl/latest/redist/ia32",
];

#[cfg(all(target_os = "windows", target_pointer_width = "64"))]
pub const SHARED_LIB_DIRS: &[&str] = &[
    "compiler/latest/windows/redist/intel64_win/compiler",
    "mkl/latest/redist/intel64",
];

#[cfg(all(target_os = "linux", target_pointer_width = "32"))]
pub const LINK_DIRS: &[&str] = &[
    "compiler/latest/linux/compiler/lib/ia32_lin",
    "mkl/latest/lib/ia32",
];

#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
pub const LINK_DIRS: &[&str] = &[
    "compiler/latest/linux/compiler/lib/intel64_lin",
    "mkl/latest/lib/intel64",
];

#[cfg(target_os = "linux")]
pub const SHARED_LIB_DIRS: &[&str] = LINK_DIRS;

#[cfg(target_os = "macos")]
const MACOS_COMPILER_PATH: &str = "compiler/latest/mac/compiler/lib";

#[cfg(target_os = "macos")]
pub const LINK_DIRS: &[&str] = &[MACOS_COMPILER_PATH, "mkl/latest/lib"];

#[cfg(target_os = "macos")]
pub const SHARED_LIB_DIRS: &[&str] = &["mkl/latest/lib"];

#[derive(Debug)]
pub enum BuildError {
    OneAPINotFound(std::path::PathBuf),
    OneAPINotADir(std::path::PathBuf),
    PathNotFound(std::env::VarError),
    AddSharedLibDirToPath(String),
}

#[cfg(feature = "intel-mkl")]
fn suggest_setvars_cmd(root: &str) -> String {
    if cfg!(windows) {
        format!("{root}/setvars.bat")
    } else {
        format!("source {root}/setvars.sh")
    }
}

#[cfg(feature = "gpu")]
mod cuda {
    pub fn build_ptx() {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let kernel_paths: Vec<std::path::PathBuf> = glob::glob("src/**/*.cu")
            .unwrap()
            .map(|p| p.unwrap())
            .collect();
        let mut include_directories: Vec<std::path::PathBuf> = glob::glob("src/**/*.cuh")
            .unwrap()
            .map(|p| p.unwrap())
            .collect();

        for path in &kernel_paths {
            println!("cargo:rerun-if-changed={}", path.display());
        }
        for path in &mut include_directories {
            println!("cargo:rerun-if-changed={}", path.display());
            // remove the filename from the path so it's just the directory
            path.pop();
        }

        include_directories.sort();
        include_directories.dedup();

        #[allow(unused)]
        let include_options: Vec<String> = include_directories
            .into_iter()
            .map(|s| "-I".to_string() + &s.into_os_string().into_string().unwrap())
            .collect::<Vec<_>>();

        #[cfg(feature = "ci-check")]
        for mut kernel_path in kernel_paths.into_iter() {
            kernel_path.set_extension("ptx");

            let mut ptx_path: std::path::PathBuf = out_dir.clone().into();
            ptx_path.push(kernel_path.as_path().file_name().unwrap());
            std::fs::File::create(ptx_path).unwrap();
        }

        #[cfg(not(feature = "ci-check"))]
        {
            let start = std::time::Instant::now();

            let compute_cap = {
                let out = std::process::Command::new("nvidia-smi")
                    .arg("--query-gpu=compute_cap")
                    .arg("--format=csv")
                    .output()
                    .unwrap();
                let out = std::str::from_utf8(&out.stdout).unwrap();
                let mut lines = out.lines();
                assert_eq!(lines.next().unwrap(), "compute_cap");
                lines.next().unwrap().replace('.', "")
            };

            kernel_paths
                .iter()
                .for_each(|p| println!("cargo:rerun-if-changed={}", p.display()));

            let children = kernel_paths
                .iter()
                .map(|p| {
                    std::process::Command::new("nvcc")
                        .arg(format!("--gpu-architecture=sm_{compute_cap}"))
                        .arg("--ptx")
                        .args(["--default-stream", "per-thread"])
                        .args(["--output-directory", &out_dir])
                        .args(&include_options)
                        .arg(p)
                        .spawn()
                        .expect("nvcc not found")
                })
                .collect::<Vec<_>>();

            for (kernel_path, child) in kernel_paths.iter().zip(children.into_iter()) {
                let output = child.wait_with_output().unwrap();
                assert!(
                    output.status.success(),
                    "nvcc error while compiling {kernel_path:?}: {output:?}",
                );
            }

            println!("cargo:warning=Compiled {:?}", kernel_paths,);

            println!(
                "cargo:warning=Compiled {:?} cuda kernels in {:?}",
                kernel_paths.len(),
                start.elapsed()
            );
        }
    }
}

fn main() -> Result<(), BuildError> {
    println!("cargo:rerun-if-changed=build.rs");

    println!("cargo:rerun-if-env-changed=STATIC");
    #[cfg(not(feature = "matrixmultiply"))]
    let library = if std::env::var("STATIC").unwrap_or_else(|_| "0".to_string()) == "1" {
        Library::Static
    } else {
        Library::Dynamic
    };

    #[cfg(not(feature = "matrixmultiply"))]
    let link_type: &str = if Library::Static == library {
        "static"
    } else {
        "dylib"
    };

    #[cfg(not(feature = "matrixmultiply"))]
    println!("cargo:rustc-link-lib={link_type}=cblas");

    #[cfg(feature = "intel-mkl")]
    {
        let root = std::env::var("ONEAPI_ROOT").unwrap_or_else(|_| DEFAULT_ONEAPI_ROOT.to_string());
        println!("Using '{root}' as ONEAPI_ROOT");

        let path = match std::env::var(LD_DIR) {
            Ok(path) => path,
            Err(e) => {
                // On macOS it's unusual to set $DYLD_LIBRARY_PATH, so we want to provide a helpful message
                println!(
                    "Library path env variable '{LD_DIR}' was not found. Run `{}`",
                    suggest_setvars_cmd(&root)
                );
                return Err(BuildError::PathNotFound(e));
            }
        };

        if library == Library::Dynamic {
            // check to make sure that things in `SHARED_LIB_DIRS` are in `$LD_DIR`.
            let path = path.replace('\\', "/");
            for shared_lib_dir in SHARED_LIB_DIRS {
                println!("cargo:rerun-if-env-changed=MKL_VERSION");
                let mkl_version =
                    std::env::var("MKL_VERSION").unwrap_or_else(|_| "2023.0.0".to_string());
                let versioned_dir = shared_lib_dir.replace("latest", &mkl_version);

                println!("Checking that '{shared_lib_dir}' or '{versioned_dir}' is in {LD_DIR}");
                if !path.contains(shared_lib_dir) && !path.contains(&versioned_dir) {
                    println!(
                        "'{shared_lib_dir}' not found in library path. Run `{}`",
                        suggest_setvars_cmd(&root)
                    );
                    return Err(BuildError::AddSharedLibDirToPath(
                        shared_lib_dir.to_string(),
                    ));
                }
            }
        }

        let root: std::path::PathBuf = root.into();

        if !root.exists() {
            return Err(BuildError::OneAPINotFound(root));
        }
        if !root.is_dir() {
            return Err(BuildError::OneAPINotADir(root));
        }

        for rel_lib_dir in LINK_DIRS {
            let lib_dir = root.join(rel_lib_dir);
            println!("cargo:rustc-link-search={}", lib_dir.display());
        }

        let lib_postfix: &str = if cfg!(windows) && library == Library::Static {
            "_dll"
        } else {
            ""
        };

        println!("cargo:rustc-link-lib={link_type}={MKL_INTERFACE}{lib_postfix}");
        println!("cargo:rustc-link-lib={link_type}={MKL_THREAD}{lib_postfix}");
        println!("cargo:rustc-link-lib={link_type}={MKL_CORE}{lib_postfix}");
        if THREADED {
            println!("cargo:rustc-link-lib=dylib={THREADING_LIB}");
        }

        if !cfg!(windows) {
            println!("cargo:rustc-link-lib=pthread");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=dl");
        }

        #[cfg(target_os = "macos")]
        {
            println!(
                "cargo:rustc-link-arg=-Wl,-rpath,{}/{MACOS_COMPILER_PATH}",
                root.display(),
            );
        }
    }

    #[cfg(feature = "gpu")]
    cuda::build_ptx();

    Ok(())
}
