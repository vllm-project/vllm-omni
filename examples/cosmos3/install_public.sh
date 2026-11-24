#!/usr/bin/env bash
# Build an isolated native runtime from public upstream sources.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$root/examples/cosmos3/environment/public-versions.env"
stage="${1:---plan}"
jobs="${COSMOS3_BUILD_JOBS:-2}"
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo 'COSMOS3_BUILD_JOBS must be positive' >&2; exit 2; }
case "$stage" in
  --plan)
    printf 'Stages: python -> official CANN installation/configuration -> graphics -> packages -> frameworks -> report\n'
    printf 'Python %s; LLVM %s (%s); Mesa %s\n' "$PUBLIC_PYTHON_VERSION" "$PUBLIC_LLVM_VERSION" "$PUBLIC_LLVM_COMMIT" "$PUBLIC_MESA_VERSION"
    printf 'vLLM %s (%s), VLLM_TARGET_DEVICE=empty\n' "$PUBLIC_VLLM_VERSION" "$PUBLIC_VLLM_COMMIT"
    printf 'Ascend %s (%s), COMPILE_CUSTOM_KERNELS=0\n' "$PUBLIC_ASCEND_VERSION" "$PUBLIC_ASCEND_COMMIT"
    printf 'Triton base %s -> Triton-Ascend backend %s\n' "$PUBLIC_TRITON_BASE_VERSION" "$PUBLIC_TRITON_VERSION"
    printf 'Prefix: %s/.runtime; venv: %s/.venv-public; build jobs: %s\n' "$root" "$root" "$jobs"
    exit 0 ;;
  python|graphics|packages|frameworks|report) ;;
  *) echo 'Usage: bash install_public.sh [--plan|python|graphics|packages|frameworks|report]' >&2; exit 2 ;;
esac
[[ "$(uname -m)" == aarch64 ]] || { echo 'NPU public build requires aarch64' >&2; exit 2; }
runtime="$root/.runtime"
downloads="$runtime/public-downloads"
sources="$runtime/public-sources"
builds="$runtime/public-builds"
logs="$runtime/public-build-logs"
venv="$root/.venv-public"
python="$venv/bin/python"
mkdir -p "$downloads" "$sources" "$builds" "$logs"
exec > >(tee -a "$logs/$stage.log") 2>&1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONNOUSERSITE=1
export CMAKE_BUILD_PARALLEL_LEVEL="$jobs" MAX_JOBS="$jobs"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

fetch() {
  local url="$1" name="$2" expected="${3:-}"
  if [[ ! -f "$downloads/$name" ]]; then
    curl -fL --retry 3 --connect-timeout 20 --max-time 1800 \
      --continue-at - "$url" -o "$downloads/$name.part"
    mv "$downloads/$name.part" "$downloads/$name"
  fi
  if [[ -n "$expected" ]]; then
    printf '%s  %s\n' "$expected" "$downloads/$name" | sha256sum -c -
  fi
  printf '%s\t%s\n' "$name" "$url" >> "$logs/download-urls.tsv"
}

checkout() {
  local url="$1" commit="$2" name="$3" path="$sources/$3"
  if [[ ! -d "$path/.git" ]]; then
    [[ ! -e "$path" ]] || { echo "Source path is not a checkout: $path" >&2; exit 2; }
    git init "$path"
    git -C "$path" remote add origin "$url"
    git -C "$path" fetch --depth 1 origin "$commit"
    git -C "$path" checkout --detach FETCH_HEAD
  fi
  [[ "$(git -C "$path" rev-parse HEAD)" == "$commit" ]] || { echo "Wrong source commit: $path" >&2; exit 2; }
  [[ -z "$(git -C "$path" status --porcelain --untracked-files=no)" ]] || { echo "Source changes found: $path" >&2; exit 2; }
  printf '%s\t%s\t%s\n' "$name" "$commit" "$url" >> "$logs/source-commits.tsv"
}

if [[ "$stage" == python ]]; then
  [[ ! -e "$venv" ]] || { echo "Use a new checkout: $venv exists" >&2; exit 2; }
  prefix="$runtime/python-public"
  [[ ! -e "$prefix" ]] || { echo "Python prefix already exists: $prefix" >&2; exit 2; }
  archive="Python-$PUBLIC_PYTHON_VERSION.tar.xz"
  fetch "https://www.python.org/ftp/python/$PUBLIC_PYTHON_VERSION/$archive" "$archive" "$PUBLIC_PYTHON_SHA256"
  mkdir "$builds/python"
  tar -xf "$downloads/$archive" -C "$builds/python" --strip-components=1
  (
    cd "$builds/python"
    LDFLAGS="-Wl,-rpath,$prefix/lib" ./configure --prefix="$prefix" --enable-shared --with-ensurepip=install
    make -j "$jobs"
    make install
  )
  "$prefix/bin/python3.12" -m venv "$venv"
  "$python" -m pip install -r "$root/examples/cosmos3/environment/public-build.lock.txt"
  "$python" -m pip install \
    -c "$root/examples/cosmos3/environment/public-runtime.lock.txt" \
    -r "$root/examples/cosmos3/environment/public-cann-python.lock.txt"
  exit 0
fi

[[ -x "$python" ]] || { echo 'Run the python stage first' >&2; exit 2; }
export PATH="$venv/bin:$PATH"

if [[ "$stage" == graphics ]]; then
  llvm_prefix="$runtime/llvm-$PUBLIC_LLVM_VERSION"
  mesa_prefix="$runtime/mesa-$PUBLIC_MESA_VERSION"
  [[ ! -e "$llvm_prefix" && ! -e "$mesa_prefix" ]] || { echo 'Use new graphics prefixes; do not overwrite an existing runtime' >&2; exit 2; }
  checkout https://github.com/llvm/llvm-project.git "$PUBLIC_LLVM_COMMIT" llvm-project
  cmake -S "$sources/llvm-project/llvm" -B "$builds/llvm" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$llvm_prefix" \
    -DLLVM_TARGETS_TO_BUILD=AArch64 -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON \
    -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_ENABLE_BINDINGS=OFF -DLLVM_ENABLE_ZSTD=OFF
  cmake --build "$builds/llvm" --parallel "$jobs"
  cmake --install "$builds/llvm"
  fetch "https://archive.mesa3d.org/mesa-$PUBLIC_MESA_VERSION.tar.xz" "mesa-$PUBLIC_MESA_VERSION.tar.xz" "$PUBLIC_MESA_SHA256"
  mkdir "$builds/mesa-source"
  tar -xf "$downloads/mesa-$PUBLIC_MESA_VERSION.tar.xz" -C "$builds/mesa-source" --strip-components=1
  export PATH="$llvm_prefix/bin:$PATH" LLVM_CONFIG="$llvm_prefix/bin/llvm-config"
  export LD_LIBRARY_PATH="$llvm_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  [[ "$(llvm-config --version)" == "$PUBLIC_LLVM_VERSION" ]]
  meson setup "$builds/mesa" "$builds/mesa-source" --prefix="$mesa_prefix" --libdir=lib \
    --buildtype=release -Dosmesa=true -Dgallium-drivers=swrast -Dvulkan-drivers=[] \
    -Dplatforms=[] -Dglx=disabled -Degl=disabled -Dgbm=disabled \
    -Dllvm=enabled -Dshared-llvm=enabled -Dshared-glapi=enabled -Dbuild-tests=false \
    -Dgallium-va=disabled -Dgallium-vdpau=disabled
  meson compile -C "$builds/mesa" -j "$jobs"
  meson install -C "$builds/mesa"
  test -f "$mesa_prefix/lib/libOSMesa.so"
  test -f "$mesa_prefix/lib/libglapi.so"
  exit 0
fi

source "$root/examples/cosmos3/env_public.sh"

if [[ "$stage" == packages ]]; then
  # Local +cpu pins prevent selection of CUDA wheels. Runtime constraints also
  # keep numpy / LLVM-related Python packages at the reference versions.
  "$python" -m pip install \
    -c "$root/examples/cosmos3/environment/public-runtime.lock.txt" \
    -r "$root/examples/cosmos3/environment/public-torch.downloads.txt"
  export ARCTIC_INFERENCE_PRECOMPILED_OPS=0
  # Triton-Ascend overlays files under triton/: install its base first and
  # the Ascend backend second, rather than depending on pip's root order.
  for name in triton triton-ascend; do
    awk -v name="$name" '$1 == name {print}' \
      "$root/examples/cosmos3/environment/public-runtime.downloads.txt" > "$builds/$name.requirements.txt"
    test -s "$builds/$name.requirements.txt"
    "$python" -m pip install --no-deps --require-hashes -r "$builds/$name.requirements.txt"
  done
  awk '$1 != "triton" && $1 != "triton-ascend"' \
    "$root/examples/cosmos3/environment/public-runtime.downloads.txt" > "$builds/runtime.requirements.txt"
  "$python" -m pip install --no-deps --no-build-isolation --require-hashes \
    -r "$builds/runtime.requirements.txt"
  native="mujoco-$PUBLIC_MUJOCO_VERSION-linux-aarch64.tar.gz"
  base="https://github.com/google-deepmind/mujoco/releases/download/$PUBLIC_MUJOCO_VERSION"
  fetch "$base/$native" "$native"
  fetch "$base/$native.sha256" "$native.sha256"
  (cd "$downloads" && sha256sum -c "$native.sha256")
  mkdir "$builds/mujoco-native" "$builds/mujoco-python"
  tar -xzf "$downloads/$native" -C "$builds/mujoco-native" --strip-components=1
  # Obtain the source distribution URL and its published hash from PyPI.
  details="$("$python" - "$PUBLIC_MUJOCO_VERSION" <<'PYPI_SDIST'
import json, sys, urllib.request
with urllib.request.urlopen('https://pypi.org/pypi/mujoco/' + sys.argv[1] + '/json', timeout=60) as response:
    files = json.load(response)['urls']
source, = [row for row in files if row['packagetype'] == 'sdist']
print(source['url'])
print(source['digests']['sha256'])
PYPI_SDIST
  )"
  mapfile -t source_details <<< "$details"
  fetch "${source_details[0]}" mujoco-python-sdist.tar.gz "${source_details[1]}"
  tar -xzf "$downloads/mujoco-python-sdist.tar.gz" -C "$builds/mujoco-python" --strip-components=1
  # Match the reference build's bounded parallelism and disabled IPO/LLD.
  sed -i "s/f'-j{os.cpu_count()}'/'-j$jobs'/" "$builds/mujoco-python/setup.py"
  export MUJOCO_PATH="$builds/mujoco-native" MUJOCO_PLUGIN_PATH="$builds/mujoco-native/plugin"
  export MUJOCO_CMAKE_ARGS='-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DSUPPORTS_LLD=OFF'
  "$python" -m pip wheel --no-deps --no-build-isolation \
    --wheel-dir "$builds/wheels" "$builds/mujoco-python"
  "$python" -m pip install --no-deps "$builds"/wheels/mujoco-2.3.7-*.whl
  exit 0
fi

if [[ "$stage" == frameworks ]]; then
  # Keep Rust and Cargo private too; do not modify the user's shell profile.
  export CARGO_HOME="$runtime/cargo" RUSTUP_HOME="$runtime/rustup"
  fetch https://static.rust-lang.org/rustup/archive/1.28.2/aarch64-unknown-linux-gnu/rustup-init rustup-init
  bash -c 'chmod +x "$1"; "$1" -y --no-modify-path --profile minimal --default-toolchain "$2"' \
    _ "$downloads/rustup-init" "$PUBLIC_RUST_VERSION"
  export PATH="$CARGO_HOME/bin:$PATH"
  checkout https://github.com/vllm-project/vllm.git "$PUBLIC_VLLM_COMMIT" vllm
  checkout https://github.com/vllm-project/vllm-ascend.git "$PUBLIC_ASCEND_COMMIT" vllm-ascend
  SETUPTOOLS_SCM_PRETEND_VERSION="$PUBLIC_VLLM_VERSION" VLLM_TARGET_DEVICE=empty \
    "$python" -m pip wheel --no-deps --no-build-isolation --wheel-dir "$builds/wheels" "$sources/vllm"
  "$python" -m pip install --no-deps "$builds"/wheels/vllm-0.33.0+empty-*.whl
  SETUPTOOLS_SCM_PRETEND_VERSION="$PUBLIC_ASCEND_VERSION" COMPILE_CUSTOM_KERNELS=0 \
    "$python" -m pip wheel --no-deps --no-build-isolation --wheel-dir "$builds/wheels" "$sources/vllm-ascend"
  "$python" -m pip install --no-deps "$builds"/wheels/vllm_ascend-0.33.0rc1-*.whl
  env -u VLLM_OMNI_VERSION_OVERRIDE SETUPTOOLS_SCM_PRETEND_VERSION=0.33.0rc1 VLLM_OMNI_TARGET_DEVICE=npu \
    "$python" -m pip install --no-deps --no-build-isolation -e "$root"
  "$python" -m pip install --no-deps --no-build-isolation -e "$root/plugins/cosmos3_client"
  exit 0
fi

"$python" -m pip list --format=json > "$logs/packages.json"
"$python" -m pip freeze --all > "$logs/packages.freeze.txt"
if "$python" -m pip check > "$logs/pip-check.txt" 2>&1; then
  echo 'pip check passed' > "$logs/pip-check.status"
else
  echo 'pip check reported dependency metadata conflicts; inspect pip-check.txt' > "$logs/pip-check.status"
fi
(cd "$downloads" && find . -type f ! -name '*.part' -print0 | sort -z | xargs -0 -r sha256sum) > "$logs/downloads.sha256"
if [[ -d "$builds/wheels" ]]; then
  (cd "$builds/wheels" && find . -name '*.whl' -print0 | sort -z | xargs -0 -r sha256sum) > "$logs/built-wheels.sha256"
fi
git -C "$root" rev-parse HEAD > "$logs/project-commit.txt"
git -C "$root" diff --stat > "$logs/project-working-tree.txt"
printf 'Build evidence: %s\nRun verify_environment.py and a real episode before full evaluation.\n' "$logs"
