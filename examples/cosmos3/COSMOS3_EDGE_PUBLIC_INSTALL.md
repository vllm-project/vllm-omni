# Cosmos3-Edge：从公开渠道下载和构建910B依赖

本指南介绍如何在昇腾910B服务器上从公开渠道安装项目依赖。项目源码和校验清单从项目仓库获取，模型权重单独准备；Python、CANN、PyTorch、Torch-NPU、LLVM、Mesa、MuJoCo、vLLM和vLLM-Ascend按下表下载或构建，无需离线运行包。

构建在用户指定的项目目录内完成，不覆盖系统Python、系统CANN或已有虚拟环境。主机的Ascend驱动、固件和设备权限仍由管理员准备。适用目标为aarch64、昇腾910B；系统基础检查见[主指南第2节](../../COSMOS3_EDGE.md#2-npu推理机系统基础依赖)。x86_64上的910B主机需要另一套wheel及构建目标，不能直接使用本流程。

软件版本、源码提交和下载地址固定如下。安装后依次完成环境检查、模型校验、smoke和全量评测；成功率、吞吐量和P95以目标机器的实测结果为准。

## 1. 固定版本与下载来源

| 组件 | 本流程使用的版本/源码 | 获取方式 |
| --- | --- | --- |
| Python | 3.12.16 | [Python官方源码](https://www.python.org/downloads/release/python-31216/)，校验官方SHA256后编译 |
| CANN及910B算子包 | 9.1.0，aarch64/A2 | [昇腾官方下载入口](https://www.hiascend.com/cann/download)，按设备和驱动配套选择 |
| PyTorch / torchvision / torchaudio | 2.10.0+cpu / 0.25.0+cpu / 2.10.0+cpu | PyTorch官方CPU wheel索引；`+cpu`用于宿主侧框架，NPU能力由Torch-NPU提供 |
| Torch-NPU | 2.10.0.post4 | [PyPI版本页](https://pypi.org/project/torch-npu/2.10.0.post4/)，cp312/aarch64 wheel |
| Triton-Ascend | 3.2.2 | [华为公开软件源](https://mirrors.huaweicloud.com/ascend/repos/pypi/triton-ascend/)，提供cp312/aarch64 wheel；该版本不从PyPI获取 |
| LLVM | 15.0.7，提交`8dfdcc7b7bf66834a761bd8de445840ef68e4d1a` | LLVM官方仓库，构建AArch64目标与共享LLVM库 |
| Mesa | 23.2.1 | [官方源码归档](https://archive.mesa3d.org/mesa-23.2.1.tar.xz)，SHA256取自[官方发布说明](https://docs.mesa3d.org/relnotes/23.2.1.html) |
| MuJoCo | 2.3.7 | 官方aarch64原生库及其SHA256、PyPI源码分发；为Python3.12编译绑定 |
| vLLM | 提交`4ba787ec9959e566d3b587a796dff07673324da5` | [官方源码](https://github.com/vllm-project/vllm/tree/4ba787ec9959e566d3b587a796dff07673324da5)，`VLLM_TARGET_DEVICE=empty`，构建版本`0.33.0+empty` |
| vLLM-Ascend | 提交`b94b12ec7d3278e40bf4c84ef25c1d595367e0f5` | [官方源码](https://github.com/vllm-project/vllm-ascend/tree/b94b12ec7d3278e40bf4c84ef25c1d595367e0f5)，`COMPILE_CUSTOM_KERNELS=0`，构建版本`0.33.0rc1` |
| 其他构建/运行依赖 | 仓库固定版本 | `environment/public-build.lock.txt`及`environment/public-runtime.lock.txt`，公开PyPI/昇腾软件源 |

软件版本和源码提交统一定义在`environment/public-versions.env`中。构建时按提交哈希获取源码，使用表中指定的包版本标签。

`public-runtime.lock.txt`固定运行依赖的版本。CANN自带的Python组件由CANN安装程序提供；四个PyTorch组件、MuJoCo以及三个项目源码包单独安装。Triton3.5.0提供基础编译器，Triton-Ascend3.2.2提供NPU后端，两者都安装；OpenCV使用固定版本的headless发行包。

## 2. 获取项目源码与配置网络

在910B主机上使用空的部署目录，设置`COSMOS3_PROJECT_REF`为需要使用的项目版本，获取指定提交或标签。

```bash
export REPRO_DIR="$HOME/cosmos3-public-repro"
export COSMOS3_ROOT="$REPRO_DIR/vllm-omni"
export COSMOS3_PROJECT_REF='<项目提交或标签>'
mkdir -p "$REPRO_DIR"
test ! -e "$COSMOS3_ROOT"
git clone git@gitee.com:zeyu_bao/vllm-omni.git "$COSMOS3_ROOT"
cd "$COSMOS3_ROOT"
git checkout --detach "$COSMOS3_PROJECT_REF"
test -f examples/cosmos3/install_public.sh
test -f examples/cosmos3/environment/model-identities.json
test -f vllm_omni/entrypoints/cosmos3/libero_native_frame_wise_relative_rot6d.json
```

本流程允许使用网络代理。仅在需要时，在当前终端设置自己的代理地址，不必修改全局Git配置。下面的地址为占位示例；远端910B的`127.0.0.1`指910B自身，不是登录电脑。

```bash
export HTTP_PROXY=http://proxy-host:proxy-port
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY"
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
# 仅对当前终端的Git生效，离开本终端后不影响其他任务。
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=http.proxy
export GIT_CONFIG_VALUE_0="$HTTP_PROXY"
curl -I --connect-timeout 20 https://pypi.org/simple/
git ls-remote https://github.com/vllm-project/vllm.git HEAD
```

若代理只监听在登录电脑或A800的本地端口，可在该机器上开启到910B的SSH反向转发；账号和SSH别名按实际连接填写：

```bash
# 在实际运行代理的机器执行，保持该会话运行。
ssh -NT -o ExitOnForwardFailure=yes \
 -R 127.0.0.1:17893:127.0.0.1:17892 cosmos3-npu
# 然后在910B的安装终端将HTTP_PROXY/HTTPS_PROXY改为http://127.0.0.1:17893。
```

这里的HTTP代理供公开依赖下载使用；Gitee的SSH访问仍使用当前用户的SSH配置。推理和渲染的本地HTTP地址通过`NO_PROXY`直连。

## 3. 安装系统编译依赖

由管理员在EulerOS/openEuler上安装以下开发包。不同发行版的包名可能不同，例如`pkgconf-pkg-config`可能对应`pkgconfig`；缺包时应安装等价的开发包，不能只安装运行库。

```bash
sudo dnf install -y \
 gcc gcc-c++ make git curl wget tar gzip xz patch binutils unzip which \
 pkgconf-pkg-config autoconf automake libtool bison flex \
 openssl-devel zlib-devel bzip2-devel xz-devel libffi-devel \
 readline-devel sqlite-devel ncurses-devel gdbm-devel libuuid-devel \
 expat-devel libxml2-devel elfutils-libelf-devel libdrm-devel \
 libX11-devel libXrandr-devel libXinerama-devel libXcursor-devel libXi-devel \
 libglvnd-devel numactl-libs libaio
```

确认`gcc`/`g++`、`make`、`git`、`curl`、`tar`、`pkg-config`可用。MuJoCo需要C++17，Arctic Inference的CPU扩展需要C++20，因此完整源码安装应使用支持C++20的编译器；如果运行wheel时出现`GLIBCXX_* not found`，先确认系统C++运行库的版本，不能复制另一台机器的整个`/lib`覆盖系统。新机驱动与CANN9.1.0的配套关系需由管理员通过官方配套表确认。

## 4. 编译Python并创建独立虚拟环境

```bash
cd "$COSMOS3_ROOT"
export COSMOS3_BUILD_JOBS=2
bash examples/cosmos3/install_public.sh --plan
bash examples/cosmos3/install_public.sh python
```

该阶段校验Python3.12.16源码的官方SHA256，将其安装到`.runtime/python-public/`，创建`.venv-public/`，再安装固定版本的构建工具及`public-cann-python.lock.txt`中的CANN Python先决依赖。Python依赖安装在独立虚拟环境中。默认限制两路编译，CPU和内存充足时可自行提高`COSMOS3_BUILD_JOBS`。

## 5. 从官方渠道安装CANN和910B算子包

打开[官方下载入口](https://www.hiascend.com/cann/download)，选择9.1.0、A2/910B、aarch64及目标操作系统，阅读下载协议并取得运行包与910B算子包。本项目固定使用9.1.0版本，应按该版本选择安装文件。

下面是对应命名和公开下载地址示例。若官方入口更新了链接，以入口给出的同版本地址为准；保存实际使用的地址。不要把950、910、310P或x86_64算子包用于910B/aarch64。

```bash
cd "$COSMOS3_ROOT"
export CANN_PREFIX="$COSMOS3_ROOT/.runtime/ascend-public"
export CANN_DOWNLOADS="$COSMOS3_ROOT/.runtime/public-downloads"
export PATH="$COSMOS3_ROOT/.venv-public/bin:$PATH"
mkdir -p "$CANN_DOWNLOADS" "$CANN_PREFIX"
export CANN_RUN_URL='https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann_9.1.0_linux-aarch64.run'
export CANN_OPS_URL='https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-910b-ops_9.1.0_linux-aarch64.run'
curl -fL --retry 3 --continue-at - "$CANN_RUN_URL" \
 -o "$CANN_DOWNLOADS/Ascend-cann_9.1.0_linux-aarch64.run"
curl -fL --retry 3 --continue-at - "$CANN_OPS_URL" \
 -o "$CANN_DOWNLOADS/Ascend-cann-910b-ops_9.1.0_linux-aarch64.run"
bash "$CANN_DOWNLOADS/Ascend-cann_9.1.0_linux-aarch64.run" --check
bash "$CANN_DOWNLOADS/Ascend-cann-910b-ops_9.1.0_linux-aarch64.run" --check
bash "$CANN_DOWNLOADS/Ascend-cann_9.1.0_linux-aarch64.run" \
 --full --install-path="$CANN_PREFIX"
bash "$CANN_DOWNLOADS/Ascend-cann-910b-ops_9.1.0_linux-aarch64.run" \
 --install --install-path="$CANN_PREFIX"
```

按安装程序输出找到主CANN的`set_env.sh`，不能选择某个调试工具或子组件的脚本。例如安装结果可能位于`$CANN_PREFIX/cann/set_env.sh`或`$CANN_PREFIX/ascend-toolkit/set_env.sh`，具体以实际版本输出为准。先查看候选路径，再填入完整路径：

```bash
find "$CANN_PREFIX" -maxdepth 5 -name set_env.sh -print
export COSMOS3_CANN_ENV='/实际安装路径/set_env.sh'
test -f "$COSMOS3_CANN_ENV"
source "$COSMOS3_CANN_ENV"
export COSMOS3_CANN_ROOT="${ASCEND_HOME_PATH:?CANN脚本应设置ASCEND_HOME_PATH}"
test -d "$COSMOS3_CANN_ROOT/opp"
{
 printf 'export COSMOS3_CANN_ENV=%q\n' "$COSMOS3_CANN_ENV"
 printf 'export COSMOS3_CANN_ROOT=%q\n' "$COSMOS3_CANN_ROOT"
} > "$COSMOS3_ROOT/.runtime/public-runtime.env"
printf '%s\n%s\n' "$CANN_RUN_URL" "$CANN_OPS_URL" \
 > "$COSMOS3_ROOT/.runtime/public-build-logs/cann-download-urls.txt"
```

不要对`.runtime/ascend-public`以外的现有CANN目录执行升级或卸载。本流程的`env_public.sh`会先加载刚保存的官方CANN环境，再加载项目的线程、JIT和NPU配置。

## 6. 从源码构建LLVM和Mesa

```bash
cd "$COSMOS3_ROOT"
bash examples/cosmos3/install_public.sh graphics
```

该阶段从LLVM官方仓库获取固定提交，构建LLVM15.0.7共享库和AArch64目标，再从Mesa官方站点下载并校验23.2.1源码，构建`llvmpipe`、OSMesa和共享glapi。输出为：

```text
.runtime/llvm-15.0.7/
.runtime/mesa-23.2.1/lib/libOSMesa.so
.runtime/mesa-23.2.1/lib/libglapi.so
```

这一阶段可能耗时较长，应在CPU和内存有余量时运行。即使选择远程GPU渲染，当前NPU评测客户端的环境初始化和CPU smoke仍会使用OSMesa，因此保留这一步。

## 7. 安装公开Python依赖并构建MuJoCo绑定

### 7.1 Triton-Ascend的公开获取渠道

`triton-ascend==3.2.2`在PyPI的版本接口返回404，但华为公开软件源提供该版本。以下公开地址可用于下载安装包：

- 软件源首页：<https://mirrors.huaweicloud.com/ascend/repos/pypi/>
- 包目录：<https://mirrors.huaweicloud.com/ascend/repos/pypi/triton-ascend/>
- 本流程使用的ARM包：[`triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl`](https://mirrors.huaweicloud.com/ascend/repos/pypi/triton-ascend/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl)。
- 官方备用下载：[GitHub v3.2.2发布页](https://github.com/triton-lang/triton-ascend/releases/tag/v3.2.2)，其Assets也提供上述cp312/aarch64 wheel，并公布SHA256：`259ef27091a11648b4813d5d590bb3118ba395418254b78a93f522bdf0468346`。
- 官方源码与说明：[GitHub源码仓库](https://github.com/triton-lang/triton-ascend)、[v3.2.2文档](https://triton-ascend.readthedocs.io/en/v3.2.2/)。本流程优先使用ARM wheel，无需重编译Triton编译器。

在已创建的`.venv-public`中，可单独安装或下载。Triton-Ascend3.2.2的实际wheel声明依赖`triton==3.5.0`，必须先安装基础Triton，再安装Ascend后端；不能只安装后端而漏掉基础编译器。这里使用`--no-deps`保留本流程固定的NumPy、SciPy等版本；其他依赖由完整运行清单安装。

```bash
cd "$COSMOS3_ROOT"
"$COSMOS3_ROOT/.venv-public/bin/python" -m pip install --no-deps triton==3.5.0
"$COSMOS3_ROOT/.venv-public/bin/python" -m pip install --no-deps \
 --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
 triton-ascend==3.2.2
```

如果先下载再安装：

```bash
mkdir -p "$COSMOS3_ROOT/.runtime/public-downloads"
"$COSMOS3_ROOT/.venv-public/bin/python" -m pip download --no-deps --only-binary=:all: \
 --dest "$COSMOS3_ROOT/.runtime/public-downloads" triton==3.5.0
"$COSMOS3_ROOT/.venv-public/bin/python" -m pip download --no-deps --only-binary=:all: \
 --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
 --dest "$COSMOS3_ROOT/.runtime/public-downloads" triton-ascend==3.2.2
"$COSMOS3_ROOT/.venv-public/bin/python" -m pip install --no-deps \
 "$COSMOS3_ROOT/.runtime/public-downloads/triton-3.5.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
"$COSMOS3_ROOT/.venv-public/bin/python" -m pip install --no-deps \
 "$COSMOS3_ROOT/.runtime/public-downloads/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
```

基础Triton和Ascend后端应同时保留，wheel选择cp312/aarch64。[vLLM-Ascend官方安装说明](https://docs.vllm.ai/projects/ascend/en/v0.27.1rc/getting_started/installation.html)也给出了从该公开软件源安装3.2.2的方式。网络需要代理时，沿用第2节的终端配置。

### 7.2 安装整套运行依赖和MuJoCo

```bash
cd "$COSMOS3_ROOT"
source examples/cosmos3/env_public.sh
bash examples/cosmos3/install_public.sh packages
```

该阶段按`public-torch.downloads.txt`从PyTorch官方CPU站点和PyPI安装四个固定框架wheel，并校验URL中的哈希，再按`public-runtime.downloads.txt`获取242项固定运行包并执行`--require-hashes`校验。脚本先安装基础Triton3.5.0，再安装Ascend后端3.2.2，最后安装其余运行包，避免共享模块目录被反向覆盖。URL和SHA256来自PyPI元数据；Triton-Ascend使用官方GitHub发布资产及其公布的SHA256，华为软件源作为第7.1节的备用渠道。`public-runtime.lock.txt`保留版本约束，用于初始框架依赖解析和人工核对。源码包构建使用当前已安装的工具，禁止隐式创建另一个带不同Torch版本的构建环境。

其中Arctic Inference、BDDL、Gym、Interegular、OpenAI Whisper和Robomimic使用公开源码包；Arctic Inference的nanobind与gRPC代码生成工具已列入构建清单，并通过`ARCTIC_INFERENCE_PRECOMPILED_OPS=0`关闭CUDA自定义算子构建。不会为安装这些源码包而替换Torch版本。

MuJoCo2.3.7没有所需的Python3.12/aarch64组合wheel，因此脚本从[官方发布页](https://github.com/google-deepmind/mujoco/releases/tag/2.3.7)获取aarch64原生库并校验发布的SHA256，从[PyPI](https://pypi.org/project/mujoco/2.3.7/)读取源码包地址及SHA256，再构建绑定。CMake会按MuJoCo2.3.7源码中固定的提交下载Abseil、Eigen、pybind11和GLFW等依赖，网络必须允许访问GitHub和GitLab。脚本只修改绑定构建的并行数，并关闭IPO/LLD，不改MuJoCo物理实现。

## 8. 从固定提交构建vLLM和vLLM-Ascend

```bash
cd "$COSMOS3_ROOT"
source examples/cosmos3/env_public.sh
bash examples/cosmos3/install_public.sh frameworks
```

该阶段将Rust工具链1.95.0安装在项目的`.runtime/cargo`和`.runtime/rustup`中，不修改用户的shell初始化文件，然后获取并核对两个固定源码提交：

```text
vLLM:        4ba787ec9959e566d3b587a796dff07673324da5
vLLM-Ascend: b94b12ec7d3278e40bf4c84ef25c1d595367e0f5
```

vLLM使用`VLLM_TARGET_DEVICE=empty`构建Python框架部分；Ascend适配通过单独的插件提供。vLLM-Ascend使用`COMPILE_CUSTOM_KERNELS=0`，关闭自定义扩展构建。项目源码和客户端插件以可编辑模式安装。

这里的`--no-deps --no-build-isolation`是复现步骤的一部分：该vLLM提交的默认构建元数据要求Torch2.14；该Ascend提交的默认依赖要求Torch-NPU2.10.0.post7、Triton-Ascend3.6.0.post2等，本项目固定使用Torch2.10.0、Torch-NPU2.10.0.post4和Triton-Ascend3.2.2。直接执行普通`pip install .`会改变这套组合。对应上游机制可参阅[vLLM构建配置](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/pyproject.toml)和[vLLM-Ascend源码安装说明](https://docs.vllm.ai/projects/ascend/en/latest/getting_started/installation.html)。

这条路线面向本项目的Cosmos3 FP32/SDPA推理，不能据此宣称该组合支持vLLM-Ascend全部模型和算子。固定依赖与上游包声明可能存在版本约束差异，`pip check`的完整输出和退出状态保存在构建记录中，供环境排查使用。出现构建失败或额外依赖缺失时，应定位具体包，不能通过取消固定版本或装入CUDA版Torch来绕过。

## 9. 保存构建记录并检查环境

```bash
cd "$COSMOS3_ROOT"
source examples/cosmos3/env_public.sh
bash examples/cosmos3/install_public.sh report
cat .runtime/public-build-logs/pip-check.status
cat .runtime/public-build-logs/pip-check.txt
```

构建证据位于`.runtime/public-build-logs/`：

| 文件 | 作用 |
| --- | --- |
| `python.log`、`graphics.log`、`packages.log`、`frameworks.log` | 各阶段原始输出 |
| `download-urls.tsv`、`cann-download-urls.txt` | 实际公开下载来源 |
| `source-commits.tsv` | 获取的源码提交 |
| `downloads.sha256` | 下载文件的SHA256 |
| `built-wheels.sha256` | 在本机生成的wheel的SHA256 |
| `packages.json`、`packages.freeze.txt` | 本机安装后的版本记录 |
| `pip-check.txt`、`pip-check.status` | 依赖元数据检查结果，不代替计算和闭环验证 |
| `project-commit.txt`、`project-working-tree.txt` | 项目代码版本和修改摘要 |

公开构建使用`.runtime/public-build-logs/`中的下载和构建哈希记录。`.cosmos3-native/artifacts.sha256`及源码归档清单用于对应的离线文件，不用于校验公开构建结果或Git工作区。

在空闲NPU上验证实际计算和OSMesa/JIT。下面的`CHECK_NPU`须替换为新机器空闲卡号：

```bash
export CHECK_NPU=4
export COSMOS3_NPU_DEVICES="$CHECK_NPU"
source examples/cosmos3/env_public.sh
source examples/cosmos3/env.sh
mkdir -p "$REPRO_DIR/results"
"$VLLM_PYTHON" examples/cosmos3/verify_environment.py \
 --device npu --output "$REPRO_DIR/results/environment.json"
```

检查Python3.12.16、Torch2.10.0+cpu、Torch-NPU2.10.0.post4、Mesa23.2.1、LLVM15.0.7及非空的Numba编译签名。vLLM模块报告`0.33.0`，其安装包元数据应为`0.33.0+empty`，两者不能混用。

在主指南生成的`npu.env`中，把`VLLM_PYTHON`设为`$COSMOS3_ROOT/.venv-public/bin/python`，`COSMOS3_CLIENT_PYTHON`使用同一路径，并把`COSMOS3_ENV_SCRIPT`设为`$COSMOS3_ROOT/examples/cosmos3/env_public.sh`。按[模型下载步骤](../../COSMOS3_EDGE.md#http下载模型权重)获取B0/F0模型，其中`example.com`为占位地址，使用时替换为实际下载地址。随后按主指南第5.2节校验模型、第6节做一回合闭环；CPU smoke通过后再配置GPU渲染和正式评测。以上小张量检查不能替代真实模型推理。

## 10. GPU渲染机也从公开渠道安装

本节只在x86_64 NVIDIA渲染机执行。NVIDIA驱动和EGL系统库由管理员准备，不能使用NPU的CANN、Torch-NPU或aarch64 wheel。设置自己的`REPRO_DIR`、`COSMOS3_RENDER_ROOT`、`RENDER_VENV`和输出目录，源码使用与NPU相同的源码提交；无需下载离线`render-support.tar.gz`。

```bash
# GPU机：先按主指南第7节生成renderer.env，然后加载。
source "$REPRO_DIR/renderer.env"
export COSMOS3_PROJECT_REF='<与NPU机相同的项目提交或标签>'
git clone git@gitee.com:zeyu_bao/vllm-omni.git "$COSMOS3_RENDER_ROOT"
git -C "$COSMOS3_RENDER_ROOT" checkout --detach "$COSMOS3_PROJECT_REF"
mkdir -p "$REPRO_DIR/downloads" "$REPRO_DIR/build/python311"
curl -fL --retry 3 https://www.python.org/ftp/python/3.11.16/Python-3.11.16.tar.xz \
 -o "$REPRO_DIR/downloads/Python-3.11.16.tar.xz"
tar -xf "$REPRO_DIR/downloads/Python-3.11.16.tar.xz" \
 -C "$REPRO_DIR/build/python311" --strip-components=1
(
 cd "$REPRO_DIR/build/python311"
 ./configure --prefix="$REPRO_DIR/python311" --with-ensurepip=install
 make -j 2
 make install
)
"$REPRO_DIR/python311/bin/python3.11" -m venv "$RENDER_VENV"
"$RENDER_VENV/bin/python" -m pip install \
 -r "$COSMOS3_RENDER_ROOT/examples/cosmos3/environment/renderer-x86_64.lock.txt"
"$RENDER_VENV/bin/python" -m pip install --no-deps robosuite==1.4.1
"$RENDER_VENV/bin/python" -m pip freeze --all > "$RENDER_OUTPUT_ROOT/packages.freeze.txt"
```

若GPU机已有独立的Python3.11.16，可直接用它创建新的`RENDER_VENV`，省略Python源码构建。新编译Python的系统依赖与第3节相同，按GPU机发行版映射包名；Python源码哈希与[官方发布页](https://www.python.org/downloads/release/python-31116/)核对。

接着执行主指南第7节中“自动查找本机NVIDIA EGL库并生成ICD文件”的代码，保留新生成的`.runtime/render-support/nvidia_egl.json`。公开安装的robosuite由GPU虚拟环境提供，不需要离线支持包的`runtime_deps`目录；`PYTHONPATH`只需包含项目客户端路径：

```bash
source "$REPRO_DIR/renderer.env"
export COSMOS3_RENDER_PYTHON="$RENDER_VENV/bin/python"
export PYTHONPATH="$COSMOS3_RENDER_ROOT/plugins/cosmos3_client"
export __EGL_VENDOR_LIBRARY_FILENAMES="$COSMOS3_RENDER_ROOT/.runtime/render-support/nvidia_egl.json"
```

执行主指南第7.1节的独立绘制检查，确认实际渲染器为NVIDIA、图像尺寸正确且不是空图，再按第8–11节启动SSH转发、B0/F0评测并汇总结果。此路线不需要在GPU机安装Torch或vLLM。

## 11. 安装失败时保留哪些证据

脚本按阶段运行。失败时保留该阶段日志、下载文件和构建目录；修复原因后，使用独立项目目录和相同源码版本重建环境。

- 官方下载404：先核对版本、CPU架构和设备型号，并从官方入口重新取得同版本地址，不能静默改成最新版本。
- TLS、代理或Git克隆失败：检查当前终端的HTTP代理和Git临时配置，使用`curl -I`及`git ls-remote`分别验证；下载代理转发与正式评测的渲染转发是两条独立连接。
- MuJoCo绑定构建失败：检查原生库和Python源码均为2.3.7、CMake输出中的依赖下载，以及IPO/LLD和并行数设置。
- LLVM/Mesa失败：检查开发包、编译器、内存、LLVM15.0.7路径及Meson配置；不要用系统旧版OSMesa替代后直接声明已对齐。
- 导入NPU失败：检查CANN9.1.0、910B算子包、驱动兼容性和设备权限；保留动态库错误和`environment.json`。
- 安装后版本检查失败：先确认加载的是`.venv-public`，没有继承其他仓库的`PYTHONPATH`；用`pip list`和实际模块路径核对。

保存构建记录、环境检查、模型校验、CPU/GPU smoke及全量评测结果，用于复核目标机器上的成功率、吞吐量和P95。
