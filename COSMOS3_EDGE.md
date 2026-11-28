# Cosmos3-Edge：安装与使用说明

本项目需部署在昇腾910B服务器上。部署前，所选部署目录应为空，并确保项目源码、Baseline和FASTER两份完整模型权重及对应校验清单已备齐。源码包含推理和评测实现、部署脚本、依赖锁定文件及LIBERO任务资源；模型权重单独提供。

环境可选择**从公开渠道下载安装依赖**或**使用离线运行包安装**。使用公开安装方式时，先按[公开下载和源码构建流程](examples/cosmos3/COSMOS3_EDGE_PUBLIC_INSTALL.md)安装NPU及GPU环境，再使用本文的模型校验、smoke和评测步骤。下文第3–4节和第7节的离线归档安装命令仅用于离线安装，公开安装路线不需要这些归档。

工作目录、仓库路径、模型路径、输出路径及 SSH 连接信息均在配置中统一定义，后续命令通过变量引用这些配置。NPU 推理机和 GPU 渲染机可以由不同用户部署，也可以使用不同的根目录。使用 CPU 渲染时只需要 NPU 推理机，整个流程无需 Docker。

FASTER 是我们引入的优化方案，通过 HAS 时间调度，按就绪情况流式返回动作，使已就绪动作的执行与后续动作的生成重叠；Baseline 则使用 constant Euler 调度，等待完整动作块生成后再执行。后文中的 Ours 均指我们使用的 FASTER 优化方案。

## 1. 准备机器与所需文件

### 1.1 区分两台机器和执行位置

已验证的系统和硬件组合如下。部署前应确认目标机器的环境兼容性。

| 机器 | 已验证配置 | 用途 |
| --- | --- | --- |
| NPU推理机 | aarch64、Huawei Cloud EulerOS 2、glibc 2.34、Ascend 910B1、驱动24.1.0.3 | 模型、物理控制、任务评测 |
| 可选GPU渲染机 | x86_64 Linux、NVIDIA A800及驱动/EGL、Python3.11.16 | 仅渲染相机图像，不执行物理仿真或模型推理 |

主机管理员需先按设备型号安装内核驱动，再分别通过 `npu-smi info`、`nvidia-smi` 验证设备状态。用户态CANN、Python和Mesa按所选安装方式从公开渠道安装或由离线运行包提供。本流程不包含内核驱动安装，也不保证不同驱动与系统组合之间的二进制兼容性。

本文中“NPU机”指目标910B服务器；“GPU机”指提供相机渲染的NVIDIA服务器。物理仿真、控制器和任务成功判定在NPU机的CPU上运行，模型在NPU上运行；GPU机只生成相机图像。

- 一回合CPU渲染检查只需一张空闲NPU。第9节按参考方式长测时，Baseline和FASTER各用一张空闲NPU，GPU0/1分别提供两路渲染服务。示例设备编号必须替换为新机器实际空闲的编号。
- `CHECK_NPU`可复用Baseline或FASTER的一张卡，先完成检查并退出，再启动正式评测，不要求另外借第三张卡。
- GPU机仅在选择CPU渲染部署时可省略。要对齐本文的GPU远程渲染参考测试，应完成第7–9节，CPU渲染结果另行标注。
- 如使用现有A800作为GPU机，只在新的渲染目录/虚拟环境中操作，不在GPU机安装NPU的CANN、Torch-NPU或aarch64 wheels。

### 1.2 所需文件与校验清单

#### 仓库内容

| 内容 | 位置与用途 |
| --- | --- |
| Cosmos3模型及动作推理实现 | `vllm_omni/diffusion/models/cosmos3/`、`vllm_omni/entrypoints/cosmos3/`；包含模型流水线、策略服务及流式动作返回实现 |
| LIBERO评测客户端与时间统计 | `plugins/cosmos3_client/vllm_omni_cosmos3_client/`；包含闭环评测、指标汇总和时间线实现 |
| 远程渲染代码 | 客户端目录下的`rendering/`；包含场景传输、GPU渲染服务和渲染时间统计 |
| LIBERO任务资源 | 客户端目录下的`_vendor/libero/`；包含环境代码、场景与物体资源、任务描述和初始状态 |
| 安装、启动和检查脚本 | `examples/cosmos3/`；包含公开依赖安装、离线安装、环境配置、服务启动、smoke和全量评测脚本 |
| 依赖版本及公开下载清单 | `examples/cosmos3/environment/`；包含固定版本、源码提交、下载地址、SHA256和构建工具清单 |
| 动作归一化统计 | `vllm_omni/entrypoints/cosmos3/libero_native_frame_wise_relative_rot6d.json`；策略服务读取的动作统计 |
| GPU离线渲染支持文件 | `.cosmos3-renderer/`；包含`render-support.tar.gz`、x86_64版`bin/uv`、`manifest.json`和`artifacts.sha256` |
| 文档和测试 | `COSMOS3_EDGE.md`及`examples/cosmos3/`中的部署、计时说明和测试代码 |

#### 校验与版本清单

| 文件 | 校验对象 |
| --- | --- |
| `examples/cosmos3/environment/model-identities.json` | Baseline和FASTER step5000 EMA模型的逐文件大小和SHA256，每份模型27个文件 |
| `examples/cosmos3/environment/public-versions.env` | 公开安装使用的软件版本和源码提交 |
| `examples/cosmos3/environment/public-runtime.downloads.txt`、`public-torch.downloads.txt` | 公开安装的运行依赖及PyTorch组件的下载地址和SHA256 |
| `.cosmos3-native/artifacts.sha256` | 离线运行包、wheel、uv、源码归档和源码清单 |
| `.cosmos3-native/source-working-tree.manifest.json` | 离线`source-working-tree.tar.gz`中的文件及符号链接 |
| `.cosmos3-renderer/manifest.json`、`artifacts.sha256` | GPU离线渲染支持包的说明及文件校验 |

公开安装使用仓库中的版本和下载清单，并保存安装后的构建记录。离线清单仅用于其对应的归档和运行包，不用于校验Git工作区或重新编译的文件。模型校验清单用于指定的两份模型权重。

#### 需单独下载的文件

| 文件 | 获取与使用方式 |
| --- | --- |
| Baseline和FASTER模型权重 | 两份step5000 EMA完整diffusers模型目录，包含transformer、VAE、vision encoder、tokenizer和元数据；下载地址见下表，命令见[模型下载](#http下载模型权重) |
| NPU离线运行包（可选） | `native-runtime.tar`、`mesa-23.2.1.tar`、aarch64 `wheels/`和`bin/uv`；仅在选择离线安装时需要 |
| 源码归档（可选） | `source-working-tree.tar.gz`；仅在选择离线安装时需要，与对应源码清单配套使用 |

Ascend/NVIDIA驱动和设备权限由主机管理员准备。SSH账号、密钥、设备编号、端口和部署路径使用目标机器的配置。

| 模型 | 下载地址 |
| --- | --- |
| Baseline（B0），step5000 EMA | <https://example.com/models/cosmos3_b0_step5000_ema_diffusers.tar.gz> |
| FASTER（F0），step5000 EMA | <https://example.com/models/cosmos3_f0_step5000_ema_diffusers.tar.gz> |

`example.com`为示例域名，下载前替换为实际可访问的地址。每个压缩包应以`diffusers/`为顶层目录，包含该模型的全部文件。模型路径分别由`COSMOS3_BASE_MODEL`和`COSMOS3_FASTER_MODEL`指定。

公开安装按[公开下载和源码构建流程](examples/cosmos3/COSMOS3_EDGE_PUBLIC_INSTALL.md)执行，无需下载上述可选离线文件。选择离线安装时，文件按以下目录布局准备：

```text
artifacts/
  npu/
    source-working-tree.tar.gz
    source-working-tree.manifest.json
    native-runtime.tar
    mesa-23.2.1.tar
    wheels/
    bin/uv
    artifacts.sha256
  renderer/
    render-support.tar.gz
    manifest.json
    bin/uv
    artifacts.sha256
  models/b0/diffusers/
  models/f0/diffusers/
```

`native-runtime.tar`包含Python、CANN、基础Python依赖和运行库；`wheels/`用于按照锁定文件安装依赖。源码目录应包含模型校验清单、动作归一化统计和LIBERO资源。GPU和NPU使用不同架构的Python环境和wheel，分别安装。

使用SSH传输安装文件时，`ARTIFACT_SSH`和`ARTIFACT_ROOT`分别填写文件服务器的SSH别名和目录。安装前校验文件，安装后保存实际环境报告。可编辑安装会生成`vllm_omni/_version.py`，该文件按第4.1节单独记录。

#### 文件获取

源码和校验清单从项目仓库获取；模型权重及可选离线文件通过以下方式获取：

| 方式 | 获取方法 |
| --- | --- |
| HTTP/HTTPS下载 | 使用模型下载地址，按第3.1节下载并解压；支持断点续传 |
| 私有对象存储，如华为云OBS | 使用文件的授权下载地址下载到部署目录；链接失效时重新获取授权地址 |
| SSH/rsync文件服务器 | 使用文件服务器的SSH账号/别名和目录，按第3.1节及第7节用`rsync`拉取 |
| 内网共享目录或移动介质 | 将同一份安装文件目录复制到新机器可读取的位置，再按本文配置路径；仍需执行相同的文件校验 |

OBS授权访问方式见[官方说明](https://support.huaweicloud.com/perms-cfg-obs/obs_40_0009.html)。下载前确认文件地址和访问权限。

无论采用哪种方式，都应一起提供文件大小、SHA256、对应源码提交和目录结构。两份模型使用`model-identities.json`逐文件校验；运行包使用`artifacts.sha256`校验。GPU渲染目录应同时保留支持包、uv、`manifest.json`和校验清单，防止拿到只有部分文件的目录。

## 2. NPU推理机：系统基础依赖

### 2.1 新910B的预检查

在新910B的独立Bash终端操作。以下命令只查看环境，不更换驱动或系统软件：

```bash
uname -m
getconf GNU_LIBC_VERSION
cat /etc/os-release
id
npu-smi info
ls -l /dev/davinci* /dev/devmm_svm /dev/hisi_hdc
free -h
df -h "$HOME" /tmp
command -v bash tar gzip sha256sum curl ssh rsync flock setsid realpath find ldd
```

检查结果按下表处理，不要把原机器的卡号或系统信息直接填入新机器的记录。

| 检查 | 要求/处理 |
| --- | --- |
| CPU架构 | NPU包要求`aarch64`；`x86_64`不能安装这套aarch64 wheel |
| 系统/glibc | 参考为Huawei Cloud EulerOS 2、glibc2.34；低于glibc2.34先停止本流程，不覆盖系统glibc。其他系统即使版本较新，也需重新通过导入和计算检查 |
| 驱动 | 记录新机实际版本及910B子型号；`24.1.0.3`是参考机记录，不是要求覆盖新机驱动的安装命令 |
| CANN兼容性 | 本项目使用CANN9.1.0；新驱动是否支持该运行栈须先由管理员确认。`npu-smi`成功只说明设备可见，后续还必须通过Torch-NPU计算和模型smoke |
| 设备权限 | 当前用户须有选定NPU的设备访问权限；设备缺失或权限不足由管理员处理，不能只检查SSH能否登录 |
| 空闲资源 | 记录实际空闲卡号；核对模型常驻进程、CPU负载及内存，不停止他人的任务 |
| 存储 | 同时容纳安装包、解包后的运行库/源码、两份模型、venv、缓存和日志；可用`du -sh`检查目录大小，不能只按压缩包大小预留 |

离线`native-runtime.tar`约14.2GB（十进制），磁盘还需容纳wheel、两份模型、解包副本、缓存和日志。该归档大小不是模型加载的最低内存要求，资源需求以目标机器实测为准。

当前`env_native.sh`会加入标准驱动库目录`/usr/local/Ascend/driver/lib64/driver`和`/usr/local/Ascend/driver/lib64/common`。检查目录是否存在：

```bash
ls -ld /usr/local/Ascend/driver/lib64/driver \
 /usr/local/Ascend/driver/lib64/common
```

如果驱动安装在非标准位置，应先由管理员确认对应库路径，再调整本次部署的`env_native.sh`驱动库配置并记录差异；仅修改模型路径不能解决驱动库查找问题。

### 2.2 安装主机基础工具

在已验证的 EulerOS 环境中，如果缺少上述工具，可由具有 sudo 权限的用户通过 DNF 安装：

```bash
sudo dnf install -y bash tar gzip coreutils curl openssh-clients rsync \
 util-linux findutils libstdc++ zlib libffi ncurses-libs libselinux
```

使用其他系统时按其包管理器提供等价依赖，并重新验证ABI；不要把上述命令当作Ubuntu命令。无需系统预装pip、conda、MuJoCo或CANN工具包；离线运行包提供Python和这些用户态依赖。驱动通常位于系统的标准Ascend安装位置，用户态CANN使用仓库独立副本，不覆盖系统全局CANN。

## 3. NPU推理机：一次设置自己的路径

下面的 `$HOME/cosmos3-repro` 只是当前用户的默认工作目录；可在执行命令前设置 `REPRO_DIR`，也可直接修改第一行或生成的配置文件中的路径。按实际存储位置设置路径。

```bash
export REPRO_DIR="${REPRO_DIR:-$HOME/cosmos3-repro}"
mkdir -p "$REPRO_DIR"
cat > "$REPRO_DIR/npu.env" <<'NPU_ENV'
export REPRO_DIR="${REPRO_DIR:-$HOME/cosmos3-repro}"
export COSMOS3_ROOT="$REPRO_DIR/vllm-omni"
export COSMOS3_NATIVE_ARTIFACTS="$REPRO_DIR/offline/npu"
export COSMOS3_BASE_MODEL="$REPRO_DIR/models/b0/diffusers"
export COSMOS3_FASTER_MODEL="$REPRO_DIR/models/f0/diffusers"
export OUTPUT_ROOT="$REPRO_DIR/results"
export VLLM_PYTHON="$COSMOS3_ROOT/.venv-native/bin/python"
export COSMOS3_CLIENT_PYTHON="$VLLM_PYTHON"
# 公开安装时改为env_public.sh，同时将上面的Python改为.venv-public/bin/python。
export COSMOS3_ENV_SCRIPT="$COSMOS3_ROOT/examples/cosmos3/env_native.sh"
# 改成自己的空闲设备编号、端口；基线与FASTER(Ours)必须用不同NPU。
export BASE_NPU=4 FASTER_NPU=5 CHECK_NPU=6
export BASE_WORKER_PORT=30100 FASTER_WORKER_PORT=30110
export BASE_RENDER_PORT=30340 FASTER_RENDER_PORT=30341
export NUMBA_DISABLE_JIT=0 NUMBA_OPT=3
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# 防止当前终端继承其他源码/运行库或线程配置。
unset COSMOS3_CANN_ROOT COSMOS3_MESA_ROOT PYTHONPATH LD_LIBRARY_PATH OMP_NUM_THREADS
NPU_ENV
source "$REPRO_DIR/npu.env"
mkdir -p "$COSMOS3_NATIVE_ARTIFACTS" "$OUTPUT_ROOT"
printf '源码目录：%s\n离线包目录：%s\n模型目录：%s\n%s\n' \
 "$COSMOS3_ROOT" "$COSMOS3_NATIVE_ARTIFACTS" \
 "$COSMOS3_BASE_MODEL" "$COSMOS3_FASTER_MODEL"
```

按下一节下载模型，或使用已有的完整模型目录。选择离线安装时，再把NPU离线目录的**内容**复制到`COSMOS3_NATIVE_ARTIFACTS`。如果模型已存放在其他磁盘，修改`npu.env`中的对应路径并重新加载配置即可。

### 3.1 传输安装文件和模型

#### HTTP下载模型权重

在NPU机上执行。先将下面两个示例URL替换为实际模型地址。两种安装方式均可使用本节下载模型；目标模型目录应不存在，避免覆盖已有文件。

```bash
source "$REPRO_DIR/npu.env"
export COSMOS3_BASE_MODEL_URL='https://example.com/models/cosmos3_b0_step5000_ema_diffusers.tar.gz'
export COSMOS3_FASTER_MODEL_URL='https://example.com/models/cosmos3_f0_step5000_ema_diffusers.tar.gz'
(
 set -euo pipefail
 model_download_dir="$REPRO_DIR/downloads/models"
 mkdir -p "$model_download_dir"
 test ! -e "$COSMOS3_BASE_MODEL" || { echo "模型目录已存在：$COSMOS3_BASE_MODEL；使用已有模型或选择新目录" >&2; exit 2; }
 test ! -e "$COSMOS3_FASTER_MODEL" || { echo "模型目录已存在：$COSMOS3_FASTER_MODEL；使用已有模型或选择新目录" >&2; exit 2; }
 for mode in b0 f0; do
  if [[ "$mode" == b0 ]]; then
   model_url="$COSMOS3_BASE_MODEL_URL"
  else
   model_url="$COSMOS3_FASTER_MODEL_URL"
  fi
  model_archive="$model_download_dir/cosmos3_${mode}_step5000_ema_diffusers.tar.gz"
  if [[ ! -f "$model_archive" ]]; then
   curl -fL --retry 3 --connect-timeout 20 --continue-at - \
    "$model_url" -o "$model_archive.part"
   mv "$model_archive.part" "$model_archive"
  fi
 done
 for mode in b0 f0; do
  if [[ "$mode" == b0 ]]; then
   model_dir="$COSMOS3_BASE_MODEL"
  else
   model_dir="$COSMOS3_FASTER_MODEL"
  fi
  model_archive="$model_download_dir/cosmos3_${mode}_step5000_ema_diffusers.tar.gz"
  mkdir -p "$model_dir"
  tar -xzf "$model_archive" -C "$model_dir" --strip-components=1
  test -f "$model_dir/model_index.json"
  printf '%s模型解压目录：%s\n' "$mode" "$model_dir"
 done
)
```

两个压缩包均下载完成后才开始解压。下载中断后可重新运行，`curl`会继续读取`.part`文件；解压中断时，保留压缩包并选择新的空模型目录重新解压。解压时去掉压缩包顶层的`diffusers/`，模型目录中应直接存在`model_index.json`、`transformer/`、`vae/`、`vision_encoder/`和`text_tokenizer/`，不能再嵌套一层`diffusers/`。安装好Python环境后，按第5.2节使用`model-identities.json`逐文件校验，两份模型均应有27个文件通过校验，然后再启动推理。

#### SSH传输安装文件和模型

若安装文件已挂载到目标服务器，可直接复制到上述目录。通过SSH传输时，在**新910B上**执行以下示例，先将两个变量替换为文件服务器的SSH别名和目录：

```bash
source "$REPRO_DIR/npu.env"
export ARTIFACT_SSH=cosmos3-files
export ARTIFACT_ROOT=/path/to/artifacts
ssh "$ARTIFACT_SSH" true
mkdir -p "$COSMOS3_BASE_MODEL" "$COSMOS3_FASTER_MODEL"
rsync -av --partial --info=progress2 \
 "$ARTIFACT_SSH:$ARTIFACT_ROOT/npu/" "$COSMOS3_NATIVE_ARTIFACTS/"
rsync -av --partial --info=progress2 \
 "$ARTIFACT_SSH:$ARTIFACT_ROOT/models/b0/diffusers/" "$COSMOS3_BASE_MODEL/"
rsync -av --partial --info=progress2 \
 "$ARTIFACT_SSH:$ARTIFACT_ROOT/models/f0/diffusers/" "$COSMOS3_FASTER_MODEL/"
```

源路径末尾的`/`表示复制目录内容；完成后应直接存在`$COSMOS3_NATIVE_ARTIFACTS/artifacts.sha256`和`$COSMOS3_BASE_MODEL/model_index.json`，而不是多嵌套一层`npu/`或`diffusers/`。大文件传输中断时可重执行同一条命令，最终仍以SHA256校验为准。

### 3.2 设备和路径配置

编辑`npu.env`，将`BASE_NPU`、`FASTER_NPU`换成第2节查到的两张不同空闲卡；将`CHECK_NPU`设为检查期间空闲的卡。检查与模型smoke退出后才启动正式长测。端口是本机监听端口，不是卡号；Baseline和FASTER的worker/policy端口不得互相冲突。

后续在 NPU 机上打开的每个终端都应先设置相同的 `REPRO_DIR`，再执行 `source "$REPRO_DIR/npu.env"`。这些变量不会跨主机自动同步，重新登录后也需要再次加载配置。

## 4. NPU推理机：解包源码、创建Python环境

请先确认目标源码目录不存在。以下命令在子 shell 中执行，遇到错误即停止；不要将源码解包到正在运行任务的仓库中。

```bash
source "$REPRO_DIR/npu.env"
(
 set -euo pipefail
 cd "$COSMOS3_NATIVE_ARTIFACTS"
 sha256sum -c artifacts.sha256
 mkdir "$COSMOS3_ROOT"
 tar -xzf source-working-tree.tar.gz -C "$COSMOS3_ROOT"
 cd "$COSMOS3_ROOT"
 bash examples/cosmos3/install_native.sh
)
```

本节只描述首次安装，`$COSMOS3_ROOT`和其中的`.venv-native`在开始前均不存在。安装脚本通过`COSMOS3_NATIVE_ARTIFACTS`读取外部包，不要求把包复制到源码目录内。若安装报错，保留安装输出定位原因，不把旧机器venv复制过来覆盖。

安装会依次完成：

1. 按照 SHA256 清单校验离线目录中的文件。
2. 解包 `.runtime/python`、`.runtime/cann-9.1.0`、基础Python依赖与运行库。
3. 解包 `.runtime/mesa-23.2.1`，包含OSMesa/glapi符号链接。
4. 使用包内 uv 创建 Python 3.12.16 虚拟环境，并按照 `native-overlay.lock.txt` 安装固定版本的 wheel。
5. 以可编辑模式安装当前源码。构建时显式设置 `VLLM_OMNI_TARGET_DEVICE=npu`、`TORCH_DEVICE_BACKEND_AUTOLOAD=0`，无需提前加载 CANN 环境脚本。

不需要额外执行pip升级、conda激活、系统CANN替换或Docker启动。后续运行通过 `source "$COSMOS3_ENV_SCRIPT"` 选择本仓库运行库。

### 4.1 安装后的源码与关键文件检查

先检查完整源码的必要资源确实已解包：

```bash
source "$REPRO_DIR/npu.env"
cd "$COSMOS3_ROOT"
test -x "$VLLM_PYTHON"
test -f examples/cosmos3/environment/model-identities.json
test -f vllm_omni/entrypoints/cosmos3/libero_native_frame_wise_relative_rot6d.json
test -d plugins/cosmos3_client/vllm_omni_cosmos3_client/_vendor/libero/init_files
source "$COSMOS3_ENV_SCRIPT"
"$VLLM_PYTHON" --version
```

再按源码清单校验文件。这里使用刚安装的Python，不要求系统预先装Python。仅对可编辑安装会重建的`vllm_omni/_version.py`单独记录当前哈希，其余清单文件须保持一致：

```bash
source "$REPRO_DIR/npu.env"
cd "$COSMOS3_ROOT"
source "$COSMOS3_ENV_SCRIPT"
"$VLLM_PYTHON" - <<'CHECK_SOURCE'
import hashlib, json, os
from pathlib import Path

root = Path(os.environ['COSMOS3_ROOT'])
manifest = json.loads((Path(os.environ['COSMOS3_NATIVE_ARTIFACTS']) /
                       'source-working-tree.manifest.json').read_text())
checked = 0
for row in manifest['files']:
    path = root / row['path']
    if row['kind'] == 'symlink':
        assert path.is_symlink() and os.readlink(path) == row['target'], path
        checked += 1
        continue
    if row['path'] == 'vllm_omni/_version.py':
        print('editable-generated version SHA256:', hashlib.sha256(path.read_bytes()).hexdigest())
        continue
    assert path.is_file() and path.stat().st_size == row['bytes'], path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], path
    checked += 1
print('source manifest verified:', checked, 'entries; created_at:', manifest['created_at'])
CHECK_SOURCE
```

## 5. NPU推理机：配置生效检查与模型校验

`CHECK_NPU` 必须空闲。配置中的编号只是示例，按 `npu-smi info` 结果修改。

```bash
source "$REPRO_DIR/npu.env"
cd "$COSMOS3_ROOT"
source "$COSMOS3_ENV_SCRIPT"
export COSMOS3_NPU_DEVICES="$CHECK_NPU"
source examples/cosmos3/env.sh
"$VLLM_PYTHON" examples/cosmos3/verify_environment.py \
 --device npu --output "$OUTPUT_ROOT/environment.json"
```

环境报告应确认：程序在宿主机原生运行、NPU 计算成功、JIT 设置为 `disabled=0`/`optimization_level=3`、编译签名非空，以及渲染库版本为 Mesa 23.2.1 / LLVM 15.0.7。`locations` 应指向当前配置的源码和虚拟环境，不能指向其他仓库。

这一步会实际分配小型NPU张量和创建OSMesa上下文，须在`CHECK_NPU`空闲时运行。它不加载模型，也不代替后续真实策略推理检查。出现`Permission denied`先检查设备权限；出现`libascend_hal`、`libhccl`或驱动初始化错误，先检查驱动/CANN和动态库路径，不通过随意升级Python包来掩盖问题。

实际固定版本：

| 依赖/行为 | 值 |
| --- | --- |
| Python、torch、torch-npu | 3.12.16、2.10.0+cpu、2.10.0.post4 |
| vLLM、vLLM-Ascend | 0.33.0+empty、0.33.0rc1；Ascend自定义kernel关闭 |
| Triton、Triton-Ascend | 3.5.0、3.2.2；分别提供基础编译器和NPU后端，公开获取方式见[公开指南第7.1节](examples/cosmos3/COSMOS3_EDGE_PUBLIC_INSTALL.md#71-triton-ascend的公开获取渠道) |
| CANN、CPU渲染 | 9.1.0、Mesa23.2.1/llvmpipe LLVM15.0.7 |
| MuJoCo、robosuite | 2.3.7、1.4.1 |
| NumPy、SciPy、Numba、llvmlite | 1.26.4、1.15.3、0.61.2、0.44.0 |
| 线程 | server/client OpenMP默认2/1；OpenBLAS/MKL各1 |
| JIT | 默认开启、OPT3；客户端先加载LLVM，再导入SciPy/Torch |

表中的vLLM版本`0.33.0+empty`是安装包元数据；`verify_environment.py`记录的模块版本为`0.33.0`。下方分别检查两者，避免将正常的版本字段差异误判为安装失败。

### 5.1 将实际版本与参考环境逐项比较

`verify_environment.py`成功退出不等于上表全部配置已自动核验。继续执行以下检查；任一断言失败，先保留`environment.json`和实际输出，确认差异后再决定是否继续评测：

```bash
source "$REPRO_DIR/npu.env"
cd "$COSMOS3_ROOT"
source "$COSMOS3_ENV_SCRIPT"
"$VLLM_PYTHON" - <<'CHECK_ENVIRONMENT'
import importlib.metadata as metadata
import json, os
from pathlib import Path

report = json.loads((Path(os.environ['OUTPUT_ROOT']) / 'environment.json').read_text())
assert report['architecture'] == 'aarch64', report['architecture']
assert report['python_version'] == '3.12.16', report['python_version']
assert report['host_native'] is True, 'Reference deployment runs on the host'
expected = {
    'torch': '2.10.0+cpu', 'torchvision': '0.25.0+cpu',
    'vllm': '0.33.0', 'torch-npu': '2.10.0.post4',
    'mujoco': '2.3.7', 'numpy': '1.26.4', 'scipy': '1.15.3',
}
for name, version in expected.items():
    assert report['versions'].get(name) == version, (name, report['versions'].get(name), version)
for name, version in {
    'vllm': '0.33.0+empty', 'vllm-ascend': '0.33.0rc1', 'numba': '0.61.2',
    'llvmlite': '0.44.0', 'robosuite': '1.4.1',
}.items():
    actual = metadata.version(name)
    assert actual == version, (name, actual, version)
    print(name, actual)
jit = report['control_jit']
assert jit['disabled'] == 0 and jit['optimization_level'] == 3, jit
assert jit['compiled_signatures'] not in ('[]', '', 'None'), jit
assert 'Mesa 23.2.1' in report['rendering']['version'], report['rendering']
assert 'LLVM 15.0.7' in report['rendering']['renderer'], report['rendering']
assert Path(report['python']).resolve() == Path(os.environ['VLLM_PYTHON']).resolve()
print('reference versions, JIT and Mesa checks passed')
CHECK_ENVIRONMENT
```

Torch版本带`+cpu`是本项目使用的配置；NPU支持来自另行安装的Torch-NPU。依赖来自固定运行包和wheel，不将包名相同或能导入当作源码/构建内容相同的证明。完整CANN运行库由归档SHA256绑定，新机驱动版本则在第2节另行记录。

### 5.2 两份参考模型校验

完整校验两份模型，读取所有权重文件，不只比较配置：

```bash
source "$REPRO_DIR/npu.env"
cd "$COSMOS3_ROOT"
source "$COSMOS3_ENV_SCRIPT"
"$VLLM_PYTHON" - <<'CHECK_MODELS'
import hashlib, json, os
from pathlib import Path
manifest=json.loads(Path('examples/cosmos3/environment/model-identities.json').read_text())
for label,variable in [('b0','COSMOS3_BASE_MODEL'),('f0','COSMOS3_FASTER_MODEL')]:
    root=Path(os.environ[variable])
    for row in manifest[label]['files']:
        path=root/row['path']
        assert path.is_file() and path.stat().st_size==row['bytes'],path
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            while block:=stream.read(8*1024*1024):digest.update(block)
        assert digest.hexdigest()==row['sha256'],path
    print(label, 'verified files:', len(manifest[label]['files']))
CHECK_MODELS
```

Baseline 和 Ours 两份参考模型应各有 27 个文件通过校验，包括 transformer 的两个权重分片、VAE、vision encoder、tokenizer 和导出元数据。不要用其他训练权重替代这两份参考模型。模型权重以 bfloat16 格式导出，运行时按 FP32 加载；使用与模型校验清单匹配的权重目录。

## 6. 先做一回合CPU smoke

请在两个终端中分别加载配置，并确认 `CHECK_NPU` 和下方使用的端口空闲。本节的 smoke test 仅使用一张 NPU。

```bash
# NPU推理机：server终端
source "$REPRO_DIR/npu.env"
cd "$COSMOS3_ROOT"
source "$COSMOS3_ENV_SCRIPT"
export COSMOS3_NPU_DEVICES="$CHECK_NPU" COSMOS3_MODEL="$COSMOS3_BASE_MODEL"
export COSMOS3_WORKER_PORT=23100 COSMOS3_POLICY_PORT=23101
export COSMOS3_LOG_DIR="$OUTPUT_ROOT/smoke_base_server"
bash examples/cosmos3/run_server.sh --device npu
```

```bash
# NPU推理机：client终端；无需GPU渲染机或SSH转发
source "$REPRO_DIR/npu.env"
cd "$COSMOS3_ROOT"
source "$COSMOS3_ENV_SCRIPT"
COSMOS3_RENDER=cpu COSMOS3_SERVER_URL=http://127.0.0.1:23101 \
 EVAL_OUT="$OUTPUT_ROOT/smoke_base" \
 bash examples/cosmos3/run_client.sh --task_suite libero_spatial \
 --task_ids 0 --num_trials_per_task 1 --exclude_dummy_time --timing_warmup_requests 0
```

请等待 policy 服务的 `/info` 接口就绪后再启动客户端。Baseline 的 CPU smoke test 完成后，停止该服务以释放 NPU。测试 Ours 时，将模型切换为 FASTER 权重，分别使用 `run_server_faster.sh` 和 `run_client_stream.sh` 启动服务端与客户端，并设置独立的端口和输出目录。

Baseline检查结束后，客户端日志应打印`Overall success rate: 1/1`，`$OUTPUT_ROOT/smoke_base/summary.json`中的`total_episodes=1`、`total_successes=1`，且回合无`error`。如果没有成功，应先排查输入、精度和环境差异，不直接开始4000回合长测。

同时查看worker的实际启动配置：

```bash
source "$REPRO_DIR/npu.env"
grep -E 'dtype=torch.float32|SDPA|strict|precision|custom ops' \
 "$OUTPUT_ROOT/smoke_base_server/worker.log"
```

应确认模型worker使用FP32和SDPA。`/info`中的`sampler_dtype=float32`只表示采样更新精度，不能单凭它判断模型前向精度。严格精度路径由`COSMOS3_STRICT_PRECISION=1`及FP32启动参数启用；自定义扩展的实际启用状态还应结合worker日志记录，不能从一次小张量计算成功推断。

首次运行可能包含Numba编译、运行时缓存创建；这个独立smoke目录不能与正式长测目录混用。CPU smoke成功仅证明基础闭环功能，不保证远程GPU流程已连通，也不代表已复现全量成功率和吞吐量。

若要使用 CPU 渲染，对同一完整子集依次评测两种模式，可执行：

```bash
source "$REPRO_DIR/npu.env"
cd "$COSMOS3_ROOT"
COSMOS3_TRIALS=3 bash examples/cosmos3/run_acceptance_suite.sh \
 libero_spatial "$CHECK_NPU" 23200 cpu "$OUTPUT_ROOT/cpu_pair"
```

该命令在同一张 NPU 上依次运行 Ours 和 Baseline。评测其他子集时，替换命令中的子集名称即可。这与下文两种模式各用一张 NPU 并行评测、每种模式内部的四个子集串行执行的安排不同。

## 7. GPU渲染机：从头创建独立环境

本节仅在 GPU 渲染机上执行。两台机器的 `REPRO_DIR` 可以不同，示例使用各自用户的主目录。请先确认 `nvidia-smi` 能识别 GPU，且系统已安装 NVIDIA EGL 驱动库、Bash、curl、SSH 和 flock。渲染支持包中的 uv 为 x86_64 版本，不要使用 NPU 离线包中的 uv。

```bash
export REPRO_DIR="${REPRO_DIR:-$HOME/cosmos3-repro}"
mkdir -p "$REPRO_DIR"
cat > "$REPRO_DIR/renderer.env" <<'RENDER_ENV'
export REPRO_DIR="${REPRO_DIR:-$HOME/cosmos3-repro}"
export COSMOS3_RENDER_ROOT="$REPRO_DIR/vllm-omni"
export RENDER_ARTIFACTS="$REPRO_DIR/offline/renderer"
export RENDER_SOURCE_ARCHIVE="$REPRO_DIR/offline/source-working-tree.tar.gz"
export RENDER_VENV="$REPRO_DIR/venvs/renderer"
export UV_PYTHON_INSTALL_DIR="$REPRO_DIR/python"
export RENDER_OUTPUT_ROOT="$REPRO_DIR/render-results"
# 自己的NPU机SSH配置别名；在下一节设置，不写固定IP、账号或密钥路径。
export NPU_SSH_TARGET=cosmos3-npu
export BASE_GPU=0 FASTER_GPU=1
# 必须与NPU侧npu.env对应端口一致。
export BASE_RENDER_PORT=30340 FASTER_RENDER_PORT=30341
RENDER_ENV
source "$REPRO_DIR/renderer.env"
mkdir -p "$RENDER_ARTIFACTS" "$RENDER_OUTPUT_ROOT"
```

将GPU渲染支持目录的内容放入 `RENDER_ARTIFACTS`，将**与NPU侧相同版本**的源码包放到 `RENDER_SOURCE_ARCHIVE`。本节同样从新的源码目录和虚拟环境开始。

通过SSH获取时，在GPU机执行以下示例，使用与NPU侧相同版本的安装文件：

```bash
source "$REPRO_DIR/renderer.env"
export ARTIFACT_SSH=cosmos3-files
export ARTIFACT_ROOT=/path/to/artifacts
rsync -av --partial --info=progress2 \
 "$ARTIFACT_SSH:$ARTIFACT_ROOT/renderer/" "$RENDER_ARTIFACTS/"
rsync -av --partial --info=progress2 \
 "$ARTIFACT_SSH:$ARTIFACT_ROOT/npu/source-working-tree.tar.gz" "$RENDER_SOURCE_ARCHIVE"
rsync -av --partial --info=progress2 \
 "$ARTIFACT_SSH:$ARTIFACT_ROOT/npu/artifacts.sha256" "$REPRO_DIR/offline/npu-artifacts.sha256"
```

GPU机需要有`rsync`和可用的SSH登录配置。这里的NPU包校验清单只用于确认源码归档与NPU机拿到的是同一份；不在GPU机安装aarch64运行包。

```bash
source "$REPRO_DIR/renderer.env"
(
 set -euo pipefail
 cd "$RENDER_ARTIFACTS"
 sha256sum -c artifacts.sha256
 expected_source_sha="$(awk '$2 == "source-working-tree.tar.gz" {print $1}' "$REPRO_DIR/offline/npu-artifacts.sha256")"
 test -n "$expected_source_sha"
 printf '%s  %s\n' "$expected_source_sha" "$RENDER_SOURCE_ARCHIVE" | sha256sum -c -
 mkdir "$COSMOS3_RENDER_ROOT"
 tar -xzf "$RENDER_SOURCE_ARCHIVE" -C "$COSMOS3_RENDER_ROOT"
 tar -xzf render-support.tar.gz -C "$COSMOS3_RENDER_ROOT"
 uv_bin="$RENDER_ARTIFACTS/bin/uv"
 "$uv_bin" python install 3.11.16
 "$uv_bin" venv --python 3.11.16 "$RENDER_VENV"
 "$uv_bin" pip sync --python "$RENDER_VENV/bin/python" \
  "$COSMOS3_RENDER_ROOT/examples/cosmos3/environment/renderer-x86_64.lock.txt"
)
```

上述 Python 安装和依赖同步步骤需要联网，全部依赖版本均由锁定文件固定。渲染支持包提供 robosuite 导入所需的文件，不依赖旧实验目录，也无需安装 Torch 或 vLLM。

自动查找本机NVIDIA EGL库并生成ICD文件，避免驱动安装路径硬编码：

```bash
source "$REPRO_DIR/renderer.env"
"$RENDER_VENV/bin/python" - <<'MAKE_EGL_CONFIG'
import json, os, subprocess
from pathlib import Path
lines=subprocess.run(['ldconfig','-p'],check=True,capture_output=True,text=True).stdout.splitlines()
paths=[line.split('=>',1)[1].strip() for line in lines if 'libEGL_nvidia.so.0 ' in line and '=>' in line]
assert paths and Path(paths[0]).is_file(), 'NVIDIA EGL库未找到；先安装/配置NVIDIA驱动'
root=Path(os.environ['COSMOS3_RENDER_ROOT'])/'.runtime/render-support'
root.mkdir(parents=True,exist_ok=True)
(root/'nvidia_egl.json').write_text(json.dumps({'file_format_version':'1.0.0','ICD':{'library_path':paths[0]}})+'\n')
print('NVIDIA EGL:',paths[0])
MAKE_EGL_CONFIG
```

如果驱动库所在目录未登记到 ldconfig，请先登记该目录，或根据驱动库的实际路径生成相同格式的 ICD 文件。GPU 渲染应使用 NVIDIA EGL 驱动，不能用 Mesa EGL 替代。后续在渲染机上打开的每个终端都应先加载配置，再设置以下变量：

```bash
source "$REPRO_DIR/renderer.env"
export COSMOS3_RENDER_PYTHON="$RENDER_VENV/bin/python"
export PYTHONPATH="$COSMOS3_RENDER_ROOT/.runtime/render-support/runtime_deps"
export __EGL_VENDOR_LIBRARY_FILENAMES="$COSMOS3_RENDER_ROOT/.runtime/render-support/nvidia_egl.json"
```

### 7.1 GPU独立渲染检查

在启动NPU模型之前，先在GPU机验证依赖导入和一次实际绘制。下面只使用`BASE_GPU`，需确保该卡空闲；不是模型推理，也不启动物理仿真回合：

```bash
source "$REPRO_DIR/renderer.env"
export COSMOS3_RENDER_PYTHON="$RENDER_VENV/bin/python"
export PYTHONPATH="$COSMOS3_RENDER_ROOT/plugins/cosmos3_client:$COSMOS3_RENDER_ROOT/.runtime/render-support/runtime_deps"
export __EGL_VENDOR_LIBRARY_FILENAMES="$COSMOS3_RENDER_ROOT/.runtime/render-support/nvidia_egl.json"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="$BASE_GPU" \
 "$COSMOS3_RENDER_PYTHON" - <<'CHECK_RENDERER'
import json, os
from pathlib import Path
import mujoco
from OpenGL import GL
from robosuite.utils.binding_utils import MjRenderContextOffscreen
from vllm_omni_cosmos3_client.rendering.protocol import DATA_FIELDS

assert mujoco.__version__ == '2.3.7', mujoco.__version__
xml = '''<mujoco><worldbody>
  <light pos="0 0 3"/>
  <camera name="probe" pos="0 0 2"/>
  <geom type="sphere" size="0.15" rgba="1 0 0 1"/>
</worldbody></mujoco>'''
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, height=64, width=64)
renderer.update_scene(data, camera='probe')
pixels = renderer.render()
backend = GL.glGetString(GL.GL_RENDERER).decode()
assert 'nvidia' in backend.lower(), backend
assert pixels.shape == (64, 64, 3) and pixels.max() > pixels.min(), pixels.shape
report = {'mujoco': mujoco.__version__, 'renderer': backend,
          'physical_gpu': int(os.environ['MUJOCO_EGL_DEVICE_ID']),
          'image_shape': list(pixels.shape)}
output = Path(os.environ['RENDER_OUTPUT_ROOT']) / 'renderer_smoke.json'
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
CHECK_RENDERER
```

若是`libGL`、`libEGL`、`libOpenGL`、`libGLdispatch`缺失，先由GPU机管理员补齐该系统对应的NVIDIA/EGL用户态依赖；若后端为Mesa或llvmpipe，检查ICD选择。不要把“nvidia-smi能运行”当成相机绘制已验证。该检查通过后，再配置SSH反向转发。

## 8. GPU渲染机：配置到NPU机的SSH

在GPU机的SSH配置中创建 `NPU_SSH_TARGET` 对应别名，`HostName`、`User`、`Port` 和 `IdentityFile` 按实际SSH连接填写。示意配置中的名称/账号需替换为真实值；已有SSH别名可直接修改 `renderer.env` 的 `NPU_SSH_TARGET` 使用它。

```sshconfig
Host cosmos3-npu
    HostName npu.example.org
    User YOUR_LOGIN
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

无法直连NPU机时，使用自己网络中的 `ProxyJump` 别名或已授权的SSH转发入口。此处不依赖某台跳板机的IP/端口。NPU端SSH服务须允许remote forwarding，且转发监听绑定在127.0.0.1。

先通过交互连接验证登录身份和主机密钥，再检查是否能够建立无需交互的批处理连接：

```bash
source "$REPRO_DIR/renderer.env"
ssh "$NPU_SSH_TARGET" true
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$NPU_SSH_TARGET" true
```

密钥需已授权；需要解锁时先配置ssh-agent。不要把私钥打包进源码或将 `StrictHostKeyChecking` 关闭。SSH 连接配置负责描述网络拓扑，脚本通过别名引用该配置。

## 9. GPU渲染长测：两种模式并行、每任务50回合

固定评测参数为：单环境、seed=0、初始状态索引 0–49、动作窗口 H16、10 次去噪、guidance=1、flow_shift=1、严格 FP32、eager 执行和 TORCH_SDPA 注意力后端。Baseline 使用 constant Euler 调度并同步返回完整动作块，Ours 使用 HAS 调度并流式返回动作。两种模式使用不同的模型权重，因此不属于同权重条件下的消融对比。

两种模式各使用一张 NPU，模式内部按 Spatial → Object → Goal → LIBERO-10 的顺序串行评测。每个子集包含 10 个任务，每个任务运行 50 回合，共 500 回合；每种模式共 2000 回合，两种模式合计 4000 回合。同一张 NPU 上不同时运行四个客户端。两路 GPU 仅负责渲染，物理仿真始终在 NPU 推理机的 CPU 上执行。

### 9.1 GPU渲染机启动两个后台服务

```bash
source "$REPRO_DIR/renderer.env"
export COSMOS3_RENDER_PYTHON="$RENDER_VENV/bin/python"
export PYTHONPATH="$COSMOS3_RENDER_ROOT/.runtime/render-support/runtime_deps"
export __EGL_VENDOR_LIBRARY_FILENAMES="$COSMOS3_RENDER_ROOT/.runtime/render-support/nvidia_egl.json"
cd "$COSMOS3_RENDER_ROOT"
render_out="$RENDER_OUTPUT_ROOT/serial50_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$render_out"
printf '%s\n' "$render_out" > "$RENDER_OUTPUT_ROOT/last-run.txt"
ssh_args=(-o BatchMode=yes -o StrictHostKeyChecking=yes "$NPU_SSH_TARGET")
nohup bash examples/cosmos3/run_render_background.sh \
 "$BASE_GPU" "$BASE_RENDER_PORT" "$render_out/base" "${ssh_args[@]}" \
 >"$render_out/base.supervisor.log" 2>&1 < /dev/null &
nohup bash examples/cosmos3/run_render_background.sh \
 "$FASTER_GPU" "$FASTER_RENDER_PORT" "$render_out/faster" "${ssh_args[@]}" \
 >"$render_out/faster.supervisor.log" 2>&1 < /dev/null &
printf '渲染输出：%s\n' "$render_out"
```

### 9.2 NPU推理机确认渲染转发，然后启动两种模式

```bash
source "$REPRO_DIR/npu.env"
curl --noproxy '*' -fsS "http://127.0.0.1:$BASE_RENDER_PORT/health"
curl --noproxy '*' -fsS "http://127.0.0.1:$FASTER_RENDER_PORT/health"
```

health 响应中的 `physical_gpu` 应与对应 GPU 服务的配置一致，且 `physics_execution=false`。首次实际渲染后，输出目录中的 `gpu_renderer_proof.json` 应显示 NVIDIA 渲染器和正确的物理 GPU 编号；仅能访问 health 接口并不代表已经成功渲染图像。

```bash
source "$REPRO_DIR/npu.env"
cd "$COSMOS3_ROOT"
output="$OUTPUT_ROOT/serial50_gpu_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$output"
printf '%s\n' "$output" > "$OUTPUT_ROOT/last-run.txt"
nohup bash examples/cosmos3/run_serial_profile.sh \
 base "$BASE_NPU" "$BASE_WORKER_PORT" "http://127.0.0.1:$BASE_RENDER_PORT" "$output" 50 \
 >"$output/base.controller.log" 2>&1 < /dev/null &
nohup bash examples/cosmos3/run_serial_profile.sh \
 faster "$FASTER_NPU" "$FASTER_WORKER_PORT" "http://127.0.0.1:$FASTER_RENDER_PORT" "$output" 50 \
 >"$output/faster.controller.log" 2>&1 < /dev/null &
printf '评测输出：%s\n' "$output"
```

每种模式只加载一次对应权重，服务端在四个子集的评测期间持续运行，每个子集完成后再启动下一个子集的客户端。脚本会检查初始状态数量是否满足 50 回合的要求、模型路径和渲染服务是否可用，以及输出目录是否存在冲突；退出时清理本次启动的客户端和服务端。`COSMOS3_BASE_MODEL`、`COSMOS3_FASTER_MODEL` 指定的路径优先于默认值；未设置时，分别使用仓库下的 `models/b0/diffusers`、`models/f0/diffusers`，不依赖特定用户的个人目录。

通过 nohup 启动的任务可以在交互式 SSH 连接断开后继续运行，但相关机器、后台渲染服务和 SSH 转发仍须保持运行。脚本没有断点续跑；失败后保留原目录，确认原因后在新目录重跑。不要覆盖已完成数据。

## 10. 重新登录后的进度、吞吐与停止

在 NPU 机上重新加载配置后，读取上次保存的输出目录，不要重新生成带有当前时间的目录名：

```bash
source "$REPRO_DIR/npu.env"
output="$(cat "$OUTPUT_ROOT/last-run.txt")"
tail -F "$output/base.controller.log" "$output/faster.controller.log"
# suite按实际正在运行的阶段修改
suite=libero_spatial
tail -F "$output/$suite/base_client.log" "$output/$suite/faster_client.log"
```

日志/证据：

| 文件 | 内容 |
| --- | --- |
| `<profile>.manifest.json` | 实际模型、NPU、renderer、JIT、初态数量、源码SHA256 |
| `<profile>.info.json` | 实际server配置 |
| `<profile>.controller.log` | 串行阶段与子集完成后的 `THROUGHPUT` 行 |
| `<suite>/<profile>_client.log` | 逐回合进度和最终吞吐 |
| `<suite>/<profile>/progress.json` | 已完成回合快照，非最终汇总 |
| `<suite>/<profile>/summary.json` | 子集结束后的完整统计 |
| `<profile>_server/{worker,policy}.log` | 常驻server日志 |
| `<profile>.COMPLETE`、`<profile>.exit_code` | 全部完成及退出码，正常应为0 |
| `<profile>.pid` | controller PID，退出后删除；结合退出码判断结果 |

每个子集全部完成后才会打印该子集的最终吞吐，运行中的逐回合日志不包含最终统计值。当前评测脚本会在 controller 日志中打印三个独立字段：

| controller字段 | summary中 `throughput.` 后的路径 | 含义 |
| --- | --- | --- |
| `raw_actions_per_s` | `evaluation_wall.executed_actions_per_s` | 任务循环全部时间，包含渲染和dummy耗时 |
| `no_render_no_dummy_actions_per_s` | `render_excluded.evaluation_no_render_no_dummy.actions_per_s` | 从整场耗时中扣除观测渲染和 dummy 动作耗时，保留环境创建、重置和记录等耗时 |
| `complete_window_no_render_actions_per_s` | `render_excluded.complete_windows_no_render.actions_per_s` | 仅统计完整动作执行窗口，不代表整场吞吐 |

客户端日志中的整场主吞吐字段为 `PRIMARY_CLOSED_LOOP_NO_RENDER_NO_DUMMY_actions_per_s`；`PRIMARY_NO_RENDER_actions_per_s` 表示完整窗口的吞吐。两者的统计范围不同，不能混用。

```bash
grep -H 'THROUGHPUT' "$output"/*.controller.log
grep -H 'PRIMARY_CLOSED_LOOP_NO_RENDER_NO_DUMMY_actions_per_s' "$output"/*/*_client.log
ls "$output"/*.COMPLETE
cat "$output/base.exit_code" "$output/faster.exit_code"
```

提前停止某一路，NPU机执行：

```bash
test ! -f "$output/base.pid" || kill -TERM "$(cat "$output/base.pid")"
test ! -f "$output/faster.pid" || kill -TERM "$(cat "$output/faster.pid")"
```

评测结束后，GPU机恢复自己的变量并停止自有renderer/SSH转发：

```bash
source "$REPRO_DIR/renderer.env"
render_out="$(cat "$RENDER_OUTPUT_ROOT/last-run.txt")"
test ! -f "$render_out/base/supervisor.pid" || kill -TERM "$(cat "$render_out/base/supervisor.pid")"
test ! -f "$render_out/faster/supervisor.pid" || kill -TERM "$(cat "$render_out/faster/supervisor.pid")"
```

渲染服务和 SSH 转发的日志分别为 `renderer.log`、`tunnel.log`，上级目录中的 `*.supervisor.log` 记录监督进程的日志。停止任务时应按 PID 精确操作，避免使用全局 pkill 误停其他作业。机器重启或进程被 SIGKILL 终止时，shell 无法执行退出清理；处理残留进程前，请先核对 PID 和端口归属。

## 11. 完整汇总与验收

整场主吞吐为实际执行模型动作数除以任务循环有效时间：

```text
T_effective = T_task_loop - duration(union(render_intervals, dummy_intervals))
throughput = executed_model_actions / T_effective
```

需要扣除的时间区间来自客户端时间线中的 `client.render_observation` 和 `client.dummy_stabilization`。先将这些区间裁剪到任务循环范围内，再取并集，以免重复扣除。渲染区间包含完整的跨机 RPC 和场景注册耗时，而非仅包含 GPU 绘制耗时。环境创建、硬重置、本地上下文初始化和结果落盘等耗时仍计入整场分母。用于稳定环境的 dummy 动作虽然实际执行，但不计入模型动作数；已生成但未执行的动作也不计入分子。

完整 H16 窗口的端到端耗时从观测采集开始计算，直到 16 个动作全部执行完成且最终响应接收完毕；仅扣除未与有效工作重叠的渲染时间。将四个子集中所有完整且非预热的窗口合并后，采用线性插值计算 P95，而非对四个子集的 P95 取平均。失败回合必须保留；末尾不足 16 个动作的窗口计入整场吞吐，但不参与完整窗口的 P95 计算。

在同一版本的代码和计时口径下，目标为 `P95_faster <= 0.9 * P95_base`。每个子集的每种模式均至少包含100个完整窗口，全部计划回合完成，运行错误和跳过的回合数均为0。正常执行但任务失败的回合必须保留。

**成功率不明显退化采用预先固定的下降容差，两项必须同时满足：**

| 范围 | 工程判据 | 每种模式的标准规模下允许减少的成功回合数 |
| --- | --- | --- |
| 四套合计 | Ours成功率不低于Baseline减1个百分点 | 2000回合时最多减少20次 |
| 每个子集 | Ours成功率不低于Baseline减2个百分点 | 每套500回合时最多减少10次 |

这里使用百分点，不是相对百分比；边界相等时工程判据通过，成功率提高不受上限限制。总体按全部回合数加权计算，不能用其他子集的提升抵消某个子集超过2个百分点的下降。例如某子集492/500变为491/500，下降0.2个百分点，不会仅因此失败；仍须检查总体和其他条件。对应参数为`--overall-success-drop-pp 1`和`--suite-success-drop-pp 2`，设为0可恢复成功率不得下降的规则。

**工程容差与统计证据分开输出。**默认`--success-policy observed`按上述观察差值判定，同时报告单侧95%置信下界。设为`--success-policy confidence`时，还要求总体及每个子集的成功率差值`Ours − Baseline`的单侧95%下界分别严格高于−1、−2个百分点。区间跨越界限时表示“尚不能证明非劣”，不等于已经证明Ours更差；差异检验不显著也不能作为基本不变的证明。

置信界使用按任务分层的配对percentile bootstrap：在同一任务内按相同回合初态配对Baseline/Ours成败，成对有放回抽样，保持各任务样本数不变。默认5000次、随机种子0；`--success-confidence`、`--success-bootstrap-samples`、`--success-bootstrap-seed`分别控制置信水平、抽样次数和随机种子。输出保留两种模式各自独有的成功数及配对数量，不能仅靠汇总成功数恢复配对置信区间。如果配对不完整则拒绝比较；退化的bootstrap分布标为证据不足，严格统计模式不能自动通过。置信界是固定测试任务与初态抽样假设下的近似推断，不代表未见任务泛化或跨训练seed的稳定性。参数与方法参考[SciPy配对bootstrap说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)。

**动作间、动作块间等待时间仅作为补充参考，不作为硬性验收指标。**执行时间上的连贯性使用客户端时间线分析：对同一回合内相邻的两个`client.execute_action`事件，计算后一次开始减去前一次结束的时间。两次动作属于同一`trace_id`时为块内等待，切换到新的动作块时为块间等待。按实际动作块边界分类，排除dummy稳定动作及跨回合重置间隔，分别报告均值、P95、P99及样本数；合并结果须汇集原始间隔计算，不能平均子集P95。

等待时间保留渲染、图像预处理、请求与客户端处理等造成的完整执行空档，不扣除渲染，也不将其称为纯模型等待。块间等待减少与流式返回期间块内等待增加可能同时出现，两类结果须同时保留。此项不设置通过阈值，不加入`checks`，不影响`passed`或退出码。成功率仍按上文独立判定，置信区间按所选模式使用。

这里描述的是客户端执行时间上的连贯性，不代表实际末端或关节轨迹已经验证平滑。`quality.motion_continuity`仍表示成功率检查没有评估实际运动平滑性。指令跳变、jerk等可用于进一步诊断，但不作为本阶段新增的硬性验收条件。若需要进行轨迹研究，串行客户端保存的`actions/task_XXX/episode_XXX.json`为送入`env.step`的7维控制指令，包含开头的dummy稳定动作；它们不是实际末端位姿或关节状态，不能直接作为物理jerk计算输入。

所有阈值、模式及置信水平应在正式评测前固定，脚本会将实际参数写入输出；命令行允许配置参数，不是不可修改的安全锁。不得为了通过验收而事后调整阈值或丢弃失败回合。对已有数据采用新规则复算，应说明是按新规则的回顾性分析。

```bash
source "$REPRO_DIR/npu.env"
cd "$COSMOS3_ROOT"
source "$COSMOS3_ENV_SCRIPT"
output="$(cat "$OUTPUT_ROOT/last-run.txt")"
args=()
for suite in libero_spatial libero_object libero_goal libero_10; do
 args+=(--baseline "$output/$suite/base" --faster "$output/$suite/faster")
done
"$VLLM_PYTHON" -m vllm_omni_cosmos3_client.acceptance \
 "${args[@]}" --minimum-windows 100 --minimum-improvement-percent 10 \
 --overall-success-drop-pp 1 --suite-success-drop-pp 2 \
 --success-policy observed --success-confidence 0.95 \
 --success-bootstrap-samples 5000 --success-bootstrap-seed 0 \
 --output "$output/e2e_acceptance.json"
```

退出码为0表示当前所选模式的全部检查通过，为2表示未满足验收要求或证据不足；数据缺失或两种模式配置不匹配时，不能判定验收成功。终端及JSON均输出`passed`、`failed_checks`和`quality`，可直接看到失败子集、实际下降幅度、容差与置信界。`quality.observed_tolerance_passed`表示工程容差是否满足；`quality.statistical_noninferiority_supported`表示置信界是否支持非劣，两者不能混用。每种模式应完成2000回合，两种模式合计4000回合。P95门槛仍独立生效，成功率容差通过不等于整体验收通过；当前阶段的吞吐量提升目标仍需按下文独立复算。应分别保留原始墙钟及扣除指定区间后的指标，不能将两卡并行运行总历时与累计设备耗时混用。

可用以下命令独立复算整体吞吐。计算方法为总执行动作数除以总有效耗时，而非对四个子集的吞吐率取平均：

```bash
"$VLLM_PYTHON" - "$output" <<'TOTAL_THROUGHPUT'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
for mode in ['base','faster']:
    actions=effective=raw=episodes=successes=0
    for suite in ['libero_spatial','libero_object','libero_goal','libero_10']:
        s=json.loads((root/suite/mode/'summary.json').read_text())
        assert s['total_episodes']==10*s['num_trials_per_task'],(suite,mode)
        m=s['throughput']['render_excluded']['evaluation_no_render_no_dummy']
        actions+=m['executed_actions'];effective+=m['elapsed_no_render_no_dummy_s']
        raw+=s['throughput']['evaluation_wall']['elapsed_s']
        episodes+=s['total_episodes'];successes+=s['total_successes']
    print(json.dumps(dict(profile=mode,episodes=episodes,successes=successes,
        executed_actions=actions,raw_actions_per_s=actions/raw,
        no_render_no_dummy_actions_per_s=actions/effective),ensure_ascii=False))
TOTAL_THROUGHPUT
```

## 12. 复现检查与排错

已验证的范围包括：在全新目录中离线解包原生运行包，并通过 NPU 计算、JIT OPT3 和 Mesa 23 检查；使用 CPU 渲染完成 Baseline 在 Spatial task 0 上的完整闭环，且任务成功；在独立的 x86 渲染环境中使用 NVIDIA GPU 实际渲染图像。这些验证仅覆盖上述前置条件和流程，不保证新机器上的具体吞吐，也不保证所有轨迹逐位一致。使用相同图像输入时，迁移前后的 Ours 实现生成的动作存在约 3.8e-5 的浮点数值差异，因此配置对齐不代表结果逐位一致。

| 现象 | 检查 |
| --- | --- |
| 不知道修改哪些路径 | 只改 `npu.env`、`renderer.env`、自己的SSH别名；source后打印变量确认 |
| 安装失败 | 离线包是否完整、venv目录是否已存在、是否使用指定版本的项目源码 |
| torch_npu/libhccl错误 | 安装脚本是否保留NPU平台参数和构建禁用自动加载；运行前是否source原生环境 |
| LLVM/JIT崩溃 | 检查 JIT 是否开启、优化级别是否为 OPT3、是否提前加载 LLVM，以及 Mesa 23 是否生效；避免导入旧仓库 |
| framebuffer错误 | OSMesa/glapi是否同包、无版本名符号链接是否存在 |
| 新910B不能加载wheel | 是否aarch64、Python3.12、glibc是否满足运行包要求；不要改名强装其他架构wheel |
| 新910B设备权限不足 | 当前账号能否访问选定davinci设备；请管理员处理驱动及设备权限 |
| 新机NPU初始化失败 | 记录驱动版本、检查用户态CANN9.1.0兼容性及标准驱动库目录 |
| 新环境版本断言失败 | 查看environment.json与metadata版本，确认使用完整安装文件和native-overlay锁；不要直接升级到最新版 |
| renderer不能启动 | NVIDIA驱动/EGL、独立venv、支持包、GPU或端口冲突 |
| NPU侧health不通 | GPU侧服务、SSH别名/转发权限、两侧端口配置是否一致 |
| 看不到最终吞吐 | 子集是否已结束、查看controller还是client、是否查对整场字段 |
| 原始吞吐低但GPU空闲 | 查看场景注册/跨机传输和reset，不只看GPU画帧耗时 |
| 任务中断 | 查exit_code和当前子集日志；没有自动断点续跑，不覆盖旧结果 |

客户端插件自带仿真所需的数据、BDDL、初始状态和纹理，无需额外提供训练用的 demonstrations 目录。默认使用TORCH_SDPA，未验证其他模型、分页、Triton或MindIE-SD路径；不要自行升级依赖后沿用原验收结论。
