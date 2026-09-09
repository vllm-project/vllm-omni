// JIT build of pytorch/pytorch#178230 all_to_all_permute (Ulysses-style fused
// permute-free all-to-all over NCCL symmetric memory).
//
// Kernel is verbatim from the PR; the host entry point is adapted to the
// NCCLDevCommManager API shipped in torch 2.11 (we obtain the host ncclComm_t
// via the manager and create our own ncclDevComm with enough LSA barriers,
// cached per group — the PR's keyed-devcomm manager API is not in 2.11).

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/macros/Macros.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <torch/library.h>

#include <map>
#include <mutex>
#include <string>

namespace {

using namespace c10d::symmetric_memory;

#ifndef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
#error "Build requires NCCL >= 2.28 with symmetric-memory device support"
#endif

void copy_rows(const at::Tensor& input, at::Tensor& out) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda(), "a2a_permute: copy_rows requires CUDA tensors");
  TORCH_CHECK(input.device() == out.device(), "a2a_permute: copy_rows device mismatch");
  TORCH_CHECK(input.scalar_type() == out.scalar_type(), "a2a_permute: copy_rows dtype mismatch");
  TORCH_CHECK(input.dim() == 3, "a2a_permute: copy_rows input must be 3-D");
  TORCH_CHECK(out.sizes() == input.sizes(), "a2a_permute: copy_rows shape mismatch");
  TORCH_CHECK(out.is_contiguous(), "a2a_permute: copy_rows output must be contiguous");
  TORCH_CHECK(
      input.stride(2) == 1 && input.stride(1) == input.size(2),
      "a2a_permute: copy_rows requires contiguous rows");

  const int64_t row_elements = input.size(1) * input.size(2);
  TORCH_CHECK(input.stride(0) >= row_elements, "a2a_permute: copy_rows input rows overlap");
  const size_t rows = static_cast<size_t>(input.size(0));
  const size_t row_bytes = static_cast<size_t>(row_elements) * input.element_size();
  const size_t input_pitch =
      static_cast<size_t>(input.stride(0) * input.element_size());

  c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemcpy2DAsync(
      out.data_ptr(),
      row_bytes,
      input.data_ptr(),
      input_pitch,
      row_bytes,
      rows,
      cudaMemcpyDeviceToDevice,
      stream));
}

// ---- kernel (verbatim from PR) ----------------------------------------------
__device__ inline void copy_bytes_vec16_aligned(
    const char* src, char* dst, size_t nbytes, size_t tid, size_t stride) {
  const size_t n_vec = nbytes / 16;
  constexpr int kUnroll = 4;
  size_t vec_idx = tid;
  for (; vec_idx + static_cast<size_t>(kUnroll - 1) * stride < n_vec;
       vec_idx += static_cast<size_t>(kUnroll) * stride) {
    at::native::memory::Vec<16> chunk[kUnroll];
#pragma unroll 4
    for (int u = 0; u < kUnroll; ++u) {
      const size_t i = vec_idx + static_cast<size_t>(u) * stride;
      chunk[u] = at::native::memory::ld_vec<16>(src + i * 16);
    }
#pragma unroll 4
    for (int u = 0; u < kUnroll; ++u) {
      const size_t i = vec_idx + static_cast<size_t>(u) * stride;
      at::native::memory::st_vec<16>(dst + i * 16, chunk[u]);
    }
  }
  for (; vec_idx < n_vec; vec_idx += stride) {
    auto v = at::native::memory::ld_vec<16>(src + vec_idx * 16);
    at::native::memory::st_vec<16>(dst + vec_idx * 16, v);
  }
}

constexpr int A2A_MAX_SLOTS = 64;
constexpr int A2A_MAX_CTAS_PER_SLOT = 128;
constexpr int A2A_THREADS_PER_CTA = 128;
constexpr int A2A_MAX_CTA_COUNT = A2A_MAX_SLOTS * A2A_MAX_CTAS_PER_SLOT;

__global__ void all_to_all_lsa_kernel(
    ncclWindow_t window,
    size_t base_src_byte_offset,
    unsigned char* out,
    int num_rows,
    size_t src_row_stride_bytes,
    size_t copy_row_bytes,
    size_t peer_stride_bytes,
    size_t dst_row_stride_bytes,
    ncclDevComm devComm) {
  const int peer_idx = blockIdx.x;
  const int local_block = blockIdx.y;
  const ncclCoopCta coop{};

  ncclLsaBarrierSession<ncclCoopCta> bar{
      coop, devComm, ncclTeamLsa(devComm), devComm.lsaBarrier,
      blockIdx.x * gridDim.y + blockIdx.y};
  bar.sync(coop, cuda::memory_order_acquire);

  unsigned char* dst_peer_base =
      out + static_cast<size_t>(peer_idx) * peer_stride_bytes;
  CUDA_KERNEL_ASSERT((base_src_byte_offset & 15) == 0);
  CUDA_KERNEL_ASSERT((reinterpret_cast<uintptr_t>(dst_peer_base) & 15) == 0);

  for (int i = local_block; i < num_rows; i += gridDim.y) {
    const size_t row_byte_offset =
        base_src_byte_offset + static_cast<size_t>(i) * src_row_stride_bytes;
    const void* src_row = ncclGetLsaPointer(window, row_byte_offset, peer_idx);
    unsigned char* dst_row =
        dst_peer_base + static_cast<size_t>(i) * dst_row_stride_bytes;
    copy_bytes_vec16_aligned(
        reinterpret_cast<const char*>(src_row),
        reinterpret_cast<char*>(dst_row),
        copy_row_bytes, coop.thread_rank(), coop.size());
  }
  bar.sync(coop, cuda::memory_order_release);
}

// ---- own devcomm cache (replaces PR's keyed manager API) --------------------
ncclDevComm get_or_create_devcomm(ncclComm_t comm, const std::string& group_name) {
  static std::mutex mu;
  static std::map<std::string, ncclDevComm> cache;
  std::lock_guard<std::mutex> lk(mu);
  auto it = cache.find(group_name);
  if (it != cache.end()) return it->second;
#ifdef NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
#else
  ncclDevCommRequirements reqs;
  memset(&reqs, 0, sizeof(ncclDevCommRequirements));
#endif
  reqs.lsaBarrierCount = A2A_MAX_CTA_COUNT;
  ncclDevComm dc;
  ncclResult_t res = ncclDevCommCreate(comm, &reqs, &dc);
  TORCH_CHECK(res == ncclSuccess, "ncclDevCommCreate (a2a_permute) failed: ", ncclGetErrorString(res));
  cache.emplace(group_name, dc);
  return dc;
}

// ---- host entry -------------------------------------------------------------
void all_to_all_permute(
    const at::Tensor& input,
    at::Tensor& out,
    int64_t scatter_dim,
    int64_t gather_dim,
    std::string group_name) {
  TORCH_CHECK(input.stride(-1) == 1, "a2a_permute: innermost dim must be contiguous");
  const bool col_scatter = (scatter_dim == 1 && gather_dim == 0);
  const bool row_scatter = (scatter_dim == 0 && gather_dim == 1);
  TORCH_CHECK(col_scatter || row_scatter,
      "a2a_permute: unsupported (scatter_dim, gather_dim)=(", scatter_dim, ",", gather_dim, ")");

  auto symm_mem = c10d::symmetric_memory::rendezvous(input, group_name);
  TORCH_CHECK(symm_mem != nullptr, "a2a_permute: input must be NCCL symmetric memory");
  auto* nccl_hdl = dynamic_cast<NCCLSymmetricMemory*>(symm_mem.get());
  TORCH_CHECK(nccl_hdl != nullptr, "a2a_permute: requires NCCL symmetric memory backend");

  c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = input.device();

  auto& manager = NCCLDevCommManager::get(device);
  ncclComm_t comm = manager.get_comm(group_name);
  ncclDevComm devcomm = get_or_create_devcomm(comm, group_name);

  const int my_rank = nccl_hdl->get_rank();
  const int p = nccl_hdl->get_world_size();
  TORCH_CHECK(p <= A2A_MAX_SLOTS, "a2a_permute: group size ", p, " exceeds ", A2A_MAX_SLOTS);
  TORCH_CHECK(out.is_contiguous(), "a2a_permute: out must be contiguous");
  TORCH_CHECK(out.scalar_type() == input.scalar_type(), "a2a_permute: dtype mismatch");

  auto window = nccl_hdl->get_window();
  TORCH_CHECK(window != nullptr, "a2a_permute: NCCL window is null");

  const size_t window_base_offset = nccl_hdl->get_offset();
  const int64_t esize = input.element_size();
  const size_t tensor_leading_offset =
      window_base_offset + static_cast<size_t>(input.storage_offset()) * static_cast<size_t>(esize);
  TORCH_CHECK(tensor_leading_offset % 16 == 0, "a2a_permute: tensor byte offset must be 16B aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0, "a2a_permute: input ptr must be 16B aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "a2a_permute: out ptr must be 16B aligned");

  const int unroll = 4 * 16 / static_cast<int>(esize);
  const int elems_per_cta = A2A_THREADS_PER_CTA * unroll;
  int ctas_per_slot = 1;

  if (col_scatter) {
    const int rows = static_cast<int>(input.size(0));
    int64_t total_cols = 0; int local_cols = 0;
    if (input.dim() == 2) {
      total_cols = input.size(1);
      TORCH_CHECK(total_cols % p == 0, "a2a_permute: cols must divide group size");
      local_cols = static_cast<int>(total_cols / p);
    } else {
      TORCH_CHECK(input.dim() == 3, "a2a_permute: input must be 2-D or 3-D");
      TORCH_CHECK(input.size(1) == p, "a2a_permute: 3-D input dim1 must equal group size");
      const int64_t lc = input.size(2);
      total_cols = static_cast<int64_t>(p) * lc;
      TORCH_CHECK(input.stride(1) == lc && input.stride(0) == total_cols,
          "a2a_permute: 3-D input must be row-major contiguous");
      local_cols = static_cast<int>(lc);
    }
    const bool ok3 = out.dim() == 3 && out.size(0) == p && out.size(1) == rows && out.size(2) == local_cols;
    const bool ok2 = out.dim() == 2 && out.size(0) == (int64_t)p * rows && out.size(1) == local_cols;
    TORCH_CHECK(ok3 || ok2, "a2a_permute: bad out shape for (1,0)");
    const size_t row_bytes = (size_t)local_cols * (size_t)esize;
    TORCH_CHECK(row_bytes % 16 == 0, "a2a_permute: local_cols*esize must be 16B-divisible");
    ctas_per_slot = std::max(1, std::min((rows * local_cols + elems_per_cta - 1) / elems_per_cta, A2A_MAX_CTAS_PER_SLOT));
    const size_t esz = (size_t)esize;
    const size_t base_src = tensor_leading_offset + (size_t)my_rank * (size_t)local_cols * esz;
    all_to_all_lsa_kernel<<<dim3(p, ctas_per_slot), A2A_THREADS_PER_CTA, 0, stream>>>(
        window, base_src, reinterpret_cast<unsigned char*>(out.data_ptr()), rows,
        (size_t)total_cols * esz, row_bytes, (size_t)rows * local_cols * esz, (size_t)local_cols * esz, devcomm);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    int local_rows = 0; int cols = 0;
    if (input.dim() == 2) {
      const int64_t total_rows = input.size(0);
      cols = static_cast<int>(input.size(1));
      TORCH_CHECK(total_rows % p == 0, "a2a_permute: rows must divide group size");
      local_rows = static_cast<int>(total_rows / p);
    } else {
      TORCH_CHECK(input.dim() == 3, "a2a_permute: input must be 2-D or 3-D");
      TORCH_CHECK(input.size(0) == p, "a2a_permute: 3-D input dim0 must equal group size");
      local_rows = static_cast<int>(input.size(1));
      const int64_t c = input.size(2); cols = static_cast<int>(c);
      const int64_t s01 = (int64_t)local_rows * c;
      TORCH_CHECK(input.stride(1) == c && input.stride(0) == s01, "a2a_permute: 3-D input must be row-major contiguous");
    }
    const bool ok3 = out.dim() == 3 && out.size(0) == local_rows && out.size(1) == p && out.size(2) == cols;
    const bool ok2 = out.dim() == 2 && out.size(0) == local_rows && out.size(1) == (int64_t)p * cols;
    TORCH_CHECK(ok3 || ok2, "a2a_permute: bad out shape for (0,1)");
    const size_t row_bytes = (size_t)cols * (size_t)esize;
    TORCH_CHECK(row_bytes % 16 == 0, "a2a_permute: cols*esize must be 16B-divisible");
    ctas_per_slot = std::max(1, std::min((local_rows * cols + elems_per_cta - 1) / elems_per_cta, A2A_MAX_CTAS_PER_SLOT));
    const size_t esz = (size_t)esize; const size_t cu = (size_t)cols;
    const size_t base_src = tensor_leading_offset + (size_t)(my_rank * local_rows) * cu * esz;
    all_to_all_lsa_kernel<<<dim3(p, ctas_per_slot), A2A_THREADS_PER_CTA, 0, stream>>>(
        window, base_src, reinterpret_cast<unsigned char*>(out.data_ptr()), local_rows,
        cu * esz, row_bytes, cu * esz, (size_t)p * cu * esz, devcomm);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

} // namespace

TORCH_LIBRARY_FRAGMENT(a2ap, m) {
  m.def("copy_rows(Tensor input, Tensor(a!) out) -> ()");
  m.def("all_to_all_permute(Tensor input, Tensor(a!) out, int scatter_dim, int gather_dim, str group_name) -> ()");
}
TORCH_LIBRARY_IMPL(a2ap, CUDA, m) {
  m.impl("copy_rows", TORCH_FN(copy_rows));
  m.impl("all_to_all_permute", TORCH_FN(all_to_all_permute));
}
