// modified from https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/CPUBlas.h>
#include <mkl.h>
#include <c10/core/SymFloat.h>

#include <Python.h>
#include <ATen/ATen.h>
#include <torch/extension.h>

#include <iostream>
#include <string>

namespace {
#define FASTMOE_DISPATCH_CASE_FLOATING_TYPES(...)                                 \
  AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__)                         \
  AT_DISPATCH_CASE(at::kHalf, __VA_ARGS__)
              
#define FASTMOE_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                          \
  AT_DISPATCH_SWITCH(TYPE, NAME, FASTMOE_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

inline c10::SymFloat calculate_scale(
    const at::Tensor& query,
    c10::optional<double> scale) {
  const auto softmax_scale = scale.has_value()
      ? scale.value()
      : (c10::SymFloat(1.0) / (c10::SymFloat(query.sym_size(-1)).sqrt()));
  return c10::SymFloat(softmax_scale);
}


// from Aten/native/cpu/utils.h
template <typename T>
inline void _store(T* dst, at::vec::Vectorized<T> src) {
  src.store(dst);
}

inline void _store(at::BFloat16* dst, at::vec::Vectorized<float> src) {
  auto res = at::vec::convert_float_bfloat16(src, src);
  res.store(dst, at::vec::Vectorized<float>::size());
}

inline void _store(at::Half* dst, at::vec::Vectorized<float> src) {
  auto res = at::vec::convert_float_half(src, src);
  res.store(dst, at::vec::Vectorized<float>::size());
}

template <typename T>
inline T data_index_init(T offset) {
  return offset;
}

template <typename T, typename... Args>
inline T data_index_init(T offset, T& x, const T& X, Args&&... args) {
  offset = data_index_init(offset, std::forward<Args>(args)...);
  x = offset % X;
  return offset / X;
}

inline bool data_index_step() {
  return true;
}

template <typename T, typename... Args>
inline bool data_index_step(T& x, const T& X, Args&&... args) {
  if (data_index_step(std::forward<Args>(args)...)) {
    x = ((x + 1) == X) ? 0 : (x + 1);
    return x == 0;
  }
  return false;
}
// Aten/native/cpu/utils.h

// 1) out = exp(a - val)
// 2) val = sum(out)
template <typename T1, typename T2>
inline void _exp_reduce_sum_fusion_kernel(
    T1* a,
    const int& size,
    T2* out,
    T1& val) {
  auto vec_size = at::vec::Vectorized<T1>::size();
  auto vec_max = at::vec::Vectorized<T1>(val);
  T1 tmp_sum = 0;
  auto vec_tmp_sum = at::vec::Vectorized<T1>(tmp_sum);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<T1>::loadu(a + i);
    auto tmp1 = tmp0 - vec_max;
    // auto tmp2 = tmp1.exp_u20();
    auto tmp2 = tmp1.exp();
    vec_tmp_sum += tmp2;
    _store(out + i, tmp2);
  }
  tmp_sum = at::vec::vec_reduce_all<T1>(
      [](at::vec::Vectorized<T1>& x, at::vec::Vectorized<T1>& y) {
        return x + y;
      },
      vec_tmp_sum);
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 - val;
    auto tmp2 = exp(tmp1);
    tmp_sum += tmp2;
    out[i] = tmp2;
  }
  val = tmp_sum;
}

// 1) max(a)
template <typename scalar_t>
inline void _vec_max_kernel(
    const scalar_t* a,
    const int& size,
    scalar_t& max) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  scalar_t tmp_max = -std::numeric_limits<scalar_t>::infinity();
  auto vec_tmp_max = at::vec::Vectorized<scalar_t>(tmp_max);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(a + i);
    vec_tmp_max = at::vec::maximum(vec_tmp_max, tmp0);
  }
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    tmp_max = std::max(tmp_max, tmp0);
  }
  max = std::max(
      tmp_max,
      at::vec::vec_reduce_all<scalar_t>(
          [](at::vec::Vectorized<scalar_t>& x, at::vec::Vectorized<scalar_t>& y) {
            return at::vec::maximum(x, y);
          },
          vec_tmp_max));
}

template <typename scalar_t>
static inline scalar_t* conditional_data_ptr(scalar_t* ptr, scalar_t* ptr2) {
  TORCH_CHECK(ptr2 == nullptr);
  return ptr;
}

template <typename scalar_t,
          typename std::enable_if_t<std::is_reduced_floating_point_v<scalar_t>, int> = 0>
static inline scalar_t* conditional_data_ptr(float* ptr, scalar_t* ptr2) {
  return ptr2;
}

template <typename scalar_t>
inline void fill_stub(scalar_t* data, scalar_t val, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  Vec data_vec = Vec(val);
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    data_vec.store(data + d);
  }
  #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
  # pragma unroll
  #endif
  for (; d < size; d++) {
    data[d] = val;
  }
}

template <typename scalar_t>
void gemm_dispatch (
    CBLAS_LAYOUT layout,
    CBLAS_TRANSPOSE transA,
    CBLAS_TRANSPOSE transB,
    int64_t m, int64_t n, int64_t k,
    at::opmath_type<scalar_t> alpha,
    const scalar_t* a, int64_t lda,
    const scalar_t* b, int64_t ldb,
    at::opmath_type<scalar_t> beta,
    at::opmath_type<scalar_t>* c, int64_t ldc);

template <>
void gemm_dispatch<float> (
    CBLAS_LAYOUT layout,
    CBLAS_TRANSPOSE transA,
    CBLAS_TRANSPOSE transB,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const float* a, int64_t lda,
    const float* b, int64_t ldb,
    float beta,
    float* c, int64_t ldc) {
  cblas_sgemm(
      layout, transA, transB,
      m, n, k,
      alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void gemm_dispatch<at::Half> (
    CBLAS_LAYOUT layout,
    CBLAS_TRANSPOSE transA,
    CBLAS_TRANSPOSE transB,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const at::Half* a, int64_t lda,
    const at::Half* b, int64_t ldb,
    float beta,
    float* c, int64_t ldc) {
  cblas_gemm_f16f16f32(
      layout, transA, transB,
      m, n, k,
      alpha, (MKL_F16*) a, lda, (MKL_F16*) b, ldb, beta, c, ldc);
}

template <typename scalar_t>
void axpy_dispatch (
    int64_t n,
    at::opmath_type<scalar_t> alpha,
    const scalar_t* x, int64_t incx,
    scalar_t* y, int64_t incy);

template <>
void axpy_dispatch<float> (
    int64_t n,
    float alpha,
    const float* x, int64_t incx,
    float* y, int64_t incy) {
  cblas_saxpy(n, alpha, x, incx, y, incy);
}

template <>
void axpy_dispatch<at::Half> (
    int64_t n,
    float alpha,
    const at::Half* x, int64_t incx,
    at::Half* y, int64_t incy) {
  cblas_saxpy(n, alpha, (float*) x, incx, (float*) y, incy);
}





template <typename scalar_t, int64_t q_split_size, int64_t kv_split_size>
void cpu_flash_decode_gqa(
    const torch::Tensor& output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& seq_lens,
    const at::Tensor& start_loc,
    c10::optional<double> scale) {
  // Query -> (Batch x 1 x Num_Q_heads  x Dim_per_head)
  // Key -> ([kv_seq_len1, kv_seq_len2, ...] x Num_KV_heads  x Dim_per_head)
  // Value -> ([kv_seq_len1, kv_seq_len2, ...] x Num_KV_heads  x Dim_per_head)

  constexpr bool is_reduced_type = std::is_reduced_floating_point_v<scalar_t>;
  using accum_t = at::opmath_type<scalar_t>;
  using Vec = at::vec::Vectorized<accum_t>;
  accum_t scaling_factor = calculate_scale(query, scale).as_float_unchecked();

  // Sizes
  TORCH_CHECK((query.size(3) == value.size(2)) && (key.size(2) == value.size(2)),
        "token_attention_cpu: Q/K/V should have the same head size");
  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(1);
  int64_t num_head = query.size(2);
  int64_t headSize = query.size(3);
  int64_t num_kv_head = key.size(1);

  // Strides
  int64_t qStrideB = query.stride(0);
  int64_t qStrideM = query.stride(1);
  int64_t qStrideH = query.stride(2);
  int64_t kStrideN = key.stride(0);
  int64_t kStrideH = key.stride(1);
  int64_t vStrideN = value.stride(0);
  int64_t vStrideH = value.stride(1);
  int64_t oStrideB = output.stride(0);
  int64_t oStrideM = output.stride(1);
  int64_t oStrideH = output.stride(2);

  int64_t qSplitSize = q_split_size > qSize ? qSize : q_split_size;
  int64_t kvSplitSize = kv_split_size;
  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
  int64_t num_thread = at::get_num_threads();
  int64_t kv_group_num = num_head / num_kv_head;

  const auto dtype = query.scalar_type();
  const auto accumulate_dtype = at::toOpMathType(dtype);

  // allocate per thread temp buf (accumulate type)
  int64_t size_per_thread =
      /* qk     */ kv_group_num * kvSplitSize +
      /* qk_max */ kv_group_num +
      /* qk_sum */ kv_group_num +
      /* dst    */ kv_group_num * headSize;

  at::Tensor buf = at::zeros({num_thread, size_per_thread}, query.options().dtype(accumulate_dtype));
  at::Tensor buf_reduced = at::zeros({num_thread, kv_group_num, is_reduced_type ? kvSplitSize : 0}, query.options());

  // Data ptrs
  const scalar_t* q_data = query.const_data_ptr<scalar_t>();
  const scalar_t* k_data = key.const_data_ptr<scalar_t>();
  const scalar_t* v_data = value.const_data_ptr<scalar_t>();
  const int64_t* seq_lens_data = seq_lens.data_ptr<int64_t>();
  const int64_t* start_loc_data = start_loc.data_ptr<int64_t>();
  scalar_t* out_data = output.data_ptr<scalar_t>();
  accum_t* buf_data = buf.data_ptr<accum_t>();
  scalar_t* buf_reduced_data = is_reduced_type ? buf_reduced.data_ptr<scalar_t>() : nullptr;

  at::parallel_for(0, batchSize * num_kv_head, 1, [&](int64_t begin, int64_t end) {
    int64_t i = 0, j = 0, k = 0;
    data_index_init(begin, i, batchSize, j, num_kv_head, k, qSlice);
    // i : batchSize
    // j : num_kv_head
    // k : qSlice
    int ompIdx = at::get_thread_num();
    accum_t* buf_ptr = buf_data + ompIdx * size_per_thread;
    accum_t* qk_data = buf_ptr;
    accum_t* qk_max_data = qk_data + kv_group_num * kvSplitSize;
    accum_t* qk_sum_data = qk_max_data + kv_group_num;
    accum_t* dst_data = qk_sum_data + kv_group_num;
    scalar_t* qk_reduced_data = is_reduced_type ? buf_reduced_data + ompIdx * kv_group_num * kvSplitSize : nullptr;

    for (const auto z : c10::irange(begin, end)) {
      (void)z; // Suppress unused variable
      int64_t m = j * kv_group_num;
      int64_t qBlockSize = 1;
      // Initialize max and sum
      fill_stub(qk_max_data,
          -std::numeric_limits<accum_t>::infinity(), kv_group_num);
      fill_stub(qk_sum_data,
          static_cast<accum_t>(0), kv_group_num);
      int64_t num_keys = seq_lens_data[i];
      int64_t start_pos = start_loc_data[i];
      for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
        int64_t kvBlockSize = std::min(kvSplitSize, num_keys - n);
        // Calculate scale * q @ k.T
        // query (kv_group_num, head_size), key (kvBlockSize, head_size),
        gemm_dispatch<scalar_t>(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            kv_group_num,
            kvBlockSize,
            headSize,
            scaling_factor,
            q_data + i * qStrideB + m * qStrideH,
            qStrideH,
            k_data + j * kStrideH + (start_pos + n) * kStrideN,
            kStrideN,
            static_cast<accum_t>(0),
            qk_data,
            kvBlockSize);
        
        // Update coefficients with Softmax
        accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;
        for (int64_t row = 0; row < kv_group_num; row++) {
          _vec_max_kernel(
              qk_data + row * kvBlockSize,
              kvBlockSize,
              tmp_max);
          tmp_max = qk_max_data[row] > tmp_max ? qk_max_data[row] : tmp_max;
          tmp_sum = tmp_max;
          _exp_reduce_sum_fusion_kernel(
              qk_data + row * kvBlockSize, 
              kvBlockSize,
              conditional_data_ptr(qk_data, qk_reduced_data)  + row * kvBlockSize,
              tmp_sum);
          exp_tmp = std::exp(qk_max_data[row] - tmp_max);
          qk_sum_data[row] = tmp_sum + exp_tmp * qk_sum_data[row];
          qk_max_data[row] = tmp_max;
          if (n > 0) {
            at::vec::map<accum_t>(
              [exp_tmp](Vec x) { return x * Vec(exp_tmp); },
              dst_data + row * headSize, dst_data + row * headSize, headSize);
          }
        }

        // Calculate Softmax(q @ k.T) @ v
        // qk (kv_group_num, kvBlockSize), v (kvBlockSize, head_size)
        gemm_dispatch<scalar_t>(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            kv_group_num,
            headSize,
            kvBlockSize,
            static_cast<accum_t>(1),
            conditional_data_ptr(qk_data, qk_reduced_data),
            kvBlockSize,
            v_data + j * vStrideH + (start_pos + n) * vStrideN,
            vStrideN,
            n == 0 ? static_cast<accum_t>(0) : static_cast<accum_t>(1),
            dst_data,
            headSize);
      }
      // dst <- dst / sum[row]
      // reorder MHA output with strides)
      for (int64_t row = 0; row < kv_group_num; ++row) {
        accum_t sum_reciprocal = 1 / qk_sum_data[row];
        at::vec::map<scalar_t>(
          [sum_reciprocal](Vec x) { return x * Vec(sum_reciprocal); },
          out_data + i * oStrideB + (m + row) * oStrideH,
          dst_data + row * headSize,
          headSize);
      }
      // Move to the next query
      data_index_step(i, batchSize, j, num_kv_head, k, qSlice);
    }
  });

}





template <typename scalar_t, int64_t q_split_size, int64_t kv_split_size>
void cpu_flash_decode_verified3(
    const torch::Tensor& output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& seq_lens,
    const at::Tensor& start_loc,
    const at::Tensor& tree_mask, // FP32!!!
    const at::Tensor& tree_mask_indptr,
    c10::optional<double> scale) {
  // Query -> (Batch x q_len x Num_Q_heads  x Dim_per_head)
  // Key -> ([kv_seq_len1, kv_seq_len2, ...] x Num_KV_heads  x Dim_per_head)
  // Value -> ([kv_seq_len1, kv_seq_len2, ...] x Num_KV_heads  x Dim_per_head)
  // tree_mask -> bs * q_len * (seq_len + tree_size(q_len))
  // tree_mask_indptr -> 每个request在tree_mask中的起始位置

  constexpr bool is_reduced_type = std::is_reduced_floating_point_v<scalar_t>;
  using accum_t = at::opmath_type<scalar_t>;
  using Vec = at::vec::Vectorized<accum_t>;
  accum_t scaling_factor = calculate_scale(query, scale).as_float_unchecked();

  // Sizes
  TORCH_CHECK((query.size(3) == value.size(2)) && (key.size(2) == value.size(2)),
        "token_attention_cpu: Q/K/V should have the same head size");
  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(1);
  int64_t num_head = query.size(2);
  int64_t headSize = query.size(3);
  int64_t num_kv_head = key.size(1);

  // Strides
  int64_t qStrideB = query.stride(0); 
  int64_t qStrideM = query.stride(1);
  int64_t qStrideH = query.stride(2);
  int64_t kStrideN = key.stride(0);
  int64_t kStrideH = key.stride(1);
  int64_t vStrideN = value.stride(0);
  int64_t vStrideH = value.stride(1);
  int64_t oStrideB = output.stride(0);
  int64_t oStrideM = output.stride(1);
  int64_t oStrideH = output.stride(2);

  int64_t qSplitSize = q_split_size > qSize ? qSize : q_split_size;
  int64_t kvSplitSize = kv_split_size;
  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
  int64_t num_thread = at::get_num_threads();
  int64_t kv_group_num = num_head / num_kv_head;

  const auto dtype = query.scalar_type();
  const auto accumulate_dtype = at::toOpMathType(dtype);

  // allocate per thread temp buf (accumulate type)
  int64_t size_per_thread =
      /* qk     */ qSplitSize * kv_group_num * kvSplitSize +
      /* qk_max */ qSplitSize * kv_group_num +
      /* qk_sum */ qSplitSize * kv_group_num +
      /* dst    */ qSplitSize * kv_group_num * headSize;

  at::Tensor buf = at::zeros({num_thread, size_per_thread}, query.options().dtype(accumulate_dtype));
  at::Tensor buf_reduced = at::zeros({num_thread, qSplitSize * kv_group_num, is_reduced_type ? kvSplitSize : 0}, query.options());
  at::Tensor q_reorder_buf = at::zeros({num_thread, qSplitSize * kv_group_num, headSize}, query.options());

  // Data ptrs
  const scalar_t* q_data = query.const_data_ptr<scalar_t>();
  const scalar_t* k_data = key.const_data_ptr<scalar_t>();
  const scalar_t* v_data = value.const_data_ptr<scalar_t>();
  const int64_t* seq_lens_data = seq_lens.data_ptr<int64_t>();
  const int64_t* start_loc_data = start_loc.data_ptr<int64_t>();
  scalar_t* out_data = output.data_ptr<scalar_t>();
  accum_t* buf_data = buf.data_ptr<accum_t>();
  scalar_t* buf_reduced_data = is_reduced_type ? buf_reduced.data_ptr<scalar_t>() : nullptr;
  const accum_t* tree_mask_data = tree_mask.const_data_ptr<accum_t>();
  const int64_t* tree_mask_indptr_data = tree_mask_indptr.data_ptr<int64_t>();
  scalar_t* q_reorder_data_all = q_reorder_buf.data_ptr<scalar_t>();

  at::parallel_for(0, batchSize * num_kv_head * qSlice, 1, [&](int64_t begin, int64_t end) {
    int64_t i = 0, j_kv = 0, k = 0;
    data_index_init(begin, i, batchSize, j_kv, num_kv_head, k, qSlice);
    int ompIdx = at::get_thread_num();
    accum_t* buf_ptr = buf_data + ompIdx * size_per_thread;
    accum_t* qk_data = buf_ptr;
    accum_t* qk_max_data = qk_data + qSplitSize * kv_group_num * kvSplitSize;
    accum_t* qk_sum_data = qk_max_data + qSplitSize * kv_group_num;
    accum_t* dst_data = qk_sum_data + qSplitSize * kv_group_num;
    scalar_t* q_reorder_data = q_reorder_data_all + ompIdx * qSplitSize * kv_group_num * headSize;
    scalar_t* qk_reduced_data = is_reduced_type ? buf_reduced_data + ompIdx * qSplitSize * kv_group_num * kvSplitSize : nullptr;

    for (const auto z : c10::irange(begin, end)) { // online softmax from begin(bs1, kv_head1, qslice1) to end(bs2, kv_head2, qslice2)
      (void)z; // Suppress unused variable
      int64_t m = k * qSplitSize;
      int64_t qBlockSize = std::min(qSplitSize, qSize - m);
      
      // Initialize max and sum for all query heads in this kv group
      fill_stub(qk_max_data,
          -std::numeric_limits<accum_t>::infinity(), qBlockSize * kv_group_num);
      fill_stub(qk_sum_data,
          static_cast<accum_t>(0), qBlockSize * kv_group_num);
      
      int64_t num_keys = seq_lens_data[i];
      int64_t start_pos = start_loc_data[i];
      int64_t tree_mask_start_pos = tree_mask_indptr_data[i];
      
      // 从 (qBlockSize, kv_group_num, headSize) 排列为 (kv_group_num * qBlockSize, headSize)
      for (int64_t qh = 0; qh < kv_group_num; qh++) {
        for (int64_t row = 0; row < qBlockSize; row++) {
          const scalar_t* src = q_data + i * qStrideB + (m + row) * qStrideM + (j_kv * kv_group_num + qh) * qStrideH;
          scalar_t* dst = q_reorder_data + (qh * qBlockSize + row) * headSize;
          for (int64_t h = 0; h < headSize; h++) {
            dst[h] = src[h];
          }
        }
      }
      
      for (int64_t n = 0; n < num_keys; n += kvSplitSize) { 
        int64_t kvBlockSize = std::min(kvSplitSize, num_keys - n); 
        
        // Calculate scale * q @ k.T for all query heads in this kv group
        gemm_dispatch<scalar_t>(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            kv_group_num * qBlockSize, // M
            kvBlockSize, // N  
            headSize, // K
            scaling_factor,
            q_reorder_data,  
            headSize,        // lda
            k_data + j_kv * kStrideH + (start_pos + n) * kStrideN,
            kStrideN,        // ldb
            static_cast<accum_t>(0),
            qk_data,
            kvBlockSize);    // ldc
        
        // Update coefficients with Softmax for each query head and query token
        for (int64_t qh = 0; qh < kv_group_num; qh++) {
          for (int64_t row = 0; row < qBlockSize; row++) { 
            int64_t qk_offset = (qh * qBlockSize + row) * kvBlockSize;
            int64_t state_offset = qh * qBlockSize + row;
            
            accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;
            
            // Add tree mask
            for (int64_t kv = 0; kv < kvBlockSize; kv++) {
              qk_data[qk_offset + kv] += tree_mask_data[tree_mask_start_pos + (m + row) * num_keys + n + kv];
            }
            
            _vec_max_kernel(
                qk_data + qk_offset,
                kvBlockSize,
                tmp_max);
            tmp_max = qk_max_data[state_offset] > tmp_max ? qk_max_data[state_offset] : tmp_max;
            tmp_sum = tmp_max;
            _exp_reduce_sum_fusion_kernel(
                qk_data + qk_offset, 
                kvBlockSize,
                conditional_data_ptr(qk_data, qk_reduced_data) + qk_offset,
                tmp_sum);
            exp_tmp = std::exp(qk_max_data[state_offset] - tmp_max);
            qk_sum_data[state_offset] = tmp_sum + exp_tmp * qk_sum_data[state_offset];
            qk_max_data[state_offset] = tmp_max;
            
            // Update previous dst values with exp correction
            if (n > 0) {
              for (int64_t h = 0; h < headSize; h++) {
                dst_data[state_offset * headSize + h] *= exp_tmp;
              }
            }
          }
        }

        // Calculate Softmax(q @ k.T) @ v for all query heads
        gemm_dispatch<scalar_t>(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            kv_group_num * qBlockSize, // M
            headSize,                  // N
            kvBlockSize,               // K
            static_cast<accum_t>(1),
            conditional_data_ptr(qk_data, qk_reduced_data),
            kvBlockSize,               // lda
            v_data + j_kv * vStrideH + (start_pos + n) * vStrideN,
            vStrideN,                  // ldb
            n == 0 ? static_cast<accum_t>(0) : static_cast<accum_t>(1),
            dst_data,
            headSize);                 // ldc
      }
      
      // dst <- dst / sum[row] and write to output
      // reorder MHA output with strides
      for (int64_t qh = 0; qh < kv_group_num; qh++) {
        for (int64_t row = 0; row < qBlockSize; ++row) {
          int64_t state_offset = qh * qBlockSize + row;
          accum_t sum_reciprocal = 1 / qk_sum_data[state_offset];
          scalar_t* out_ptr = out_data + i * oStrideB + (m + row) * oStrideM + (j_kv * kv_group_num + qh) * oStrideH;
          accum_t* src_ptr = dst_data + state_offset * headSize;
          
          for (int64_t h = 0; h < headSize; h++) {
            out_ptr[h] = static_cast<scalar_t>(src_ptr[h] * sum_reciprocal);
          }
        }
      }
      
      // Move to the next query
      data_index_step(i, batchSize, j_kv, num_kv_head, k, qSlice);
    }
  });
}




template <typename scalar_t, int64_t q_split_size, int64_t kv_split_size>
void cpu_flash_decode_draft_extend_with_mask_optimized_2(
    const torch::Tensor& output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& seq_lens,
    const at::Tensor& start_loc,
    const at::Tensor& qo_indptr,
    const at::Tensor& tree_mask,
    const at::Tensor& tree_mask_start_pos,
    c10::optional<double> scale) {
  // Query -> ([q_len1, q_len2, ...] x Num_Q_heads  x Dim_per_head)
  // Key -> ([kv_seq_len1, kv_seq_len2, ...] x Num_KV_heads  x Dim_per_head)
  // Value -> ([kv_seq_len1, kv_seq_len2, ...] x Num_KV_heads  x Dim_per_head)
  // qo_indptr -> 每个request在query中的起始位置
  // tree_mask -> custom attention mask for each request
  // tree_mask_start_pos -> starting position for each request in tree_mask

  constexpr bool is_reduced_type = std::is_reduced_floating_point_v<scalar_t>;
  using accum_t = at::opmath_type<scalar_t>;
  using Vec = at::vec::Vectorized<accum_t>;
  accum_t scaling_factor = calculate_scale(query, scale).as_float_unchecked();

  // Sizes
  TORCH_CHECK((query.size(2) == value.size(2)) && (key.size(2) == value.size(2)),
        "token_attention_cpu: Q/K/V should have the same head size");
  int64_t batchSize = qo_indptr.size(0) - 1;  // qo_indptr的长度比batch多1
  int64_t num_head = query.size(1);
  int64_t headSize = query.size(2);
  int64_t num_kv_head = key.size(1);

  // Strides
  int64_t qStrideN = query.stride(0);
  int64_t qStrideH = query.stride(1);
  int64_t kStrideN = key.stride(0);
  int64_t kStrideH = key.stride(1);
  int64_t vStrideN = value.stride(0);
  int64_t vStrideH = value.stride(1);
  int64_t oStrideN = output.stride(0);
  int64_t oStrideH = output.stride(1);

  int64_t qSplitSize = q_split_size;
  int64_t kvSplitSize = kv_split_size;
  int64_t num_thread = at::get_num_threads();
  int64_t kv_group_num = num_head / num_kv_head;

  const auto dtype = query.scalar_type();
  const auto accumulate_dtype = at::toOpMathType(dtype);

  int64_t max_q_len = 0;
  const int64_t* qo_indptr_data = qo_indptr.data_ptr<int64_t>();
  for (int64_t i = 0; i < batchSize; i++) {
    int64_t q_len = qo_indptr_data[i + 1] - qo_indptr_data[i];
    max_q_len = std::max(max_q_len, q_len);
  }

  int64_t max_qSlice = (max_q_len - 1) / qSplitSize + 1;

  // allocate per thread temp buf (accumulate type)
  int64_t size_per_thread =
      /* qk     */ qSplitSize * kv_group_num * kvSplitSize +
      /* qk_max */ qSplitSize * kv_group_num +
      /* qk_sum */ qSplitSize * kv_group_num +
      /* dst    */ qSplitSize * kv_group_num * headSize;

  at::Tensor buf = at::zeros({num_thread, size_per_thread}, query.options().dtype(accumulate_dtype));
  at::Tensor buf_reduced = at::zeros({num_thread, qSplitSize * kv_group_num, is_reduced_type ? kvSplitSize : 0}, query.options());
  at::Tensor q_reorder_buf = at::zeros({num_thread, qSplitSize * kv_group_num, headSize}, query.options());

  // Data ptrs
  const scalar_t* q_data = query.const_data_ptr<scalar_t>();
  const scalar_t* k_data = key.const_data_ptr<scalar_t>();
  const scalar_t* v_data = value.const_data_ptr<scalar_t>();
  const int64_t* seq_lens_data = seq_lens.data_ptr<int64_t>();
  const int64_t* start_loc_data = start_loc.data_ptr<int64_t>();
  scalar_t* out_data = output.data_ptr<scalar_t>();
  accum_t* buf_data = buf.data_ptr<accum_t>();
  scalar_t* buf_reduced_data = is_reduced_type ? buf_reduced.data_ptr<scalar_t>() : nullptr;
  const accum_t* tree_mask_data = tree_mask.const_data_ptr<accum_t>();
  const int64_t* tree_mask_start_pos_data = tree_mask_start_pos.data_ptr<int64_t>();
  scalar_t* q_reorder_data_all = q_reorder_buf.data_ptr<scalar_t>();

  at::parallel_for(0, batchSize * num_kv_head * max_qSlice, 1, [&](int64_t begin, int64_t end) {
    int64_t i = 0, j_kv = 0, k = 0;
    data_index_init(begin, i, batchSize, j_kv, num_kv_head, k, max_qSlice);
    int ompIdx = at::get_thread_num();
    accum_t* buf_ptr = buf_data + ompIdx * size_per_thread;
    accum_t* qk_data = buf_ptr;
    accum_t* qk_max_data = qk_data + qSplitSize * kv_group_num * kvSplitSize;
    accum_t* qk_sum_data = qk_max_data + qSplitSize * kv_group_num;
    accum_t* dst_data = qk_sum_data + qSplitSize * kv_group_num;
    scalar_t* q_reorder_data = q_reorder_data_all + ompIdx * qSplitSize * kv_group_num * headSize;
    scalar_t* qk_reduced_data = is_reduced_type ? buf_reduced_data + ompIdx * qSplitSize * kv_group_num * kvSplitSize : nullptr;

    for (const auto z : c10::irange(begin, end)) {
      (void)z; // Suppress unused variable
      
      int64_t q_start = qo_indptr_data[i];
      int64_t q_end = qo_indptr_data[i + 1];
      int64_t q_len = q_end - q_start;
      
      int64_t m = k * qSplitSize;
      if (m >= q_len) {
        data_index_step(i, batchSize, j_kv, num_kv_head, k, max_qSlice);
        continue;
      }
      
      int64_t qBlockSize = std::min(qSplitSize, q_len - m);
      
      // Initialize max and sum for all query heads in this kv group
      fill_stub(qk_max_data,
          -std::numeric_limits<accum_t>::infinity(), qBlockSize * kv_group_num);
      fill_stub(qk_sum_data,
          static_cast<accum_t>(0), qBlockSize * kv_group_num);
      
      int64_t num_keys = seq_lens_data[i];
      int64_t start_pos = start_loc_data[i];
      int64_t mask_start_pos = tree_mask_start_pos_data[i];
      
      for (int64_t qh = 0; qh < kv_group_num; qh++) {
        for (int64_t row = 0; row < qBlockSize; row++) {
          int64_t q_pos = q_start + m + row;
          const scalar_t* src = q_data + q_pos * qStrideN + (j_kv * kv_group_num + qh) * qStrideH;
          scalar_t* dst = q_reorder_data + (qh * qBlockSize + row) * headSize;
          for (int64_t h = 0; h < headSize; h++) {
            dst[h] = src[h];
          }
        }
      }
      
      for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
        int64_t kvBlockSize = std::min(kvSplitSize, num_keys - n); 
        
        int64_t min_visible_kv_len = num_keys;
        for (int64_t row = 0; row < qBlockSize; row++) {
          int64_t visible_kv_len = num_keys - (q_len - 1 - (m + row));
          min_visible_kv_len = std::min(min_visible_kv_len, visible_kv_len);
        }
        
        if (n >= min_visible_kv_len) {
          continue;
        }
        
        kvBlockSize = std::min(kvBlockSize, min_visible_kv_len - n);
        
        // Calculate scale * q @ k.T for all query heads in this kv group
        gemm_dispatch<scalar_t>(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            kv_group_num * qBlockSize, // M
            kvBlockSize, // N  
            headSize, // K
            scaling_factor,
            q_reorder_data,  
            headSize,        // lda
            k_data + j_kv * kStrideH + (start_pos + n) * kStrideN,
            kStrideN,        // ldb
            static_cast<accum_t>(0),
            qk_data,
            kvBlockSize);    // ldc
        
        // Update coefficients with Softmax for each query head and query token
        for (int64_t qh = 0; qh < kv_group_num; qh++) { 
          for (int64_t row = 0; row < qBlockSize; row++) { 
            int64_t q_pos = q_start + m + row;
            int64_t qk_offset = (qh * qBlockSize + row) * kvBlockSize;
            int64_t state_offset = qh * qBlockSize + row;
            
            int64_t visible_kv_len = num_keys - (q_len - 1 - (m + row));
            
            if (n >= visible_kv_len) {
              continue;
            }
            
            int64_t actual_kvBlockSize = std::min(kvBlockSize, visible_kv_len - n);
            
            accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;
            
            // Add tree mask: qk += mask
            int64_t mask_offset = mask_start_pos + (m + row) * num_keys + n;
            for (int64_t kv = 0; kv < actual_kvBlockSize; kv++) {
              qk_data[qk_offset + kv] += tree_mask_data[mask_offset + kv];
            }
            
            _vec_max_kernel(
                qk_data + qk_offset,
                actual_kvBlockSize,
                tmp_max);
            tmp_max = qk_max_data[state_offset] > tmp_max ? qk_max_data[state_offset] : tmp_max;
            tmp_sum = tmp_max;
            _exp_reduce_sum_fusion_kernel(
                qk_data + qk_offset, 
                actual_kvBlockSize,
                conditional_data_ptr(qk_data, qk_reduced_data) + qk_offset,
                tmp_sum);
            exp_tmp = std::exp(qk_max_data[state_offset] - tmp_max);
            qk_sum_data[state_offset] = tmp_sum + exp_tmp * qk_sum_data[state_offset];
            qk_max_data[state_offset] = tmp_max;
            
            // Update previous dst values with exp correction
            if (n > 0) {
              for (int64_t h = 0; h < headSize; h++) {
                dst_data[state_offset * headSize + h] *= exp_tmp;
              }
            }
          }
        }

        // Calculate Softmax(q @ k.T) @ v for all query heads
        gemm_dispatch<scalar_t>(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            kv_group_num * qBlockSize, // M
            headSize,                  // N
            kvBlockSize,               // K
            static_cast<accum_t>(1),
            conditional_data_ptr(qk_data, qk_reduced_data),
            kvBlockSize,               // lda
            v_data + j_kv * vStrideH + (start_pos + n) * vStrideN,
            vStrideN,                  // ldb
            n == 0 ? static_cast<accum_t>(0) : static_cast<accum_t>(1),
            dst_data,
            headSize);                 // ldc
      }
      
      // dst <- dst / sum[row] and write to output
      // reorder MHA output with strides
      for (int64_t qh = 0; qh < kv_group_num; qh++) {
        for (int64_t row = 0; row < qBlockSize; ++row) {
          int64_t q_pos = q_start + m + row;
          int64_t state_offset = qh * qBlockSize + row;
          accum_t sum_reciprocal = 1 / qk_sum_data[state_offset];
          scalar_t* out_ptr = out_data + q_pos * oStrideN + (j_kv * kv_group_num + qh) * oStrideH;
          accum_t* src_ptr = dst_data + state_offset * headSize;
          
          for (int64_t h = 0; h < headSize; h++) {
            out_ptr[h] = static_cast<scalar_t>(src_ptr[h] * sum_reciprocal);
          }
        }
      }
      
      // Move to the next query
      data_index_step(i, batchSize, j_kv, num_kv_head, k, max_qSlice);
    }
  });
}

template <typename scalar_t, int64_t q_split_size, int64_t kv_split_size>
void cpu_flash_decode_draft_extend_optimized_2(
    const torch::Tensor& output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& seq_lens,
    const at::Tensor& start_loc,
    const at::Tensor& qo_indptr,
    c10::optional<double> scale) {
  // Query -> ([q_len1, q_len2, ...] x Num_Q_heads  x Dim_per_head)
  // Key -> ([kv_seq_len1, kv_seq_len2, ...] x Num_KV_heads  x Dim_per_head)
  // Value -> ([kv_seq_len1, kv_seq_len2, ...] x Num_KV_heads  x Dim_per_head)
  // qo_indptr -> 每个request在query中的起始位置

  constexpr bool is_reduced_type = std::is_reduced_floating_point_v<scalar_t>;
  using accum_t = at::opmath_type<scalar_t>;
  using Vec = at::vec::Vectorized<accum_t>;
  accum_t scaling_factor = calculate_scale(query, scale).as_float_unchecked();

  // Sizes
  TORCH_CHECK((query.size(2) == value.size(2)) && (key.size(2) == value.size(2)),
        "token_attention_cpu: Q/K/V should have the same head size");
  int64_t batchSize = qo_indptr.size(0) - 1;  // qo_indptr的长度比batch多1
  int64_t num_head = query.size(1);
  int64_t headSize = query.size(2);
  int64_t num_kv_head = key.size(1);

  // Strides
  int64_t qStrideN = query.stride(0);
  int64_t qStrideH = query.stride(1);
  int64_t kStrideN = key.stride(0);
  int64_t kStrideH = key.stride(1);
  int64_t vStrideN = value.stride(0);
  int64_t vStrideH = value.stride(1);
  int64_t oStrideN = output.stride(0);
  int64_t oStrideH = output.stride(1);

  int64_t qSplitSize = q_split_size;
  int64_t kvSplitSize = kv_split_size;
  int64_t num_thread = at::get_num_threads();
  int64_t kv_group_num = num_head / num_kv_head;

  const auto dtype = query.scalar_type();
  const auto accumulate_dtype = at::toOpMathType(dtype);

  int64_t max_q_len = 0;
  const int64_t* qo_indptr_data = qo_indptr.data_ptr<int64_t>();
  for (int64_t i = 0; i < batchSize; i++) {
    int64_t q_len = qo_indptr_data[i + 1] - qo_indptr_data[i];
    max_q_len = std::max(max_q_len, q_len);
  }

  int64_t max_qSlice = (max_q_len - 1) / qSplitSize + 1;

  // allocate per thread temp buf (accumulate type)
  int64_t size_per_thread =
      /* qk     */ qSplitSize * kv_group_num * kvSplitSize +
      /* qk_max */ qSplitSize * kv_group_num +
      /* qk_sum */ qSplitSize * kv_group_num +
      /* dst    */ qSplitSize * kv_group_num * headSize;

  at::Tensor buf = at::zeros({num_thread, size_per_thread}, query.options().dtype(accumulate_dtype));
  at::Tensor buf_reduced = at::zeros({num_thread, qSplitSize * kv_group_num, is_reduced_type ? kvSplitSize : 0}, query.options());
  at::Tensor q_reorder_buf = at::zeros({num_thread, qSplitSize * kv_group_num, headSize}, query.options());

  // Data ptrs
  const scalar_t* q_data = query.const_data_ptr<scalar_t>();
  const scalar_t* k_data = key.const_data_ptr<scalar_t>();
  const scalar_t* v_data = value.const_data_ptr<scalar_t>();
  const int64_t* seq_lens_data = seq_lens.data_ptr<int64_t>();
  const int64_t* start_loc_data = start_loc.data_ptr<int64_t>();
  scalar_t* out_data = output.data_ptr<scalar_t>();
  accum_t* buf_data = buf.data_ptr<accum_t>();
  scalar_t* buf_reduced_data = is_reduced_type ? buf_reduced.data_ptr<scalar_t>() : nullptr;
  scalar_t* q_reorder_data_all = q_reorder_buf.data_ptr<scalar_t>();

  at::parallel_for(0, batchSize * num_kv_head * max_qSlice, 1, [&](int64_t begin, int64_t end) {
    int64_t i = 0, j_kv = 0, k = 0;
    data_index_init(begin, i, batchSize, j_kv, num_kv_head, k, max_qSlice);
    int ompIdx = at::get_thread_num();
    accum_t* buf_ptr = buf_data + ompIdx * size_per_thread;
    accum_t* qk_data = buf_ptr;
    accum_t* qk_max_data = qk_data + qSplitSize * kv_group_num * kvSplitSize;
    accum_t* qk_sum_data = qk_max_data + qSplitSize * kv_group_num;
    accum_t* dst_data = qk_sum_data + qSplitSize * kv_group_num;
    scalar_t* q_reorder_data = q_reorder_data_all + ompIdx * qSplitSize * kv_group_num * headSize;
    scalar_t* qk_reduced_data = is_reduced_type ? buf_reduced_data + ompIdx * qSplitSize * kv_group_num * kvSplitSize : nullptr;

    for (const auto z : c10::irange(begin, end)) {
      (void)z; // Suppress unused variable
      
      int64_t q_start = qo_indptr_data[i];
      int64_t q_end = qo_indptr_data[i + 1];
      int64_t q_len = q_end - q_start;
      
      int64_t m = k * qSplitSize;
      if (m >= q_len) {
        data_index_step(i, batchSize, j_kv, num_kv_head, k, max_qSlice);
        continue;
      }
      
      int64_t qBlockSize = std::min(qSplitSize, q_len - m);
      
      // Initialize max and sum for all query heads in this kv group
      fill_stub(qk_max_data,
          -std::numeric_limits<accum_t>::infinity(), qBlockSize * kv_group_num);
      fill_stub(qk_sum_data,
          static_cast<accum_t>(0), qBlockSize * kv_group_num);
      
      int64_t num_keys = seq_lens_data[i];
      int64_t start_pos = start_loc_data[i];
      
      // 从 (qBlockSize, kv_group_num, headSize) 排列为 (kv_group_num * qBlockSize, headSize)
      for (int64_t qh = 0; qh < kv_group_num; qh++) {
        for (int64_t row = 0; row < qBlockSize; row++) {
          int64_t q_pos = q_start + m + row;
          const scalar_t* src = q_data + q_pos * qStrideN + (j_kv * kv_group_num + qh) * qStrideH;
          scalar_t* dst = q_reorder_data + (qh * qBlockSize + row) * headSize;
          for (int64_t h = 0; h < headSize; h++) {
            dst[h] = src[h];
          }
        }
      }
      
      for (int64_t n = 0; n < num_keys; n += kvSplitSize) { 
        int64_t kvBlockSize = std::min(kvSplitSize, num_keys - n); 
        
        int64_t min_visible_kv_len = num_keys;
        for (int64_t row = 0; row < qBlockSize; row++) {
          int64_t visible_kv_len = num_keys - (q_len - 1 - (m + row));
          min_visible_kv_len = std::min(min_visible_kv_len, visible_kv_len);
        }
        
        if (n >= min_visible_kv_len) {
          continue;
        }
        
        kvBlockSize = std::min(kvBlockSize, min_visible_kv_len - n);
        
        // Calculate scale * q @ k.T for all query heads in this kv group
        gemm_dispatch<scalar_t>(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            kv_group_num * qBlockSize, // M
            kvBlockSize, // N  
            headSize, // K
            scaling_factor,
            q_reorder_data,  
            headSize,        // lda
            k_data + j_kv * kStrideH + (start_pos + n) * kStrideN,
            kStrideN,        // ldb
            static_cast<accum_t>(0),
            qk_data,
            kvBlockSize);    // ldc
        
        // Update coefficients with Softmax for each query head and query token
        for (int64_t qh = 0; qh < kv_group_num; qh++) { 
          for (int64_t row = 0; row < qBlockSize; row++) {
            int64_t q_pos = q_start + m + row;
            int64_t qk_offset = (qh * qBlockSize + row) * kvBlockSize;
            int64_t state_offset = qh * qBlockSize + row;
            
            int64_t visible_kv_len = num_keys - (q_len - 1 - (m + row));
            
            if (n >= visible_kv_len) {
              continue;
            }
            
            int64_t actual_kvBlockSize = std::min(kvBlockSize, visible_kv_len - n);
            
            accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;
            
            _vec_max_kernel(
                qk_data + qk_offset,
                actual_kvBlockSize,
                tmp_max);
            tmp_max = qk_max_data[state_offset] > tmp_max ? qk_max_data[state_offset] : tmp_max;
            tmp_sum = tmp_max;
            _exp_reduce_sum_fusion_kernel(
                qk_data + qk_offset, 
                actual_kvBlockSize,
                conditional_data_ptr(qk_data, qk_reduced_data) + qk_offset,
                tmp_sum);
            exp_tmp = std::exp(qk_max_data[state_offset] - tmp_max);
            qk_sum_data[state_offset] = tmp_sum + exp_tmp * qk_sum_data[state_offset];
            qk_max_data[state_offset] = tmp_max;
            
            // Update previous dst values with exp correction
            if (n > 0) {
              for (int64_t h = 0; h < headSize; h++) {
                dst_data[state_offset * headSize + h] *= exp_tmp;
              }
            }
          }
        }

        // Calculate Softmax(q @ k.T) @ v for all query heads
        gemm_dispatch<scalar_t>(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            kv_group_num * qBlockSize, // M
            headSize,                  // N
            kvBlockSize,               // K
            static_cast<accum_t>(1),
            conditional_data_ptr(qk_data, qk_reduced_data),
            kvBlockSize,               // lda
            v_data + j_kv * vStrideH + (start_pos + n) * vStrideN,
            vStrideN,                  // ldb
            n == 0 ? static_cast<accum_t>(0) : static_cast<accum_t>(1),
            dst_data,
            headSize);                 // ldc
      }
      
      // dst <- dst / sum[row] and write to output
      // reorder MHA output with strides
      for (int64_t qh = 0; qh < kv_group_num; qh++) {
        for (int64_t row = 0; row < qBlockSize; ++row) {
          int64_t q_pos = q_start + m + row;
          int64_t state_offset = qh * qBlockSize + row;
          accum_t sum_reciprocal = 1 / qk_sum_data[state_offset];
          scalar_t* out_ptr = out_data + q_pos * oStrideN + (j_kv * kv_group_num + qh) * oStrideH;
          accum_t* src_ptr = dst_data + state_offset * headSize;
          
          for (int64_t h = 0; h < headSize; h++) {
            out_ptr[h] = static_cast<scalar_t>(src_ptr[h] * sum_reciprocal);
          }
        }
      }
      
      // Move to the next query
      data_index_step(i, batchSize, j_kv, num_kv_head, k, max_qSlice);
    }
  });
}



template <typename scalar_t, int64_t q_split_size, int64_t kv_split_size>
void cpu_flash_decode_verified3_less_mask(
    const torch::Tensor& output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& seq_lens,
    const at::Tensor& start_loc,
    const at::Tensor& tree_mask, // FP32!!!
    const at::Tensor& tree_mask_indptr,
    int64_t mask_len,
    c10::optional<double> scale) {
  // 每个query只对sequence的最后mask_len个token应用mask
  // tree_mask现在只包含最后mask_len个token的数据

  constexpr bool is_reduced_type = std::is_reduced_floating_point_v<scalar_t>;
  using accum_t = at::opmath_type<scalar_t>;
  using Vec = at::vec::Vectorized<accum_t>;
  accum_t scaling_factor = calculate_scale(query, scale).as_float_unchecked();

  // Sizes
  TORCH_CHECK((query.size(3) == value.size(2)) && (key.size(2) == value.size(2)),
        "token_attention_cpu: Q/K/V should have the same head size");
  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(1);
  int64_t num_head = query.size(2);
  int64_t headSize = query.size(3);
  int64_t num_kv_head = key.size(1);

  // Strides
  int64_t qStrideB = query.stride(0);
  int64_t qStrideM = query.stride(1);
  int64_t qStrideH = query.stride(2);
  int64_t kStrideN = key.stride(0);
  int64_t kStrideH = key.stride(1);
  int64_t vStrideN = value.stride(0);
  int64_t vStrideH = value.stride(1);
  int64_t oStrideB = output.stride(0);
  int64_t oStrideM = output.stride(1);
  int64_t oStrideH = output.stride(2);

  int64_t qSplitSize = q_split_size > qSize ? qSize : q_split_size;
  int64_t kvSplitSize = kv_split_size;
  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
  int64_t num_thread = at::get_num_threads();
  int64_t kv_group_num = num_head / num_kv_head;

  const auto dtype = query.scalar_type();
  const auto accumulate_dtype = at::toOpMathType(dtype);

  // allocate per thread temp buf (accumulate type)
  int64_t size_per_thread =
      /* qk     */ qSplitSize * kv_group_num * kvSplitSize +
      /* qk_max */ qSplitSize * kv_group_num +
      /* qk_sum */ qSplitSize * kv_group_num +
      /* dst    */ qSplitSize * kv_group_num * headSize;

  at::Tensor buf = at::zeros({num_thread, size_per_thread}, query.options().dtype(accumulate_dtype));
  at::Tensor buf_reduced = at::zeros({num_thread, qSplitSize * kv_group_num, is_reduced_type ? kvSplitSize : 0}, query.options());
  at::Tensor q_reorder_buf = at::zeros({num_thread, qSplitSize * kv_group_num, headSize}, query.options());

  // Data ptrs
  const scalar_t* q_data = query.const_data_ptr<scalar_t>();
  const scalar_t* k_data = key.const_data_ptr<scalar_t>();
  const scalar_t* v_data = value.const_data_ptr<scalar_t>();
  const int64_t* seq_lens_data = seq_lens.data_ptr<int64_t>();
  const int64_t* start_loc_data = start_loc.data_ptr<int64_t>();
  scalar_t* out_data = output.data_ptr<scalar_t>();
  accum_t* buf_data = buf.data_ptr<accum_t>();
  scalar_t* buf_reduced_data = is_reduced_type ? buf_reduced.data_ptr<scalar_t>() : nullptr;
  const accum_t* tree_mask_data = tree_mask.const_data_ptr<accum_t>();
  const int64_t* tree_mask_indptr_data = tree_mask_indptr.data_ptr<int64_t>();
  scalar_t* q_reorder_data_all = q_reorder_buf.data_ptr<scalar_t>();

  at::parallel_for(0, batchSize * num_kv_head * qSlice, 1, [&](int64_t begin, int64_t end) {
    int64_t i = 0, j_kv = 0, k = 0;
    data_index_init(begin, i, batchSize, j_kv, num_kv_head, k, qSlice);
    int ompIdx = at::get_thread_num();
    accum_t* buf_ptr = buf_data + ompIdx * size_per_thread;
    accum_t* qk_data = buf_ptr;
    accum_t* qk_max_data = qk_data + qSplitSize * kv_group_num * kvSplitSize;
    accum_t* qk_sum_data = qk_max_data + qSplitSize * kv_group_num;
    accum_t* dst_data = qk_sum_data + qSplitSize * kv_group_num;
    scalar_t* q_reorder_data = q_reorder_data_all + ompIdx * qSplitSize * kv_group_num * headSize;
    scalar_t* qk_reduced_data = is_reduced_type ? buf_reduced_data + ompIdx * qSplitSize * kv_group_num * kvSplitSize : nullptr;

    for (const auto z : c10::irange(begin, end)) {
      (void)z; // Suppress unused variable
      int64_t m = k * qSplitSize;
      int64_t qBlockSize = std::min(qSplitSize, qSize - m);

      // Initialize max and sum for all query heads in this kv group
      fill_stub(qk_max_data, -std::numeric_limits<accum_t>::infinity(), qBlockSize * kv_group_num);
      fill_stub(qk_sum_data, static_cast<accum_t>(0), qBlockSize * kv_group_num);

      int64_t num_keys = seq_lens_data[i];
      int64_t start_pos = start_loc_data[i];
      int64_t tree_mask_start_pos = tree_mask_indptr_data[i];

      int64_t mask_apply_start = std::max(static_cast<int64_t>(0), num_keys - mask_len);

      for (int64_t qh = 0; qh < kv_group_num; qh++) {
        for (int64_t row = 0; row < qBlockSize; row++) {
          const scalar_t* src = q_data + i * qStrideB + (m + row) * qStrideM + (j_kv * kv_group_num + qh) * qStrideH;
          scalar_t* dst = q_reorder_data + (qh * qBlockSize + row) * headSize;
          for (int64_t h = 0; h < headSize; h++) {
            dst[h] = src[h];
          }
        }
      }

      for (int64_t n = 0; n < mask_apply_start; n += kvSplitSize) {
        int64_t kvBlockSize = std::min(kvSplitSize, mask_apply_start - n);

        // GEMM for all query heads in this kv group
        gemm_dispatch<scalar_t>(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            kv_group_num * qBlockSize, kvBlockSize, headSize, scaling_factor,
            q_reorder_data, headSize,
            k_data + j_kv * kStrideH + (start_pos + n) * kStrideN, kStrideN,
            static_cast<accum_t>(0), qk_data, kvBlockSize);

        // Softmax without mask application
        for (int64_t qh = 0; qh < kv_group_num; qh++) {
          for (int64_t row = 0; row < qBlockSize; row++) {
            int64_t qk_offset = (qh * qBlockSize + row) * kvBlockSize;
            int64_t state_offset = qh * qBlockSize + row;

            accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;

            _vec_max_kernel(qk_data + qk_offset, kvBlockSize, tmp_max);
            tmp_max = qk_max_data[state_offset] > tmp_max ? qk_max_data[state_offset] : tmp_max;
            tmp_sum = tmp_max;
            _exp_reduce_sum_fusion_kernel(qk_data + qk_offset, kvBlockSize,
                conditional_data_ptr(qk_data, qk_reduced_data) + qk_offset, tmp_sum);
            exp_tmp = std::exp(qk_max_data[state_offset] - tmp_max);
            qk_sum_data[state_offset] = tmp_sum + exp_tmp * qk_sum_data[state_offset];
            qk_max_data[state_offset] = tmp_max;

            if (n > 0) {
              for (int64_t h = 0; h < headSize; h++) {
                dst_data[state_offset * headSize + h] *= exp_tmp;
              }
            }
          }
        }

        // QV GEMM for all query heads
        gemm_dispatch<scalar_t>(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            kv_group_num * qBlockSize, headSize, kvBlockSize, static_cast<accum_t>(1),
            conditional_data_ptr(qk_data, qk_reduced_data), kvBlockSize,
            v_data + j_kv * vStrideH + (start_pos + n) * vStrideN, vStrideN,
            n == 0 ? static_cast<accum_t>(0) : static_cast<accum_t>(1),
            dst_data, headSize);
      }

      for (int64_t n = mask_apply_start; n < num_keys; n += kvSplitSize) {
        int64_t kvBlockSize = std::min(kvSplitSize, num_keys - n);

        // GEMM for all query heads in this kv group
        gemm_dispatch<scalar_t>(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            kv_group_num * qBlockSize, kvBlockSize, headSize, scaling_factor,
            q_reorder_data, headSize,
            k_data + j_kv * kStrideH + (start_pos + n) * kStrideN, kStrideN,
            static_cast<accum_t>(0), qk_data, kvBlockSize);

        // Softmax with mask application
        for (int64_t qh = 0; qh < kv_group_num; qh++) {
          for (int64_t row = 0; row < qBlockSize; row++) {
            int64_t qk_offset = (qh * qBlockSize + row) * kvBlockSize;
            int64_t state_offset = qh * qBlockSize + row;

            accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;

            for (int64_t kv = 0; kv < kvBlockSize; kv++) {
              int64_t mask_offset = tree_mask_start_pos + (m + row) * mask_len + (n + kv - mask_apply_start);
              qk_data[qk_offset + kv] += tree_mask_data[mask_offset];
            }

            _vec_max_kernel(qk_data + qk_offset, kvBlockSize, tmp_max);
            tmp_max = qk_max_data[state_offset] > tmp_max ? qk_max_data[state_offset] : tmp_max;
            tmp_sum = tmp_max;
            _exp_reduce_sum_fusion_kernel(qk_data + qk_offset, kvBlockSize,
                conditional_data_ptr(qk_data, qk_reduced_data) + qk_offset, tmp_sum);
            exp_tmp = std::exp(qk_max_data[state_offset] - tmp_max);
            qk_sum_data[state_offset] = tmp_sum + exp_tmp * qk_sum_data[state_offset];
            qk_max_data[state_offset] = tmp_max;

            if (n > 0) {
              for (int64_t h = 0; h < headSize; h++) {
                dst_data[state_offset * headSize + h] *= exp_tmp;
              }
            }
          }
        }

        // QV GEMM for all query heads
        gemm_dispatch<scalar_t>(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            kv_group_num * qBlockSize, headSize, kvBlockSize, static_cast<accum_t>(1),
            conditional_data_ptr(qk_data, qk_reduced_data), kvBlockSize,
            v_data + j_kv * vStrideH + (start_pos + n) * vStrideN, vStrideN,
            n == 0 ? static_cast<accum_t>(0) : static_cast<accum_t>(1),
            dst_data, headSize);
      }

      // dst <- dst / sum[row] and write to output
      for (int64_t qh = 0; qh < kv_group_num; qh++) {
        for (int64_t row = 0; row < qBlockSize; ++row) {
          int64_t state_offset = qh * qBlockSize + row;
          accum_t sum_reciprocal = 1 / qk_sum_data[state_offset];
          scalar_t* out_ptr = out_data + i * oStrideB + (m + row) * oStrideM + (j_kv * kv_group_num + qh) * oStrideH;
          accum_t* src_ptr = dst_data + state_offset * headSize;

          for (int64_t h = 0; h < headSize; h++) {
            out_ptr[h] = static_cast<scalar_t>(src_ptr[h] * sum_reciprocal);
          }
        }
      }

      data_index_step(i, batchSize, j_kv, num_kv_head, k, qSlice);
    }
  });
}

template <typename scalar_t, int64_t kv_split_size>
void cpu_flash_decode_draft_extend_with_less_mask_optimized(
    const torch::Tensor& output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& seq_lens,
    const at::Tensor& start_loc,
    const at::Tensor& qo_indptr,
    const at::Tensor& tree_mask,
    const at::Tensor& tree_mask_start_pos,
    int64_t mask_len,
    c10::optional<double> scale) {
  constexpr bool is_reduced_type = std::is_reduced_floating_point_v<scalar_t>;
  using accum_t = at::opmath_type<scalar_t>;
  using Vec = at::vec::Vectorized<accum_t>;
  accum_t scaling_factor = calculate_scale(query, scale).as_float_unchecked();

  // Sizes
  TORCH_CHECK((query.size(2) == value.size(2)) && (key.size(2) == value.size(2)),
        "token_attention_cpu: Q/K/V should have the same head size");
  int64_t batchSize = qo_indptr.size(0) - 1;
  int64_t num_head = query.size(1);
  int64_t headSize = query.size(2);
  int64_t num_kv_head = key.size(1);

  // Strides
  int64_t qStrideN = query.stride(0);
  int64_t qStrideH = query.stride(1);
  int64_t kStrideN = key.stride(0);
  int64_t kStrideH = key.stride(1);
  int64_t vStrideN = value.stride(0);
  int64_t vStrideH = value.stride(1);
  int64_t oStrideN = output.stride(0);
  int64_t oStrideH = output.stride(1);

  int64_t kvSplitSize = kv_split_size;
  int64_t num_thread = at::get_num_threads();
  int64_t kv_group_num = num_head / num_kv_head;

  const auto dtype = query.scalar_type();
  const auto accumulate_dtype = at::toOpMathType(dtype);

  int64_t max_q_len = 0;
  const int64_t* qo_indptr_data = qo_indptr.data_ptr<int64_t>();
  for (int64_t i = 0; i < batchSize; i++) {
    int64_t q_len = qo_indptr_data[i + 1] - qo_indptr_data[i];
    max_q_len = std::max(max_q_len, q_len);
  }

  int64_t size_per_thread =
      /* qk     */ max_q_len * kv_group_num * kvSplitSize +
      /* qk_max */ max_q_len * kv_group_num +
      /* qk_sum */ max_q_len * kv_group_num +
      /* dst    */ max_q_len * kv_group_num * headSize;

  at::Tensor buf = at::zeros({num_thread, size_per_thread}, query.options().dtype(accumulate_dtype));
  at::Tensor buf_reduced = at::zeros({num_thread, max_q_len * kv_group_num, is_reduced_type ? kvSplitSize : 0}, query.options());

  // Data ptrs
  const scalar_t* q_data = query.const_data_ptr<scalar_t>();
  const scalar_t* k_data = key.const_data_ptr<scalar_t>();
  const scalar_t* v_data = value.const_data_ptr<scalar_t>();
  const int64_t* seq_lens_data = seq_lens.data_ptr<int64_t>();
  const int64_t* start_loc_data = start_loc.data_ptr<int64_t>();
  scalar_t* out_data = output.data_ptr<scalar_t>();
  accum_t* buf_data = buf.data_ptr<accum_t>();
  scalar_t* buf_reduced_data = is_reduced_type ? buf_reduced.data_ptr<scalar_t>() : nullptr;
  const accum_t* tree_mask_data = tree_mask.const_data_ptr<accum_t>();
  const int64_t* tree_mask_start_pos_data = tree_mask_start_pos.data_ptr<int64_t>();

  at::parallel_for(0, batchSize * num_kv_head, 1, [&](int64_t begin, int64_t end) {
    int64_t i = 0, j_kv = 0;
    data_index_init(begin, i, batchSize, j_kv, num_kv_head);

    int ompIdx = at::get_thread_num();
    accum_t* buf_ptr = buf_data + ompIdx * size_per_thread;
    accum_t* qk_data = buf_ptr;
    accum_t* qk_max_data = qk_data + max_q_len * kv_group_num * kvSplitSize;
    accum_t* qk_sum_data = qk_max_data + max_q_len * kv_group_num;
    accum_t* dst_data = qk_sum_data + max_q_len * kv_group_num;
    scalar_t* qk_reduced_data = is_reduced_type ? buf_reduced_data + ompIdx * max_q_len * kv_group_num * kvSplitSize : nullptr;

    for (const auto z : c10::irange(begin, end)) {
      (void)z; // Suppress unused variable

      int64_t q_start = qo_indptr_data[i];
      int64_t q_end = qo_indptr_data[i + 1];
      int64_t q_len = q_end - q_start;

      int64_t num_keys = seq_lens_data[i];
      int64_t start_pos = start_loc_data[i];
      int64_t mask_start_pos = tree_mask_start_pos_data[i];

      int64_t mask_apply_start = std::max(static_cast<int64_t>(0), num_keys - mask_len);

      // Initialize max and sum
      fill_stub(qk_max_data, -std::numeric_limits<accum_t>::infinity(), q_len * kv_group_num);
      fill_stub(qk_sum_data, static_cast<accum_t>(0), q_len * kv_group_num);

      // 高效处理非mask区域 - 无条件分支，统一流程
      for (int64_t n = 0; n < mask_apply_start; n += kvSplitSize) {
        int64_t kvBlockSize = std::min(kvSplitSize, mask_apply_start - n);

        // Stage 1: 高效GEMM - 按query token合并处理
        for (int64_t q_idx = 0; q_idx < q_len; q_idx++) {
          int64_t visible_kv_len = num_keys - (q_len - 1 - q_idx);
          if (n >= visible_kv_len) continue;

          int64_t actual_kvBlockSize = std::min(kvBlockSize, visible_kv_len - n);
          int64_t q_pos = q_start + q_idx;

          gemm_dispatch<scalar_t>(
              CblasRowMajor, CblasNoTrans, CblasTrans,
              kv_group_num, actual_kvBlockSize, headSize, scaling_factor,
              q_data + q_pos * qStrideN + (j_kv * kv_group_num) * qStrideH, qStrideH,
              k_data + j_kv * kStrideH + (start_pos + n) * kStrideN, kStrideN,
              static_cast<accum_t>(0), qk_data + q_idx * kv_group_num * kvSplitSize, kvSplitSize);
        }

        for (int64_t q_idx = 0; q_idx < q_len; q_idx++) {
          int64_t visible_kv_len = num_keys - (q_len - 1 - q_idx);
          if (n >= visible_kv_len) continue;

          int64_t actual_kvBlockSize = std::min(kvBlockSize, visible_kv_len - n);

          for (int64_t qh = 0; qh < kv_group_num; qh++) {
            int64_t qk_offset = q_idx * kv_group_num * kvSplitSize + qh * kvSplitSize;
            int64_t state_offset = q_idx * kv_group_num + qh;

            accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;

            _vec_max_kernel(qk_data + qk_offset, actual_kvBlockSize, tmp_max);
            tmp_max = qk_max_data[state_offset] > tmp_max ? qk_max_data[state_offset] : tmp_max;
            tmp_sum = tmp_max;
            _exp_reduce_sum_fusion_kernel(qk_data + qk_offset, actual_kvBlockSize,
                conditional_data_ptr(qk_data, qk_reduced_data) + qk_offset, tmp_sum);
            exp_tmp = std::exp(qk_max_data[state_offset] - tmp_max);
            qk_sum_data[state_offset] = tmp_sum + exp_tmp * qk_sum_data[state_offset];
            qk_max_data[state_offset] = tmp_max;

            if (n > 0) {
              at::vec::map<accum_t>([exp_tmp](Vec x) { return x * Vec(exp_tmp); },
                dst_data + state_offset * headSize, dst_data + state_offset * headSize, headSize);
            }
          }
        }

        for (int64_t q_idx = 0; q_idx < q_len; q_idx++) {
          int64_t visible_kv_len = num_keys - (q_len - 1 - q_idx);
          if (n >= visible_kv_len) continue;

          int64_t actual_kvBlockSize = std::min(kvBlockSize, visible_kv_len - n);

          gemm_dispatch<scalar_t>(
              CblasRowMajor, CblasNoTrans, CblasNoTrans,
              kv_group_num, headSize, actual_kvBlockSize, static_cast<accum_t>(1),
              conditional_data_ptr(qk_data, qk_reduced_data) + q_idx * kv_group_num * kvSplitSize, kvSplitSize,
              v_data + j_kv * vStrideH + (start_pos + n) * vStrideN, vStrideN,
              n == 0 ? static_cast<accum_t>(0) : static_cast<accum_t>(1),
              dst_data + q_idx * kv_group_num * headSize, headSize);
        }
      }

      for (int64_t n = mask_apply_start; n < num_keys; n += kvSplitSize) {
        int64_t kvBlockSize = std::min(kvSplitSize, num_keys - n);

        for (int64_t q_idx = 0; q_idx < q_len; q_idx++) {
          int64_t visible_kv_len = num_keys - (q_len - 1 - q_idx);
          if (n >= visible_kv_len) continue;

          int64_t actual_kvBlockSize = std::min(kvBlockSize, visible_kv_len - n);
          int64_t q_pos = q_start + q_idx;

          gemm_dispatch<scalar_t>(
              CblasRowMajor, CblasNoTrans, CblasTrans,
              kv_group_num, actual_kvBlockSize, headSize, scaling_factor,
              q_data + q_pos * qStrideN + (j_kv * kv_group_num) * qStrideH, qStrideH,
              k_data + j_kv * kStrideH + (start_pos + n) * kStrideN, kStrideN,
              static_cast<accum_t>(0), qk_data + q_idx * kv_group_num * kvSplitSize, kvSplitSize);
        }

        // Stage 2: Apply mask + Softmax - 总是应用mask
        for (int64_t q_idx = 0; q_idx < q_len; q_idx++) {
          int64_t visible_kv_len = num_keys - (q_len - 1 - q_idx);
          if (n >= visible_kv_len) continue;

          int64_t actual_kvBlockSize = std::min(kvBlockSize, visible_kv_len - n);
          int64_t mask_offset_in_tree = mask_start_pos + q_idx * mask_len + (n - mask_apply_start);

          for (int64_t qh = 0; qh < kv_group_num; qh++) {
            int64_t qk_offset = q_idx * kv_group_num * kvSplitSize + qh * kvSplitSize;
            int64_t state_offset = q_idx * kv_group_num + qh;

            cblas_saxpy(actual_kvBlockSize, static_cast<scalar_t>(1),
                       (float*)(tree_mask_data + mask_offset_in_tree), 1,
                       (float*)(qk_data + qk_offset), 1);

            accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;

            _vec_max_kernel(qk_data + qk_offset, actual_kvBlockSize, tmp_max);
            tmp_max = qk_max_data[state_offset] > tmp_max ? qk_max_data[state_offset] : tmp_max;
            tmp_sum = tmp_max;
            _exp_reduce_sum_fusion_kernel(qk_data + qk_offset, actual_kvBlockSize,
                conditional_data_ptr(qk_data, qk_reduced_data) + qk_offset, tmp_sum);
            exp_tmp = std::exp(qk_max_data[state_offset] - tmp_max);
            qk_sum_data[state_offset] = tmp_sum + exp_tmp * qk_sum_data[state_offset];
            qk_max_data[state_offset] = tmp_max;

            if (n > 0) {
              at::vec::map<accum_t>([exp_tmp](Vec x) { return x * Vec(exp_tmp); },
                dst_data + state_offset * headSize, dst_data + state_offset * headSize, headSize);
            }
          }
        }

        for (int64_t q_idx = 0; q_idx < q_len; q_idx++) {
          int64_t visible_kv_len = num_keys - (q_len - 1 - q_idx);
          if (n >= visible_kv_len) continue;

          int64_t actual_kvBlockSize = std::min(kvBlockSize, visible_kv_len - n);

          gemm_dispatch<scalar_t>(
              CblasRowMajor, CblasNoTrans, CblasNoTrans,
              kv_group_num, headSize, actual_kvBlockSize, static_cast<accum_t>(1),
              conditional_data_ptr(qk_data, qk_reduced_data) + q_idx * kv_group_num * kvSplitSize, kvSplitSize,
              v_data + j_kv * vStrideH + (start_pos + n) * vStrideN, vStrideN,
              n == 0 ? static_cast<accum_t>(0) : static_cast<accum_t>(1),
              dst_data + q_idx * kv_group_num * headSize, headSize);
        }
      }

      for (int64_t q_idx = 0; q_idx < q_len; q_idx++) {
        for (int64_t qh = 0; qh < kv_group_num; qh++) {
          int64_t q_pos = q_start + q_idx;
          int64_t state_offset = q_idx * kv_group_num + qh;
          accum_t sum_reciprocal = 1 / qk_sum_data[state_offset];

          at::vec::map<scalar_t>(
            [sum_reciprocal](Vec x) { return x * Vec(sum_reciprocal); },
            out_data + q_pos * oStrideN + (j_kv * kv_group_num + qh) * oStrideH,
            dst_data + state_offset * headSize, headSize);
        }
      }

      data_index_step(i, batchSize, j_kv, num_kv_head);
    }
  });
}


void flash_attention_kernel_impl(
    const torch::Tensor& output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& seq_lens,
    const at::Tensor& start_loc,
    c10::optional<double> scale) {
  Py_BEGIN_ALLOW_THREADS
  auto q_head = query.size(2);
  auto kv_head = key.size(1);

  FASTMOE_DISPATCH_FLOATING_TYPES(query.scalar_type(), "cpu_flash_decode", [&] {
    if (q_head == kv_head) {
      cpu_flash_decode_gqa<scalar_t, 32, 1024>(
        output, query, key, value, seq_lens, start_loc, scale);
    } else {
      cpu_flash_decode_gqa<scalar_t, 32, 1024>(
        output, query, key, value, seq_lens, start_loc, scale);
    }
  });
  Py_END_ALLOW_THREADS
}


void flash_attention_kernel_impl_verified3(
    const torch::Tensor& output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& seq_lens,
    const at::Tensor& start_loc,
    const at::Tensor& tree_mask,
    const at::Tensor& tree_mask_indptr,
    c10::optional<double> scale) {
  Py_BEGIN_ALLOW_THREADS
  auto q_head = query.size(2);
  auto kv_head = key.size(1);

  FASTMOE_DISPATCH_FLOATING_TYPES(query.scalar_type(), "cpu_flash_decode_verified3", [&] {
    if (q_head == kv_head) {
      cpu_flash_decode_verified3<scalar_t, 32, 1024>(
        output, query, key, value, seq_lens, start_loc, tree_mask, tree_mask_indptr, scale);
    } else {
      cpu_flash_decode_verified3<scalar_t, 32, 1024>(
        output, query, key, value, seq_lens, start_loc, tree_mask, tree_mask_indptr, scale);
    }
  });
  Py_END_ALLOW_THREADS
}





void flash_attention_kernel_impl_draft_extend_optimized_2(
    const torch::Tensor& output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& seq_lens,
    const at::Tensor& start_loc,
    const at::Tensor& qo_indptr,
    c10::optional<double> scale) {
  Py_BEGIN_ALLOW_THREADS
  auto q_head = query.size(1);
  auto kv_head = key.size(1);

  FASTMOE_DISPATCH_FLOATING_TYPES(query.scalar_type(), "cpu_flash_decode_draft_extend_optimized_2", [&] {
    if (q_head == kv_head) {
      cpu_flash_decode_draft_extend_optimized_2<scalar_t, 64, 4096>(
        output, query, key, value, seq_lens, start_loc, qo_indptr, scale);
    } else {
      cpu_flash_decode_draft_extend_optimized_2<scalar_t, 64, 4096>(
        output, query, key, value, seq_lens, start_loc, qo_indptr, scale);
    }
  });
  Py_END_ALLOW_THREADS
}

void flash_attention_kernel_impl_draft_extend_with_mask_optimized_2(
    const torch::Tensor& output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& seq_lens,
    const at::Tensor& start_loc,
    const at::Tensor& qo_indptr,
    const at::Tensor& tree_mask,
    const at::Tensor& tree_mask_start_pos,
    c10::optional<double> scale) {
  Py_BEGIN_ALLOW_THREADS
  auto q_head = query.size(1);
  auto kv_head = key.size(1);

  FASTMOE_DISPATCH_FLOATING_TYPES(query.scalar_type(), "cpu_flash_decode_draft_extend_with_mask_optimized_2", [&] {
    if (q_head == kv_head) {
      cpu_flash_decode_draft_extend_with_mask_optimized_2<scalar_t, 64, 4096>(
        output, query, key, value, seq_lens, start_loc, qo_indptr, tree_mask, tree_mask_start_pos, scale);
    } else {
      cpu_flash_decode_draft_extend_with_mask_optimized_2<scalar_t, 64, 4096>(
        output, query, key, value, seq_lens, start_loc, qo_indptr, tree_mask, tree_mask_start_pos, scale);
    }
  });
  Py_END_ALLOW_THREADS
}

void flash_attention_kernel_impl_verified3_less_mask(
    const torch::Tensor& output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& seq_lens,
    const at::Tensor& start_loc,
    const at::Tensor& tree_mask,
    const at::Tensor& tree_mask_indptr,
    int64_t mask_len,
    c10::optional<double> scale) {
  Py_BEGIN_ALLOW_THREADS
  auto q_head = query.size(2);
  auto kv_head = key.size(1);

  FASTMOE_DISPATCH_FLOATING_TYPES(query.scalar_type(), "cpu_flash_decode_verified3_less_mask", [&] {
    if (q_head == kv_head) {
      cpu_flash_decode_verified3_less_mask<scalar_t, 32, 4096>(
        output, query, key, value, seq_lens, start_loc, tree_mask, tree_mask_indptr, mask_len, scale);
    } else {
      cpu_flash_decode_verified3_less_mask<scalar_t, 32, 4096>(
        output, query, key, value, seq_lens, start_loc, tree_mask, tree_mask_indptr, mask_len, scale);
    }
  });
  Py_END_ALLOW_THREADS
}

void flash_attention_kernel_impl_draft_extend_with_less_mask_optimized(
    const torch::Tensor& output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& seq_lens,
    const at::Tensor& start_loc,
    const at::Tensor& qo_indptr,
    const at::Tensor& tree_mask,
    const at::Tensor& tree_mask_start_pos,
    int64_t mask_len,
    c10::optional<double> scale) {

  Py_BEGIN_ALLOW_THREADS
  FASTMOE_DISPATCH_FLOATING_TYPES(
    query.scalar_type(),
    "flash_attention_kernel_impl_draft_extend_with_less_mask_optimized", [&] {
        cpu_flash_decode_draft_extend_with_less_mask_optimized<scalar_t, /*kv_split_size=*/4096>(
            output, query, key, value, seq_lens, start_loc, qo_indptr, tree_mask, tree_mask_start_pos, mask_len, scale);
  });
  Py_END_ALLOW_THREADS
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("token_attention_cpu", &flash_attention_kernel_impl, "Token Attention CPU");
  m.def("token_attention_cpu_verified3", &flash_attention_kernel_impl_verified3, "Token Attention CPU Verified3 - Single GEMM optimization for GQA");
  m.def("token_attention_cpu_draft_extend_optimized_2", &flash_attention_kernel_impl_draft_extend_optimized_2, "Token Attention CPU Draft Extend Optimized 2 - High-performance version using verified3 optimization techniques");
  m.def("token_attention_cpu_draft_extend_with_mask_optimized_2", &flash_attention_kernel_impl_draft_extend_with_mask_optimized_2, "Token Attention CPU Draft Extend with Mask Optimized 2 - High-performance version using verified3 optimization techniques");
  m.def("token_attention_cpu_verified3_less_mask", &flash_attention_kernel_impl_verified3_less_mask, "Token Attention CPU Verified3 Less Mask");
  m.def("token_attention_cpu_draft_extend_with_less_mask_optimized", &flash_attention_kernel_impl_draft_extend_with_less_mask_optimized, "Token Attention CPU Draft Extend with Less Mask Optimized");
}
}
