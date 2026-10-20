// Metal Shading Language on the CPU: just enough of <metal_stdlib> for kernels.metal, so the shaders can be run
// and tested on machines without an Apple GPU (scripts/build-metal-emu.sh, docs/08 §8.13).
//
// Address spaces stay distinct pointer types (clang address_space attributes), so overloads such as
// as_f4(device const float *) / as_f4(threadgroup const float *) still resolve. Vectors are clang ext vectors;
// MSL's constructor syntax float4(x) is rewritten by transpile.py to mk_float4(x). The collective operations
// (simd_sum, threadgroup_barrier, ...) are implemented by the fiber scheduler in runtime.cpp: every GPU thread of
// a threadgroup is a fiber, and a collective yields until all threads of its simdgroup / threadgroup arrived.
#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <type_traits>
#include "emu_args.h"

#define device __attribute__((address_space(1)))
#define constant __attribute__((address_space(2)))
#define threadgroup __attribute__((address_space(3)))
#define thread
#define kernel

typedef _Float16 half;

#define EMU_VEC(T)                                                   \
    typedef T T##2 __attribute__((ext_vector_type(2)));              \
    typedef T T##3 __attribute__((ext_vector_type(3)));              \
    typedef T T##4 __attribute__((ext_vector_type(4)));              \
    typedef T T##8 __attribute__((ext_vector_type(8)));
EMU_VEC(float)
EMU_VEC(half)
EMU_VEC(uchar)
EMU_VEC(char)
EMU_VEC(ushort)
EMU_VEC(short)
EMU_VEC(int)
EMU_VEC(ulong)
typedef uchar4 packed_uchar4;
typedef char4 packed_char4;
typedef float4 packed_float4;

// ---- constructors: mk_float4(scalar) splats, mk_float4(a, b, c, d) composes, mk_float4(vector) converts
template <typename V> struct emu_elem;
#define EMU_ELEM(T)                                                                    \
    template <> struct emu_elem<T##2> { typedef T type; static const int n = 2; };    \
    template <> struct emu_elem<T##3> { typedef T type; static const int n = 3; };    \
    template <> struct emu_elem<T##4> { typedef T type; static const int n = 4; };    \
    template <> struct emu_elem<T##8> { typedef T type; static const int n = 8; };
EMU_ELEM(float)
EMU_ELEM(half)
EMU_ELEM(uchar)
EMU_ELEM(char)
EMU_ELEM(ushort)
EMU_ELEM(short)
EMU_ELEM(int)
EMU_ELEM(uint)
EMU_ELEM(ulong)

template <typename V, typename = void> struct emu_is_vec : std::false_type {};
template <typename V> struct emu_is_vec<V, std::void_t<typename emu_elem<V>::type>> : std::true_type {};

template <typename V, typename S>
static inline V emu_make(S s) {
    if constexpr (emu_is_vec<S>::value) {
        return __builtin_convertvector(s, V);
    } else {
        V v;
        for (int i = 0; i < emu_elem<V>::n; i++) v[i] = (typename emu_elem<V>::type) s;
        return v;
    }
}
template <typename V, typename... S>
static inline V emu_make(S... s) {
    static_assert(sizeof...(S) == emu_elem<V>::n, "component count");
    typename emu_elem<V>::type a[] = {(typename emu_elem<V>::type) s...};
    V v;
    for (int i = 0; i < emu_elem<V>::n; i++) v[i] = a[i];
    return v;
}
#define EMU_MK(T)                                                                            \
    template <typename... S> static inline T##2 mk_##T##2(S... s) { return emu_make<T##2>(s...); } \
    template <typename... S> static inline T##3 mk_##T##3(S... s) { return emu_make<T##3>(s...); } \
    template <typename... S> static inline T##4 mk_##T##4(S... s) { return emu_make<T##4>(s...); } \
    template <typename... S> static inline T##8 mk_##T##8(S... s) { return emu_make<T##8>(s...); }
EMU_MK(float)
EMU_MK(half)
EMU_MK(uchar)
EMU_MK(char)
EMU_MK(ushort)
EMU_MK(short)
EMU_MK(int)
EMU_MK(uint)
EMU_MK(ulong)

template <typename T, typename S>
static inline T as_type(S s) {
    static_assert(sizeof(T) == sizeof(S), "as_type needs equal sizes");
    T t;
    memcpy(&t, &s, sizeof t);
    return t;
}

// ---- math
#ifndef INFINITY
#define INFINITY __builtin_inff()
#endif
#ifndef MAXFLOAT
#define MAXFLOAT 3.40282346e38f
#endif
#define FLT_MAX 3.40282346e38f

template <typename V, std::enable_if_t<emu_is_vec<V>::value, int> = 0>
static inline typename emu_elem<V>::type dot(V a, V b) {
    typename emu_elem<V>::type s = 0;
    for (int i = 0; i < emu_elem<V>::n; i++) s += a[i] * b[i];
    return s;
}
static inline float rsqrt(float x) { return 1.0f / sqrtf(x); }
template <typename A, typename B, std::enable_if_t<!emu_is_vec<A>::value && !emu_is_vec<B>::value, int> = 0>
static inline std::common_type_t<A, B> max(A a, B b) {
    typedef std::common_type_t<A, B> C;
    if constexpr (std::is_floating_point_v<C>) return fmaxf((float) a, (float) b);
    else return (C) a > (C) b ? (C) a : (C) b;
}
template <typename A, typename B, std::enable_if_t<!emu_is_vec<A>::value && !emu_is_vec<B>::value, int> = 0>
static inline std::common_type_t<A, B> min(A a, B b) {
    typedef std::common_type_t<A, B> C;
    if constexpr (std::is_floating_point_v<C>) return fminf((float) a, (float) b);
    else return (C) a < (C) b ? (C) a : (C) b;
}
template <typename V, std::enable_if_t<emu_is_vec<V>::value, int> = 0>
static inline V max(V a, V b) {
    V r;
    for (int i = 0; i < emu_elem<V>::n; i++) r[i] = a[i] > b[i] ? a[i] : b[i];
    return r;
}
template <typename V, std::enable_if_t<emu_is_vec<V>::value, int> = 0>
static inline V min(V a, V b) {
    V r;
    for (int i = 0; i < emu_elem<V>::n; i++) r[i] = a[i] < b[i] ? a[i] : b[i];
    return r;
}
template <typename T, typename L, typename H>
static inline T clamp(T v, L lo, H hi) { return min(max(v, (T) lo), (T) hi); }
template <typename T> static inline T select(T a, T b, bool c) { return c ? b : a; }
static inline float fast_exp(float x) { return expf(x); }
namespace precise {
static inline float exp(float x) { return expf(x); }
static inline float log(float x) { return logf(x); }
static inline float tanh(float x) { return tanhf(x); }
}
namespace fast {
static inline float exp(float x) { return expf(x); }
static inline float log(float x) { return logf(x); }
static inline float tanh(float x) { return tanhf(x); }
}

// ---- collectives (runtime.cpp)
enum class mem_flags { mem_none = 0, mem_device = 1, mem_threadgroup = 2, mem_texture = 4 };
static inline mem_flags operator|(mem_flags a, mem_flags b) { return (mem_flags) ((int) a | (int) b); }
void emu_barrier();
// all = 32 slots of `size` bytes in lane order; returns the mask of the lanes that took part
uint32_t emu_simd_exchange(const void *mine, void *all, int size);
static inline void threadgroup_barrier(mem_flags) { emu_barrier(); }
static inline void simdgroup_barrier(mem_flags) { // every live lane of the simdgroup arrives
    char none = 0, all[32];
    emu_simd_exchange(&none, all, 1);
}

template <typename T, typename F>
static inline T emu_simd_reduce(T v, F f) {
    T all[32];
    uint32_t m = emu_simd_exchange(&v, all, sizeof(T));
    bool first = true;
    T s = v;
    for (int i = 0; i < 32; i++) {
        if (m & (1u << i)) {
            s = first ? all[i] : f(s, all[i]);
            first = false;
        }
    }
    return s;
}
template <typename T>
static inline T simd_sum(T v) { return emu_simd_reduce(v, [](T a, T b) { return a + b; }); }
template <typename T>
static inline T simd_max(T v) { return emu_simd_reduce(v, [](T a, T b) { return max(a, b); }); }
template <typename T>
static inline T simd_min(T v) { return emu_simd_reduce(v, [](T a, T b) { return min(a, b); }); }
int emu_lane();
template <typename T>
static inline T simd_shuffle(T v, ushort lane) {
    T all[32];
    emu_simd_exchange(&v, all, sizeof(T));
    return all[lane];
}
template <typename T>
static inline T simd_broadcast(T v, ushort lane) { return simd_shuffle(v, lane); }
template <typename T>
static inline T simd_shuffle_xor(T v, ushort mask) {
    T all[32];
    emu_simd_exchange(&v, all, sizeof(T));
    return all[emu_lane() ^ mask];
}
template <typename T>
static inline T simd_shuffle_down(T v, ushort d) {
    T all[32];
    uint32_t m = emu_simd_exchange(&v, all, sizeof(T));
    int l = emu_lane() + d;
    return l < 32 && (m & (1u << l)) ? all[l] : v;
}
template <typename T>
static inline T simd_prefix_exclusive_sum(T v) {
    T all[32];
    emu_simd_exchange(&v, all, sizeof(T));
    T s = 0;
    for (int i = 0; i < emu_lane(); i++) s += all[i];
    return s;
}

// ---- simdgroup matrices: every lane holds the whole 8x8 tile (loads and stores are uniform across the
// simdgroup, so computing it once per lane gives the collective result)
template <typename T, int R, int C>
struct simdgroup_matrix {
    T m[R][C];
};
typedef simdgroup_matrix<float, 8, 8> simdgroup_float8x8;
typedef simdgroup_matrix<half, 8, 8> simdgroup_half8x8;
template <typename T, int R, int C>
static inline simdgroup_matrix<T, R, C> make_filled_simdgroup_matrix(T v) {
    simdgroup_matrix<T, R, C> s;
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) s.m[i][j] = v;
    return s;
}
template <typename T, int R, int C, typename P>
static inline void simdgroup_load(thread simdgroup_matrix<T, R, C> &d, P src, ulong stride, ulong2 origin = 0,
                                  bool transpose = false) {
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            ulong r = i + origin.y, c = j + origin.x;
            d.m[i][j] = (T) (transpose ? src[c * stride + r] : src[r * stride + c]);
        }
}
template <typename T, int R, int C, typename P>
static inline void simdgroup_store(simdgroup_matrix<T, R, C> s, P dst, ulong stride, ulong2 origin = 0,
                                   bool transpose = false) {
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            ulong r = i + origin.y, c = j + origin.x;
            if (transpose) dst[c * stride + r] = s.m[i][j];
            else dst[r * stride + c] = s.m[i][j];
        }
}
template <typename T, int M, int K, int N>
static inline void simdgroup_multiply_accumulate(thread simdgroup_matrix<T, M, N> &d,
                                                 simdgroup_matrix<T, M, K> a, simdgroup_matrix<T, K, N> b,
                                                 simdgroup_matrix<T, M, N> c) {
    simdgroup_matrix<T, M, N> r;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = c.m[i][j];
            for (int k = 0; k < K; k++) s += (float) a.m[i][k] * (float) b.m[k][j];
            r.m[i][j] = (T) s;
        }
    d = r;
}
template <typename T, int M, int K, int N>
static inline void simdgroup_multiply(thread simdgroup_matrix<T, M, N> &d, simdgroup_matrix<T, M, K> a,
                                      simdgroup_matrix<T, K, N> b) {
    simdgroup_multiply_accumulate(d, a, b, make_filled_simdgroup_matrix<T, M, N>((T) 0));
}

template <typename T>
static inline T emu_builtin(uint3 v) {
    if constexpr (emu_is_vec<T>::value) {
        T r;
        for (int i = 0; i < emu_elem<T>::n && i < 3; i++) r[i] = (typename emu_elem<T>::type) v[i];
        return r;
    } else {
        return (T) v.x;
    }
}
