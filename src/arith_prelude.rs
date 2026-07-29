//! Integer arithmetic that refuses a result it cannot represent.
//!
//! Wrapping is a defined answer and a wrong one: it keeps going, so a count that
//! overflowed becomes a small number and every check downstream runs against it.
//! An operation whose result does not fit the type it is computed at has no
//! right answer, so it stops there.
//!
//! Both C backends emit this text and then call into it, so the two agree by
//! sharing the source rather than by two people writing the same conditions.
//! Each is a `static inline` over a comparison the hardware was doing anyway,
//! and the call to the runtime sits on the branch that has already failed.
//!
//! Written without `__builtin_add_overflow` so the emitted unit compiles with
//! any C compiler, and so the condition is one a reader can check.

/// The declaration every helper below calls, and the helpers themselves.
pub const ARITH_PRELUDE: &str = r#"void frost_rt_arith_trap(int64_t);

static inline int64_t frost_add_i64(int64_t a, int64_t b) {
  if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b))
    frost_rt_arith_trap(0);
  return a + b;
}
static inline int64_t frost_sub_i64(int64_t a, int64_t b) {
  if ((b < 0 && a > INT64_MAX + b) || (b > 0 && a < INT64_MIN + b))
    frost_rt_arith_trap(1);
  return a - b;
}
static inline int64_t frost_mul_i64(int64_t a, int64_t b) {
  if (a != 0 && b != 0) {
    int64_t r = (int64_t)((uint64_t)a * (uint64_t)b);
    if (r / b != a || (a == -1 && b == INT64_MIN) || (b == -1 && a == INT64_MIN))
      frost_rt_arith_trap(2);
    return r;
  }
  return 0;
}
static inline uint64_t frost_add_u64(uint64_t a, uint64_t b) {
  if (a > UINT64_MAX - b) frost_rt_arith_trap(0);
  return a + b;
}
static inline uint64_t frost_sub_u64(uint64_t a, uint64_t b) {
  if (a < b) frost_rt_arith_trap(1);
  return a - b;
}
static inline uint64_t frost_mul_u64(uint64_t a, uint64_t b) {
  if (a != 0 && b > UINT64_MAX / a)
    frost_rt_arith_trap(2);
  return a * b;
}

/* A narrower type is computed at 64 bits, where neither operand can overflow,
   and the answer is held to its own range. One rule for eight widths, and the
   range is what the type means. */
static inline int64_t frost_fit(int64_t v, int64_t lo, int64_t hi) {
  if (v < lo || v > hi) frost_rt_arith_trap(8);
  return v;
}
static inline uint64_t frost_fit_u(uint64_t v, uint64_t hi) {
  if (v > hi) frost_rt_arith_trap(8);
  return v;
}
/* The same rule with the width passed rather than the bounds, which is what a
   backend emits when it knows the type but not the numbers. */
static inline int64_t frost_narrow(int64_t v, int64_t bits, int64_t is_signed,
                                   int64_t what) {
  if (is_signed) {
    int64_t hi = ((int64_t)1 << (bits - 1)) - 1;
    if (v < -hi - 1 || v > hi) frost_rt_arith_trap(what);
  } else {
    if (v < 0 || (uint64_t)v > (((uint64_t)1 << bits) - 1))
      frost_rt_arith_trap(what);
  }
  return v;
}

static inline int64_t frost_div_i64(int64_t a, int64_t b) {
  if (b == 0) frost_rt_arith_trap(3);
  if (a == INT64_MIN && b == -1) frost_rt_arith_trap(5);
  return a / b;
}
static inline int64_t frost_rem_i64(int64_t a, int64_t b) {
  if (b == 0) frost_rt_arith_trap(4);
  if (a == INT64_MIN && b == -1) return 0;
  return a % b;
}
static inline uint64_t frost_div_u64(uint64_t a, uint64_t b) {
  if (b == 0) frost_rt_arith_trap(3);
  return a / b;
}
static inline uint64_t frost_rem_u64(uint64_t a, uint64_t b) {
  if (b == 0) frost_rt_arith_trap(4);
  return a % b;
}
static inline int64_t frost_neg_i64(int64_t a) {
  if (a == INT64_MIN) frost_rt_arith_trap(6);
  return -a;
}
static inline int64_t frost_shift(int64_t by, int64_t bits) {
  if (by < 0 || by >= bits)
    frost_rt_arith_trap(7);
  return by;
}
"#;
