#include <stdint.h>

void frost_rt_bounds_check(int64_t index, int64_t length);
void frost_rt_generation_check(int64_t stored, int64_t expected);

void frost_rt_mem_set(void *destination, int64_t value, int64_t size);

void frost_rt_print_i64(int64_t value);
void frost_rt_print_f64(double value);


int64_t strlen(char* a0);

int8_t frost_u_bound(char* a0);

int8_t frost_u_bound(char* a0) {
  char* _0;
  int64_t _1;
  int8_t _2;
  _0 = a0;
 block0:;
  _1 = strlen(_0);
  _2 = (_1 > 0LL);
  return (int8_t)(_2);
}

int32_t main(void) {
  int8_t _0;
 block0:;
  _0 = frost_u_bound((char*)"hi");
  frost_rt_print_i64(_0);
  return (int32_t)(0LL);
}

