#include <stdint.h>

void frost_bounds_check(int64_t index, int64_t length);
void frost_generation_check(int64_t stored, int64_t expected);

int32_t printf(char* a0, int64_t a1);


int32_t main(void) {
  int64_t _0;
  int64_t _1;
  int64_t _2;
  int64_t _3;
  int32_t _4;
  int64_t _5;
  int64_t _6;
  int64_t _7;
  int8_t _8;
  int64_t _9;
  int64_t _10;
  int64_t _11;
  int64_t _12;
  int64_t _13;
  int64_t _14;
  int64_t _15;
  int64_t _16;
  int32_t _17;
  int32_t _18;
  int64_t _19;
  int64_t _20;
  int64_t _21;
  int64_t _22;
  int64_t _23;
  int32_t _24;
 block0:;
  _0 = (-9LL);
  _1 = (_0 % 1LL);
  _2 = (_1 - 9LL);
  _3 = (_2 | 178LL);
  _4 = printf((char*)"%lld\n", _3);
  _5 = (-7LL);
  _6 = (_5 & 176LL);
  _7 = (_6 * 3LL);
  _8 = (_7 >= 2LL);
  if (_8) goto block1; else goto block2;
 block1:;
  _9 = (3LL - 1LL);
  _10 = (_9 - 8LL);
  _11 = (3LL % 3LL);
  _12 = (_11 | 26LL);
  _13 = (_10 - _12);
  _14 = _13;
  goto block3;
 block2:;
  _15 = (-10LL);
  _16 = (_15 * 0LL);
  _14 = _16;
  goto block3;
 block3:;
  _17 = printf((char*)"%lld\n", _14);
  _18 = printf((char*)"%lld\n", 9LL);
  _19 = (-8LL);
  _20 = (_19 + 0LL);
  _21 = (1LL - _20);
  _22 = (9LL + _21);
  _23 = (_22 % 5LL);
  _24 = printf((char*)"%lld\n", _23);
  return (int32_t)(0LL);
}

