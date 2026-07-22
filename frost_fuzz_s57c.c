#include <stdint.h>

void frost_bounds_check(int64_t index, int64_t length);
void frost_generation_check(int64_t stored, int64_t expected);

int32_t printf(char* a0, int64_t a1);


int32_t main(void) {
  int64_t _0;
  int64_t _1;
  int64_t _2;
  int64_t _3;
  int64_t _4;
  int64_t _5;
  int64_t _6;
  int32_t _7;
  int64_t _8;
  int64_t _9;
  int64_t _10;
  int64_t _11;
  int64_t _12;
  int8_t _13;
  int64_t _14;
  int64_t _15;
  int64_t _16;
  int32_t _17;
  int32_t _18;
  int64_t _19;
  int8_t _20;
  int64_t _21;
  int64_t _22;
  int64_t _23;
  int64_t _24;
  int64_t _25;
  int32_t _26;
 block0:;
  _0 = (8LL % 7LL);
  _1 = (_0 % 4LL);
  _2 = (9LL - _1);
  _3 = (3LL % 2LL);
  _4 = (_3 & 129LL);
  _5 = (_4 + 2LL);
  _6 = (_2 + _5);
  _7 = printf((char*)"%lld\n", _6);
  _8 = (-5LL);
  _9 = (-6LL);
  _10 = (_8 + _9);
  _11 = (3LL | 241LL);
  _12 = (_10 - _11);
  _13 = (_12 == 9LL);
  if (_13) goto block1; else goto block2;
 block1:;
  _14 = 1LL;
  goto block3;
 block2:;
  _15 = (8LL % 1LL);
  _16 = (_15 * 2LL);
  _14 = _16;
  goto block3;
 block3:;
  _17 = printf((char*)"%lld\n", _14);
  _18 = printf((char*)"%lld\n", 8LL);
  _19 = (-6LL);
  _20 = (_19 <= 4LL);
  if (_20) goto block4; else goto block5;
 block4:;
  _21 = 7LL;
  goto block6;
 block5:;
  _22 = (-3LL);
  _23 = (_22 + 6LL);
  _24 = (_23 * 2LL);
  _25 = (_24 % 3LL);
  _21 = _25;
  goto block6;
 block6:;
  _26 = printf((char*)"%lld\n", _21);
  return (int32_t)(0LL);
}

