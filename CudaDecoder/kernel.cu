#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string>
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>

using namespace std;

constexpr uint32_t d0[256] = {
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x000000f8, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x000000fc,
0x000000d0, 0x000000d4, 0x000000d8, 0x000000dc, 0x000000e0, 0x000000e4,
0x000000e8, 0x000000ec, 0x000000f0, 0x000000f4, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x00000000,
0x00000004, 0x00000008, 0x0000000c, 0x00000010, 0x00000014, 0x00000018,
0x0000001c, 0x00000020, 0x00000024, 0x00000028, 0x0000002c, 0x00000030,
0x00000034, 0x00000038, 0x0000003c, 0x00000040, 0x00000044, 0x00000048,
0x0000004c, 0x00000050, 0x00000054, 0x00000058, 0x0000005c, 0x00000060,
0x00000064, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x00000068, 0x0000006c, 0x00000070, 0x00000074, 0x00000078,
0x0000007c, 0x00000080, 0x00000084, 0x00000088, 0x0000008c, 0x00000090,
0x00000094, 0x00000098, 0x0000009c, 0x000000a0, 0x000000a4, 0x000000a8,
0x000000ac, 0x000000b0, 0x000000b4, 0x000000b8, 0x000000bc, 0x000000c0,
0x000000c4, 0x000000c8, 0x000000cc, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff
};
constexpr uint32_t d1[256] = {
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x0000e003, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x0000f003,
0x00004003, 0x00005003, 0x00006003, 0x00007003, 0x00008003, 0x00009003,
0x0000a003, 0x0000b003, 0x0000c003, 0x0000d003, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x00000000,
0x00001000, 0x00002000, 0x00003000, 0x00004000, 0x00005000, 0x00006000,
0x00007000, 0x00008000, 0x00009000, 0x0000a000, 0x0000b000, 0x0000c000,
0x0000d000, 0x0000e000, 0x0000f000, 0x00000001, 0x00001001, 0x00002001,
0x00003001, 0x00004001, 0x00005001, 0x00006001, 0x00007001, 0x00008001,
0x00009001, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x0000a001, 0x0000b001, 0x0000c001, 0x0000d001, 0x0000e001,
0x0000f001, 0x00000002, 0x00001002, 0x00002002, 0x00003002, 0x00004002,
0x00005002, 0x00006002, 0x00007002, 0x00008002, 0x00009002, 0x0000a002,
0x0000b002, 0x0000c002, 0x0000d002, 0x0000e002, 0x0000f002, 0x00000003,
0x00001003, 0x00002003, 0x00003003, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff
};
constexpr uint32_t d2[256] = {
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x00800f00, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x00c00f00,
0x00000d00, 0x00400d00, 0x00800d00, 0x00c00d00, 0x00000e00, 0x00400e00,
0x00800e00, 0x00c00e00, 0x00000f00, 0x00400f00, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x00000000,
0x00400000, 0x00800000, 0x00c00000, 0x00000100, 0x00400100, 0x00800100,
0x00c00100, 0x00000200, 0x00400200, 0x00800200, 0x00c00200, 0x00000300,
0x00400300, 0x00800300, 0x00c00300, 0x00000400, 0x00400400, 0x00800400,
0x00c00400, 0x00000500, 0x00400500, 0x00800500, 0x00c00500, 0x00000600,
0x00400600, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x00800600, 0x00c00600, 0x00000700, 0x00400700, 0x00800700,
0x00c00700, 0x00000800, 0x00400800, 0x00800800, 0x00c00800, 0x00000900,
0x00400900, 0x00800900, 0x00c00900, 0x00000a00, 0x00400a00, 0x00800a00,
0x00c00a00, 0x00000b00, 0x00400b00, 0x00800b00, 0x00c00b00, 0x00000c00,
0x00400c00, 0x00800c00, 0x00c00c00, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff
};
constexpr uint32_t d3[256] = {
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x003e0000, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x003f0000,
0x00340000, 0x00350000, 0x00360000, 0x00370000, 0x00380000, 0x00390000,
0x003a0000, 0x003b0000, 0x003c0000, 0x003d0000, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x00000000,
0x00010000, 0x00020000, 0x00030000, 0x00040000, 0x00050000, 0x00060000,
0x00070000, 0x00080000, 0x00090000, 0x000a0000, 0x000b0000, 0x000c0000,
0x000d0000, 0x000e0000, 0x000f0000, 0x00100000, 0x00110000, 0x00120000,
0x00130000, 0x00140000, 0x00150000, 0x00160000, 0x00170000, 0x00180000,
0x00190000, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x001a0000, 0x001b0000, 0x001c0000, 0x001d0000, 0x001e0000,
0x001f0000, 0x00200000, 0x00210000, 0x00220000, 0x00230000, 0x00240000,
0x00250000, 0x00260000, 0x00270000, 0x00280000, 0x00290000, 0x002a0000,
0x002b0000, 0x002c0000, 0x002d0000, 0x002e0000, 0x002f0000, 0x00300000,
0x00310000, 0x00320000, 0x00330000, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff,
0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff
};

__device__ constexpr unsigned char unb64[256] = {
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //10 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //20 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //30 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //40 
  0,   0,   0,  62,   0,   0,   0,  63,  52,  53, //50 
 54,  55,  56,  57,  58,  59,  60,  61,   0,   0, //60 
  0,   0,   0,   0,   0,   0,   1,   2,   3,   4, //70 
  5,   6,   7,   8,   9,  10,  11,  12,  13,  14, //80 
 15,  16,  17,  18,  19,  20,  21,  22,  23,  24, //90 
 25,   0,   0,   0,   0,   0,   0,  26,  27,  28, //100 
 29,  30,  31,  32,  33,  34,  35,  36,  37,  38, //110 
 39,  40,  41,  42,  43,  44,  45,  46,  47,  48, //120 
 49,  50,  51,   0,   0,   0,   0,   0,   0,   0, //130 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //140 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //150 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //160 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //170 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //180 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //190 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //200 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //210 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //220 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //230 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //240 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, //250 
  0,   0,   0,   0,   0,   0,
};

#define CHARPAD 'A'
#define BADCHAR 0x01FFFFFF

int modp_b64_decode_h(char* dest, char* src, int len) {

	if (len < 4 || (len % 4 != 0)) return -1; /* error */
	/* there can be at most 2 pad chars at the end */
	if (src[len - 1] == CHARPAD) {
		len--;
		if (src[len - 1] == CHARPAD) {
			len--;
		}
	}

	int i;
	int leftover = len % 4;
	int chunks = (leftover == 0) ? len / 4 - 1 : len / 4;

	uint8_t* p = (uint8_t*)dest;
	uint32_t x = 0;
	uint32_t* destInt = (uint32_t*)p;
	uint32_t* srcInt = (uint32_t*)src;
	uint32_t y = *srcInt++;
	for (i = 0; i < chunks; ++i) {
		x = d0[y & 0xff] |
			d1[(y >> 8) & 0xff] |
			d2[(y >> 16) & 0xff] |
			d3[(y >> 24) & 0xff];

		if (x >= BADCHAR) return -1;
		*destInt = x;
		p += 3;
		destInt = (uint32_t*)p;
		y = *srcInt++;
	}


	switch (leftover) {
	case 0:
		x = d0[y & 0xff] |
			d1[(y >> 8) & 0xff] |
			d2[(y >> 16) & 0xff] |
			d3[(y >> 24) & 0xff];

		if (x >= BADCHAR) return -1;
		*p++ = ((uint8_t*)(&x))[0];
		*p++ = ((uint8_t*)(&x))[1];
		*p = ((uint8_t*)(&x))[2];
		return (chunks + 1) * 3;
		break;
	case 1:  /* with padding this is an impossible case */
		x = d0[y & 0xff];
		*p = *((uint8_t*)(&x)); // i.e. first char/byte in int
		break;
	case 2: // * case 2, 1  output byte */
		x = d0[y & 0xff] | d1[y >> 8 & 0xff];
		*p = *((uint8_t*)(&x)); // i.e. first char
		break;
	default: /* case 3, 2 output bytes */
		x = d0[y & 0xff] |
			d1[y >> 8 & 0xff] |
			d2[y >> 16 & 0xff];  /* 0x3c */
		*p++ = ((uint8_t*)(&x))[0];
		*p = ((uint8_t*)(&x))[1];
		break;
	}

	if (x >= BADCHAR) return -1;

	return 0;// 3 * chunks + (6 * leftover) / 8;
}

#define CIPHERTEXT_LEN 68
#define UTF8_LEN 50

//__device__ const char* CIPHERTEXT = "DePk6rqSKIcsDzx177WKCeD6uEYOo3iRkMszgy1sMJLD8rbSSP2J+FGF3L3yL8GmQQAA";
__device__ constexpr char CIPHERTEXTd[CIPHERTEXT_LEN] = {3,30,15,36,58,43,42,18,10,8,28,44,3,51,49,53,59,59,22,10,2,30,3,58,46,4,24,14,40,55,34,17,36,12,44,51,32,50,53,44,12,9,11,3,60,43,27,18,18,15,54,9,62,5,6,5,55,11,55,50,11,60,6,38,16,16,0,0};
__device__ constexpr char* ALPHABETd = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
constexpr char CIPHERTEXT[CIPHERTEXT_LEN] = {3,30,15,36,58,43,42,18,10,8,28,44,3,51,49,53,59,59,22,10,2,30,3,58,46,4,24,14,40,55,34,17,36,12,44,51,32,50,53,44,12,9,11,3,60,43,27,18,18,15,54,9,62,5,6,5,55,11,55,50,11,60,6,38,16,16,0,0};
constexpr char* ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
__device__ constexpr char* DECODE_TEST = "U0RG";

__global__ void compute(
	size_t key_length,
	size_t max_key_length,
	char* keys,
	char* decrypted_base64
) {
	size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	//printf("thread #%u is doing a thing\n", idx);

	char* base64_ptr = decrypted_base64 + idx * CIPHERTEXT_LEN;
	char* current = keys + idx * max_key_length;

	for (uint8_t i = 0; i < CIPHERTEXT_LEN; i++) {
		//*/
		char* out = base64_ptr + i;
		char new_ch = CIPHERTEXTd[i] - current[i % key_length];
		if (new_ch < 0) {
			*out = ALPHABETd[new_ch + 64];
		} else {
			*out = ALPHABETd[new_ch];
		}
		//*/
		
		//base64_ptr[i] = DECODE_TEST[i % 4];
	}
}

__global__ void decode_base64(char* decrypted_base64, char* decoded_utf8) {
	size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	char* base64_ptr = decrypted_base64 + idx * 4; //1 thread per 4 bytes
	char* utf8_ptr = decoded_utf8 + idx * 3;	   //1 thread per 3 bytes

	//int A = unb64[*base64_ptr];
	uint8_t B = unb64[base64_ptr[1]];
	uint8_t C = unb64[base64_ptr[2]];
	//int D = unb64[base64_ptr[3]];

	// Just unmap each sextet to THE NUMBER it represents.
	// You then have to pack it in bin,
	// we go in groups of 4 sextets, 
	// and pull out 3 octets per quad of sextets.
	//    bin[0]       bin[1]      bin[2]
	// +-----------+-----------+-----------+
	// | 0000 0011   0111 1011   1010 1101 |
	// +-AAAA AABB   BBBB CCCC   CCDD DDDD
	// or them
	*utf8_ptr = (/*A*/unb64[*base64_ptr] << 2) | (B >> 4); // OR in last 2 bits of B

	// The 2nd byte is the bottom 4 bits of B for the upper nibble,
	// and the top 4 bits of C for the lower nibble.
	utf8_ptr[1] = (B << 4) | (C >> 2);
	utf8_ptr[2] = (C << 6) | (unb64[base64_ptr[3]]);//(D); // shove C up to top 2 bits, or with D
}

__global__ void validate_utf8(size_t max_key_length, char* keys, char* decoded_utf8, char* decode_success) {
	size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	bool valid = true;
	char* utf8_ptr = decoded_utf8 + idx * UTF8_LEN;

	for (char* ptr = utf8_ptr + UTF8_LEN; ptr > utf8_ptr; --ptr) {
		uint8_t c = *ptr;
		if (c < 0x20 || c > 0x7E) {
			valid = false;
		}
	}

	if (valid) {
		char* key_ptr = keys + idx*max_key_length;
		char* success_ptr = decode_success + idx*max_key_length;
		for (uint8_t i = 0; i < max_key_length; ++i) {
			success_ptr[i] = ALPHABET[key_ptr[i]];
		}
	}
}

#define THREADS_PER_BLOCK 224

void add_one(char* ptr, size_t &len) {
	for (int i = 1;; ++i) {
		if (*ptr < 63) {(*ptr)++; return;}
		*ptr++ = 0;
		if (i == len) {len++; return;}
	}
}

void swap(void** ptr_a, void** ptr_b) {
	void* mid = *ptr_a;
	*ptr_a = *ptr_b;
	*ptr_b = mid;
}

int main(int argc, char *argv[]) {
	cudaSetDeviceFlags(cudaDeviceMapHost);
	size_t block_count = 89286;
	size_t max_key_size = 16;
	size_t current_length = 3;
	string current = string(max_key_size, 'A');
	char* current_ptr = &current[0];

	cudaError_t cuda_err;
	int device_id = 0;

	if (argc > 1) {
		try {
			device_id = stoi(argv[1]);
		} catch (std::exception /*const &e*/) {
			//cout << " ERROR: invalid argument #2; number in range 0-" << UINT_MAX << " expected" << endl;
			//return 1;
		}

		try {
			block_count = stoi(argv[2]);
		} catch(std::exception /*const &e*/) {
			//cout << " ERROR: invalid argument #2; number in range 0-" << UINT_MAX << " expected" << endl;
			//return 1;
		}
		if (argc > 4 && argv[4] != NULL && argv[4][0] != '\0') {
			try {
				max_key_size = stoi(argv[3]);
				current = string('A', max_key_size);
			} catch(std::exception /*const &e*/) {
				//cout << " ERROR: invalid argument #4; number in range 0-" << UINT_MAX << " expected" << endl;
				//return 1;
			}
		}
		if (argc > 3 && argv[3] != NULL && argv[3][0] != '\0') {
			//cout << " ERROR: invalid argument #3; string expected" << endl;
			//return 1;
			current_length = 0;
			for (char* ch = argv[3];; ++ch) {
				if (*ch == '\0') {break;}
				current[current_length] = *ch;
				++current_length;
			}
		}
	}

	ofstream file;
	file.open("OUTPUT.txt");

	if (cudaSetDevice(device_id) != cudaSuccess) {
		cout << "invalid device ID, defaulting to 0";
		if (cudaSetDevice(0) != cudaSuccess) {
			cout << "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?";
			exit(1);
		}
	}

	cout << "starting on device #" <<  device_id << " at " << current << " with " << block_count << " blocks, each with " << THREADS_PER_BLOCK << " threads" << endl
	     << "if you ever want to pause the program then that's too bad, I can't be bothered to implement it" << endl << endl;

	size_t total_thread_count = block_count * THREADS_PER_BLOCK;
	size_t decode_base64_block_count = block_count * max_key_size/4;

	char* current_ptr_end = current_ptr + current_length;
	for (char* ptr = current_ptr; ptr != current_ptr_end; ++ptr) {
		*ptr = find(ALPHABET, ALPHABET + 65, *ptr) - ALPHABET;
	}

	/*/
	char* decrypted_base64;
	if (cudaHostAlloc(&decrypted_base64, total_thread_count * CIPHERTEXT_LEN, cudaHostAllocDefault) != cudaSuccess) {
		cout << "cudaHostAlloc failed";
		goto Err;
	}
	char* decrypted_base64_end = decrypted_base64 + total_thread_count * CIPHERTEXT_LEN;
	//*/
	/*/
	char* decoded_utf8;
	if (cudaHostAlloc(&decoded_utf8, total_thread_count * UTF8_LEN, cudaHostAllocDefault) != cudaSuccess) {
		cout << "cudaHostAlloc failed";
		goto Err;
	}
	char* decoded_utf8_end = decoded_utf8 + total_thread_count * UTF8_LEN;
	//*/
	char* keys;
	if (cudaHostAlloc(&keys, total_thread_count * max_key_size, cudaHostAllocWriteCombined) != cudaSuccess) {
		cout << "cudaHostAlloc failed";
		goto Err;
	}
	char* decode_success;
	if (cudaHostAlloc(&decode_success, total_thread_count * max_key_size, cudaHostAllocMapped) != cudaSuccess) {
		cout << "cudaHostAlloc failed";
		goto Err;
	}
	char* decode_success_end = decode_success + total_thread_count;
	char* decode_success_device = 0;
	if (cudaHostGetDevicePointer((void**)&decode_success_device,(void*)decode_success,0) != cudaSuccess) {
		cout << "cudaHostGetDevicePointer failed";
		goto Err;
	}
	char* keys_device;
	if (cudaMalloc(&keys_device, total_thread_count * max_key_size) != cudaSuccess) {
		cout << "cudaMalloc failed";
		goto Err;
	}
	char* decrypted_base64_device;
	if (cudaMalloc(&decrypted_base64_device, total_thread_count * CIPHERTEXT_LEN) != cudaSuccess) {
		cout << "cudaMalloc failed";
		goto Err;
	}
	char* decoded_utf8_device;
	if (cudaMalloc(&decoded_utf8_device, (total_thread_count * UTF8_LEN)) != cudaSuccess) {
		cout << "cudaMalloc failed";
		goto Err;
	}


	char* keys_buffer;
	if (cudaHostAlloc(&keys_buffer, total_thread_count * max_key_size, cudaHostAllocWriteCombined) != cudaSuccess) {
		cout << "cudaHostAlloc failed";
		goto Err;
	}
	char* decode_success_buffer;
	if (cudaHostAlloc(&decode_success_buffer, total_thread_count * max_key_size, cudaHostAllocMapped) != cudaSuccess) {
		cout << "cudaHostAlloc failed";
		goto Err;
	}
	char* decode_success_buffer_end = decode_success_buffer + total_thread_count;
	char* decode_success_buffer_device = 0;
	if (cudaHostGetDevicePointer((void**)&decode_success_buffer_device,(void*)decode_success_buffer,0) != cudaSuccess) {
		cout << "cudaHostGetDevicePointer failed";
		goto Err;
	}
	char* keys_device_buffer;
	if (cudaMalloc(&keys_device_buffer, total_thread_count * max_key_size) != cudaSuccess) {
		cout << "cudaMalloc failed";
		goto Err;
	}
	char* decrypted_base64_device_buffer;
	if (cudaMalloc(&decrypted_base64_device_buffer, total_thread_count * CIPHERTEXT_LEN) != cudaSuccess) {
		cout << "cudaMalloc failed";
		goto Err;
	}
	char* decoded_utf8_device_buffer;
	if (cudaMalloc(&decoded_utf8_device_buffer, (total_thread_count * UTF8_LEN)) != cudaSuccess) {
		cout << "cudaMalloc failed";
		goto Err;
	}
	/*bool* success_device_buffer;
	if (cudaMalloc(&success_device_buffer, total_thread_count * sizeof(bool)) != cudaSuccess) {
		cout << "cudaMalloc failed";
		goto Err;
	}*/

	cudaStream_t validate_stream, copy_stream;
	cudaStreamCreate(&validate_stream);
	cudaStreamCreate(&copy_stream);

	auto sync = [&] () {
		cuda_err = cudaDeviceSynchronize();
		if (cuda_err != cudaSuccess) {
			cout << endl << "cudaDeviceSynchronize returned error after launching kernel: " << cudaGetErrorString(cuda_err) << endl;
			cudaDeviceReset();
			file.close();
			system("pause");
			exit(1);
		}
	};

	auto sync_stream = [&](cudaStream_t stream) {
		cuda_err = cudaStreamSynchronize(stream);
		if (cuda_err != cudaSuccess) {
			cout << "cudaDeviceSynchronize returned error after launching kernel: " << cudaGetErrorString(cuda_err) << endl;
			cudaDeviceReset();
			file.close();
			system("pause");
			exit(1);
		}
	};

	auto start = chrono::high_resolution_clock::now();
	auto decrypt_start = start;

	uint8_t i = 0;
	for (;;++i) {
		/* #region GENERATE */
		cout << "generating " << total_thread_count << " new keys, starting at ";
		
		char* current_ptr_end = current_ptr + current_length;
		for (char* ptr = current_ptr; ptr != current_ptr_end; ++ptr) {
			cout << ALPHABET[*ptr];
		}

		start = chrono::high_resolution_clock::now();
		for (size_t i = 0; i < total_thread_count * max_key_size; i += max_key_size) {
			char* key_ptr = keys + i;
			for (size_t j = 0; j < max_key_size; ++j) { key_ptr[j] = current_ptr[j]; }

			add_one(current_ptr, current_length);

			/*/
			char* key_ptr_end = key_ptr + max_key_size;
			for (char* ch = key_ptr; ch < key_ptr_end; ++ch) {
				if (*ch < 10) {cout << 0;}
				cout << (int)*ch << " ";
			}
			cout << " " << (int)current_length << endl;
			//*/
		}
		cout << endl << "generated new keys in " << (chrono::high_resolution_clock::now() - start) / chrono::milliseconds(1) << "ms, transferring keys to device" << endl;
		/* #endregion */

		/* #region TRANSFER */
		start = chrono::high_resolution_clock::now();
		if (cudaMemcpyAsync(keys_device, keys, total_thread_count * max_key_size, cudaMemcpyHostToDevice, copy_stream) != cudaSuccess) {
			cout << "cudaMalloc failed";
			goto Err;
		}
		sync_stream(copy_stream);

		cout << "transferred in " << (chrono::high_resolution_clock::now() - start) / chrono::milliseconds(1) << "ms" << endl;
		/* #endregion */

		sync();
		cout << "decrypted in " << (chrono::high_resolution_clock::now() - decrypt_start) / chrono::milliseconds(1) << "ms" << endl;

		/* #region DECODE */
		start = chrono::high_resolution_clock::now();
		decode_base64<<<decode_base64_block_count, THREADS_PER_BLOCK>>>(decrypted_base64_device, decoded_utf8_device);

		cuda_err = cudaGetLastError();
		if (cuda_err != cudaSuccess) {
			cout << "decode_base64 kernel launch failed: " << cudaGetErrorString(cuda_err) << endl;
			exit(1);
		}

		cout << "decoded keys in " << (chrono::high_resolution_clock::now() - start) / chrono::milliseconds(1) << "ms, validating UTF-8 and decrypting buffered keys" << endl;
		/* #endregion */

		sync();

		/* #region VALIDATE START */
		start = chrono::high_resolution_clock::now();
		validate_utf8<<<block_count, THREADS_PER_BLOCK>>>(max_key_size, keys_device_buffer, decoded_utf8_device, decode_success_device);

		cuda_err = cudaGetLastError();
		if (cuda_err != cudaSuccess) {
			cout << "decode_base64 kernel launch failed: " << cudaGetErrorString(cuda_err) << endl;
			exit(1);
		}
		/* #endregion */

		sync();

		/*if(cudaMemcpyAsync(decode_success,success_device,total_thread_count,cudaMemcpyDeviceToHost,copy_stream) != cudaSuccess) {
			cout << "cudaMemcpy failed" << endl;
			goto Err;
		}
		sync_stream(copy_stream);*/

		decrypt_start = chrono::high_resolution_clock::now();
		compute<<<block_count, THREADS_PER_BLOCK>>>(current_length, max_key_size, keys_device, decrypted_base64_device);

		cout << "validated keys in " << (chrono::high_resolution_clock::now() - start) / chrono::milliseconds(1) << "ms, processing on CPU" << endl;
		start = chrono::high_resolution_clock::now();

		/*/
		if (cudaMemcpy(decoded_utf8, decoded_utf8_device, total_thread_count * UTF8_LEN, cudaMemcpyDeviceToHost) != cudaSuccess) {
			cout << "cudaMemcpy failed";
			goto Err;
		}

		for (char* ptr = decoded_utf8; ptr < decoded_utf8_end; ptr += UTF8_LEN) {
			char* ptr_end = ptr + CIPHERTEXT_LEN;
			for (char* ch = ptr; ch < ptr_end; ++ch) {
				cout << *ch;
			}
			cout << endl;
		}
		//*/
		/*/
		if (cudaMemcpy(decrypted_base64, decrypted_base64_device, total_thread_count * CIPHERTEXT_LEN, cudaMemcpyDeviceToHost) != cudaSuccess) {
			cout << "cudaMemcpy failed";
			goto Err;
		}

		for (char* ptr = decrypted_base64; ptr < decrypted_base64_end; ptr += CIPHERTEXT_LEN) {
			char* ptr_end = ptr + CIPHERTEXT_LEN;
			for (char* ch = ptr; ch < ptr_end; ++ch) {
				cout << *ch;
			}
			cout << endl;
		}
		//*/

		cout << "copied in " << (chrono::high_resolution_clock::now() - start) / chrono::milliseconds(1) << "ms, processing results" << endl;
		start = chrono::high_resolution_clock::now();

		uint16_t found = 0;
		for (char* ptr = decode_success; ptr < decode_success_end; ptr+=max_key_size) {
			if (*ptr != 0) {
				found++;
				size_t i = ptr - decode_success;
				cout << endl << "DECODED SUCCESSFULLY WITH KEY ";
				for (char* ch = ptr; ch < ptr + current_length; ++ch) {
					cout << *ch;
					file << *ch;
				}
				cout << endl;
				file << ": ";

				char* decrypted_base64 = (char*)malloc(CIPHERTEXT_LEN);
				if (cudaMemcpy(decrypted_base64, decrypted_base64_device + i * CIPHERTEXT_LEN, CIPHERTEXT_LEN, cudaMemcpyDeviceToHost) != cudaSuccess) {
					cout << "cudaMemcpy failed";
					goto Err;
				}
				char* decoded_utf8 = (char*)malloc(UTF8_LEN);
				modp_b64_decode_h(decoded_utf8, decrypted_base64, CIPHERTEXT_LEN);

				for (char* ch = decoded_utf8; ch < decoded_utf8 + UTF8_LEN; ++ch) {
					file << *ch;
					cout << *ch;
				}
				file << endl;
				cout << endl << endl;

				free(decoded_utf8); free(decrypted_base64);
				*ptr = 0;
			}
		}

		cout << found << " matches found in " << (chrono::high_resolution_clock::now() - start) / chrono::milliseconds(1) << "ms" << endl;
		if (found > 0) {system("pause");}
		/* #endregion */

		swap((void**)&keys_device, (void**)&keys_device_buffer);
		swap((void**)&decrypted_base64_device, (void**)&decrypted_base64_device_buffer);
		swap((void**)&decoded_utf8_device, (void**)&decoded_utf8_device_buffer);
		
		swap((void**)&keys, (void**)&keys_buffer);
		swap((void**)&decode_success, (void**)&decode_success_buffer);
		swap((void**)&decode_success_device, (void**)&decode_success_buffer_device);

		//this_thread::sleep_for(chrono::milliseconds(5000));
	}

	exit(0);

Err:
	cudaDeviceReset();
	file.close();
	system("pause");
	exit(1);
}