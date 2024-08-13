#include "matrix.h"
#define LOOP_UNROLL 16

void transpose(int N, UINT B[][2048]) {
    #pragma omp parallel for schedule(dynamic, 8)   // non-rectangle
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < r; ++c) {
            UINT temp = B[r][c];
            B[r][c] = B[c][r];
            B[c][r] = temp;
        }
    }
}

void multiply(int N, UINT A[][2048], UINT B[][2048], UINT C[][2048]) {
    // improve cache locality in multiplication for row-based memory
    transpose(N, B);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            register UINT sum = 0;  // overflow, let it go.
            UINT *a = &A[r][0], *b = &B[c][0];

            int k = 0;
            for (; k + LOOP_UNROLL < N; k += LOOP_UNROLL) {
                sum += *a * *b, ++a, ++b;
                sum += *a * *b, ++a, ++b;
                sum += *a * *b, ++a, ++b;
                sum += *a * *b, ++a, ++b;
                sum += *a * *b, ++a, ++b;
                sum += *a * *b, ++a, ++b;
                sum += *a * *b, ++a, ++b;
                sum += *a * *b, ++a, ++b;
                sum += *a * *b, ++a, ++b;
                sum += *a * *b, ++a, ++b;
                sum += *a * *b, ++a, ++b;
                sum += *a * *b, ++a, ++b;
                sum += *a * *b, ++a, ++b;
                sum += *a * *b, ++a, ++b;
                sum += *a * *b, ++a, ++b;
                sum += *a * *b, ++a, ++b;
            }

            for (; k < N; ++k) {
                sum += *a * *b, ++a, ++b;
            }

            C[r][c] = sum;
        }
    }
}