#ifndef TS
#define TS 8
#endif

__kernel void matmul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M, const int N, const int K
) {
    __local float As[TS][TS];
    __local float Bs[TS][TS];

    int row = get_local_id(0);
    int col = get_local_id(1);
    int gRow = TS * get_group_id(0) + row;
    int gCol = TS * get_group_id(1) + col;

    float acc = 0.0f;

    for (int t = 0; t < K; t += TS) {
        As[row][col] = (gRow < M && t + col < K)
            ? A[gRow * K + t + col] : 0.0f;
        Bs[row][col] = (t + row < K && gCol < N)
            ? B[(t + row) * N + gCol] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++)
            acc += As[row][k] * Bs[k][col];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gRow < M && gCol < N)
        C[gRow * N + gCol] = acc;
}
