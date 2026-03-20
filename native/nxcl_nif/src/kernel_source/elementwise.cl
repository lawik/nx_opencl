__kernel void elementwise_add(
    __global const float* a,
    __global const float* b,
    __global float* out,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    out[gid] = a[gid] + b[gid];
}

__kernel void elementwise_subtract(
    __global const float* a,
    __global const float* b,
    __global float* out,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    out[gid] = a[gid] - b[gid];
}

__kernel void elementwise_multiply(
    __global const float* a,
    __global const float* b,
    __global float* out,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    out[gid] = a[gid] * b[gid];
}

__kernel void elementwise_divide(
    __global const float* a,
    __global const float* b,
    __global float* out,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    out[gid] = a[gid] / b[gid];
}

__kernel void elementwise_negate(
    __global const float* a,
    __global float* out,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    out[gid] = -a[gid];
}

__kernel void elementwise_exp(
    __global const float* a,
    __global float* out,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    out[gid] = exp(a[gid]);
}

__kernel void elementwise_log(
    __global const float* a,
    __global float* out,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    out[gid] = log(a[gid]);
}

__kernel void elementwise_tanh(
    __global const float* a,
    __global float* out,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    out[gid] = tanh(a[gid]);
}

__kernel void elementwise_sigmoid(
    __global const float* a,
    __global float* out,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    out[gid] = 1.0f / (1.0f + exp(-a[gid]));
}

__kernel void elementwise_relu(
    __global const float* a,
    __global float* out,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    out[gid] = fmax(a[gid], 0.0f);
}

__kernel void elementwise_rsqrt(
    __global const float* a,
    __global float* out,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    out[gid] = rsqrt(a[gid]);
}

__kernel void elementwise_abs(
    __global const float* a,
    __global float* out,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    out[gid] = fabs(a[gid]);
}
