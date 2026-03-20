__kernel void reduce_sum(
    __global const float* input,
    __global float* partials,
    __local  float* scratch,
    const int n
) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);

    scratch[lid] = (gid < n) ? input[gid] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (lid < stride)
            scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
        partials[get_group_id(0)] = scratch[0];
}

__kernel void reduce_max(
    __global const float* input,
    __global float* partials,
    __local  float* scratch,
    const int n
) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);

    scratch[lid] = (gid < n) ? input[gid] : -INFINITY;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (lid < stride)
            scratch[lid] = fmax(scratch[lid], scratch[lid + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
        partials[get_group_id(0)] = scratch[0];
}

__kernel void reduce_min(
    __global const float* input,
    __global float* partials,
    __local  float* scratch,
    const int n
) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);

    scratch[lid] = (gid < n) ? input[gid] : INFINITY;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (lid < stride)
            scratch[lid] = fmin(scratch[lid], scratch[lid + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
        partials[get_group_id(0)] = scratch[0];
}
