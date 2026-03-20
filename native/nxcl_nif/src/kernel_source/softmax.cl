// Softmax: computes softmax over the last dimension
// Input is treated as [rows x cols] where softmax is applied per row

__kernel void softmax(
    __global const float* input,
    __global float* output,
    const int cols
) {
    int row = get_global_id(0);
    int base = row * cols;

    // Find max for numerical stability
    float max_val = input[base];
    for (int j = 1; j < cols; j++) {
        max_val = fmax(max_val, input[base + j]);
    }

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
        float e = exp(input[base + j] - max_val);
        output[base + j] = e;
        sum += e;
    }

    // Normalize
    for (int j = 0; j < cols; j++) {
        output[base + j] /= sum;
    }
}
