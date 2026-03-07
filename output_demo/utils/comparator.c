#include "comparator.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

static float read_elem(const void* buf, int index, const char* dtype) {
    if (strcmp(dtype, "fp32") == 0) {
        return ((const float*)buf)[index];
    }
    /* fp16: stored as uint16_t, convert via bit manipulation */
    unsigned short h = ((const unsigned short*)buf)[index];
    unsigned int sign = (h >> 15) & 1;
    unsigned int exp  = (h >> 10) & 0x1f;
    unsigned int mant = h & 0x3ff;
    float f;
    if (exp == 0) {
        f = (float)(mant) / 1024.0f * (1.0f / 16384.0f);
    } else if (exp == 31) {
        f = mant ? NAN : INFINITY;
    } else {
        f = (1.0f + (float)(mant) / 1024.0f) * powf(2.0f, (float)(exp) - 15.0f);
    }
    return sign ? -f : f;
}

int compare_tensors(const char* actual_path, const char* golden_path,
                    const char* desc_path, float abs_tol, float cos_tol,
                    compare_result_t* result) {
    tensor_desc_t desc;
    if (parse_desc(desc_path, &desc) != 0) return -1;

    FILE* fa = fopen(actual_path, "rb");
    FILE* fg = fopen(golden_path, "rb");
    if (!fa || !fg) {
        if (fa) fclose(fa);
        if (fg) fclose(fg);
        return -1;
    }

    void* actual_buf = malloc(desc.total_bytes);
    void* golden_buf = malloc(desc.total_bytes);
    fread(actual_buf, 1, desc.total_bytes, fa);
    fread(golden_buf, 1, desc.total_bytes, fg);
    fclose(fa);
    fclose(fg);

    int elem_size = (strcmp(desc.dtype, "fp32") == 0) ? 4 : 2;
    int n = (int)(desc.total_bytes / elem_size);

    result->total_elements = n;
    result->max_abs_diff = 0.0f;
    result->max_rel_diff = 0.0f;
    result->mismatch_count = 0;
    result->first_mismatch_index = -1;
    result->mse = 0.0f;

    double dot = 0.0, norm_a = 0.0, norm_g = 0.0;
    for (int i = 0; i < n; i++) {
        float a = read_elem(actual_buf, i, desc.dtype);
        float g = read_elem(golden_buf, i, desc.dtype);
        float diff = fabsf(a - g);
        float rel = (fabsf(g) > 1e-8f) ? diff / fabsf(g) : 0.0f;

        if (diff > result->max_abs_diff) result->max_abs_diff = diff;
        if (rel  > result->max_rel_diff) result->max_rel_diff = rel;
        result->mse += diff * diff;

        if (diff > abs_tol) {
            result->mismatch_count++;
            if (result->first_mismatch_index < 0)
                result->first_mismatch_index = i;
        }
        dot    += (double)a * g;
        norm_a += (double)a * a;
        norm_g += (double)g * g;
    }
    result->mse /= (n > 0 ? n : 1);
    result->cosine_similarity = (norm_a > 0 && norm_g > 0)
        ? (float)(dot / (sqrt(norm_a) * sqrt(norm_g))) : 0.0f;

    free(actual_buf);
    free(golden_buf);
    return (result->max_abs_diff <= abs_tol && result->cosine_similarity >= cos_tol) ? 0 : 1;
}
