#ifndef COMPARATOR_H
#define COMPARATOR_H

#include "data_loader.h"

typedef struct {
    float max_abs_diff;
    float max_rel_diff;
    float cosine_similarity;
    float mse;
    int mismatch_count;
    int total_elements;
    int first_mismatch_index;
} compare_result_t;

int compare_tensors(const char* actual_path, const char* golden_path,
                    const char* desc_path, float abs_tol, float cos_tol,
                    compare_result_t* result);

#endif
