#include "npu_api.h"

/* Global L1 base pointer — set by generated model_run */
unsigned char* npu_l1_base = 0;

void npu_set_dependency(int src_id, int dst_id) {
    (void)src_id; (void)dst_id;
    /* no-op on CPU */
}

void npu_barrier(void) {
    /* no-op on CPU */
}
