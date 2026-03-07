#include "npu_api.h"

void npu_set_dependency(int src_id, int dst_id) {
    (void)src_id; (void)dst_id;
    /* no-op on CPU */
}

void npu_barrier(void) {
    /* no-op on CPU */
}
