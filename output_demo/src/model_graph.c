#include "model_graph.h"
#include "model_memory.h"
#include "model_weights.h"
#include "npu_mock.h"


/* Model dimension constants */
#define DIM_FF  512
#define BATCH  1
#define SEQ_LEN  32
#define D_MODEL  256

#define T_FP16_ND(base, off)  {(base) + (off), NPU_DTYPE_FP16, NPU_FORMAT_ND}
#define T_FP16_NZ(base, off)  {(base) + (off), NPU_DTYPE_FP16, NPU_FORMAT_NZ}
#define T_FP32_ND(base, off)  {(base) + (off), NPU_DTYPE_FP32, NPU_FORMAT_ND}
#define T_FP32_NZ(base, off)  {(base) + (off), NPU_DTYPE_FP32, NPU_FORMAT_NZ}


typedef struct {
    struct {
        npu_tensor_t in_32;
        npu_tensor_t in_33;
    } inputs;
    struct {
        npu_tensor_t l0_sa_q_proj_weight;
        npu_tensor_t l0_sa_k_proj_weight;
        npu_tensor_t l0_sa_v_proj_weight;
        npu_tensor_t l0_sa_o_proj_weight;
        npu_tensor_t l0_linear1_weight;
        npu_tensor_t l0_linear2_weight;
        npu_tensor_t l1_sa_q_proj_weight;
        npu_tensor_t l1_sa_k_proj_weight;
        npu_tensor_t l1_sa_v_proj_weight;
        npu_tensor_t l1_sa_o_proj_weight;
        npu_tensor_t l1_linear1_weight;
        npu_tensor_t l1_linear2_weight;
        npu_tensor_t l0_sa_q_proj_bias;
        npu_tensor_t l0_sa_k_proj_bias;
        npu_tensor_t l0_sa_v_proj_bias;
        npu_tensor_t l0_sa_o_proj_bias;
        npu_tensor_t l0_norm1_weight;
        npu_tensor_t l0_norm1_bias;
        npu_tensor_t l0_linear1_bias;
        npu_tensor_t l0_linear2_bias;
        npu_tensor_t l0_norm2_weight;
        npu_tensor_t l0_norm2_bias;
        npu_tensor_t l1_sa_q_proj_bias;
        npu_tensor_t l1_sa_k_proj_bias;
        npu_tensor_t l1_sa_v_proj_bias;
        npu_tensor_t l1_sa_o_proj_bias;
        npu_tensor_t l1_norm1_weight;
        npu_tensor_t l1_norm1_bias;
        npu_tensor_t l1_linear1_bias;
        npu_tensor_t l1_linear2_bias;
        npu_tensor_t l1_norm2_weight;
        npu_tensor_t l1_norm2_bias;
    } weights;
    struct {
        npu_tensor_t reshape_0;
        npu_tensor_t trans2d_1;
        npu_tensor_t reshape_4;
        npu_tensor_t trans2d_5;
        npu_tensor_t reshape_8;
        npu_tensor_t trans2d_9;
        npu_tensor_t trans2d_18;
        npu_tensor_t reformat_t_34;
        npu_tensor_t reformat_t_35;
        npu_tensor_t reformat_t_38;
        npu_tensor_t reformat_t_39;
        npu_tensor_t reformat_t_42;
        npu_tensor_t reformat_t_43;
        npu_tensor_t reformat_t_52;
        npu_tensor_t mm_bias_2;
        npu_tensor_t mm_bias_6;
        npu_tensor_t mm_bias_10;
        npu_tensor_t reshape_3;
        npu_tensor_t reshape_7;
        npu_tensor_t reshape_11;
        npu_tensor_t reformat_t_37;
        npu_tensor_t trans_12;
        npu_tensor_t reformat_t_45;
        npu_tensor_t reformat_t_46;
        npu_tensor_t mm_13;
        npu_tensor_t add_14;
        npu_tensor_t softmax_15;
        npu_tensor_t reformat_t_49;
        npu_tensor_t mm_16;
        npu_tensor_t reshape_17;
        npu_tensor_t reformat_t_51;
        npu_tensor_t mm_bias_19;
        npu_tensor_t reshape_20;
    } layer0_self_attn;
    struct {
        npu_tensor_t trans2d_24;
        npu_tensor_t trans2d_29;
        npu_tensor_t reformat_t_60;
        npu_tensor_t reformat_t_65;
        npu_tensor_t add_21;
        npu_tensor_t layernorm_22;
        npu_tensor_t reshape_23;
        npu_tensor_t reformat_t_59;
        npu_tensor_t mm_bias_25;
        npu_tensor_t reshape_26;
        npu_tensor_t gelu_27;
        npu_tensor_t reshape_28;
        npu_tensor_t reformat_t_64;
        npu_tensor_t mm_bias_30;
        npu_tensor_t reshape_31;
        npu_tensor_t add_32;
        npu_tensor_t layernorm_33;
    } layer0;
    struct {
        npu_tensor_t trans2d_35;
        npu_tensor_t trans2d_39;
        npu_tensor_t trans2d_43;
        npu_tensor_t trans2d_52;
        npu_tensor_t reformat_t_73;
        npu_tensor_t reformat_t_77;
        npu_tensor_t reformat_t_81;
        npu_tensor_t reformat_t_90;
        npu_tensor_t reshape_34;
        npu_tensor_t reshape_38;
        npu_tensor_t reshape_42;
        npu_tensor_t reformat_t_72;
        npu_tensor_t reformat_t_76;
        npu_tensor_t reformat_t_80;
        npu_tensor_t mm_bias_36;
        npu_tensor_t mm_bias_40;
        npu_tensor_t mm_bias_44;
        npu_tensor_t reshape_37;
        npu_tensor_t reshape_41;
        npu_tensor_t reshape_45;
        npu_tensor_t reformat_t_75;
        npu_tensor_t trans_46;
        npu_tensor_t reformat_t_83;
        npu_tensor_t reformat_t_84;
        npu_tensor_t mm_47;
        npu_tensor_t add_48;
        npu_tensor_t softmax_49;
        npu_tensor_t reformat_t_87;
        npu_tensor_t mm_50;
        npu_tensor_t reshape_51;
        npu_tensor_t reformat_t_89;
        npu_tensor_t mm_bias_53;
        npu_tensor_t reshape_54;
    } layer1_self_attn;
    struct {
        npu_tensor_t trans2d_58;
        npu_tensor_t trans2d_63;
        npu_tensor_t reformat_t_98;
        npu_tensor_t reformat_t_103;
        npu_tensor_t add_55;
        npu_tensor_t layernorm_56;
        npu_tensor_t reshape_57;
        npu_tensor_t reformat_t_97;
        npu_tensor_t mm_bias_59;
        npu_tensor_t reshape_60;
        npu_tensor_t gelu_61;
        npu_tensor_t reshape_62;
        npu_tensor_t reformat_t_102;
        npu_tensor_t mm_bias_64;
        npu_tensor_t reshape_65;
        npu_tensor_t add_66;
    } layer1;
    struct {
        npu_tensor_t out_107;
    } outputs;
} model_tensors_t;


static void layer0_self_attn(model_tensors_t* t) {
    /* === node_0: scalar_reshape (scalar) === */
    scalar_reshape(t->inputs.in_32, t->layer0_self_attn.reshape_0, 16384, NPU_DTYPE_FP32);

    /* === node_1: vector_transpose_2d (vector) === */
    vector_transpose_2d(t->weights.l0_sa_q_proj_weight, t->layer0_self_attn.trans2d_1, D_MODEL, D_MODEL, NPU_DTYPE_FP32);

    /* === node_4: scalar_reshape (scalar) === */
    scalar_reshape(t->inputs.in_32, t->layer0_self_attn.reshape_4, 16384, NPU_DTYPE_FP32);

    /* === node_5: vector_transpose_2d (vector) === */
    vector_transpose_2d(t->weights.l0_sa_k_proj_weight, t->layer0_self_attn.trans2d_5, D_MODEL, D_MODEL, NPU_DTYPE_FP32);

    /* === node_8: scalar_reshape (scalar) === */
    scalar_reshape(t->inputs.in_32, t->layer0_self_attn.reshape_8, 16384, NPU_DTYPE_FP32);

    /* === node_9: vector_transpose_2d (vector) === */
    vector_transpose_2d(t->weights.l0_sa_v_proj_weight, t->layer0_self_attn.trans2d_9, D_MODEL, D_MODEL, NPU_DTYPE_FP32);

    /* === node_18: vector_transpose_2d (vector) === */
    vector_transpose_2d(t->weights.l0_sa_o_proj_weight, t->layer0_self_attn.trans2d_18, D_MODEL, D_MODEL, NPU_DTYPE_FP32);

    /* === node_2: cube_matmul_bias (cube) === */
    cube_matmul_bias(t->layer0_self_attn.reformat_t_34, t->layer0_self_attn.reformat_t_35, t->weights.l0_sa_q_proj_bias, t->layer0_self_attn.mm_bias_2, 1, SEQ_LEN, D_MODEL, D_MODEL, NPU_DTYPE_FP32);

    /* === node_6: cube_matmul_bias (cube) === */
    cube_matmul_bias(t->layer0_self_attn.reformat_t_38, t->layer0_self_attn.reformat_t_39, t->weights.l0_sa_k_proj_bias, t->layer0_self_attn.mm_bias_6, 1, SEQ_LEN, D_MODEL, D_MODEL, NPU_DTYPE_FP32);

    /* === node_10: cube_matmul_bias (cube) === */
    cube_matmul_bias(t->layer0_self_attn.reformat_t_42, t->layer0_self_attn.reformat_t_43, t->weights.l0_sa_v_proj_bias, t->layer0_self_attn.mm_bias_10, 1, SEQ_LEN, D_MODEL, D_MODEL, NPU_DTYPE_FP32);

    /* === node_3: scalar_reshape (scalar) === */
    scalar_reshape(t->layer0_self_attn.mm_bias_2, t->layer0_self_attn.reshape_3, 16384, NPU_DTYPE_FP32);

    /* === node_7: scalar_reshape (scalar) === */
    scalar_reshape(t->layer0_self_attn.mm_bias_6, t->layer0_self_attn.reshape_7, 16384, NPU_DTYPE_FP32);

    /* === node_11: scalar_reshape (scalar) === */
    scalar_reshape(t->layer0_self_attn.mm_bias_10, t->layer0_self_attn.reshape_11, 16384, NPU_DTYPE_FP32);

    /* === reformat_t_37: dma_reformat (idma) === */
    dma_reformat(t->layer0_self_attn.reshape_3, t->layer0_self_attn.reformat_t_37, 8192);

    /* === node_12: vector_transpose (vector) === */
    vector_transpose(t->layer0_self_attn.reshape_7, t->layer0_self_attn.trans_12, 3, (const int[]){1, 32, 256}, BATCH, 2, NPU_DTYPE_FP16);

    /* === reformat_t_45: dma_reformat (idma) === */
    dma_reformat(t->layer0_self_attn.reshape_11, t->layer0_self_attn.reformat_t_45, 8192);

    /* === reformat_t_46: dma_reformat (idma) === */
    dma_reformat(t->layer0_self_attn.trans_12, t->layer0_self_attn.reformat_t_46, 8192);

    /* === node_13: cube_matmul (cube) === */
    cube_matmul(t->layer0_self_attn.reformat_t_37, t->layer0_self_attn.reformat_t_46, t->layer0_self_attn.mm_13, 1, SEQ_LEN, SEQ_LEN, D_MODEL, NPU_DTYPE_FP16);

    /* === node_14: vector_add (vector) === */
    vector_add(t->layer0_self_attn.mm_13, t->inputs.in_33, t->layer0_self_attn.add_14, 1024, NPU_DTYPE_FP16);

    /* === node_15: vector_softmax (vector) === */
    vector_softmax(t->layer0_self_attn.add_14, t->layer0_self_attn.softmax_15, SEQ_LEN, 1024, NPU_DTYPE_FP16);

    /* === reformat_t_49: dma_reformat (idma) === */
    dma_reformat(t->layer0_self_attn.softmax_15, t->layer0_self_attn.reformat_t_49, 1024);

    /* === node_16: cube_matmul (cube) === */
    cube_matmul(t->layer0_self_attn.reformat_t_49, t->layer0_self_attn.reformat_t_45, t->layer0_self_attn.mm_16, 1, SEQ_LEN, D_MODEL, SEQ_LEN, NPU_DTYPE_FP16);

    /* === node_17: scalar_reshape (scalar) === */
    scalar_reshape(t->layer0_self_attn.mm_16, t->layer0_self_attn.reshape_17, 32768, NPU_DTYPE_FP32);

    /* === reformat_t_51: dma_reformat (idma) === */
    dma_reformat(t->layer0_self_attn.reshape_17, t->layer0_self_attn.reformat_t_51, 8192);

    /* === node_19: cube_matmul_bias (cube) === */
    cube_matmul_bias(t->layer0_self_attn.reformat_t_51, t->layer0_self_attn.reformat_t_52, t->weights.l0_sa_o_proj_bias, t->layer0_self_attn.mm_bias_19, 1, SEQ_LEN, D_MODEL, D_MODEL, NPU_DTYPE_FP32);

    /* === node_20: scalar_reshape (scalar) === */
    scalar_reshape(t->layer0_self_attn.mm_bias_19, t->layer0_self_attn.reshape_20, 16384, NPU_DTYPE_FP32);
}


static void layer0(model_tensors_t* t) {
    /* === node_24: vector_transpose_2d (vector) === */
    vector_transpose_2d(t->weights.l0_linear1_weight, t->layer0.trans2d_24, DIM_FF, D_MODEL, NPU_DTYPE_FP32);

    /* === node_29: vector_transpose_2d (vector) === */
    vector_transpose_2d(t->weights.l0_linear2_weight, t->layer0.trans2d_29, D_MODEL, DIM_FF, NPU_DTYPE_FP32);

    /* === node_21: vector_add (vector) === */
    vector_add(t->inputs.in_32, t->layer0_self_attn.reshape_20, t->layer0.add_21, 8192, NPU_DTYPE_FP16);

    /* === node_22: vector_layernorm (vector) === */
    vector_layernorm(t->layer0.add_21, t->weights.l0_norm1_weight, t->weights.l0_norm1_bias, t->layer0.layernorm_22, D_MODEL, SEQ_LEN, 0.000010f, NPU_DTYPE_FP32);

    /* === node_23: scalar_reshape (scalar) === */
    scalar_reshape(t->layer0.layernorm_22, t->layer0.reshape_23, 32768, NPU_DTYPE_FP32);

    /* === reformat_t_59: dma_reformat (idma) === */
    dma_reformat(t->layer0.reshape_23, t->layer0.reformat_t_59, 8192);

    /* === node_25: cube_matmul_bias (cube) === */
    cube_matmul_bias(t->layer0.reformat_t_59, t->layer0.reformat_t_60, t->weights.l0_linear1_bias, t->layer0.mm_bias_25, 1, SEQ_LEN, DIM_FF, D_MODEL, NPU_DTYPE_FP32);

    /* === node_26: scalar_reshape (scalar) === */
    scalar_reshape(t->layer0.mm_bias_25, t->layer0.reshape_26, 65536, NPU_DTYPE_FP32);

    /* === node_27: vector_gelu (vector) === */
    vector_gelu(t->layer0.reshape_26, t->layer0.gelu_27, 16384, NPU_DTYPE_FP16);

    /* === node_28: scalar_reshape (scalar) === */
    scalar_reshape(t->layer0.gelu_27, t->layer0.reshape_28, 32768, NPU_DTYPE_FP32);

    /* === reformat_t_64: dma_reformat (idma) === */
    dma_reformat(t->layer0.reshape_28, t->layer0.reformat_t_64, 16384);

    /* === node_30: cube_matmul_bias (cube) === */
    cube_matmul_bias(t->layer0.reformat_t_64, t->layer0.reformat_t_65, t->weights.l0_linear2_bias, t->layer0.mm_bias_30, 1, SEQ_LEN, D_MODEL, DIM_FF, NPU_DTYPE_FP32);

    /* === node_31: scalar_reshape (scalar) === */
    scalar_reshape(t->layer0.mm_bias_30, t->layer0.reshape_31, 16384, NPU_DTYPE_FP32);

    /* === node_32: vector_add (vector) === */
    vector_add(t->layer0.layernorm_22, t->layer0.reshape_31, t->layer0.add_32, 8192, NPU_DTYPE_FP16);

    /* === node_33: vector_layernorm (vector) === */
    vector_layernorm(t->layer0.add_32, t->weights.l0_norm2_weight, t->weights.l0_norm2_bias, t->layer0.layernorm_33, D_MODEL, SEQ_LEN, 0.000010f, NPU_DTYPE_FP32);
}


static void layer1_self_attn(model_tensors_t* t) {
    /* === node_35: vector_transpose_2d (vector) === */
    vector_transpose_2d(t->weights.l1_sa_q_proj_weight, t->layer1_self_attn.trans2d_35, D_MODEL, D_MODEL, NPU_DTYPE_FP32);

    /* === node_39: vector_transpose_2d (vector) === */
    vector_transpose_2d(t->weights.l1_sa_k_proj_weight, t->layer1_self_attn.trans2d_39, D_MODEL, D_MODEL, NPU_DTYPE_FP32);

    /* === node_43: vector_transpose_2d (vector) === */
    vector_transpose_2d(t->weights.l1_sa_v_proj_weight, t->layer1_self_attn.trans2d_43, D_MODEL, D_MODEL, NPU_DTYPE_FP32);

    /* === node_52: vector_transpose_2d (vector) === */
    vector_transpose_2d(t->weights.l1_sa_o_proj_weight, t->layer1_self_attn.trans2d_52, D_MODEL, D_MODEL, NPU_DTYPE_FP32);

    /* === node_34: scalar_reshape (scalar) === */
    scalar_reshape(t->layer0.layernorm_33, t->layer1_self_attn.reshape_34, 16384, NPU_DTYPE_FP32);

    /* === node_38: scalar_reshape (scalar) === */
    scalar_reshape(t->layer0.layernorm_33, t->layer1_self_attn.reshape_38, 16384, NPU_DTYPE_FP32);

    /* === node_42: scalar_reshape (scalar) === */
    scalar_reshape(t->layer0.layernorm_33, t->layer1_self_attn.reshape_42, 16384, NPU_DTYPE_FP32);

    /* === reformat_t_72: dma_reformat (idma) === */
    dma_reformat(t->layer1_self_attn.reshape_34, t->layer1_self_attn.reformat_t_72, 8192);

    /* === reformat_t_76: dma_reformat (idma) === */
    dma_reformat(t->layer1_self_attn.reshape_38, t->layer1_self_attn.reformat_t_76, 8192);

    /* === reformat_t_80: dma_reformat (idma) === */
    dma_reformat(t->layer1_self_attn.reshape_42, t->layer1_self_attn.reformat_t_80, 8192);

    /* === node_36: cube_matmul_bias (cube) === */
    cube_matmul_bias(t->layer1_self_attn.reformat_t_72, t->layer1_self_attn.reformat_t_73, t->weights.l1_sa_q_proj_bias, t->layer1_self_attn.mm_bias_36, 1, SEQ_LEN, D_MODEL, D_MODEL, NPU_DTYPE_FP32);

    /* === node_40: cube_matmul_bias (cube) === */
    cube_matmul_bias(t->layer1_self_attn.reformat_t_76, t->layer1_self_attn.reformat_t_77, t->weights.l1_sa_k_proj_bias, t->layer1_self_attn.mm_bias_40, 1, SEQ_LEN, D_MODEL, D_MODEL, NPU_DTYPE_FP32);

    /* === node_44: cube_matmul_bias (cube) === */
    cube_matmul_bias(t->layer1_self_attn.reformat_t_80, t->layer1_self_attn.reformat_t_81, t->weights.l1_sa_v_proj_bias, t->layer1_self_attn.mm_bias_44, 1, SEQ_LEN, D_MODEL, D_MODEL, NPU_DTYPE_FP32);

    /* === node_37: scalar_reshape (scalar) === */
    scalar_reshape(t->layer1_self_attn.mm_bias_36, t->layer1_self_attn.reshape_37, 16384, NPU_DTYPE_FP32);

    /* === node_41: scalar_reshape (scalar) === */
    scalar_reshape(t->layer1_self_attn.mm_bias_40, t->layer1_self_attn.reshape_41, 16384, NPU_DTYPE_FP32);

    /* === node_45: scalar_reshape (scalar) === */
    scalar_reshape(t->layer1_self_attn.mm_bias_44, t->layer1_self_attn.reshape_45, 16384, NPU_DTYPE_FP32);

    /* === reformat_t_75: dma_reformat (idma) === */
    dma_reformat(t->layer1_self_attn.reshape_37, t->layer1_self_attn.reformat_t_75, 8192);

    /* === node_46: vector_transpose (vector) === */
    vector_transpose(t->layer1_self_attn.reshape_41, t->layer1_self_attn.trans_46, 3, (const int[]){1, 32, 256}, BATCH, 2, NPU_DTYPE_FP16);

    /* === reformat_t_83: dma_reformat (idma) === */
    dma_reformat(t->layer1_self_attn.reshape_45, t->layer1_self_attn.reformat_t_83, 8192);

    /* === reformat_t_84: dma_reformat (idma) === */
    dma_reformat(t->layer1_self_attn.trans_46, t->layer1_self_attn.reformat_t_84, 8192);

    /* === node_47: cube_matmul (cube) === */
    cube_matmul(t->layer1_self_attn.reformat_t_75, t->layer1_self_attn.reformat_t_84, t->layer1_self_attn.mm_47, 1, SEQ_LEN, SEQ_LEN, D_MODEL, NPU_DTYPE_FP16);

    /* === node_48: vector_add (vector) === */
    vector_add(t->layer1_self_attn.mm_47, t->inputs.in_33, t->layer1_self_attn.add_48, 1024, NPU_DTYPE_FP16);

    /* === node_49: vector_softmax (vector) === */
    vector_softmax(t->layer1_self_attn.add_48, t->layer1_self_attn.softmax_49, SEQ_LEN, 1024, NPU_DTYPE_FP16);

    /* === reformat_t_87: dma_reformat (idma) === */
    dma_reformat(t->layer1_self_attn.softmax_49, t->layer1_self_attn.reformat_t_87, 1024);

    /* === node_50: cube_matmul (cube) === */
    cube_matmul(t->layer1_self_attn.reformat_t_87, t->layer1_self_attn.reformat_t_83, t->layer1_self_attn.mm_50, 1, SEQ_LEN, D_MODEL, SEQ_LEN, NPU_DTYPE_FP16);

    /* === node_51: scalar_reshape (scalar) === */
    scalar_reshape(t->layer1_self_attn.mm_50, t->layer1_self_attn.reshape_51, 32768, NPU_DTYPE_FP32);

    /* === reformat_t_89: dma_reformat (idma) === */
    dma_reformat(t->layer1_self_attn.reshape_51, t->layer1_self_attn.reformat_t_89, 8192);

    /* === node_53: cube_matmul_bias (cube) === */
    cube_matmul_bias(t->layer1_self_attn.reformat_t_89, t->layer1_self_attn.reformat_t_90, t->weights.l1_sa_o_proj_bias, t->layer1_self_attn.mm_bias_53, 1, SEQ_LEN, D_MODEL, D_MODEL, NPU_DTYPE_FP32);

    /* === node_54: scalar_reshape (scalar) === */
    scalar_reshape(t->layer1_self_attn.mm_bias_53, t->layer1_self_attn.reshape_54, 16384, NPU_DTYPE_FP32);
}


static void layer1(model_tensors_t* t) {
    /* === node_58: vector_transpose_2d (vector) === */
    vector_transpose_2d(t->weights.l1_linear1_weight, t->layer1.trans2d_58, DIM_FF, D_MODEL, NPU_DTYPE_FP32);

    /* === node_63: vector_transpose_2d (vector) === */
    vector_transpose_2d(t->weights.l1_linear2_weight, t->layer1.trans2d_63, D_MODEL, DIM_FF, NPU_DTYPE_FP32);

    /* === reformat_t_34: dma_reformat (idma) === */
    dma_reformat(t->layer0_self_attn.reshape_0, t->layer0_self_attn.reformat_t_34, 8192);

    /* === reformat_t_35: dma_reformat (idma) === */
    dma_reformat(t->layer0_self_attn.trans2d_1, t->layer0_self_attn.reformat_t_35, 65536);

    /* === reformat_t_38: dma_reformat (idma) === */
    dma_reformat(t->layer0_self_attn.reshape_4, t->layer0_self_attn.reformat_t_38, 8192);

    /* === reformat_t_39: dma_reformat (idma) === */
    dma_reformat(t->layer0_self_attn.trans2d_5, t->layer0_self_attn.reformat_t_39, 65536);

    /* === reformat_t_42: dma_reformat (idma) === */
    dma_reformat(t->layer0_self_attn.reshape_8, t->layer0_self_attn.reformat_t_42, 8192);

    /* === reformat_t_43: dma_reformat (idma) === */
    dma_reformat(t->layer0_self_attn.trans2d_9, t->layer0_self_attn.reformat_t_43, 65536);

    /* === reformat_t_52: dma_reformat (idma) === */
    dma_reformat(t->layer0_self_attn.trans2d_18, t->layer0_self_attn.reformat_t_52, 65536);

    /* === reformat_t_60: dma_reformat (idma) === */
    dma_reformat(t->layer0.trans2d_24, t->layer0.reformat_t_60, 131072);

    /* === reformat_t_65: dma_reformat (idma) === */
    dma_reformat(t->layer0.trans2d_29, t->layer0.reformat_t_65, 131072);

    /* === reformat_t_73: dma_reformat (idma) === */
    dma_reformat(t->layer1_self_attn.trans2d_35, t->layer1_self_attn.reformat_t_73, 65536);

    /* === reformat_t_77: dma_reformat (idma) === */
    dma_reformat(t->layer1_self_attn.trans2d_39, t->layer1_self_attn.reformat_t_77, 65536);

    /* === reformat_t_81: dma_reformat (idma) === */
    dma_reformat(t->layer1_self_attn.trans2d_43, t->layer1_self_attn.reformat_t_81, 65536);

    /* === reformat_t_90: dma_reformat (idma) === */
    dma_reformat(t->layer1_self_attn.trans2d_52, t->layer1_self_attn.reformat_t_90, 65536);

    /* === reformat_t_98: dma_reformat (idma) === */
    dma_reformat(t->layer1.trans2d_58, t->layer1.reformat_t_98, 131072);

    /* === reformat_t_103: dma_reformat (idma) === */
    dma_reformat(t->layer1.trans2d_63, t->layer1.reformat_t_103, 131072);

    /* === node_55: vector_add (vector) === */
    vector_add(t->layer0.layernorm_33, t->layer1_self_attn.reshape_54, t->layer1.add_55, 8192, NPU_DTYPE_FP16);

    /* === node_56: vector_layernorm (vector) === */
    vector_layernorm(t->layer1.add_55, t->weights.l1_norm1_weight, t->weights.l1_norm1_bias, t->layer1.layernorm_56, D_MODEL, SEQ_LEN, 0.000010f, NPU_DTYPE_FP32);

    /* === node_57: scalar_reshape (scalar) === */
    scalar_reshape(t->layer1.layernorm_56, t->layer1.reshape_57, 32768, NPU_DTYPE_FP32);

    /* === reformat_t_97: dma_reformat (idma) === */
    dma_reformat(t->layer1.reshape_57, t->layer1.reformat_t_97, 8192);

    /* === node_59: cube_matmul_bias (cube) === */
    cube_matmul_bias(t->layer1.reformat_t_97, t->layer1.reformat_t_98, t->weights.l1_linear1_bias, t->layer1.mm_bias_59, 1, SEQ_LEN, DIM_FF, D_MODEL, NPU_DTYPE_FP32);

    /* === node_60: scalar_reshape (scalar) === */
    scalar_reshape(t->layer1.mm_bias_59, t->layer1.reshape_60, 65536, NPU_DTYPE_FP32);

    /* === node_61: vector_gelu (vector) === */
    vector_gelu(t->layer1.reshape_60, t->layer1.gelu_61, 16384, NPU_DTYPE_FP16);

    /* === node_62: scalar_reshape (scalar) === */
    scalar_reshape(t->layer1.gelu_61, t->layer1.reshape_62, 32768, NPU_DTYPE_FP32);

    /* === reformat_t_102: dma_reformat (idma) === */
    dma_reformat(t->layer1.reshape_62, t->layer1.reformat_t_102, 16384);

    /* === node_64: cube_matmul_bias (cube) === */
    cube_matmul_bias(t->layer1.reformat_t_102, t->layer1.reformat_t_103, t->weights.l1_linear2_bias, t->layer1.mm_bias_64, 1, SEQ_LEN, D_MODEL, DIM_FF, NPU_DTYPE_FP32);

    /* === node_65: scalar_reshape (scalar) === */
    scalar_reshape(t->layer1.mm_bias_64, t->layer1.reshape_65, 16384, NPU_DTYPE_FP32);

    /* === node_66: vector_add (vector) === */
    vector_add(t->layer1.layernorm_56, t->layer1.reshape_65, t->layer1.add_66, 8192, NPU_DTYPE_FP16);

    /* === node_67: vector_layernorm (vector) === */
    vector_layernorm(t->layer1.add_66, t->weights.l1_norm2_weight, t->weights.l1_norm2_bias, t->outputs.out_107, D_MODEL, SEQ_LEN, 0.000010f, NPU_DTYPE_FP32);
}


void model_run(unsigned char* hbm, unsigned char* l1) {
    model_tensors_t t = {
        .inputs = {
            .in_32 = T_FP16_ND(l1, 2636800),
            .in_33 = T_FP16_ND(l1, 2653184),
        },
        .weights = {
            .l0_sa_q_proj_weight = T_FP16_NZ(l1, 0),
            .l0_sa_k_proj_weight = T_FP16_NZ(l1, 131584),
            .l0_sa_v_proj_weight = T_FP16_NZ(l1, 263168),
            .l0_sa_o_proj_weight = T_FP16_NZ(l1, 394752),
            .l0_linear1_weight = T_FP32_ND(l1, 528384),
            .l0_linear2_weight = T_FP16_NZ(l1, 1054720),
            .l1_sa_q_proj_weight = T_FP16_NZ(l1, 1318400),
            .l1_sa_k_proj_weight = T_FP16_NZ(l1, 1449984),
            .l1_sa_v_proj_weight = T_FP16_NZ(l1, 1581568),
            .l1_sa_o_proj_weight = T_FP16_NZ(l1, 1713152),
            .l1_linear1_weight = T_FP32_ND(l1, 1846784),
            .l1_linear2_weight = T_FP16_NZ(l1, 2373120),
            .l0_sa_q_proj_bias = T_FP16_NZ(l1, 131072),
            .l0_sa_k_proj_bias = T_FP16_NZ(l1, 262656),
            .l0_sa_v_proj_bias = T_FP16_NZ(l1, 394240),
            .l0_sa_o_proj_bias = T_FP16_NZ(l1, 525824),
            .l0_norm1_weight = T_FP32_ND(l1, 526336),
            .l0_norm1_bias = T_FP32_ND(l1, 527360),
            .l0_linear1_bias = T_FP32_ND(l1, 1052672),
            .l0_linear2_bias = T_FP16_NZ(l1, 1316864),
            .l0_norm2_weight = T_FP16_ND(l1, 1317376),
            .l0_norm2_bias = T_FP16_ND(l1, 1317888),
            .l1_sa_q_proj_bias = T_FP16_NZ(l1, 1449472),
            .l1_sa_k_proj_bias = T_FP16_NZ(l1, 1581056),
            .l1_sa_v_proj_bias = T_FP16_NZ(l1, 1712640),
            .l1_sa_o_proj_bias = T_FP16_NZ(l1, 1844224),
            .l1_norm1_weight = T_FP32_ND(l1, 1844736),
            .l1_norm1_bias = T_FP32_ND(l1, 1845760),
            .l1_linear1_bias = T_FP32_ND(l1, 2371072),
            .l1_linear2_bias = T_FP16_NZ(l1, 2635264),
            .l1_norm2_weight = T_FP16_ND(l1, 2635776),
            .l1_norm2_bias = T_FP16_ND(l1, 2636288),
        },
        .layer0_self_attn = {
            .reshape_0 = T_FP16_ND(l1, 2655232),
            .trans2d_1 = T_FP16_ND(l1, 2671616),
            .reshape_4 = T_FP16_ND(l1, 2835456),
            .trans2d_5 = T_FP16_ND(l1, 2851840),
            .reshape_8 = T_FP16_ND(l1, 3015680),
            .trans2d_9 = T_FP16_ND(l1, 3032064),
            .trans2d_18 = T_FP16_ND(l1, 3290112),
            .reformat_t_34 = T_FP16_NZ(l1, 6579968),
            .reformat_t_35 = T_FP16_NZ(l1, 6596352),
            .reformat_t_38 = T_FP16_NZ(l1, 6727424),
            .reformat_t_39 = T_FP16_NZ(l1, 6743808),
            .reformat_t_42 = T_FP16_NZ(l1, 6874880),
            .reformat_t_43 = T_FP16_NZ(l1, 6891264),
            .reformat_t_52 = T_FP16_NZ(l1, 7108352),
            .mm_bias_2 = T_FP16_ND(l1, 2802688),
            .mm_bias_6 = T_FP16_ND(l1, 2982912),
            .mm_bias_10 = T_FP16_ND(l1, 3163136),
            .reshape_3 = T_FP16_ND(l1, 2819072),
            .reshape_7 = T_FP16_ND(l1, 2999296),
            .reshape_11 = T_FP16_ND(l1, 3179520),
            .reformat_t_37 = T_FP16_NZ(l1, 7022336),
            .trans_12 = T_FP32_ND(l1, 3195904),
            .reformat_t_45 = T_FP16_NZ(l1, 7075584),
            .reformat_t_46 = T_FP32_NZ(l1, 7038720),
            .mm_13 = T_FP32_ND(l1, 3228672),
            .add_14 = T_FP32_ND(l1, 3232768),
            .softmax_15 = T_FP32_ND(l1, 3236864),
            .reformat_t_49 = T_FP32_NZ(l1, 7071488),
            .mm_16 = T_FP32_ND(l1, 3240960),
            .reshape_17 = T_FP16_ND(l1, 3273728),
            .reformat_t_51 = T_FP16_NZ(l1, 7091968),
            .mm_bias_19 = T_FP16_ND(l1, 3421184),
            .reshape_20 = T_FP16_ND(l1, 3437568),
        },
        .layer0 = {
            .trans2d_24 = T_FP32_ND(l1, 3552512),
            .trans2d_29 = T_FP16_ND(l1, 4273408),
            .reformat_t_60 = T_FP32_NZ(l1, 7272192),
            .reformat_t_65 = T_FP16_NZ(l1, 7829248),
            .add_21 = T_FP32_ND(l1, 3453952),
            .layernorm_22 = T_FP32_ND(l1, 3486720),
            .reshape_23 = T_FP32_ND(l1, 3519744),
            .reformat_t_59 = T_FP32_NZ(l1, 7239424),
            .mm_bias_25 = T_FP32_ND(l1, 4076800),
            .reshape_26 = T_FP32_ND(l1, 4142336),
            .gelu_27 = T_FP16_ND(l1, 4207872),
            .reshape_28 = T_FP16_ND(l1, 4240640),
            .reformat_t_64 = T_FP16_NZ(l1, 7796480),
            .mm_bias_30 = T_FP16_ND(l1, 4535552),
            .reshape_31 = T_FP16_ND(l1, 4551936),
            .add_32 = T_FP32_ND(l1, 4568320),
            .layernorm_33 = T_FP16_ND(l1, 4601088),
        },
        .layer1_self_attn = {
            .trans2d_35 = T_FP16_ND(l1, 4633984),
            .trans2d_39 = T_FP16_ND(l1, 4814208),
            .trans2d_43 = T_FP16_ND(l1, 4994432),
            .trans2d_52 = T_FP16_ND(l1, 5252480),
            .reformat_t_73 = T_FP16_NZ(l1, 8107776),
            .reformat_t_77 = T_FP16_NZ(l1, 8255232),
            .reformat_t_81 = T_FP16_NZ(l1, 8402688),
            .reformat_t_90 = T_FP16_NZ(l1, 8619776),
            .reshape_34 = T_FP16_ND(l1, 4617600),
            .reshape_38 = T_FP16_ND(l1, 4797824),
            .reshape_42 = T_FP16_ND(l1, 4978048),
            .reformat_t_72 = T_FP16_NZ(l1, 8091392),
            .reformat_t_76 = T_FP16_NZ(l1, 8238848),
            .reformat_t_80 = T_FP16_NZ(l1, 8386304),
            .mm_bias_36 = T_FP16_ND(l1, 4765056),
            .mm_bias_40 = T_FP16_ND(l1, 4945280),
            .mm_bias_44 = T_FP16_ND(l1, 5125504),
            .reshape_37 = T_FP16_ND(l1, 4781440),
            .reshape_41 = T_FP16_ND(l1, 4961664),
            .reshape_45 = T_FP16_ND(l1, 5141888),
            .reformat_t_75 = T_FP16_NZ(l1, 8533760),
            .trans_46 = T_FP32_ND(l1, 5158272),
            .reformat_t_83 = T_FP16_NZ(l1, 8587008),
            .reformat_t_84 = T_FP32_NZ(l1, 8550144),
            .mm_47 = T_FP32_ND(l1, 5191040),
            .add_48 = T_FP32_ND(l1, 5195136),
            .softmax_49 = T_FP32_ND(l1, 5199232),
            .reformat_t_87 = T_FP32_NZ(l1, 8582912),
            .mm_50 = T_FP32_ND(l1, 5203328),
            .reshape_51 = T_FP16_ND(l1, 5236096),
            .reformat_t_89 = T_FP16_NZ(l1, 8603392),
            .mm_bias_53 = T_FP16_ND(l1, 5383552),
            .reshape_54 = T_FP16_ND(l1, 5399936),
        },
        .layer1 = {
            .trans2d_58 = T_FP32_ND(l1, 5514880),
            .trans2d_63 = T_FP16_ND(l1, 6235776),
            .reformat_t_98 = T_FP32_NZ(l1, 8783616),
            .reformat_t_103 = T_FP16_NZ(l1, 9340672),
            .add_55 = T_FP32_ND(l1, 5416320),
            .layernorm_56 = T_FP32_ND(l1, 5449088),
            .reshape_57 = T_FP32_ND(l1, 5482112),
            .reformat_t_97 = T_FP32_NZ(l1, 8750848),
            .mm_bias_59 = T_FP32_ND(l1, 6039168),
            .reshape_60 = T_FP32_ND(l1, 6104704),
            .gelu_61 = T_FP16_ND(l1, 6170240),
            .reshape_62 = T_FP16_ND(l1, 6203008),
            .reformat_t_102 = T_FP16_NZ(l1, 9307904),
            .mm_bias_64 = T_FP16_ND(l1, 6497920),
            .reshape_65 = T_FP16_ND(l1, 6514304),
            .add_66 = T_FP32_ND(l1, 6530688),
        },
        .outputs = {
            .out_107 = T_FP16_ND(l1, 6563456),
        },
    };

    /* === Bulk DMA Load === */
    npu_dma_load((void*)(l1 + 0), (void*)(hbm + 0), 131072, NPU_FORMAT_NZ, NPU_FORMAT_NZ);
    npu_dma_load((void*)(l1 + 131072), (void*)(hbm + 131072), 512, NPU_FORMAT_NZ, NPU_FORMAT_NZ);
    npu_dma_load((void*)(l1 + 131584), (void*)(hbm + 131584), 131072, NPU_FORMAT_NZ, NPU_FORMAT_NZ);
    npu_dma_load((void*)(l1 + 262656), (void*)(hbm + 262656), 512, NPU_FORMAT_NZ, NPU_FORMAT_NZ);
    npu_dma_load((void*)(l1 + 263168), (void*)(hbm + 263168), 131072, NPU_FORMAT_NZ, NPU_FORMAT_NZ);
    npu_dma_load((void*)(l1 + 394240), (void*)(hbm + 394240), 512, NPU_FORMAT_NZ, NPU_FORMAT_NZ);
    npu_dma_load((void*)(l1 + 394752), (void*)(hbm + 394752), 131072, NPU_FORMAT_NZ, NPU_FORMAT_NZ);
    npu_dma_load((void*)(l1 + 525824), (void*)(hbm + 525824), 512, NPU_FORMAT_NZ, NPU_FORMAT_NZ);
    npu_dma_load((void*)(l1 + 526336), (void*)(hbm + 526336), 1024, NPU_FORMAT_ND, NPU_FORMAT_ND);
    npu_dma_load((void*)(l1 + 527360), (void*)(hbm + 527360), 1024, NPU_FORMAT_ND, NPU_FORMAT_ND);
    npu_dma_load((void*)(l1 + 528384), (void*)(hbm + 528384), 524288, NPU_FORMAT_ND, NPU_FORMAT_ND);
    npu_dma_load((void*)(l1 + 1052672), (void*)(hbm + 1052672), 2048, NPU_FORMAT_ND, NPU_FORMAT_ND);
    npu_dma_load((void*)(l1 + 1054720), (void*)(hbm + 1054720), 262144, NPU_FORMAT_NZ, NPU_FORMAT_NZ);
    npu_dma_load((void*)(l1 + 1316864), (void*)(hbm + 1316864), 512, NPU_FORMAT_NZ, NPU_FORMAT_NZ);
    npu_dma_load((void*)(l1 + 1317376), (void*)(hbm + 1317376), 512, NPU_FORMAT_ND, NPU_FORMAT_ND);
    npu_dma_load((void*)(l1 + 1317888), (void*)(hbm + 1317888), 512, NPU_FORMAT_ND, NPU_FORMAT_ND);
    npu_dma_load((void*)(l1 + 1318400), (void*)(hbm + 1318400), 131072, NPU_FORMAT_NZ, NPU_FORMAT_NZ);
    npu_dma_load((void*)(l1 + 1449472), (void*)(hbm + 1449472), 512, NPU_FORMAT_NZ, NPU_FORMAT_NZ);
    npu_dma_load((void*)(l1 + 1449984), (void*)(hbm + 1449984), 131072, NPU_FORMAT_NZ, NPU_FORMAT_NZ);
    npu_dma_load((void*)(l1 + 1581056), (void*)(hbm + 1581056), 512, NPU_FORMAT_NZ, NPU_FORMAT_NZ);
    npu_dma_load((void*)(l1 + 1581568), (void*)(hbm + 1581568), 131072, NPU_FORMAT_NZ, NPU_FORMAT_NZ);
    npu_dma_load((void*)(l1 + 1712640), (void*)(hbm + 1712640), 512, NPU_FORMAT_NZ, NPU_FORMAT_NZ);
    npu_dma_load((void*)(l1 + 1713152), (void*)(hbm + 1713152), 131072, NPU_FORMAT_NZ, NPU_FORMAT_NZ);
    npu_dma_load((void*)(l1 + 1844224), (void*)(hbm + 1844224), 512, NPU_FORMAT_NZ, NPU_FORMAT_NZ);
    npu_dma_load((void*)(l1 + 1844736), (void*)(hbm + 1844736), 1024, NPU_FORMAT_ND, NPU_FORMAT_ND);
    npu_dma_load((void*)(l1 + 1845760), (void*)(hbm + 1845760), 1024, NPU_FORMAT_ND, NPU_FORMAT_ND);
    npu_dma_load((void*)(l1 + 1846784), (void*)(hbm + 1846784), 524288, NPU_FORMAT_ND, NPU_FORMAT_ND);
    npu_dma_load((void*)(l1 + 2371072), (void*)(hbm + 2371072), 2048, NPU_FORMAT_ND, NPU_FORMAT_ND);
    npu_dma_load((void*)(l1 + 2373120), (void*)(hbm + 2373120), 262144, NPU_FORMAT_NZ, NPU_FORMAT_NZ);
    npu_dma_load((void*)(l1 + 2635264), (void*)(hbm + 2635264), 512, NPU_FORMAT_NZ, NPU_FORMAT_NZ);
    npu_dma_load((void*)(l1 + 2635776), (void*)(hbm + 2635776), 512, NPU_FORMAT_ND, NPU_FORMAT_ND);
    npu_dma_load((void*)(l1 + 2636288), (void*)(hbm + 2636288), 512, NPU_FORMAT_ND, NPU_FORMAT_ND);
    npu_dma_load((void*)(l1 + 2636800), (void*)(hbm + 2636800), 16384, NPU_FORMAT_ND, NPU_FORMAT_ND);
    npu_dma_load((void*)(l1 + 2653184), (void*)(hbm + 2653184), 2048, NPU_FORMAT_ND, NPU_FORMAT_ND);
    npu_dma_barrier();

    layer0_self_attn(&t);
    layer0(&t);
    layer1_self_attn(&t);
    layer1(&t);

    /* === Bulk DMA Store === */
    npu_dma_store((void*)(hbm + 6418432), (void*)(l1 + 6563456), 16384, NPU_FORMAT_ND, NPU_FORMAT_ND);
    npu_dma_barrier();
}
