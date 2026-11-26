; SPDX-License-Identifier: MIT
; SPDX-FileCopyrightText: Copyright (c) 2026 The MoonMath.ai
; SPDX-FileCopyrightText: Copyright (C) 2025-2026 carlus.huang@amd.com (Opus)
;
; BF16 MiniMax-H3 attention for gfx950 (CDNA4), head dimension 128.
; Adapted from MoonMath amd-kernels/csrc/gfx950 at revision
; 73961f8ccebf057b9a223a46528a65fc35ba9273. See NOTICE in this directory.
;
; h3_pack_v stages V into the operand layout used by the PV matrix multiply.
; h3_wide_attention holds Q in registers and streams 64 keys at a time.
; QK and PV use BF16 MFMA with FP32 accumulation. The online softmax updates
; the row maximum, rescales pairs of FP32 accumulators, then updates the row
; sum. Each queue descriptor addresses exactly one head and one sequence.
; Groups of eight wave64s process 384 or 256 queries. Keep the three wide
; regions adjacent and ahead of the narrow regions for instruction fetch.
;
; Kernel arguments, block dimensions and scratch ownership are documented
; alongside the launch code. Both entry points run on the caller's stream.

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 6

; Preserve the tuned issue order while grouping repeated accumulator operations.
.macro zero_accumulators registers:vararg
    .irp reg, \registers
        v_mov_b32_e32 v\reg, 0
    .endr
.endm

.macro rescale_accumulator_pairs scale, registers:vararg
    .irp reg, \registers
        v_pk_mul_f32 v[\reg:\reg+1], v[\scale:\scale+1], v[\reg:\reg+1] op_sel_hi:[0,1]
    .endr
.endm

; One workgroup starts at each hardware work slot. Subsequent descriptors
; come from the stream-owned atomic work queue.
.text
.p2align 8
.globl h3_wide_attention
.protected h3_wide_attention
.type h3_wide_attention,@function
h3_wide_attention:
    s_load_dword s33, s[0:1], 0xe0
    s_waitcnt lgkmcnt(0)
    s_cmp_ge_i32 s2, s33
    s_cbranch_scc1 .Lh3_wide_attention_block_101
    v_lshrrev_b32_e32 v5, 1, v0
    v_bfe_u32 v6, v0, 1, 3
    v_bfe_u32 v192, v0, 4, 2
    v_bfe_u32 v5, v5, 2, 1
    v_mul_u32_u24_e32 v6, 0x2a00, v6
    v_and_b32_e32 v2, 63, v0
    v_bitop3_b32 v5, v5, v0, 1 bitop3:0x78
    v_lshl_or_b32 v6, v192, 6, v6
    s_load_dwordx4 s[36:39], s[0:1], 0x0
    s_load_dwordx2 s[34:35], s[0:1], 0x18
    s_load_dwordx2 s[48:49], s[0:1], 0xe8
    s_load_dword s72, s[0:1], 0xf0
    s_load_dwordx2 s[50:51], s[0:1], 0x80
    s_load_dwordx2 s[52:53], s[0:1], 0x90
    v_lshl_or_b32 v139, v5, 5, v6
    v_lshlrev_b32_e32 v1, 4, v2
    v_bfe_u32 v6, v2, 4, 1
    v_cmp_gt_u32_e32 vcc, 32, v2
    v_and_b32_e32 v2, 2, v0
    v_cmp_eq_u32_e64 s[0:1], 0, v2
    v_mov_b32_e32 v2, 0x3f80
    s_xor_b64 s[0:1], vcc, s[0:1]
    v_lshlrev_b32_e32 v4, 4, v0
    v_and_b32_e32 v7, 1, v0
    v_cndmask_b32_e64 v2, v2, 0, s[0:1]
    s_mov_b32 s0, 0x5040100
    v_and_b32_e32 v4, 0x70, v4
    v_perm_b32 v26, v2, v2, s0
    v_mov_b32_e32 v2, 0xeeeeeeee
    v_mov_b32_e32 v8, 0x44444444
    v_cmp_eq_u32_e32 vcc, 0, v7
    v_and_or_b32 v168, v0, 8, v4
    v_lshlrev_b32_e32 v4, 1, v4
    v_cndmask_b32_e32 v181, v2, v8, vcc
    v_bfe_u32 v2, v0, 3, 1
    s_movk_i32 s0, 0x880
    v_bfe_u32 v5, v0, 5, 1
    v_lshrrev_b32_e32 v7, 2, v0
    v_mad_u32_u24 v2, v2, s0, v4
    s_movk_i32 s0, 0x2200
    v_and_b32_e32 v3, 15, v0
    v_bitop3_b32 v5, v5, v7, 1 bitop3:0x78
    v_mad_u32_u24 v2, v6, s0, v2
    v_mov_b32_e32 v171, 0
    s_movk_i32 s73, 0x2a00
    v_or_b32_e32 v169, 16, v139
    v_add_u32_e32 v180, 0xa8000, v139
    v_and_b32_e32 v193, 32, v0
    v_mul_u32_u24_e32 v194, 0x120, v3
    v_lshlrev_b32_e32 v195, 4, v6
    v_mul_u32_u24_e32 v196, 0x120, v192
    s_mov_b32 s55, 0
    v_mov_b32_e32 v27, v26
    v_mov_b32_e32 v28, v26
    v_mov_b32_e32 v29, v26
    v_or_b32_e32 v7, 0xcc00, v1
    v_lshl_or_b32 v183, v5, 4, v2
    v_lshlrev_b32_e32 v182, 3, v192
    s_movk_i32 s74, 0x110
    v_mul_u32_u24_e32 v197, 0x110, v3
    v_cmp_eq_u32_e64 s[0:1], 0, v0
    v_mov_b32_e32 v2, s2
    s_movk_i32 s75, 0xe00
    s_mov_b32 s43, 0x20000
    s_mov_b64 s[56:57], 0xffffffff
    s_mov_b32 s76, 0xf2177617
    s_mov_b32 s77, 0x7f800001
    v_mov_b32_e32 v198, 0x23400
    v_lshlrev_b32_e32 v170, 1, v168
    scratch_store_dword off, v7, off
    scratch_store_dwordx2 off, v[168:169], off offset:4
    scratch_store_dword off, v1, off offset:12
    scratch_store_dword off, v195, off offset:16
    scratch_store_dword off, v182, off offset:20
    scratch_store_dword off, v197, off offset:24
    s_branch .Lh3_wide_attention_block_27

; 384-query tile, first MFMA and online-softmax pipeline region.
.Lwide_stage_a:
    v_xor_b32_e32 v138, 0x80000000, v135
    v_xor_b32_e32 v136, 0x80000000, v166
    v_xor_b32_e32 v134, 0x80000000, v167

.Lh3_wide_attention_block_1:
    v_mul_f32_e32 v135, 0x3e0293ee, v138
    v_fmamk_f32 v156, v156, 0x3e0293ee, v135
    v_fmamk_f32 v157, v157, 0x3e0293ee, v135
    v_fmamk_f32 v158, v158, 0x3e0293ee, v135
    v_mul_f32_e32 v166, 0x3e0293ee, v136
    v_mul_f32_e32 v167, 0x3e0293ee, v134
    v_exp_f32_e32 v156, v156
    s_mov_b64 s[6:7], src_shared_base
    v_fmamk_f32 v159, v159, 0x3e0293ee, v135
    v_fmamk_f32 v10, v10, 0x3e0293ee, v166
    v_fmamk_f32 v11, v11, 0x3e0293ee, v166
    v_fmamk_f32 v12, v12, 0x3e0293ee, v166
    v_fmamk_f32 v13, v13, 0x3e0293ee, v166
    v_fmamk_f32 v2, v2, 0x3e0293ee, v167
    v_fmamk_f32 v3, v3, 0x3e0293ee, v167
    v_fmamk_f32 v4, v4, 0x3e0293ee, v167
    v_fmamk_f32 v5, v5, 0x3e0293ee, v167
    v_fmamk_f32 v14, v14, 0x3e0293ee, v166
    v_fmamk_f32 v15, v15, 0x3e0293ee, v166
    v_fmamk_f32 v16, v16, 0x3e0293ee, v166
    v_fmamk_f32 v17, v17, 0x3e0293ee, v166
    v_fmamk_f32 v152, v152, 0x3e0293ee, v167
    v_fmamk_f32 v153, v153, 0x3e0293ee, v167
    v_fmamk_f32 v154, v154, 0x3e0293ee, v167
    v_fmamk_f32 v155, v155, 0x3e0293ee, v167
    v_fmamk_f32 v140, v140, 0x3e0293ee, v166
    v_fmamk_f32 v141, v141, 0x3e0293ee, v166
    v_fmamk_f32 v142, v142, 0x3e0293ee, v166
    v_fmamk_f32 v143, v143, 0x3e0293ee, v166
    v_fmamk_f32 v6, v6, 0x3e0293ee, v167
    v_fmamk_f32 v7, v7, 0x3e0293ee, v167
    v_fmamk_f32 v8, v8, 0x3e0293ee, v167
    v_fmamk_f32 v9, v9, 0x3e0293ee, v167
    v_fmamk_f32 v22, v22, 0x3e0293ee, v166
    v_fmamk_f32 v23, v23, 0x3e0293ee, v166
    v_fmamk_f32 v24, v24, 0x3e0293ee, v166
    v_fmac_f32_e32 v166, 0x3e0293ee, v25
    v_fmamk_f32 v25, v144, 0x3e0293ee, v167
    v_fmamk_f32 v144, v145, 0x3e0293ee, v167
    v_fmamk_f32 v145, v146, 0x3e0293ee, v167
    v_fmac_f32_e32 v167, 0x3e0293ee, v147
    v_exp_f32_e32 v146, v157
    s_and_b64 s[2:3], s[8:9], s[4:5]
    v_exp_f32_e32 v147, v158
    s_add_i32 s82, s82, 1
    v_fmamk_f32 v18, v18, 0x3e0293ee, v135
    v_fmamk_f32 v19, v19, 0x3e0293ee, v135
    v_fmamk_f32 v20, v20, 0x3e0293ee, v135
    v_fmamk_f32 v21, v21, 0x3e0293ee, v135
    v_fmamk_f32 v160, v160, 0x3e0293ee, v135
    v_fmamk_f32 v161, v161, 0x3e0293ee, v135
    v_fmamk_f32 v162, v162, 0x3e0293ee, v135
    v_fmamk_f32 v163, v163, 0x3e0293ee, v135
    v_fmamk_f32 v148, v148, 0x3e0293ee, v135
    v_fmamk_f32 v149, v149, 0x3e0293ee, v135
    v_fmamk_f32 v150, v150, 0x3e0293ee, v135
    v_fmac_f32_e32 v135, 0x3e0293ee, v151
    v_exp_f32_e32 v151, v159
    s_and_b32 s11, s81, 3
    v_exp_f32_e32 v10, v10
    s_lshr_b32 s13, s81, 8
    v_exp_f32_e32 v11, v11
    s_bfe_u32 s4, s81, 0x20008
    v_exp_f32_e32 v12, v12
    s_lshr_b32 s12, s81, 16
    v_exp_f32_e32 v13, v13
    s_add_i32 s5, s82, s67
    v_exp_f32_e32 v2, v2
    s_mul_i32 s9, s5, 0xa8000
    v_exp_f32_e32 v3, v3
    s_add_i32 s5, s82, s68
    v_exp_f32_e32 v4, v4
    s_lshl_b32 s8, s5, 14
    v_exp_f32_e32 v5, v5
    s_mulk_i32 s4, 0x4400
    v_exp_f32_e32 v157, v18
    s_add_u32 s14, s4, s62
    v_exp_f32_e32 v158, v19
    s_addc_u32 s15, s7, s63
    v_exp_f32_e32 v159, v20
    s_mul_i32 s5, s11, 0x4800
    v_exp_f32_e32 v21, v21
    s_add_i32 s7, s60, s5
    v_exp_f32_e32 v14, v14
    s_add_i32 s6, s7, 0xcc00
    v_exp_f32_e32 v15, v15
    s_add_i32 s4, s59, s4
    v_exp_f32_e32 v16, v16
    s_add_i32 s5, s4, 0x2200
    v_exp_f32_e32 v17, v17
    s_add_i32 s4, s7, 0xd000
    v_exp_f32_e32 v152, v152
    s_cmp_lg_u64 s[14:15], 0
    v_exp_f32_e32 v153, v153
    s_cselect_b32 s7, s14, -1
    v_exp_f32_e32 v154, v154
    s_mov_b32 m0, s7
    v_exp_f32_e32 v155, v155
    s_lshl_b32 s7, s81, 16
    v_exp_f32_e32 v160, v160
    s_and_b32 s7, s7, 0xff0000
    v_exp_f32_e32 v161, v161
    s_or_b32 s81, s7, s13
    v_exp_f32_e32 v162, v162
    v_exp_f32_e32 v163, v163
    v_exp_f32_e32 v140, v140
    v_exp_f32_e32 v141, v141
    v_exp_f32_e32 v142, v142
    v_exp_f32_e32 v143, v143
    v_exp_f32_e32 v6, v6
    v_exp_f32_e32 v7, v7
    v_exp_f32_e32 v8, v8
    v_exp_f32_e32 v9, v9
    v_exp_f32_e32 v148, v148
    v_exp_f32_e32 v149, v149
    v_exp_f32_e32 v150, v150
    v_exp_f32_e32 v135, v135
    v_exp_f32_e32 v176, v22
    v_exp_f32_e32 v177, v23
    v_exp_f32_e32 v178, v24
    v_exp_f32_e32 v166, v166
    v_exp_f32_e32 v179, v25
    v_exp_f32_e32 v144, v144
    v_exp_f32_e32 v145, v145
    v_exp_f32_e32 v167, v167
    s_mulk_i32 s12, 0x4800
    s_mulk_i32 s11, 0x4400
    v_cvt_pk_bf16_f32 v23, v12, v13
    v_cvt_pk_bf16_f32 v13, v150, v135
    v_cvt_pk_bf16_f32 v6, v6, v7
    v_cvt_pk_bf16_f32 v7, v8, v9
    v_cvt_pk_bf16_f32 v8, v179, v144
    v_add_u32_e32 v135, s12, v190
    v_add_u32_e32 v144, s11, v183
    v_cvt_pk_bf16_f32 v18, v156, v146
    v_cvt_pk_bf16_f32 v19, v147, v151
    v_cvt_pk_bf16_f32 v20, v157, v158
    v_cvt_pk_bf16_f32 v21, v159, v21
    v_cvt_pk_bf16_f32 v22, v10, v11
    v_cvt_pk_bf16_f32 v24, v14, v15
    v_cvt_pk_bf16_f32 v25, v16, v17
    v_cvt_pk_bf16_f32 v2, v2, v3
    v_cvt_pk_bf16_f32 v3, v4, v5
    v_cvt_pk_bf16_f32 v4, v152, v153
    v_cvt_pk_bf16_f32 v5, v154, v155
    v_cvt_pk_bf16_f32 v10, v160, v161
    v_cvt_pk_bf16_f32 v11, v162, v163
    v_cvt_pk_bf16_f32 v12, v148, v149
    v_cvt_pk_bf16_f32 v14, v140, v141
    v_cvt_pk_bf16_f32 v15, v142, v143
    v_cvt_pk_bf16_f32 v16, v176, v177
    v_cvt_pk_bf16_f32 v17, v178, v166
    v_cvt_pk_bf16_f32 v9, v145, v167
    ds_read_b128 v[146:149], v135
    ds_read_b128 v[140:143], v135 offset:1024
    v_smfmac_f32_16x16x64_bf16 v[30:33], v[26:29], v[18:25], v181
    s_sub_i32 s71, s71, 64
    s_add_i32 s70, s70, 0xa8000
    s_cmp_eq_u32 s69, s82
    v_smfmac_f32_16x16x64_bf16 v[30:33], v[26:29], v[10:17], v181
    v_smfmac_f32_16x16x64_bf16 v[34:37], v[26:29], v[2:9], v181
    s_cbranch_scc1 .Lh3_wide_attention_block_67

.Lh3_wide_attention_block_2:
    s_andn2_b64 vcc, exec, s[2:3]
    s_cbranch_vccz .Lh3_wide_attention_block_64

.Lh3_wide_attention_block_3:
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[130:133], v[146:149], v[22:25], v[130:133]
    ds_read_b128 v[150:153], v135 offset:2048
    ; Form the next K address ahead of the independent MFMA.
    v_add_u32_e32 v137, s9, v139
    v_mfma_f32_16x16x32_bf16 v[126:129], v[146:149], v[18:21], v[126:129]
    v_mfma_f32_16x16x32_bf16 v[122:125], v[146:149], v[2:5], v[122:125]
    s_waitcnt lgkmcnt(1)
    buffer_load_dwordx4 v137, s[40:43], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[118:121], v[140:143], v[2:5], v[118:121]
    ds_read_b128 v[146:149], v135 offset:3072
    v_mfma_f32_16x16x32_bf16 v[114:117], v[140:143], v[18:21], v[114:117]
    v_mfma_f32_16x16x32_bf16 v[110:113], v[140:143], v[22:25], v[110:113]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[106:109], v[150:153], v[22:25], v[106:109]
    ds_read_b128 v[154:157], v135 offset:4096
    v_mfma_f32_16x16x32_bf16 v[102:105], v[150:153], v[18:21], v[102:105]
    v_mfma_f32_16x16x32_bf16 v[98:101], v[150:153], v[2:5], v[98:101]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[94:97], v[146:149], v[2:5], v[94:97]
    ds_read_b128 v[140:143], v135 offset:5120
    v_mfma_f32_16x16x32_bf16 v[90:93], v[146:149], v[18:21], v[90:93]
    v_mfma_f32_16x16x32_bf16 v[86:89], v[146:149], v[22:25], v[86:89]
    s_mov_b32 m0, s5
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[82:85], v[154:157], v[22:25], v[82:85]
    ds_read_b128 v[146:149], v135 offset:6144
    ; Form the next K address ahead of the independent MFMA.
    v_add_u32_e32 v137, s9, v169
    v_mfma_f32_16x16x32_bf16 v[78:81], v[154:157], v[18:21], v[78:81]
    v_mfma_f32_16x16x32_bf16 v[74:77], v[154:157], v[2:5], v[74:77]
    s_waitcnt lgkmcnt(1)
    buffer_load_dwordx4 v137, s[40:43], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[70:73], v[140:143], v[2:5], v[70:73]
    ds_read_b128 v[150:153], v135 offset:7168
    v_mfma_f32_16x16x32_bf16 v[66:69], v[140:143], v[18:21], v[66:69]
    v_mfma_f32_16x16x32_bf16 v[62:65], v[140:143], v[22:25], v[62:65]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[58:61], v[146:149], v[22:25], v[58:61]
    ds_read_b128 v[140:143], v135 offset:8192
    v_mfma_f32_16x16x32_bf16 v[54:57], v[146:149], v[18:21], v[54:57]
    v_mfma_f32_16x16x32_bf16 v[50:53], v[146:149], v[2:5], v[50:53]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[2:5], v[150:153], v[2:5], v[38:41]
    s_nop 2
    ds_read_b128 v[38:41], v135 offset:9216
    v_mfma_f32_16x16x32_bf16 v[18:21], v[150:153], v[18:21], v[46:49]
    v_mfma_f32_16x16x32_bf16 v[22:25], v[150:153], v[22:25], v[42:45]
    s_mov_b32 m0, s6
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[130:133], v[140:143], v[14:17], v[130:133]
    s_nop 0
    ds_read_b128 v[44:47], v135 offset:10240
    v_mfma_f32_16x16x32_bf16 v[126:129], v[140:143], v[10:13], v[126:129]
    v_mfma_f32_16x16x32_bf16 v[122:125], v[140:143], v[6:9], v[122:125]
    v_or_b32_e32 v42, s8, v188
    s_waitcnt lgkmcnt(1)
    buffer_load_dwordx4 v42, s[44:47], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[118:121], v[38:41], v[6:9], v[118:121]
    ds_read_b128 v[140:143], v135 offset:11264
    v_mfma_f32_16x16x32_bf16 v[114:117], v[38:41], v[10:13], v[114:117]
    v_mfma_f32_16x16x32_bf16 v[110:113], v[38:41], v[14:17], v[110:113]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[106:109], v[44:47], v[14:17], v[106:109]
    ds_read_b128 v[146:149], v135 offset:12288
    v_mfma_f32_16x16x32_bf16 v[102:105], v[44:47], v[10:13], v[102:105]
    v_mfma_f32_16x16x32_bf16 v[98:101], v[44:47], v[6:9], v[98:101]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[94:97], v[140:143], v[6:9], v[94:97]
    ds_read_b128 v[38:41], v135 offset:13312
    v_mfma_f32_16x16x32_bf16 v[90:93], v[140:143], v[10:13], v[90:93]
    v_mfma_f32_16x16x32_bf16 v[86:89], v[140:143], v[14:17], v[86:89]
    s_mov_b32 m0, s4
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[82:85], v[146:149], v[14:17], v[82:85]
    ds_read_b128 v[44:47], v135 offset:14336
    v_mfma_f32_16x16x32_bf16 v[78:81], v[146:149], v[10:13], v[78:81]
    v_mfma_f32_16x16x32_bf16 v[74:77], v[146:149], v[6:9], v[74:77]
    v_or_b32_e32 v42, 0x400, v42
    s_waitcnt lgkmcnt(1)
    buffer_load_dwordx4 v42, s[44:47], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[70:73], v[38:41], v[6:9], v[70:73]
    ds_read_b128 v[140:143], v135 offset:15360
    v_mfma_f32_16x16x32_bf16 v[66:69], v[38:41], v[10:13], v[66:69]
    v_mfma_f32_16x16x32_bf16 v[62:65], v[38:41], v[14:17], v[62:65]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[58:61], v[44:47], v[14:17], v[58:61]
    ds_read_b128 v[146:149], v144
    v_mfma_f32_16x16x32_bf16 v[54:57], v[44:47], v[10:13], v[54:57]
    v_mfma_f32_16x16x32_bf16 v[50:53], v[44:47], v[6:9], v[50:53]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[38:41], v[140:143], v[6:9], v[2:5]
    ds_read_b128 v[150:153], v144 offset:1088
    v_mfma_f32_16x16x32_bf16 v[46:49], v[140:143], v[10:13], v[18:21]
    v_mfma_f32_16x16x32_bf16 v[42:45], v[140:143], v[14:17], v[22:25]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[2:5], v[146:149], v[226:229], 0
    ds_read_b128 v[154:157], v144 offset:4352
    v_mfma_f32_16x16x32_bf16 v[6:9], v[146:149], v[210:213], 0
    v_mfma_f32_16x16x32_bf16 v[10:13], v[146:149], v[194:197], 0
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[14:17], v[150:153], v[194:197], 0
    ds_read_b128 v[162:165], v144 offset:5440
    v_mfma_f32_16x16x32_bf16 v[18:21], v[150:153], v[210:213], 0
    v_mfma_f32_16x16x32_bf16 v[22:25], v[150:153], v[226:229], 0
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[140:143], v[154:157], v[226:229], 0
    ds_read_b128 v[176:179], v144 offset:256
    v_mfma_f32_16x16x32_bf16 v[146:149], v[154:157], v[210:213], 0
    v_mfma_f32_16x16x32_bf16 v[150:153], v[154:157], v[194:197], 0
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[154:157], v[162:165], v[194:197], 0
    ds_read_b128 v[184:187], v144 offset:1344
    v_mfma_f32_16x16x32_bf16 v[158:161], v[162:165], v[210:213], 0
    v_mfma_f32_16x16x32_bf16 v[162:165], v[162:165], v[226:229], 0
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[2:5], v[176:179], v[230:233], v[2:5]
    ds_read_b128 v[242:245], v144 offset:4608
    v_mfma_f32_16x16x32_bf16 v[6:9], v[176:179], v[214:217], v[6:9]
    v_mfma_f32_16x16x32_bf16 v[10:13], v[176:179], v[198:201], v[10:13]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[14:17], v[184:187], v[198:201], v[14:17]
    ds_read_b128 v[176:179], v144 offset:5696
    v_mfma_f32_16x16x32_bf16 v[18:21], v[184:187], v[214:217], v[18:21]
    v_mfma_f32_16x16x32_bf16 v[22:25], v[184:187], v[230:233], v[22:25]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[140:143], v[242:245], v[230:233], v[140:143]
    ds_read_b128 v[184:187], v144 offset:512
    v_mfma_f32_16x16x32_bf16 v[146:149], v[242:245], v[214:217], v[146:149]
    v_mfma_f32_16x16x32_bf16 v[150:153], v[242:245], v[198:201], v[150:153]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[154:157], v[176:179], v[198:201], v[154:157]
    ds_read_b128 v[246:249], v144 offset:1600
    v_mfma_f32_16x16x32_bf16 v[158:161], v[176:179], v[214:217], v[158:161]
    v_mfma_f32_16x16x32_bf16 v[162:165], v[176:179], v[230:233], v[162:165]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[2:5], v[184:187], v[234:237], v[2:5]
    ds_read_b128 v[250:253], v144 offset:4864
    v_mfma_f32_16x16x32_bf16 v[6:9], v[184:187], v[218:221], v[6:9]
    v_mfma_f32_16x16x32_bf16 v[176:179], v[184:187], v[202:205], v[10:13]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[14:17], v[246:249], v[202:205], v[14:17]
    s_nop 0
    ds_read_b128 v[10:13], v144 offset:5952
    v_mfma_f32_16x16x32_bf16 v[242:245], v[246:249], v[218:221], v[18:21]
    v_mfma_f32_16x16x32_bf16 v[22:25], v[246:249], v[234:237], v[22:25]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[140:143], v[250:253], v[234:237], v[140:143]
    ds_read_b128 v[18:21], v144 offset:768
    v_mfma_f32_16x16x32_bf16 v[146:149], v[250:253], v[218:221], v[146:149]
    v_mfma_f32_16x16x32_bf16 v[246:249], v[250:253], v[202:205], v[150:153]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[250:253], v[10:13], v[202:205], v[154:157]
    s_nop 0
    ds_read_b128 v[150:153], v144 offset:1856
    v_mfma_f32_16x16x32_bf16 v[184:187], v[10:13], v[218:221], v[158:161]
    v_mfma_f32_16x16x32_bf16 v[164:167], v[10:13], v[234:237], v[162:165]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[2:5], v[18:21], v[238:241], v[2:5]
    s_nop 0
    ds_read_b128 v[160:163], v144 offset:5120
    v_mfma_f32_16x16x32_bf16 v[10:13], v[18:21], v[222:225], v[6:9]
    v_mfma_f32_16x16x32_bf16 v[156:159], v[18:21], v[206:209], v[176:179]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[18:21], v[150:153], v[206:209], v[14:17]
    s_nop 0
    ds_read_b128 v[176:179], v144 offset:6208
    v_mfma_f32_16x16x32_bf16 v[14:17], v[150:153], v[222:225], v[242:245]
    v_mfma_f32_16x16x32_bf16 v[152:155], v[150:153], v[238:241], v[22:25]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[6:9], v[160:163], v[238:241], v[140:143]
    v_mfma_f32_16x16x32_bf16 v[140:143], v[160:163], v[222:225], v[146:149]
    v_mfma_f32_16x16x32_bf16 v[160:163], v[160:163], v[206:209], v[246:249]
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_16x16x32_bf16 v[148:151], v[176:179], v[206:209], v[250:253]
    v_mfma_f32_16x16x32_bf16 v[22:25], v[176:179], v[222:225], v[184:187]
    v_mfma_f32_16x16x32_bf16 v[144:147], v[176:179], v[238:241], v[164:167]
    s_waitcnt vmcnt(0)
    s_barrier
    s_cmp_lt_i32 s82, s10
    s_cbranch_scc0 .Lh3_wide_attention_block_65

.Lh3_wide_attention_block_4:
    ; Start the local three-input row-maximum reductions.
    v_max3_f32 v135, v156, v157, v158
    v_max3_f32 v137, v159, v18, v19
    v_max3_f32 v164, v20, v21, v160
    v_max3_f32 v165, v161, v162, v163
    v_max3_f32 v166, v148, v149, v150
    v_max3_f32 v135, v135, v137, v164
    v_max3_f32 v137, v165, v166, s76
    v_max3_f32 v165, v135, v137, v151
    v_max3_f32 v135, v10, v11, v12
    v_max3_f32 v137, v13, v14, v15
    v_max3_f32 v164, v16, v17, v140
    v_max3_f32 v166, v141, v142, v143
    v_max3_f32 v167, v22, v23, v24
    v_max3_f32 v135, v135, v137, v164
    v_max3_f32 v137, v166, v167, s76
    v_max3_f32 v164, v135, v137, v25
    v_max3_f32 v135, v2, v3, v4
    v_max3_f32 v137, v5, v152, v153
    v_max3_f32 v166, v154, v155, v6
    v_max3_f32 v167, v7, v8, v9
    v_max3_f32 v176, v144, v145, v146
    v_max3_f32 v135, v135, v137, v166
    s_cmp_eq_u32 s82, 0
    v_max3_f32 v137, v167, v176, s76
    s_cselect_b64 s[2:3], -1, 0
    v_max3_f32 v137, v135, v137, v147
    s_mov_b64 s[4:5], -1
    s_and_b64 vcc, exec, s[2:3]
    s_cbranch_vccnz .Lh3_wide_attention_block_5
    v_cmp_gt_f32_e64 s[4:5], v165, -v138
    v_cmp_gt_f32_e64 s[6:7], v164, -v136
    s_or_b64 s[4:5], s[4:5], s[6:7]
    v_cmp_gt_f32_e64 s[6:7], v137, -v134
    s_or_b64 s[4:5], s[4:5], s[6:7]
    s_and_b64 vcc, exec, s[4:5]
    s_cselect_b64 s[4:5], -1, 0

.Lh3_wide_attention_block_5:
    s_cmp_lg_u32 s82, 0
    s_cselect_b64 s[8:9], -1, 0
    s_andn2_b64 vcc, exec, s[4:5]
    s_cbranch_vccnz .Lh3_wide_attention_block_1
    v_mov_b32_e32 v135, v165
    v_mov_b32_e32 v166, v164
    v_mov_b32_e32 v167, v137
    v_permlane32_swap_b32_e64 v165, v135 bound_ctrl:1
    v_permlane32_swap_b32_e64 v164, v166 bound_ctrl:1
    v_permlane32_swap_b32_e64 v137, v167 bound_ctrl:1
    s_andn2_b64 vcc, exec, s[8:9]
    v_max_f32_e32 v135, v165, v135
    v_max_f32_e32 v164, v164, v166
    v_max_f32_e32 v137, v137, v167
    v_mov_b32_e32 v165, v135
    v_mov_b32_e32 v166, v164
    v_mov_b32_e32 v167, v137
    v_permlane16_swap_b32_e64 v135, v165 bound_ctrl:1
    v_permlane16_swap_b32_e64 v164, v166 bound_ctrl:1
    v_permlane16_swap_b32_e64 v137, v167 bound_ctrl:1
    s_nop 0
    v_max_f32_e32 v135, v135, v165
    v_max_f32_e64 v165, -v138, v135
    v_cndmask_b32_e64 v135, v165, v135, s[2:3]
    v_max_f32_e32 v164, v164, v166
    v_add_f32_e32 v138, v135, v138
    v_mul_f32_e32 v165, 0x3e0293ee, v138
    v_max_f32_e64 v138, -v136, v164
    v_cndmask_b32_e64 v166, v138, v164, s[2:3]
    v_max_f32_e32 v137, v137, v167
    v_add_f32_e32 v136, v166, v136
    v_mul_f32_e32 v164, 0x3e0293ee, v136
    v_max_f32_e64 v136, -v134, v137
    v_cndmask_b32_e64 v167, v136, v137, s[2:3]
    v_add_f32_e32 v134, v167, v134
    v_mul_f32_e32 v137, 0x3e0293ee, v134
    s_cbranch_vccnz .Lwide_stage_a
    v_exp_f32_e64 v134, -v165
    v_exp_f32_e64 v136, -v164
    v_exp_f32_e64 v138, -v137
    v_mul_f32_e32 v30, v134, v30
    v_mul_f32_e32 v31, v134, v31
    v_mul_f32_e32 v32, v136, v32
    v_mul_f32_e32 v33, v136, v33
    v_mul_f32_e32 v34, v138, v34
    v_mul_f32_e32 v35, v138, v35
    v_mul_f32_e32 v36, v138, v36
    v_mul_f32_e32 v37, v138, v37
    s_branch .Lwide_stage_a

; 384-query tile, second pipeline region.
.Lwide_stage_b:
    v_mul_f32_e32 v141, 0x3e0293ee, v142
    v_fmamk_f32 v160, v160, 0x3e0293ee, v141
    v_fmamk_f32 v161, v161, 0x3e0293ee, v141
    v_mul_f32_e32 v176, 0x3e0293ee, v140
    v_mul_f32_e32 v177, 0x3e0293ee, v138
    v_fmamk_f32 v162, v162, 0x3e0293ee, v141
    v_fmamk_f32 v163, v163, 0x3e0293ee, v141
    v_fmamk_f32 v10, v10, 0x3e0293ee, v176
    v_fmamk_f32 v11, v11, 0x3e0293ee, v176
    v_fmamk_f32 v12, v12, 0x3e0293ee, v176
    v_fmamk_f32 v13, v13, 0x3e0293ee, v176
    v_fmamk_f32 v18, v18, 0x3e0293ee, v141
    v_fmamk_f32 v19, v19, 0x3e0293ee, v141
    v_fmamk_f32 v20, v20, 0x3e0293ee, v141
    v_fmamk_f32 v21, v21, 0x3e0293ee, v141
    v_fmamk_f32 v14, v14, 0x3e0293ee, v176
    v_fmamk_f32 v15, v15, 0x3e0293ee, v176
    v_fmamk_f32 v16, v16, 0x3e0293ee, v176
    v_fmamk_f32 v17, v17, 0x3e0293ee, v176
    v_fmamk_f32 v164, v164, 0x3e0293ee, v141
    v_fmamk_f32 v165, v165, 0x3e0293ee, v141
    v_fmamk_f32 v166, v166, 0x3e0293ee, v141
    v_fmamk_f32 v167, v167, 0x3e0293ee, v141
    v_fmamk_f32 v22, v22, 0x3e0293ee, v176
    v_fmamk_f32 v23, v23, 0x3e0293ee, v176
    v_fmamk_f32 v24, v24, 0x3e0293ee, v176
    v_fmamk_f32 v25, v25, 0x3e0293ee, v176
    v_fmamk_f32 v152, v152, 0x3e0293ee, v141
    v_fmamk_f32 v153, v153, 0x3e0293ee, v141
    v_fmamk_f32 v154, v154, 0x3e0293ee, v141
    v_fmac_f32_e32 v141, 0x3e0293ee, v155
    v_fmamk_f32 v148, v148, 0x3e0293ee, v176
    v_fmamk_f32 v149, v149, 0x3e0293ee, v176
    v_fmamk_f32 v150, v150, 0x3e0293ee, v176
    v_fmac_f32_e32 v176, 0x3e0293ee, v151
    v_fmamk_f32 v151, v156, 0x3e0293ee, v177
    v_fmamk_f32 v155, v157, 0x3e0293ee, v177
    v_fmamk_f32 v156, v158, 0x3e0293ee, v177
    v_exp_f32_e32 v157, v160
    s_mov_b64 s[4:5], src_shared_base
    v_exp_f32_e32 v158, v161
    s_add_i32 s81, s81, 1
    v_fmamk_f32 v2, v2, 0x3e0293ee, v177
    v_fmamk_f32 v3, v3, 0x3e0293ee, v177
    v_fmamk_f32 v4, v4, 0x3e0293ee, v177
    v_fmamk_f32 v5, v5, 0x3e0293ee, v177
    v_fmamk_f32 v144, v144, 0x3e0293ee, v177
    v_fmamk_f32 v145, v145, 0x3e0293ee, v177
    v_fmamk_f32 v146, v146, 0x3e0293ee, v177
    v_fmamk_f32 v147, v147, 0x3e0293ee, v177
    v_fmamk_f32 v6, v6, 0x3e0293ee, v177
    v_fmamk_f32 v7, v7, 0x3e0293ee, v177
    v_fmamk_f32 v8, v8, 0x3e0293ee, v177
    v_fmamk_f32 v9, v9, 0x3e0293ee, v177
    v_fmac_f32_e32 v177, 0x3e0293ee, v159
    v_exp_f32_e32 v159, v162
    s_and_b32 s12, s80, 3
    v_exp_f32_e32 v160, v163
    s_lshr_b32 s13, s80, 8
    v_exp_f32_e32 v161, v10
    s_bfe_u32 s4, s80, 0x20008
    v_exp_f32_e32 v162, v11
    s_lshr_b32 s14, s80, 16
    v_exp_f32_e32 v163, v12
    s_add_i32 s6, s81, s67
    v_exp_f32_e32 v178, v13
    s_mul_i32 s9, s6, 0xa8000
    v_exp_f32_e32 v2, v2
    s_add_i32 s6, s81, s68
    v_exp_f32_e32 v3, v3
    s_lshl_b32 s7, s6, 14
    v_exp_f32_e32 v4, v4
    s_mul_i32 s6, s14, 0x4400
    v_exp_f32_e32 v5, v5
    s_add_u32 s10, s6, s62
    v_exp_f32_e32 v179, v18
    s_addc_u32 s11, s5, s63
    v_exp_f32_e32 v184, v19
    s_mulk_i32 s4, 0x4800
    v_exp_f32_e32 v185, v20
    s_add_i32 s4, s60, s4
    v_exp_f32_e32 v21, v21
    s_add_i32 s5, s4, 0xcc00
    v_exp_f32_e32 v18, v14
    s_add_i32 s6, s59, s6
    v_exp_f32_e32 v19, v15
    s_addk_i32 s6, 0x2200
    v_exp_f32_e32 v20, v16
    s_add_i32 s4, s4, 0xd000
    v_exp_f32_e32 v186, v17
    s_cmp_lg_u64 s[10:11], 0
    v_exp_f32_e32 v187, v144
    s_cselect_b32 s10, s10, -1
    v_exp_f32_e32 v145, v145
    s_mov_b32 m0, s10
    v_exp_f32_e32 v146, v146
    s_lshl_b32 s10, s80, 16
    v_exp_f32_e32 v147, v147
    s_and_b32 s10, s10, 0xff0000
    v_exp_f32_e32 v10, v164
    s_or_b32 s80, s10, s13
    v_exp_f32_e32 v11, v165
    v_exp_f32_e32 v12, v166
    v_exp_f32_e32 v13, v167
    v_exp_f32_e32 v14, v22
    v_exp_f32_e32 v15, v23
    v_exp_f32_e32 v16, v24
    v_exp_f32_e32 v17, v25
    v_exp_f32_e32 v6, v6
    v_exp_f32_e32 v7, v7
    v_exp_f32_e32 v8, v8
    v_exp_f32_e32 v9, v9
    v_exp_f32_e32 v22, v152
    v_exp_f32_e32 v23, v153
    v_exp_f32_e32 v24, v154
    v_exp_f32_e32 v25, v141
    v_exp_f32_e32 v152, v148
    v_exp_f32_e32 v149, v149
    v_exp_f32_e32 v150, v150
    v_exp_f32_e32 v153, v176
    v_exp_f32_e32 v151, v151
    v_exp_f32_e32 v154, v155
    v_exp_f32_e32 v155, v156
    v_exp_f32_e32 v156, v177
    s_mulk_i32 s14, 0x4800
    s_mul_i32 s10, s12, 0x4400
    s_mulk_i32 s12, 0x4800
    s_sub_i32 s71, s71, 64
    s_add_i32 s70, s70, 0xa8000
    v_add_u32_e32 v141, s14, v190
    v_add_u32_e32 v144, s10, v183
    v_add_u32_e32 v148, s12, v190
    v_cvt_pk_bf16_f32 v6, v6, v7
    v_cvt_pk_bf16_f32 v7, v8, v9
    v_cvt_pk_bf16_f32 v8, v151, v154
    v_cvt_pk_bf16_f32 v9, v155, v156
    v_cvt_pk_bf16_f32 v14, v14, v15
    v_cvt_pk_bf16_f32 v15, v16, v17
    v_cvt_pk_bf16_f32 v16, v152, v149
    v_cvt_pk_bf16_f32 v17, v150, v153
    v_cvt_pk_bf16_f32 v10, v10, v11
    v_cvt_pk_bf16_f32 v11, v12, v13
    v_cvt_pk_bf16_f32 v12, v22, v23
    v_cvt_pk_bf16_f32 v13, v24, v25
    v_cvt_pk_bf16_f32 v2, v2, v3
    v_cvt_pk_bf16_f32 v3, v4, v5
    v_cvt_pk_bf16_f32 v4, v187, v145
    v_cvt_pk_bf16_f32 v5, v146, v147
    v_cvt_pk_bf16_f32 v22, v161, v162
    v_cvt_pk_bf16_f32 v23, v163, v178
    v_cvt_pk_bf16_f32 v24, v18, v19
    v_cvt_pk_bf16_f32 v25, v20, v186
    v_cvt_pk_bf16_f32 v18, v157, v158
    v_cvt_pk_bf16_f32 v19, v159, v160
    v_cvt_pk_bf16_f32 v20, v179, v184
    s_cmp_eq_u32 s69, s81
    v_cvt_pk_bf16_f32 v21, v185, v21
    s_cbranch_scc1 .Lh3_wide_attention_block_78

.Lh3_wide_attention_block_7:
    s_andn2_b64 vcc, exec, s[2:3]
    s_cbranch_vccz .Lh3_wide_attention_block_72

.Lh3_wide_attention_block_8:
    s_waitcnt vmcnt(0)
    v_smfmac_f32_16x16x64_bf16 v[30:33], v[26:29], v[18:25], v181
    s_barrier
    v_smfmac_f32_16x16x64_bf16 v[30:33], v[26:29], v[10:17], v181
    v_smfmac_f32_16x16x64_bf16 v[34:37], v[26:29], v[2:9], v181
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[130:133], v[242:245], v[22:25], v[130:133]
    ds_read_b128 v[150:153], v141 offset:2048
    ; Form the next K address ahead of the independent MFMA.
    v_add_u32_e32 v143, s9, v139
    v_mfma_f32_16x16x32_bf16 v[126:129], v[242:245], v[18:21], v[126:129]
    v_mfma_f32_16x16x32_bf16 v[122:125], v[242:245], v[2:5], v[122:125]
    s_waitcnt lgkmcnt(1)
    buffer_load_dwordx4 v143, s[40:43], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[118:121], v[134:137], v[2:5], v[118:121]
    ds_read_b128 v[154:157], v141 offset:3072
    v_mfma_f32_16x16x32_bf16 v[114:117], v[134:137], v[18:21], v[114:117]
    v_mfma_f32_16x16x32_bf16 v[110:113], v[134:137], v[22:25], v[110:113]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[106:109], v[150:153], v[22:25], v[106:109]
    ds_read_b128 v[158:161], v141 offset:4096
    v_mfma_f32_16x16x32_bf16 v[102:105], v[150:153], v[18:21], v[102:105]
    v_mfma_f32_16x16x32_bf16 v[98:101], v[150:153], v[2:5], v[98:101]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[94:97], v[154:157], v[2:5], v[94:97]
    ds_read_b128 v[134:137], v141 offset:5120
    v_mfma_f32_16x16x32_bf16 v[90:93], v[154:157], v[18:21], v[90:93]
    v_mfma_f32_16x16x32_bf16 v[86:89], v[154:157], v[22:25], v[86:89]
    s_mov_b32 m0, s6
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[82:85], v[158:161], v[22:25], v[82:85]
    ds_read_b128 v[150:153], v141 offset:6144
    ; Form the next K address ahead of the independent MFMA.
    v_add_u32_e32 v143, s9, v169
    v_mfma_f32_16x16x32_bf16 v[78:81], v[158:161], v[18:21], v[78:81]
    v_mfma_f32_16x16x32_bf16 v[74:77], v[158:161], v[2:5], v[74:77]
    s_waitcnt lgkmcnt(1)
    buffer_load_dwordx4 v143, s[40:43], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[70:73], v[134:137], v[2:5], v[70:73]
    ds_read_b128 v[154:157], v141 offset:7168
    v_mfma_f32_16x16x32_bf16 v[66:69], v[134:137], v[18:21], v[66:69]
    v_mfma_f32_16x16x32_bf16 v[62:65], v[134:137], v[22:25], v[62:65]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[58:61], v[150:153], v[22:25], v[58:61]
    ds_read_b128 v[134:137], v141 offset:8192
    v_mfma_f32_16x16x32_bf16 v[54:57], v[150:153], v[18:21], v[54:57]
    v_mfma_f32_16x16x32_bf16 v[50:53], v[150:153], v[2:5], v[50:53]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[2:5], v[154:157], v[2:5], v[46:49]
    s_nop 2
    ds_read_b128 v[46:49], v141 offset:9216
    v_mfma_f32_16x16x32_bf16 v[18:21], v[154:157], v[18:21], v[42:45]
    v_mfma_f32_16x16x32_bf16 v[22:25], v[154:157], v[22:25], v[38:41]
    s_mov_b32 m0, s5
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[130:133], v[134:137], v[14:17], v[130:133]
    s_nop 0
    ds_read_b128 v[40:43], v141 offset:10240
    v_mfma_f32_16x16x32_bf16 v[126:129], v[134:137], v[10:13], v[126:129]
    v_mfma_f32_16x16x32_bf16 v[122:125], v[134:137], v[6:9], v[122:125]
    s_nop 0
    s_waitcnt lgkmcnt(1)
    ; Fold the packed V tile displacement into the scalar buffer offset.
    buffer_load_dwordx4 v188, s[44:47], s7 offen lds
    v_mfma_f32_16x16x32_bf16 v[118:121], v[46:49], v[6:9], v[118:121]
    ds_read_b128 v[134:137], v141 offset:11264
    v_mfma_f32_16x16x32_bf16 v[114:117], v[46:49], v[10:13], v[114:117]
    v_mfma_f32_16x16x32_bf16 v[110:113], v[46:49], v[14:17], v[110:113]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[106:109], v[40:43], v[14:17], v[106:109]
    ds_read_b128 v[44:47], v141 offset:12288
    v_mfma_f32_16x16x32_bf16 v[102:105], v[40:43], v[10:13], v[102:105]
    v_mfma_f32_16x16x32_bf16 v[98:101], v[40:43], v[6:9], v[98:101]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[94:97], v[134:137], v[6:9], v[94:97]
    ds_read_b128 v[38:41], v141 offset:13312
    v_mfma_f32_16x16x32_bf16 v[90:93], v[134:137], v[10:13], v[90:93]
    v_mfma_f32_16x16x32_bf16 v[86:89], v[134:137], v[14:17], v[86:89]
    s_mov_b32 m0, s4
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[82:85], v[44:47], v[14:17], v[82:85]
    ds_read_b128 v[134:137], v141 offset:14336
    v_mfma_f32_16x16x32_bf16 v[78:81], v[44:47], v[10:13], v[78:81]
    v_mfma_f32_16x16x32_bf16 v[74:77], v[44:47], v[6:9], v[74:77]
    s_nop 0
    s_waitcnt lgkmcnt(1)
    ; Fold the packed V tile displacement into the scalar buffer offset.
    buffer_load_dwordx4 v189, s[44:47], s7 offen lds
    v_mfma_f32_16x16x32_bf16 v[70:73], v[38:41], v[6:9], v[70:73]
    ds_read_b128 v[150:153], v141 offset:15360
    v_mfma_f32_16x16x32_bf16 v[66:69], v[38:41], v[10:13], v[66:69]
    v_mfma_f32_16x16x32_bf16 v[62:65], v[38:41], v[14:17], v[62:65]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[58:61], v[134:137], v[14:17], v[58:61]
    ds_read_b128 v[154:157], v144
    v_mfma_f32_16x16x32_bf16 v[54:57], v[134:137], v[10:13], v[54:57]
    v_mfma_f32_16x16x32_bf16 v[50:53], v[134:137], v[6:9], v[50:53]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[46:49], v[150:153], v[6:9], v[2:5]
    ds_read_b128 v[134:137], v144 offset:1088
    v_mfma_f32_16x16x32_bf16 v[42:45], v[150:153], v[10:13], v[18:21]
    v_mfma_f32_16x16x32_bf16 v[38:41], v[150:153], v[14:17], v[22:25]
    s_waitcnt lgkmcnt(1)
    s_setprio 1
    v_mfma_f32_16x16x32_bf16 v[2:5], v[154:157], v[226:229], 0
    ds_read_b128 v[158:161], v144 offset:4352
    v_mfma_f32_16x16x32_bf16 v[6:9], v[154:157], v[210:213], 0
    v_mfma_f32_16x16x32_bf16 v[10:13], v[154:157], v[194:197], 0
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[14:17], v[134:137], v[194:197], 0
    ds_read_b128 v[176:179], v144 offset:5440
    v_mfma_f32_16x16x32_bf16 v[18:21], v[134:137], v[210:213], 0
    v_mfma_f32_16x16x32_bf16 v[22:25], v[134:137], v[226:229], 0
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[134:137], v[158:161], v[226:229], 0
    ds_read_b128 v[184:187], v144 offset:256
    v_mfma_f32_16x16x32_bf16 v[150:153], v[158:161], v[210:213], 0
    v_mfma_f32_16x16x32_bf16 v[154:157], v[158:161], v[194:197], 0
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[158:161], v[176:179], v[194:197], 0
    ds_read_b128 v[242:245], v144 offset:1344
    v_mfma_f32_16x16x32_bf16 v[162:165], v[176:179], v[210:213], 0
    v_mfma_f32_16x16x32_bf16 v[176:179], v[176:179], v[226:229], 0
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[2:5], v[184:187], v[230:233], v[2:5]
    ds_read_b128 v[246:249], v144 offset:4608
    v_mfma_f32_16x16x32_bf16 v[6:9], v[184:187], v[214:217], v[6:9]
    v_mfma_f32_16x16x32_bf16 v[10:13], v[184:187], v[198:201], v[10:13]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[14:17], v[242:245], v[198:201], v[14:17]
    ds_read_b128 v[184:187], v144 offset:5696
    v_mfma_f32_16x16x32_bf16 v[18:21], v[242:245], v[214:217], v[18:21]
    v_mfma_f32_16x16x32_bf16 v[22:25], v[242:245], v[230:233], v[22:25]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[134:137], v[246:249], v[230:233], v[134:137]
    ds_read_b128 v[242:245], v144 offset:512
    v_mfma_f32_16x16x32_bf16 v[150:153], v[246:249], v[214:217], v[150:153]
    v_mfma_f32_16x16x32_bf16 v[154:157], v[246:249], v[198:201], v[154:157]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[158:161], v[184:187], v[198:201], v[158:161]
    ds_read_b128 v[246:249], v144 offset:1600
    v_mfma_f32_16x16x32_bf16 v[162:165], v[184:187], v[214:217], v[162:165]
    v_mfma_f32_16x16x32_bf16 v[176:179], v[184:187], v[230:233], v[176:179]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[2:5], v[242:245], v[234:237], v[2:5]
    ds_read_b128 v[250:253], v144 offset:4864
    v_mfma_f32_16x16x32_bf16 v[6:9], v[242:245], v[218:221], v[6:9]
    v_mfma_f32_16x16x32_bf16 v[184:187], v[242:245], v[202:205], v[10:13]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[14:17], v[246:249], v[202:205], v[14:17]
    s_nop 0
    ds_read_b128 v[10:13], v144 offset:5952
    v_mfma_f32_16x16x32_bf16 v[242:245], v[246:249], v[218:221], v[18:21]
    v_mfma_f32_16x16x32_bf16 v[22:25], v[246:249], v[234:237], v[22:25]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[134:137], v[250:253], v[234:237], v[134:137]
    ds_read_b128 v[18:21], v144 offset:768
    v_mfma_f32_16x16x32_bf16 v[150:153], v[250:253], v[218:221], v[150:153]
    v_mfma_f32_16x16x32_bf16 v[154:157], v[250:253], v[202:205], v[154:157]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[246:249], v[10:13], v[202:205], v[158:161]
    ds_read_b128 v[172:175], v144 offset:1856
    v_mfma_f32_16x16x32_bf16 v[250:253], v[10:13], v[218:221], v[162:165]
    v_mfma_f32_16x16x32_bf16 v[176:179], v[10:13], v[234:237], v[176:179]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[2:5], v[18:21], v[238:241], v[2:5]
    ds_read_b128 v[164:167], v144 offset:5120
    v_mfma_f32_16x16x32_bf16 v[10:13], v[18:21], v[222:225], v[6:9]
    v_mfma_f32_16x16x32_bf16 v[160:163], v[18:21], v[206:209], v[184:187]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[18:21], v[172:175], v[206:209], v[14:17]
    s_nop 0
    ds_read_b128 v[184:187], v144 offset:6208
    v_mfma_f32_16x16x32_bf16 v[14:17], v[172:175], v[222:225], v[242:245]
    v_mfma_f32_16x16x32_bf16 v[144:147], v[172:175], v[238:241], v[22:25]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[6:9], v[164:167], v[238:241], v[134:137]
    ds_read_b128 v[242:245], v148
    v_mfma_f32_16x16x32_bf16 v[22:25], v[164:167], v[222:225], v[150:153]
    v_mfma_f32_16x16x32_bf16 v[164:167], v[164:167], v[206:209], v[154:157]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[152:155], v[184:187], v[206:209], v[246:249]
    ds_read_b128 v[134:137], v148 offset:1024
    v_mfma_f32_16x16x32_bf16 v[148:151], v[184:187], v[222:225], v[250:253]
    v_mfma_f32_16x16x32_bf16 v[156:159], v[184:187], v[238:241], v[176:179]
    s_setprio 0
    s_cmp_lt_i32 s81, s8
    s_cbranch_scc0 .Lh3_wide_attention_block_73

.Lh3_wide_attention_block_9:
    s_cmp_lg_u32 s81, 0
    s_cselect_b64 s[4:5], -1, 0
    s_and_b64 vcc, exec, s[4:5]
    v_xor_b32_e32 v193, 0x80000000, v140
    v_xor_b32_e32 v141, 0x80000000, v138
    s_cbranch_vccnz .Lwide_drain
    s_mov_b64 s[6:7], 0
    s_mov_b64 s[2:3], -1
    s_branch .Lh3_wide_attention_block_11

; 384-query tile, drain the pipeline.
.Lwide_drain:
    s_mov_b64 s[6:7], -1
    s_mov_b64 s[2:3], 0

.Lh3_wide_attention_block_11:
    v_max3_f32 v143, v160, v161, v162
    v_max3_f32 v176, v163, v18, v19
    v_max3_f32 v177, v20, v21, v164
    v_max3_f32 v178, v165, v166, v167
    v_max3_f32 v179, v152, v153, v154
    v_max3_f32 v143, v143, v176, v177
    v_max3_f32 v176, v178, v179, s76
    v_max3_f32 v192, v143, v176, v155
    v_max3_f32 v143, v10, v11, v12
    v_max3_f32 v176, v13, v14, v15
    v_max3_f32 v177, v16, v17, v22
    v_max3_f32 v178, v23, v24, v25
    v_max3_f32 v179, v148, v149, v150
    v_max3_f32 v143, v143, v176, v177
    v_max3_f32 v176, v178, v179, s76
    v_max3_f32 v191, v143, v176, v151
    v_max3_f32 v143, v2, v3, v4
    v_max3_f32 v176, v5, v144, v145
    v_max3_f32 v177, v146, v147, v6
    v_max3_f32 v178, v7, v8, v9
    v_max3_f32 v179, v156, v157, v158
    v_max3_f32 v143, v143, v176, v177
    v_max3_f32 v176, v178, v179, s76
    s_andn2_b64 vcc, exec, s[6:7]
    v_max3_f32 v143, v143, v176, v159
    s_cbranch_vccnz .Lh3_wide_attention_block_12
    v_cmp_gt_f32_e64 s[2:3], v192, -v142
    v_cmp_gt_f32_e64 s[6:7], v191, -v140
    s_or_b64 s[2:3], s[2:3], s[6:7]
    v_cmp_gt_f32_e64 s[6:7], v143, -v138
    s_or_b64 s[2:3], s[2:3], s[6:7]
    s_and_b64 vcc, exec, s[2:3]
    s_cselect_b64 s[2:3], -1, 0

.Lh3_wide_attention_block_12:
    s_andn2_b64 vcc, exec, s[2:3]
    s_mov_b64 s[2:3], 0
    s_cbranch_vccnz .Lwide_stage_b
    v_mov_b32_e32 v176, v192
    v_mov_b32_e32 v177, v191
    v_mov_b32_e32 v178, v143
    v_permlane32_swap_b32_e64 v192, v176 bound_ctrl:1
    v_permlane32_swap_b32_e64 v191, v177 bound_ctrl:1
    v_permlane32_swap_b32_e64 v143, v178 bound_ctrl:1
    s_mov_b64 s[2:3], s[4:5]
    v_max_f32_e32 v176, v192, v176
    v_max_f32_e32 v177, v191, v177
    v_max_f32_e32 v143, v143, v178
    v_mov_b32_e32 v178, v176
    v_mov_b32_e32 v179, v177
    v_mov_b32_e32 v184, v143
    v_permlane16_swap_b32_e64 v176, v178 bound_ctrl:1
    v_permlane16_swap_b32_e64 v177, v179 bound_ctrl:1
    v_permlane16_swap_b32_e64 v143, v184 bound_ctrl:1
    s_nop 0
    v_max_f32_e32 v176, v176, v178
    v_max_f32_e64 v178, -v142, v176
    v_cndmask_b32_e64 v176, v176, v178, s[4:5]
    v_max_f32_e32 v177, v177, v179
    v_max_f32_e32 v143, v143, v184
    v_add_f32_e32 v142, v176, v142
    v_mul_f32_e32 v192, 0x3e0293ee, v142
    v_xor_b32_e32 v142, 0x80000000, v176
    v_max_f32_e32 v176, v193, v177
    v_max_f32_e32 v141, v141, v143
    v_cndmask_b32_e64 v176, v177, v176, s[4:5]
    v_cndmask_b32_e64 v141, v143, v141, s[4:5]
    v_add_f32_e32 v140, v176, v140
    v_add_f32_e32 v138, v141, v138
    v_mul_f32_e32 v191, 0x3e0293ee, v140
    v_xor_b32_e32 v140, 0x80000000, v176
    v_mul_f32_e32 v143, 0x3e0293ee, v138
    v_xor_b32_e32 v138, 0x80000000, v141
    s_branch .Lwide_stage_b

; 256-query tile, first pipeline region.
.Lnarrow_stage_a:
    v_mul_f32_e32 v91, 0x3e0293ee, v92
    v_fmamk_f32 v102, v102, 0x3e0293ee, v91
    v_fmamk_f32 v103, v103, 0x3e0293ee, v91
    v_fmamk_f32 v104, v104, 0x3e0293ee, v91
    v_mul_f32_e32 v111, 0x3e0293ee, v90
    v_exp_f32_e32 v102, v102
    s_mov_b64 s[4:5], src_shared_base
    v_fmamk_f32 v105, v105, 0x3e0293ee, v91
    v_fmamk_f32 v2, v2, 0x3e0293ee, v111
    v_fmamk_f32 v3, v3, 0x3e0293ee, v111
    v_fmamk_f32 v4, v4, 0x3e0293ee, v111
    v_fmamk_f32 v5, v5, 0x3e0293ee, v111
    v_fmamk_f32 v10, v10, 0x3e0293ee, v91
    v_fmamk_f32 v11, v11, 0x3e0293ee, v91
    v_fmamk_f32 v12, v12, 0x3e0293ee, v91
    v_fmamk_f32 v13, v13, 0x3e0293ee, v91
    v_fmamk_f32 v14, v14, 0x3e0293ee, v111
    v_fmamk_f32 v15, v15, 0x3e0293ee, v111
    v_fmamk_f32 v16, v16, 0x3e0293ee, v111
    v_fmamk_f32 v17, v17, 0x3e0293ee, v111
    v_fmamk_f32 v106, v106, 0x3e0293ee, v91
    v_fmamk_f32 v107, v107, 0x3e0293ee, v91
    v_fmamk_f32 v108, v108, 0x3e0293ee, v91
    v_fmamk_f32 v109, v109, 0x3e0293ee, v91
    v_fmamk_f32 v6, v6, 0x3e0293ee, v111
    v_fmamk_f32 v7, v7, 0x3e0293ee, v111
    v_fmamk_f32 v8, v8, 0x3e0293ee, v111
    v_fmamk_f32 v9, v9, 0x3e0293ee, v111
    v_fmamk_f32 v94, v94, 0x3e0293ee, v91
    v_fmamk_f32 v95, v95, 0x3e0293ee, v91
    v_fmamk_f32 v96, v96, 0x3e0293ee, v91
    v_fmac_f32_e32 v91, 0x3e0293ee, v97
    v_fmamk_f32 v97, v98, 0x3e0293ee, v111
    v_fmamk_f32 v98, v99, 0x3e0293ee, v111
    v_fmamk_f32 v99, v100, 0x3e0293ee, v111
    v_fmac_f32_e32 v111, 0x3e0293ee, v101
    v_exp_f32_e32 v100, v103
    s_and_b64 s[2:3], s[2:3], s[6:7]
    v_exp_f32_e32 v101, v104
    s_add_i32 s92, s92, 1
    v_exp_f32_e32 v103, v105
    s_and_b32 s12, s91, 3
    v_exp_f32_e32 v2, v2
    s_lshr_b32 s13, s91, 8
    v_exp_f32_e32 v3, v3
    s_bfe_u32 s4, s91, 0x20008
    v_exp_f32_e32 v4, v4
    s_lshr_b32 s14, s91, 16
    v_exp_f32_e32 v5, v5
    s_add_i32 s6, s92, s86
    v_exp_f32_e32 v104, v10
    s_mul_i32 s9, s6, 0xa8000
    v_exp_f32_e32 v105, v11
    s_add_i32 s6, s92, s87
    v_exp_f32_e32 v112, v12
    s_lshl_b32 s7, s6, 14
    v_exp_f32_e32 v13, v13
    s_mulk_i32 s4, 0x4400
    v_exp_f32_e32 v113, v14
    s_add_u32 s10, s4, s70
    v_exp_f32_e32 v121, v15
    s_addc_u32 s11, s5, s71
    v_exp_f32_e32 v138, v16
    s_mul_i32 s5, s12, 0x4800
    v_exp_f32_e32 v17, v17
    s_add_i32 s15, s63, s5
    v_exp_f32_e32 v106, v106
    s_add_i32 s6, s15, 0xcc00
    v_exp_f32_e32 v107, v107
    s_add_i32 s4, s59, s4
    v_exp_f32_e32 v108, v108
    s_add_i32 s5, s4, 0x2200
    v_exp_f32_e32 v109, v109
    s_add_i32 s4, s15, 0xd000
    v_exp_f32_e32 v6, v6
    s_cmp_lg_u64 s[10:11], 0
    v_exp_f32_e32 v7, v7
    s_cselect_b32 s10, s10, -1
    v_exp_f32_e32 v8, v8
    s_mov_b32 m0, s10
    v_exp_f32_e32 v9, v9
    s_lshl_b32 s10, s91, 16
    v_exp_f32_e32 v94, v94
    s_and_b32 s10, s10, 0xff0000
    v_exp_f32_e32 v95, v95
    s_or_b32 s91, s10, s13
    v_exp_f32_e32 v96, v96
    v_exp_f32_e32 v91, v91
    v_exp_f32_e32 v97, v97
    v_exp_f32_e32 v98, v98
    v_exp_f32_e32 v99, v99
    v_exp_f32_e32 v111, v111
    s_mulk_i32 s14, 0x4800
    s_mulk_i32 s12, 0x4400
    v_cvt_pk_bf16_f32 v15, v4, v5
    v_cvt_pk_bf16_f32 v5, v96, v91
    v_cvt_pk_bf16_f32 v6, v6, v7
    v_cvt_pk_bf16_f32 v7, v8, v9
    v_cvt_pk_bf16_f32 v8, v97, v98
    v_add_u32_e32 v91, s14, v120
    v_add_u32_e32 v98, s12, v183
    v_cvt_pk_bf16_f32 v10, v102, v100
    v_cvt_pk_bf16_f32 v11, v101, v103
    v_cvt_pk_bf16_f32 v12, v104, v105
    v_cvt_pk_bf16_f32 v13, v112, v13
    v_cvt_pk_bf16_f32 v14, v2, v3
    v_cvt_pk_bf16_f32 v16, v113, v121
    v_cvt_pk_bf16_f32 v17, v138, v17
    v_cvt_pk_bf16_f32 v2, v106, v107
    v_cvt_pk_bf16_f32 v3, v108, v109
    v_cvt_pk_bf16_f32 v4, v94, v95
    v_cvt_pk_bf16_f32 v9, v99, v111
    ds_read_b128 v[104:107], v91
    ds_read_b128 v[94:97], v91 offset:1024
    ds_read_b128 v[100:103], v91 offset:2048
    v_smfmac_f32_16x16x64_bf16 v[18:21], v[26:29], v[10:17], v181
    s_sub_i32 s90, s90, 64
    s_add_i32 s89, s89, 0xa8000
    s_cmp_eq_u32 s88, s92
    v_smfmac_f32_16x16x64_bf16 v[18:21], v[26:29], v[2:9], v181
    s_cbranch_scc1 .Lh3_wide_attention_block_35

.Lh3_wide_attention_block_14:
    s_andn2_b64 vcc, exec, s[2:3]
    s_cbranch_vccz .Lh3_wide_attention_block_32

.Lh3_wide_attention_block_15:
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[86:89], v[104:107], v[14:17], v[86:89]
    ds_read_b128 v[108:111], v91 offset:3072
    ; Form the next K address ahead of the independent MFMA.
    v_add_u32_e32 v93, s9, v139
    v_mfma_f32_16x16x32_bf16 v[58:61], v[104:107], v[10:13], v[58:61]
    s_waitcnt lgkmcnt(2)
    buffer_load_dwordx4 v93, s[40:43], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[82:85], v[94:97], v[10:13], v[82:85]
    ds_read_b128 v[104:107], v91 offset:4096
    v_mfma_f32_16x16x32_bf16 v[78:81], v[94:97], v[14:17], v[78:81]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[74:77], v[100:103], v[14:17], v[74:77]
    ds_read_b128 v[94:97], v91 offset:5120
    v_mfma_f32_16x16x32_bf16 v[70:73], v[100:103], v[10:13], v[70:73]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[66:69], v[108:111], v[10:13], v[66:69]
    ds_read_b128 v[100:103], v91 offset:6144
    v_mfma_f32_16x16x32_bf16 v[62:65], v[108:111], v[14:17], v[62:65]
    s_mov_b32 m0, s5
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[54:57], v[104:107], v[14:17], v[54:57]
    ds_read_b128 v[108:111], v91 offset:7168
    ; Form the next K address ahead of the independent MFMA.
    v_add_u32_e32 v93, s9, v169
    v_mfma_f32_16x16x32_bf16 v[50:53], v[104:107], v[10:13], v[50:53]
    s_waitcnt lgkmcnt(2)
    buffer_load_dwordx4 v93, s[40:43], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[46:49], v[94:97], v[10:13], v[46:49]
    ds_read_b128 v[104:107], v91 offset:8192
    v_mfma_f32_16x16x32_bf16 v[42:45], v[94:97], v[14:17], v[42:45]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[38:41], v[100:103], v[14:17], v[38:41]
    ds_read_b128 v[94:97], v91 offset:9216
    v_mfma_f32_16x16x32_bf16 v[34:37], v[100:103], v[10:13], v[34:37]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[10:13], v[108:111], v[10:13], v[30:33]
    ds_read_b128 v[100:103], v91 offset:10240
    v_mfma_f32_16x16x32_bf16 v[14:17], v[108:111], v[14:17], v[22:25]
    s_mov_b32 m0, s6
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[86:89], v[104:107], v[6:9], v[86:89]
    ds_read_b128 v[108:111], v91 offset:11264
    v_mfma_f32_16x16x32_bf16 v[58:61], v[104:107], v[2:5], v[58:61]
    v_or_b32_e32 v30, s7, v118
    s_waitcnt lgkmcnt(2)
    buffer_load_dwordx4 v30, s[44:47], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[82:85], v[94:97], v[2:5], v[82:85]
    ds_read_b128 v[104:107], v91 offset:12288
    v_mfma_f32_16x16x32_bf16 v[78:81], v[94:97], v[6:9], v[78:81]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[74:77], v[100:103], v[6:9], v[74:77]
    ds_read_b128 v[22:25], v91 offset:13312
    v_mfma_f32_16x16x32_bf16 v[70:73], v[100:103], v[2:5], v[70:73]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[66:69], v[108:111], v[2:5], v[66:69]
    ds_read_b128 v[94:97], v91 offset:14336
    v_mfma_f32_16x16x32_bf16 v[62:65], v[108:111], v[6:9], v[62:65]
    s_mov_b32 m0, s4
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[54:57], v[104:107], v[6:9], v[54:57]
    ds_read_b128 v[100:103], v91 offset:15360
    v_mfma_f32_16x16x32_bf16 v[50:53], v[104:107], v[2:5], v[50:53]
    v_or_b32_e32 v30, 0x400, v30
    s_waitcnt lgkmcnt(2)
    buffer_load_dwordx4 v30, s[44:47], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[46:49], v[22:25], v[2:5], v[46:49]
    ds_read_b128 v[104:107], v98
    v_mfma_f32_16x16x32_bf16 v[42:45], v[22:25], v[6:9], v[42:45]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[38:41], v[94:97], v[6:9], v[38:41]
    ds_read_b128 v[108:111], v98 offset:1088
    v_mfma_f32_16x16x32_bf16 v[34:37], v[94:97], v[2:5], v[34:37]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[30:33], v[100:103], v[2:5], v[10:13]
    ds_read_b128 v[156:159], v98 offset:4352
    v_mfma_f32_16x16x32_bf16 v[22:25], v[100:103], v[6:9], v[14:17]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[2:5], v[104:107], v[140:143], 0
    ds_read_b128 v[160:163], v98 offset:5440
    v_mfma_f32_16x16x32_bf16 v[6:9], v[104:107], v[122:125], 0
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[10:13], v[108:111], v[122:125], 0
    ds_read_b128 v[164:167], v98 offset:256
    v_mfma_f32_16x16x32_bf16 v[14:17], v[108:111], v[140:143], 0
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[94:97], v[156:159], v[140:143], 0
    ds_read_b128 v[172:175], v98 offset:1344
    v_mfma_f32_16x16x32_bf16 v[100:103], v[156:159], v[122:125], 0
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[104:107], v[160:163], v[122:125], 0
    ds_read_b128 v[156:159], v98 offset:4608
    v_mfma_f32_16x16x32_bf16 v[108:111], v[160:163], v[140:143], 0
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[2:5], v[164:167], v[144:147], v[2:5]
    ds_read_b128 v[160:163], v98 offset:5696
    v_mfma_f32_16x16x32_bf16 v[6:9], v[164:167], v[126:129], v[6:9]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[10:13], v[172:175], v[126:129], v[10:13]
    ds_read_b128 v[164:167], v98 offset:512
    v_mfma_f32_16x16x32_bf16 v[14:17], v[172:175], v[144:147], v[14:17]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[94:97], v[156:159], v[144:147], v[94:97]
    ds_read_b128 v[172:175], v98 offset:1600
    v_mfma_f32_16x16x32_bf16 v[100:103], v[156:159], v[126:129], v[100:103]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[104:107], v[160:163], v[126:129], v[104:107]
    ds_read_b128 v[156:159], v98 offset:4864
    v_mfma_f32_16x16x32_bf16 v[108:111], v[160:163], v[144:147], v[108:111]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[2:5], v[164:167], v[148:151], v[2:5]
    ds_read_b128 v[176:179], v98 offset:5952
    v_mfma_f32_16x16x32_bf16 v[6:9], v[164:167], v[130:133], v[6:9]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[10:13], v[172:175], v[130:133], v[10:13]
    ds_read_b128 v[164:167], v98 offset:768
    v_mfma_f32_16x16x32_bf16 v[14:17], v[172:175], v[148:151], v[14:17]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[94:97], v[156:159], v[148:151], v[94:97]
    ds_read_b128 v[172:175], v98 offset:1856
    v_mfma_f32_16x16x32_bf16 v[156:159], v[156:159], v[130:133], v[100:103]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[160:163], v[176:179], v[130:133], v[104:107]
    ds_read_b128 v[184:187], v98 offset:5120
    v_mfma_f32_16x16x32_bf16 v[110:113], v[176:179], v[148:151], v[108:111]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[2:5], v[164:167], v[152:155], v[2:5]
    ds_read_b128 v[98:101], v98 offset:6208
    v_mfma_f32_16x16x32_bf16 v[102:105], v[164:167], v[134:137], v[6:9]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[10:13], v[172:175], v[134:137], v[10:13]
    v_mfma_f32_16x16x32_bf16 v[14:17], v[172:175], v[152:155], v[14:17]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[6:9], v[184:187], v[152:155], v[94:97]
    v_mfma_f32_16x16x32_bf16 v[106:109], v[184:187], v[134:137], v[156:159]
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_16x16x32_bf16 v[94:97], v[98:101], v[134:137], v[160:163]
    v_mfma_f32_16x16x32_bf16 v[98:101], v[98:101], v[152:155], v[110:113]
    s_waitcnt vmcnt(0)
    s_barrier
    s_cmp_lt_i32 s92, s8
    s_cbranch_scc0 .Lh3_wide_attention_block_33

.Lh3_wide_attention_block_16:
    v_max3_f32 v91, v102, v103, v104
    v_max3_f32 v93, v105, v10, v11
    v_max3_f32 v110, v12, v13, v106
    v_max3_f32 v111, v107, v108, v109
    v_max3_f32 v112, v94, v95, v96
    v_max3_f32 v91, v91, v93, v110
    v_max3_f32 v93, v111, v112, s76
    v_max3_f32 v110, v91, v93, v97
    v_max3_f32 v91, v2, v3, v4
    v_max3_f32 v93, v5, v14, v15
    v_max3_f32 v111, v16, v17, v6
    v_max3_f32 v112, v7, v8, v9
    v_max3_f32 v113, v98, v99, v100
    v_max3_f32 v91, v91, v93, v111
    s_cmp_eq_u32 s92, 0
    v_max3_f32 v93, v112, v113, s76
    s_cselect_b64 s[4:5], -1, 0
    v_max3_f32 v93, v91, v93, v101
    s_mov_b64 s[6:7], -1
    s_and_b64 vcc, exec, s[4:5]
    s_cbranch_vccnz .Lh3_wide_attention_block_17
    v_cmp_gt_f32_e64 s[2:3], v110, -v92
    v_cmp_gt_f32_e64 s[6:7], v93, -v90
    s_or_b64 s[2:3], s[2:3], s[6:7]
    s_and_b64 vcc, exec, s[2:3]
    s_cselect_b64 s[6:7], -1, 0

.Lh3_wide_attention_block_17:
    s_cmp_lg_u32 s92, 0
    s_cselect_b64 s[2:3], -1, 0
    s_andn2_b64 vcc, exec, s[6:7]
    s_cbranch_vccnz .Lnarrow_stage_a
    v_mov_b32_e32 v91, v110
    v_mov_b32_e32 v111, v110
    v_mov_b32_e32 v112, v93
    v_mov_b32_e32 v113, v91
    v_permlane32_swap_b32_e64 v110, v111 bound_ctrl:1
    v_permlane32_swap_b32_e64 v93, v112 bound_ctrl:1
    v_permlane32_swap_b32_e64 v91, v113 bound_ctrl:1
    v_mov_b64_e32 v[158:159], v[20:21]
    v_max_f32_e32 v91, v110, v111
    v_max_f32_e32 v93, v93, v112
    v_mov_b32_e32 v110, v91
    v_mov_b32_e32 v111, v110
    v_mov_b32_e32 v112, v93
    v_mov_b32_e32 v113, v91
    v_permlane16_swap_b32_e64 v110, v111 bound_ctrl:1
    v_permlane16_swap_b32_e64 v93, v112 bound_ctrl:1
    v_permlane16_swap_b32_e64 v91, v113 bound_ctrl:1
    v_cndmask_b32_e64 v121, v21, v159, s[2:3]
    v_max_f32_e32 v91, v110, v111
    v_max_f32_e64 v110, -v92, v91
    v_cndmask_b32_e64 v91, v110, v91, s[4:5]
    v_add_f32_e32 v92, v91, v92
    v_mul_f32_e32 v110, 0x3e0293ee, v92
    v_exp_f32_e64 v111, -v110
    v_max_f32_e32 v93, v93, v112
    v_xor_b32_e32 v92, 0x80000000, v91
    v_cndmask_b32_e64 v138, v20, v158, s[2:3]
    v_mul_f32_e32 v112, v111, v18
    v_mul_f32_e32 v91, v111, v19
    v_max_f32_e64 v111, -v90, v93
    v_cndmask_b32_e64 v111, v111, v93, s[4:5]
    v_add_f32_e32 v90, v111, v90
    v_mul_f32_e32 v93, 0x3e0293ee, v90
    v_exp_f32_e64 v113, -v93
    v_xor_b32_e32 v90, 0x80000000, v111
    v_mov_b64_e32 v[156:157], v[18:19]
    v_cndmask_b32_e64 v19, v19, v91, s[2:3]
    v_mul_f32_e32 v111, v138, v113
    v_mul_f32_e32 v113, v121, v113
    v_cndmask_b32_e64 v18, v18, v112, s[2:3]
    v_cndmask_b32_e64 v21, v21, v113, s[2:3]
    v_cndmask_b32_e64 v20, v20, v111, s[2:3]
    s_branch .Lnarrow_stage_a

; 256-query tile, second pipeline region.
.Lnarrow_stage_b:
    v_mul_f32_e32 v95, 0x3e0293ee, v96
    v_fmamk_f32 v98, v98, 0x3e0293ee, v95
    v_fmamk_f32 v99, v99, 0x3e0293ee, v95
    v_mul_f32_e32 v138, 0x3e0293ee, v94
    v_fmamk_f32 v100, v100, 0x3e0293ee, v95
    v_fmamk_f32 v101, v101, 0x3e0293ee, v95
    v_fmamk_f32 v2, v2, 0x3e0293ee, v138
    v_fmamk_f32 v3, v3, 0x3e0293ee, v138
    v_fmamk_f32 v4, v4, 0x3e0293ee, v138
    v_fmamk_f32 v6, v6, 0x3e0293ee, v95
    v_fmamk_f32 v7, v7, 0x3e0293ee, v95
    v_fmamk_f32 v8, v8, 0x3e0293ee, v95
    v_fmamk_f32 v9, v9, 0x3e0293ee, v95
    v_fmamk_f32 v110, v110, 0x3e0293ee, v95
    v_fmamk_f32 v111, v111, 0x3e0293ee, v95
    v_fmamk_f32 v112, v112, 0x3e0293ee, v95
    v_fmamk_f32 v113, v113, 0x3e0293ee, v95
    v_fmamk_f32 v102, v102, 0x3e0293ee, v95
    v_fmamk_f32 v103, v103, 0x3e0293ee, v95
    v_fmamk_f32 v104, v104, 0x3e0293ee, v95
    v_fmac_f32_e32 v95, 0x3e0293ee, v105
    v_fmamk_f32 v105, v106, 0x3e0293ee, v138
    v_fmamk_f32 v106, v107, 0x3e0293ee, v138
    v_fmamk_f32 v107, v108, 0x3e0293ee, v138
    v_exp_f32_e32 v108, v98
    s_mov_b64 s[4:5], src_shared_base
    v_exp_f32_e32 v99, v99
    s_add_i32 s92, s92, 1
    v_fmamk_f32 v5, v5, 0x3e0293ee, v138
    v_fmamk_f32 v14, v14, 0x3e0293ee, v138
    v_fmamk_f32 v15, v15, 0x3e0293ee, v138
    v_fmamk_f32 v16, v16, 0x3e0293ee, v138
    v_fmamk_f32 v17, v17, 0x3e0293ee, v138
    v_fmamk_f32 v10, v10, 0x3e0293ee, v138
    v_fmamk_f32 v11, v11, 0x3e0293ee, v138
    v_fmamk_f32 v12, v12, 0x3e0293ee, v138
    v_fmamk_f32 v13, v13, 0x3e0293ee, v138
    v_fmac_f32_e32 v138, 0x3e0293ee, v109
    v_exp_f32_e32 v100, v100
    s_and_b32 s12, s91, 3
    v_exp_f32_e32 v101, v101
    s_lshr_b32 s13, s91, 8
    v_exp_f32_e32 v109, v2
    s_bfe_u32 s4, s91, 0x20008
    v_exp_f32_e32 v164, v3
    s_lshr_b32 s14, s91, 16
    v_exp_f32_e32 v165, v4
    s_add_i32 s6, s92, s86
    v_exp_f32_e32 v166, v5
    s_mul_i32 s9, s6, 0xa8000
    v_exp_f32_e32 v167, v6
    s_add_i32 s6, s92, s87
    v_exp_f32_e32 v172, v7
    s_lshl_b32 s7, s6, 14
    v_exp_f32_e32 v173, v8
    s_mul_i32 s6, s14, 0x4400
    v_exp_f32_e32 v176, v9
    s_add_u32 s10, s6, s70
    v_exp_f32_e32 v177, v14
    s_addc_u32 s11, s5, s71
    v_exp_f32_e32 v178, v15
    s_mulk_i32 s4, 0x4800
    v_exp_f32_e32 v179, v16
    s_add_i32 s4, s63, s4
    v_exp_f32_e32 v17, v17
    s_add_i32 s5, s4, 0xcc00
    v_exp_f32_e32 v2, v110
    s_add_i32 s6, s59, s6
    v_exp_f32_e32 v3, v111
    s_addk_i32 s6, 0x2200
    v_exp_f32_e32 v4, v112
    s_add_i32 s4, s4, 0xd000
    v_exp_f32_e32 v5, v113
    s_cmp_lg_u64 s[10:11], 0
    v_exp_f32_e32 v6, v10
    s_cselect_b32 s10, s10, -1
    v_exp_f32_e32 v7, v11
    s_mov_b32 m0, s10
    v_exp_f32_e32 v8, v12
    s_lshl_b32 s10, s91, 16
    v_exp_f32_e32 v9, v13
    s_and_b32 s10, s10, 0xff0000
    v_exp_f32_e32 v10, v102
    s_or_b32 s91, s10, s13
    v_exp_f32_e32 v11, v103
    v_exp_f32_e32 v12, v104
    v_exp_f32_e32 v13, v95
    v_exp_f32_e32 v14, v105
    v_exp_f32_e32 v15, v106
    v_exp_f32_e32 v16, v107
    v_exp_f32_e32 v102, v138
    s_mulk_i32 s14, 0x4800
    s_mul_i32 s10, s12, 0x4400
    s_mulk_i32 s12, 0x4800
    s_sub_i32 s90, s90, 64
    s_add_i32 s89, s89, 0xa8000
    v_add_u32_e32 v95, s14, v120
    v_add_u32_e32 v98, s10, v183
    v_add_u32_e32 v106, s12, v120
    v_cvt_pk_bf16_f32 v6, v6, v7
    v_cvt_pk_bf16_f32 v7, v8, v9
    v_cvt_pk_bf16_f32 v8, v14, v15
    v_cvt_pk_bf16_f32 v9, v16, v102
    v_cvt_pk_bf16_f32 v2, v2, v3
    v_cvt_pk_bf16_f32 v3, v4, v5
    v_cvt_pk_bf16_f32 v4, v10, v11
    v_cvt_pk_bf16_f32 v5, v12, v13
    v_cvt_pk_bf16_f32 v14, v109, v164
    v_cvt_pk_bf16_f32 v15, v165, v166
    v_cvt_pk_bf16_f32 v16, v177, v178
    v_cvt_pk_bf16_f32 v17, v179, v17
    v_cvt_pk_bf16_f32 v10, v108, v99
    v_cvt_pk_bf16_f32 v11, v100, v101
    v_cvt_pk_bf16_f32 v12, v167, v172
    s_cmp_eq_u32 s88, s92
    v_cvt_pk_bf16_f32 v13, v173, v176
    s_cbranch_scc1 .Lh3_wide_attention_block_46

.Lh3_wide_attention_block_19:
    s_andn2_b64 vcc, exec, s[2:3]
    s_cbranch_vccz .Lh3_wide_attention_block_40

.Lh3_wide_attention_block_20:
    s_waitcnt vmcnt(0)
    v_smfmac_f32_16x16x64_bf16 v[18:21], v[26:29], v[10:17], v181
    s_barrier
    v_smfmac_f32_16x16x64_bf16 v[18:21], v[26:29], v[2:9], v181
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[86:89], v[160:163], v[14:17], v[86:89]
    ds_read_b128 v[100:103], v95 offset:3072
    ; Form the next K address ahead of the independent MFMA.
    v_add_u32_e32 v97, s9, v139
    v_mfma_f32_16x16x32_bf16 v[82:85], v[160:163], v[10:13], v[82:85]
    s_waitcnt lgkmcnt(2)
    buffer_load_dwordx4 v97, s[40:43], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[78:81], v[90:93], v[10:13], v[78:81]
    ds_read_b128 v[108:111], v95 offset:4096
    v_mfma_f32_16x16x32_bf16 v[74:77], v[90:93], v[14:17], v[74:77]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[70:73], v[156:159], v[14:17], v[70:73]
    ds_read_b128 v[90:93], v95 offset:5120
    v_mfma_f32_16x16x32_bf16 v[66:69], v[156:159], v[10:13], v[66:69]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[62:65], v[100:103], v[10:13], v[62:65]
    ds_read_b128 v[156:159], v95 offset:6144
    v_mfma_f32_16x16x32_bf16 v[58:61], v[100:103], v[14:17], v[58:61]
    s_mov_b32 m0, s6
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[54:57], v[108:111], v[14:17], v[54:57]
    ds_read_b128 v[100:103], v95 offset:7168
    ; Form the next K address ahead of the independent MFMA.
    v_add_u32_e32 v97, s9, v169
    v_mfma_f32_16x16x32_bf16 v[50:53], v[108:111], v[10:13], v[50:53]
    s_waitcnt lgkmcnt(2)
    buffer_load_dwordx4 v97, s[40:43], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[46:49], v[90:93], v[10:13], v[46:49]
    ds_read_b128 v[108:111], v95 offset:8192
    v_mfma_f32_16x16x32_bf16 v[42:45], v[90:93], v[14:17], v[42:45]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[38:41], v[156:159], v[14:17], v[38:41]
    ds_read_b128 v[90:93], v95 offset:9216
    v_mfma_f32_16x16x32_bf16 v[34:37], v[156:159], v[10:13], v[34:37]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[10:13], v[100:103], v[10:13], v[30:33]
    s_nop 2
    ds_read_b128 v[30:33], v95 offset:10240
    v_mfma_f32_16x16x32_bf16 v[14:17], v[100:103], v[14:17], v[22:25]
    s_mov_b32 m0, s5
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[86:89], v[108:111], v[6:9], v[86:89]
    ds_read_b128 v[100:103], v95 offset:11264
    v_mfma_f32_16x16x32_bf16 v[82:85], v[108:111], v[2:5], v[82:85]
    s_nop 0
    s_waitcnt lgkmcnt(2)
    ; Fold the packed V tile displacement into the scalar buffer offset.
    buffer_load_dwordx4 v118, s[44:47], s7 offen lds
    v_mfma_f32_16x16x32_bf16 v[78:81], v[90:93], v[2:5], v[78:81]
    ds_read_b128 v[108:111], v95 offset:12288
    v_mfma_f32_16x16x32_bf16 v[74:77], v[90:93], v[6:9], v[74:77]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[70:73], v[30:33], v[6:9], v[70:73]
    ds_read_b128 v[22:25], v95 offset:13312
    v_mfma_f32_16x16x32_bf16 v[66:69], v[30:33], v[2:5], v[66:69]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[62:65], v[100:103], v[2:5], v[62:65]
    ds_read_b128 v[90:93], v95 offset:14336
    v_mfma_f32_16x16x32_bf16 v[58:61], v[100:103], v[6:9], v[58:61]
    s_mov_b32 m0, s4
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[54:57], v[108:111], v[6:9], v[54:57]
    ds_read_b128 v[100:103], v95 offset:15360
    v_mfma_f32_16x16x32_bf16 v[50:53], v[108:111], v[2:5], v[50:53]
    s_nop 0
    s_waitcnt lgkmcnt(2)
    ; Fold the packed V tile displacement into the scalar buffer offset.
    buffer_load_dwordx4 v119, s[44:47], s7 offen lds
    v_mfma_f32_16x16x32_bf16 v[46:49], v[22:25], v[2:5], v[46:49]
    ds_read_b128 v[108:111], v98
    v_mfma_f32_16x16x32_bf16 v[42:45], v[22:25], v[6:9], v[42:45]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[38:41], v[90:93], v[6:9], v[38:41]
    ds_read_b128 v[156:159], v98 offset:1088
    v_mfma_f32_16x16x32_bf16 v[34:37], v[90:93], v[2:5], v[34:37]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[30:33], v[100:103], v[2:5], v[10:13]
    ds_read_b128 v[160:163], v98 offset:4352
    v_mfma_f32_16x16x32_bf16 v[22:25], v[100:103], v[6:9], v[14:17]
    s_waitcnt lgkmcnt(2)
    s_setprio 1
    v_mfma_f32_16x16x32_bf16 v[2:5], v[108:111], v[140:143], 0
    ds_read_b128 v[164:167], v98 offset:5440
    v_mfma_f32_16x16x32_bf16 v[6:9], v[108:111], v[122:125], 0
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[10:13], v[156:159], v[122:125], 0
    ds_read_b128 v[172:175], v98 offset:256
    v_mfma_f32_16x16x32_bf16 v[14:17], v[156:159], v[140:143], 0
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[90:93], v[160:163], v[140:143], 0
    ds_read_b128 v[176:179], v98 offset:1344
    v_mfma_f32_16x16x32_bf16 v[100:103], v[160:163], v[122:125], 0
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[108:111], v[164:167], v[122:125], 0
    ds_read_b128 v[160:163], v98 offset:4608
    v_mfma_f32_16x16x32_bf16 v[156:159], v[164:167], v[140:143], 0
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[2:5], v[172:175], v[144:147], v[2:5]
    ds_read_b128 v[164:167], v98 offset:5696
    v_mfma_f32_16x16x32_bf16 v[6:9], v[172:175], v[126:129], v[6:9]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[10:13], v[176:179], v[126:129], v[10:13]
    ds_read_b128 v[172:175], v98 offset:512
    v_mfma_f32_16x16x32_bf16 v[14:17], v[176:179], v[144:147], v[14:17]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[90:93], v[160:163], v[144:147], v[90:93]
    ds_read_b128 v[176:179], v98 offset:1600
    v_mfma_f32_16x16x32_bf16 v[100:103], v[160:163], v[126:129], v[100:103]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[108:111], v[164:167], v[126:129], v[108:111]
    ds_read_b128 v[160:163], v98 offset:4864
    v_mfma_f32_16x16x32_bf16 v[156:159], v[164:167], v[144:147], v[156:159]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[2:5], v[172:175], v[148:151], v[2:5]
    ds_read_b128 v[184:187], v98 offset:5952
    v_mfma_f32_16x16x32_bf16 v[6:9], v[172:175], v[130:133], v[6:9]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[10:13], v[176:179], v[130:133], v[10:13]
    ds_read_b128 v[172:175], v98 offset:768
    v_mfma_f32_16x16x32_bf16 v[14:17], v[176:179], v[148:151], v[14:17]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[90:93], v[160:163], v[148:151], v[90:93]
    ds_read_b128 v[188:191], v98 offset:1856
    v_mfma_f32_16x16x32_bf16 v[102:105], v[160:163], v[130:133], v[100:103]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[164:167], v[184:187], v[130:133], v[108:111]
    s_nop 2
    ds_read_b128 v[108:111], v98 offset:5120
    v_mfma_f32_16x16x32_bf16 v[176:179], v[184:187], v[148:151], v[156:159]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[2:5], v[172:175], v[152:155], v[2:5]
    ds_read_b128 v[184:187], v98 offset:6208
    v_mfma_f32_16x16x32_bf16 v[98:101], v[172:175], v[134:137], v[6:9]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[6:9], v[188:191], v[134:137], v[10:13]
    ds_read_b128 v[160:163], v106
    v_mfma_f32_16x16x32_bf16 v[14:17], v[188:191], v[152:155], v[14:17]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[10:13], v[108:111], v[152:155], v[90:93]
    s_nop 2
    ds_read_b128 v[90:93], v106 offset:1024
    v_mfma_f32_16x16x32_bf16 v[110:113], v[108:111], v[134:137], v[102:105]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[102:105], v[184:187], v[134:137], v[164:167]
    ds_read_b128 v[156:159], v106 offset:2048
    v_mfma_f32_16x16x32_bf16 v[106:109], v[184:187], v[152:155], v[176:179]
    s_setprio 0
    s_cmp_lt_i32 s92, s8
    s_cbranch_scc0 .Lh3_wide_attention_block_41

.Lh3_wide_attention_block_21:
    s_cmp_lg_u32 s92, 0
    s_cselect_b64 s[4:5], -1, 0
    s_and_b64 vcc, exec, s[4:5]
    v_xor_b32_e32 v95, 0x80000000, v94
    s_cbranch_vccnz .Lnarrow_drain
    s_mov_b64 s[6:7], 0
    s_mov_b64 s[2:3], -1
    s_branch .Lh3_wide_attention_block_23

; 256-query tile, drain the pipeline.
.Lnarrow_drain:
    s_mov_b64 s[6:7], -1
    s_mov_b64 s[2:3], 0

.Lh3_wide_attention_block_23:
    v_max3_f32 v97, v98, v99, v100
    v_max3_f32 v121, v101, v6, v7
    v_max3_f32 v138, v8, v9, v110
    v_max3_f32 v164, v111, v112, v113
    v_max3_f32 v165, v102, v103, v104
    v_max3_f32 v97, v97, v121, v138
    v_max3_f32 v121, v164, v165, s76
    v_max3_f32 v121, v97, v121, v105
    v_max3_f32 v97, v2, v3, v4
    v_max3_f32 v138, v5, v14, v15
    v_max3_f32 v164, v16, v17, v10
    v_max3_f32 v165, v11, v12, v13
    v_max3_f32 v166, v106, v107, v108
    v_max3_f32 v97, v97, v138, v164
    v_max3_f32 v138, v165, v166, s76
    s_andn2_b64 vcc, exec, s[6:7]
    v_max3_f32 v97, v97, v138, v109
    s_cbranch_vccnz .Lh3_wide_attention_block_24
    v_cmp_gt_f32_e64 s[2:3], v121, -v96
    v_cmp_gt_f32_e64 s[6:7], v97, -v94
    s_or_b64 s[2:3], s[2:3], s[6:7]
    s_and_b64 vcc, exec, s[2:3]
    s_cselect_b64 s[2:3], -1, 0

.Lh3_wide_attention_block_24:
    s_andn2_b64 vcc, exec, s[2:3]
    s_mov_b64 s[2:3], 0
    s_cbranch_vccnz .Lnarrow_stage_b
    v_mov_b32_e32 v138, v121
    v_mov_b32_e32 v164, v138
    v_mov_b32_e32 v165, v97
    v_mov_b32_e32 v166, v121
    v_permlane32_swap_b32_e64 v138, v164 bound_ctrl:1
    v_permlane32_swap_b32_e64 v97, v165 bound_ctrl:1
    v_permlane32_swap_b32_e64 v121, v166 bound_ctrl:1
    s_mov_b64 s[2:3], s[4:5]
    v_max_f32_e32 v121, v138, v164
    v_max_f32_e32 v97, v97, v165
    v_mov_b32_e32 v138, v121
    v_mov_b32_e32 v164, v138
    v_mov_b32_e32 v165, v97
    v_mov_b32_e32 v166, v121
    v_permlane16_swap_b32_e64 v138, v164 bound_ctrl:1
    v_permlane16_swap_b32_e64 v97, v165 bound_ctrl:1
    v_permlane16_swap_b32_e64 v121, v166 bound_ctrl:1
    s_nop 0
    v_max_f32_e32 v121, v138, v164
    v_max_f32_e32 v97, v97, v165
    v_max_f32_e64 v138, -v96, v121
    v_max_f32_e32 v95, v95, v97
    v_cndmask_b32_e64 v138, v121, v138, s[4:5]
    v_cndmask_b32_e64 v95, v97, v95, s[4:5]
    v_add_f32_e32 v96, v138, v96
    v_add_f32_e32 v94, v95, v94
    v_mul_f32_e32 v121, 0x3e0293ee, v96
    v_xor_b32_e32 v96, 0x80000000, v138
    v_mul_f32_e32 v97, 0x3e0293ee, v94
    v_xor_b32_e32 v94, 0x80000000, v95
    s_branch .Lnarrow_stage_b
    s_nop 0

.Lh3_wide_attention_block_25:
    s_or_b64 exec, exec, s[4:5]
    s_waitcnt vmcnt(0)
    v_readfirstlane_b32 s4, v3
    s_add_i32 s4, s72, s4
    s_nop 0
    v_add_u32_e32 v2, s4, v2
    ds_write_b32 v198, v2

.Lh3_wide_attention_block_26:
    s_or_b64 exec, exec, s[2:3]
    s_waitcnt lgkmcnt(0)
    s_barrier
    ds_read_b32 v2, v198
    s_waitcnt lgkmcnt(0)
    v_cmp_gt_i32_e32 vcc, s33, v2
    s_cbranch_vccz .Lh3_wide_attention_block_101

.Lh3_wide_attention_block_27:
    v_mul_hi_i32 v3, v2, 44
    v_mul_lo_u32 v2, v2, 44
    s_waitcnt lgkmcnt(0)
    v_lshl_add_u64 v[114:115], s[50:51], 0, v[2:3]
    global_load_dwordx4 v[2:5], v[114:115], off offset:4
    global_load_dword v6, v[114:115], off offset:36
    global_load_dword v7, v[114:115], off offset:28
    v_readfirstlane_b32 s2, v0
    s_lshr_b32 s2, s2, 6
    s_mov_b64 s[64:65], -1
    s_waitcnt vmcnt(2)
    v_readfirstlane_b32 s80, v4
    s_waitcnt vmcnt(1)
    v_readfirstlane_b32 s4, v6
    s_min_i32 s4, s2, s4
    s_add_i32 s5, s80, 63
    v_readfirstlane_b32 s3, v5
    v_cmp_ge_i32_e32 vcc, s2, v6
    s_mul_i32 s6, s4, 48
    s_sub_i32 s2, s2, s4
    s_ashr_i32 s4, s5, 31
    s_add_i32 s83, s6, s3
    s_lshr_b32 s3, s4, 26
    s_lshl_b32 s2, s2, 5
    s_add_i32 s5, s5, s3
    v_readfirstlane_b32 s78, v2
    s_add_i32 s83, s83, s2
    s_ashr_i32 s58, s5, 6
    s_cmp_lt_i32 s83, s78
    s_cselect_b64 s[60:61], -1, 0
    s_ashr_i32 s82, s80, 31
    s_lshr_b32 s2, s82, 26
    s_add_i32 s2, s80, s2
    s_ashr_i32 s79, s2, 6
    s_cmp_lg_u32 s78, 0
    v_readfirstlane_b32 s81, v3
    s_waitcnt vmcnt(0)
    v_readfirstlane_b32 s62, v7
    s_cselect_b64 s[66:67], -1, 0
    s_and_b64 vcc, exec, vcc
    s_mov_b64 s[2:3], 0
    s_cbranch_vccz .Lh3_wide_attention_block_59
    v_readfirstlane_b32 s20, v0
    s_mov_b64 s[64:65], 0
    s_and_b64 vcc, exec, s[66:67]
    s_cbranch_vccz .Lh3_wide_attention_block_59
    global_load_dword v4, v[114:115], off
    s_lshr_b32 s85, s20, 6
    s_and_b64 s[2:3], s[60:61], exec
    s_cselect_b32 s84, s83, 0xffffffe0
    v_add_u32_e32 v2, s84, v192
    v_cmp_gt_u32_e64 s[8:9], s78, v2
    v_add_u32_e32 v3, 4, v2
    v_add_u32_e32 v5, 8, v2
    v_cndmask_b32_e64 v6, 0, v2, s[8:9]
    v_add_u32_e32 v7, 12, v2
    v_add_u32_e32 v8, 16, v2
    v_add_u32_e32 v9, 20, v2
    v_add_u32_e32 v10, 24, v2
    v_add_u32_e32 v2, 28, v2
    v_cmp_gt_u32_e64 s[16:17], s78, v3
    v_cmp_gt_u32_e64 s[2:3], s78, v2
    s_mov_b32 s63, s55
    v_cndmask_b32_e64 v11, 0, v3, s[16:17]
    v_cndmask_b32_e64 v17, 0, v2, s[2:3]
    s_lshl_b64 s[18:19], s[62:63], 8
    v_cmp_gt_u32_e64 s[14:15], s78, v5
    v_cmp_gt_u32_e64 s[12:13], s78, v7
    v_cmp_gt_u32_e64 s[10:11], s78, v8
    v_cndmask_b32_e64 v12, 0, v5, s[14:15]
    v_cmp_gt_u32_e64 s[6:7], s78, v9
    v_cmp_gt_u32_e64 s[4:5], s78, v10
    v_cndmask_b32_e64 v13, 0, v7, s[12:13]
    v_cndmask_b32_e64 v14, 0, v8, s[10:11]
    v_cndmask_b32_e64 v15, 0, v9, s[6:7]
    v_cndmask_b32_e64 v16, 0, v10, s[4:5]
    s_lshl_b32 s54, s85, 3
    s_cmp_le_u32 s80, s54
    s_mov_b32 s42, 0
    s_waitcnt vmcnt(0)
    v_mul_hi_u32 v3, v4, s73
    v_mul_lo_u32 v2, v4, s73
    v_lshl_add_u64 v[2:3], s[36:37], 0, v[2:3]
    v_mul_hi_u32 v5, v4, s75
    v_mul_lo_u32 v4, v4, s75
    v_lshl_add_u64 v[2:3], v[2:3], 0, s[18:19]
    v_lshl_add_u64 v[4:5], s[34:35], 0, v[4:5]
    v_lshl_add_u64 v[2:3], v[2:3], 0, v[170:171]
    v_lshl_add_u64 v[116:117], v[4:5], 0, s[18:19]
    v_mad_u64_u32 v[4:5], s[18:19], v6, s73, v[2:3]
    v_mad_u64_u32 v[6:7], s[18:19], v11, s73, v[2:3]
    v_mad_u64_u32 v[8:9], s[18:19], v12, s73, v[2:3]
    v_mad_u64_u32 v[10:11], s[18:19], v13, s73, v[2:3]
    v_mad_u64_u32 v[12:13], s[18:19], v14, s73, v[2:3]
    v_mad_u64_u32 v[38:39], s[18:19], v15, s73, v[2:3]
    v_mad_u64_u32 v[40:41], s[18:19], v16, s73, v[2:3]
    v_mad_u64_u32 v[2:3], s[18:19], v17, s73, v[2:3]
    global_load_dwordx4 v[34:37], v[4:5], off
    global_load_dwordx4 v[30:33], v[6:7], off
    global_load_dwordx4 v[22:25], v[8:9], off
    global_load_dwordx4 v[18:21], v[10:11], off
    global_load_dwordx4 v[14:17], v[12:13], off
    s_nop 0
    global_load_dwordx4 v[10:13], v[38:39], off
    global_load_dwordx4 v[6:9], v[40:41], off
    s_nop 0
    global_load_dwordx4 v[2:5], v[2:3], off
    s_cbranch_scc1 .Lh3_wide_attention_block_28
    s_not_b64 s[18:19], s[54:55]
    s_add_u32 s18, s18, s80
    s_addc_u32 s19, s19, s82
    s_mulk_i32 s19, 0x2a00
    s_mul_hi_u32 s21, s18, 0x2a00
    s_add_i32 s19, s21, s19
    s_mulk_i32 s18, 0x2a00
    s_bitset1_b32 s18, 8
    s_cmp_eq_u32 s19, 0
    s_cselect_b32 s42, s18, -1

.Lh3_wide_attention_block_28:
    s_lshl_b64 s[18:19], s[62:63], 7
    s_mul_i32 s22, s81, 0x2a00
    s_mul_hi_u32 s21, s81, 0x2a00
    s_add_u32 s22, s38, s22
    s_addc_u32 s21, s39, s21
    s_lshl_b64 s[18:19], s[18:19], 1
    s_add_u32 s22, s22, s18
    s_addc_u32 s23, s21, s19
    s_cmpk_gt_u32 s20, 0xff
    s_cselect_b64 s[18:19], -1, 0
    s_cmpk_lt_u32 s20, 0x100
    s_mul_i32 s25, s54, 0x2a00
    s_cselect_b64 s[20:21], -1, 0
    s_mul_hi_u32 s24, s54, 0x2a00
    s_add_u32 s40, s22, s25
    s_addc_u32 s41, s23, s24
    s_ashr_i32 s59, s58, 31
    s_add_i32 s26, s80, 0xbe
    s_lshl_b64 s[22:23], s[58:59], 14
    s_add_u32 s22, s22, 0x4000
    s_addc_u32 s23, s23, 0
    v_mov_b64_e32 v[38:39], s[56:57]
    v_cmp_lt_u64_e32 vcc, s[22:23], v[38:39]
    s_and_b64 s[24:25], vcc, exec
    s_cselect_b32 s24, s22, -1
    s_cmp_eq_u32 s81, 0
    s_mul_i32 s22, s62, 0x251
    s_cselect_b32 s23, 0, 0x24f
    s_add_i32 s54, s22, s23
    s_lshl_b64 s[22:23], s[54:55], 14
    s_add_u32 s44, s52, s22
    s_addc_u32 s45, s53, s23
    s_cmp_gt_u32 s26, 63
    s_mul_i32 s54, s85, 0x220
    s_cselect_b32 s46, s24, 0
    s_add_i32 s22, s54, 0x1100
    s_mov_b32 s23, s55
    s_lshl_b32 s59, s54, 1
    s_lshl_b64 s[22:23], s[22:23], 1
    s_mov_b64 s[68:69], src_shared_base
    s_add_u32 s24, 0, s22
    s_addc_u32 s25, s69, s23
    s_cmp_lg_u64 s[24:25], 0
    s_cselect_b32 s24, s24, -1
    s_lshl_b32 s63, s85, 11
    s_mov_b32 m0, s59
    s_nop 0
    buffer_load_dwordx4 v139, s[40:43], 0 offen lds
    s_add_i32 s68, s63, 0xcc00
    s_mov_b32 m0, s24
    s_nop 0
    buffer_load_dwordx4 v169, s[40:43], 0 offen lds
    s_add_u32 s24, s68, 0x400
    s_addc_u32 s25, s69, 0
    s_mov_b32 s47, s43
    v_or_b32_e32 v118, s63, v1
    s_mov_b32 m0, s68
    s_nop 0
    buffer_load_dwordx4 v118, s[44:47], 0 offen lds
    s_cmp_lg_u64 s[24:25], 0
    s_cselect_b32 s24, s24, -1
    v_or_b32_e32 v119, 0x400, v118
    s_mov_b32 m0, s24
    s_nop 0
    buffer_load_dwordx4 v119, s[44:47], 0 offen lds
    s_and_b64 vcc, exec, s[20:21]
    s_cbranch_vccnz .Lh3_wide_attention_block_29
    s_lshl_b64 s[26:27], s[54:55], 1
    s_mov_b64 s[24:25], src_shared_base
    s_add_u32 s26, 0x4400, s26
    s_addc_u32 s27, s25, s27
    s_cmp_lg_u64 s[26:27], 0
    s_cselect_b32 s24, s26, -1
    s_add_u32 s22, 0x4400, s22
    s_addc_u32 s23, s25, s23
    s_mov_b32 m0, s24
    s_nop 0
    buffer_load_dwordx4 v180, s[40:43], 0 offen lds
    s_cmp_lg_u64 s[22:23], 0
    s_cselect_b32 s22, s22, -1
    v_add_u32_e32 v38, 0xa8010, v139
    s_mov_b32 m0, s22
    s_nop 0
    buffer_load_dwordx4 v38, s[40:43], 0 offen lds

.Lh3_wide_attention_block_29:
    s_mul_i32 s22, s85, 0x1200
    s_add_i32 s22, s22, 0x1a400
    v_or_b32_e32 v38, s22, v193
    v_lshl_or_b32 v39, v168, 1, s22
    v_add3_u32 v38, v38, v194, v195
    s_waitcnt vmcnt(7)
    v_cndmask_b32_e64 v37, 0, v37, s[8:9]
    v_cndmask_b32_e64 v36, 0, v36, s[8:9]
    v_cndmask_b32_e64 v35, 0, v35, s[8:9]
    v_cndmask_b32_e64 v34, 0, v34, s[8:9]
    v_add_u32_e32 v39, v39, v196
    s_waitcnt vmcnt(6)
    v_cndmask_b32_e64 v33, 0, v33, s[16:17]
    v_cndmask_b32_e64 v32, 0, v32, s[16:17]
    v_cndmask_b32_e64 v31, 0, v31, s[16:17]
    v_cndmask_b32_e64 v30, 0, v30, s[16:17]
    s_waitcnt vmcnt(5)
    v_cndmask_b32_e64 v25, 0, v25, s[14:15]
    v_cndmask_b32_e64 v24, 0, v24, s[14:15]
    v_cndmask_b32_e64 v23, 0, v23, s[14:15]
    v_cndmask_b32_e64 v22, 0, v22, s[14:15]
    s_waitcnt vmcnt(4)
    v_cndmask_b32_e64 v21, 0, v21, s[12:13]
    v_cndmask_b32_e64 v20, 0, v20, s[12:13]
    v_cndmask_b32_e64 v19, 0, v19, s[12:13]
    v_cndmask_b32_e64 v18, 0, v18, s[12:13]
    ds_write_b128 v39, v[34:37]
    ds_write_b128 v39, v[30:33] offset:1152
    ds_write_b128 v39, v[22:25] offset:2304
    ds_write_b128 v39, v[18:21] offset:3456
    s_waitcnt vmcnt(3)
    v_cndmask_b32_e64 v17, 0, v17, s[10:11]
    v_cndmask_b32_e64 v16, 0, v16, s[10:11]
    v_cndmask_b32_e64 v15, 0, v15, s[10:11]
    v_cndmask_b32_e64 v14, 0, v14, s[10:11]
    s_waitcnt vmcnt(2)
    v_cndmask_b32_e64 v13, 0, v13, s[6:7]
    v_cndmask_b32_e64 v12, 0, v12, s[6:7]
    v_cndmask_b32_e64 v11, 0, v11, s[6:7]
    v_cndmask_b32_e64 v10, 0, v10, s[6:7]
    s_waitcnt vmcnt(1)
    v_cndmask_b32_e64 v9, 0, v9, s[4:5]
    v_cndmask_b32_e64 v8, 0, v8, s[4:5]
    v_cndmask_b32_e64 v7, 0, v7, s[4:5]
    v_cndmask_b32_e64 v6, 0, v6, s[4:5]
    s_waitcnt vmcnt(0)
    v_cndmask_b32_e64 v5, 0, v5, s[2:3]
    v_cndmask_b32_e64 v4, 0, v4, s[2:3]
    v_cndmask_b32_e64 v3, 0, v3, s[2:3]
    v_cndmask_b32_e64 v2, 0, v2, s[2:3]
    ds_read_b128 v[122:125], v38
    ds_read_b128 v[126:129], v38 offset:64
    ds_read_b128 v[130:133], v38 offset:128
    ds_read_b128 v[134:137], v38 offset:192
    ds_write_b128 v39, v[14:17]
    ds_write_b128 v39, v[10:13] offset:1152
    ds_write_b128 v39, v[6:9] offset:2304
    ds_write_b128 v39, v[2:5] offset:3456
    ds_read_b128 v[140:143], v38
    ds_read_b128 v[144:147], v38 offset:64
    ds_read_b128 v[148:151], v38 offset:128
    ds_read_b128 v[152:155], v38 offset:192
    scratch_load_dword v120, off, off
    v_cndmask_b32_e64 v2, 0, 1, s[20:21]
    s_andn2_b64 vcc, exec, s[20:21]
    s_mov_b64 s[2:3], -1
    s_waitcnt vmcnt(0)
    s_cbranch_vccnz .Lh3_wide_attention_block_30
    s_waitcnt vmcnt(0)
    s_mov_b64 s[2:3], 0

.Lh3_wide_attention_block_30:
    s_andn2_b64 vcc, exec, s[2:3]
    s_cbranch_vccnz .Lh3_wide_attention_block_31
    s_waitcnt vmcnt(2)

.Lh3_wide_attention_block_31:
    s_and_b64 s[2:3], s[18:19], exec
    s_barrier
    s_cselect_b32 s86, 2, 1
    v_cndmask_b32_e64 v3, 0, 1, s[18:19]
    s_cmpk_gt_i32 s84, 0xffe0
    v_cmp_ne_u32_e32 vcc, 1, v2
    v_readfirstlane_b32 s87, v3
    s_cselect_b32 s88, s58, 0
    s_and_b64 vcc, exec, vcc
    s_mov_b64 s[2:3], -1
    s_cbranch_vccnz .Lh3_wide_attention_block_39
    s_mov_b64 s[2:3], src_shared_base
    s_lshl_b64 s[70:71], s[54:55], 1
    s_add_u32 s2, 0x4400, s70
    s_addc_u32 s3, s3, s71
    s_add_i32 s5, s59, 0x6600
    s_add_i32 s4, s63, 0xd000
    s_cmp_lg_u64 s[2:3], 0
    v_mov_b32_e32 v91, v120
    s_cselect_b32 s2, s2, -1
    s_mov_b32 m0, s2
    v_mov_b32_e32 v98, v183
    ds_read_b128 v[104:107], v91
    ds_read_b128 v[94:97], v91 offset:1024
    ds_read_b128 v[100:103], v91 offset:2048
    s_cmp_lt_i32 s88, 1
    v_mov_b32_e32 v25, 0
    s_cbranch_scc1 .Lh3_wide_attention_block_43
    v_mov_b32_e32 v6, 0
    v_mov_b32_e32 v58, 0
    s_add_i32 s90, s80, -1
    s_mov_b32 s9, 0xa8000
    s_mul_i32 s89, s86, 0xa8000
    s_mov_b32 s7, 0
    v_mov_b32_e32 v18, v171
    v_mov_b32_e32 v19, v171
    v_mov_b32_e32 v20, v171
    v_mov_b32_e32 v21, v171
    s_mov_b64 s[2:3], 0
    s_movk_i32 s91, 0x201
    v_mov_b32_e32 v90, 0xc29cd760
    v_mov_b32_e32 v92, 0xc29cd760
    s_mov_b32 s8, 0
    s_mov_b32 s6, s68
    s_mov_b32 s92, 0
    v_mov_b32_e32 v7, v6
    v_mov_b32_e32 v8, v6
    v_mov_b32_e32 v9, v6
    v_mov_b32_e32 v2, v6
    v_mov_b32_e32 v3, v6
    v_mov_b32_e32 v4, v6
    v_mov_b32_e32 v5, v6
    v_mov_b32_e32 v14, v6
    v_mov_b32_e32 v15, v6
    v_mov_b32_e32 v16, v6
    v_mov_b32_e32 v17, v6
    v_mov_b32_e32 v10, v6
    v_mov_b32_e32 v11, v6
    v_mov_b32_e32 v12, v6
    v_mov_b32_e32 v13, v6
    v_mov_b32_e32 v59, v58
    v_mov_b32_e32 v60, v58
    v_mov_b32_e32 v61, v58
    v_mov_b32_e32 v86, v58
    v_mov_b32_e32 v87, v58
    v_mov_b32_e32 v88, v58
    v_mov_b32_e32 v89, v58
    v_mov_b32_e32 v82, v58
    v_mov_b32_e32 v83, v58
    v_mov_b32_e32 v84, v58
    v_mov_b32_e32 v85, v58
    v_mov_b32_e32 v78, v58
    v_mov_b32_e32 v79, v58
    v_mov_b32_e32 v80, v58
    v_mov_b32_e32 v81, v58
    v_mov_b32_e32 v70, v58
    v_mov_b32_e32 v71, v58
    v_mov_b32_e32 v72, v58
    v_mov_b32_e32 v73, v58
    v_mov_b32_e32 v74, v58
    v_mov_b32_e32 v75, v58
    v_mov_b32_e32 v76, v58
    v_mov_b32_e32 v77, v58
    v_mov_b32_e32 v66, v58
    v_mov_b32_e32 v67, v58
    v_mov_b32_e32 v68, v58
    v_mov_b32_e32 v69, v58
    v_mov_b32_e32 v62, v58
    v_mov_b32_e32 v63, v58
    v_mov_b32_e32 v64, v58
    v_mov_b32_e32 v65, v58
    v_mov_b32_e32 v50, v58
    v_mov_b32_e32 v51, v58
    v_mov_b32_e32 v52, v58
    v_mov_b32_e32 v53, v58
    v_mov_b32_e32 v54, v58
    v_mov_b32_e32 v55, v58
    v_mov_b32_e32 v56, v58
    v_mov_b32_e32 v57, v58
    v_mov_b32_e32 v46, v58
    v_mov_b32_e32 v47, v58
    v_mov_b32_e32 v48, v58
    v_mov_b32_e32 v49, v58
    v_mov_b32_e32 v42, v58
    v_mov_b32_e32 v43, v58
    v_mov_b32_e32 v44, v58
    v_mov_b32_e32 v45, v58
    v_mov_b32_e32 v34, v58
    v_mov_b32_e32 v35, v58
    v_mov_b32_e32 v36, v58
    v_mov_b32_e32 v37, v58
    v_mov_b32_e32 v38, v58
    v_mov_b32_e32 v39, v58
    v_mov_b32_e32 v40, v58
    v_mov_b32_e32 v41, v58
    v_mov_b32_e32 v30, v58
    v_mov_b32_e32 v31, v58
    v_mov_b32_e32 v32, v58
    v_mov_b32_e32 v33, v58
    v_mov_b32_e32 v22, v58
    v_mov_b32_e32 v23, v58
    v_mov_b32_e32 v24, v58
    v_mov_b32_e32 v25, v58
    s_branch .Lh3_wide_attention_block_14

.Lh3_wide_attention_block_32:
    v_exp_f32_e64 v99, -v110
    v_exp_f32_e64 v93, -v93
    ; Apply one row rescale to both FP32 accumulators. op_sel_hi broadcasts the low scale.
    v_pk_mul_f32 v[58:59], v[98:99], v[58:59] op_sel:[1,0]
    v_pk_mul_f32 v[60:61], v[98:99], v[60:61] op_sel:[1,0]
    v_pk_mul_f32 v[82:83], v[98:99], v[82:83] op_sel:[1,0]
    v_pk_mul_f32 v[84:85], v[98:99], v[84:85] op_sel:[1,0]
    v_pk_mul_f32 v[70:71], v[98:99], v[70:71] op_sel:[1,0]
    v_pk_mul_f32 v[72:73], v[98:99], v[72:73] op_sel:[1,0]
    v_pk_mul_f32 v[66:67], v[98:99], v[66:67] op_sel:[1,0]
    v_pk_mul_f32 v[68:69], v[98:99], v[68:69] op_sel:[1,0]
    v_pk_mul_f32 v[50:51], v[98:99], v[50:51] op_sel:[1,0]
    v_pk_mul_f32 v[52:53], v[98:99], v[52:53] op_sel:[1,0]
    v_pk_mul_f32 v[46:47], v[98:99], v[46:47] op_sel:[1,0]
    v_pk_mul_f32 v[48:49], v[98:99], v[48:49] op_sel:[1,0]
    v_pk_mul_f32 v[34:35], v[98:99], v[34:35] op_sel:[1,0]
    v_pk_mul_f32 v[36:37], v[98:99], v[36:37] op_sel:[1,0]
    v_pk_mul_f32 v[30:31], v[98:99], v[30:31] op_sel:[1,0]
    v_pk_mul_f32 v[32:33], v[98:99], v[32:33] op_sel:[1,0]
    v_pk_mul_f32 v[86:87], v[92:93], v[86:87] op_sel:[1,0]
    v_pk_mul_f32 v[88:89], v[92:93], v[88:89] op_sel:[1,0]
    v_pk_mul_f32 v[78:79], v[92:93], v[78:79] op_sel:[1,0]
    v_pk_mul_f32 v[80:81], v[92:93], v[80:81] op_sel:[1,0]
    v_pk_mul_f32 v[74:75], v[92:93], v[74:75] op_sel:[1,0]
    v_pk_mul_f32 v[76:77], v[92:93], v[76:77] op_sel:[1,0]
    v_pk_mul_f32 v[62:63], v[92:93], v[62:63] op_sel:[1,0]
    v_pk_mul_f32 v[64:65], v[92:93], v[64:65] op_sel:[1,0]
    v_pk_mul_f32 v[54:55], v[92:93], v[54:55] op_sel:[1,0]
    v_pk_mul_f32 v[56:57], v[92:93], v[56:57] op_sel:[1,0]
    v_pk_mul_f32 v[42:43], v[92:93], v[42:43] op_sel:[1,0]
    v_pk_mul_f32 v[44:45], v[92:93], v[44:45] op_sel:[1,0]
    v_pk_mul_f32 v[38:39], v[92:93], v[38:39] op_sel:[1,0]
    v_pk_mul_f32 v[40:41], v[92:93], v[40:41] op_sel:[1,0]
    v_pk_mul_f32 v[22:23], v[92:93], v[22:23] op_sel:[1,0]
    v_pk_mul_f32 v[24:25], v[92:93], v[24:25] op_sel:[1,0]
    s_branch .Lh3_wide_attention_block_15

.Lh3_wide_attention_block_33:
    s_cmp_lt_i32 s92, s79
    s_cbranch_scc1 .Lh3_wide_attention_block_34
    v_mbcnt_lo_u32_b32 v91, -1, 0
    v_mbcnt_hi_u32_b32 v91, -1, v91
    s_nop 0
    v_ashrrev_i32_e32 v93, 4, v91
    v_lshrrev_b32_e32 v91, 31, v91
    v_add_u32_e32 v91, v93, v91
    v_lshrrev_b32_e32 v91, 1, v91
    v_sub_u32_e32 v91, 0, v91
    v_lshlrev_b32_e32 v91, 3, v91
    v_lshlrev_b32_e32 v93, 2, v93
    v_sub_u32_e32 v91, v91, v93
    v_add_u32_e32 v91, s90, v91
    v_cmp_gt_i32_e32 vcc, 0, v91
    v_mov_b32_e32 v93, 0xf2177617
    v_cmp_gt_i32_e64 s[2:3], 1, v91
    v_cmp_gt_i32_e64 s[4:5], 2, v91
    v_cmp_gt_i32_e64 s[6:7], 3, v91
    v_cmp_gt_i32_e64 s[8:9], 8, v91
    v_cmp_gt_i32_e64 s[10:11], 9, v91
    v_cmp_gt_i32_e64 s[12:13], 10, v91
    v_cmp_gt_i32_e64 s[14:15], 11, v91
    v_cmp_gt_i32_e64 s[16:17], 32, v91
    v_cmp_gt_i32_e64 s[18:19], 33, v91
    v_cmp_gt_i32_e64 s[20:21], 34, v91
    v_cmp_gt_i32_e64 s[22:23], 35, v91
    v_cmp_gt_i32_e64 s[24:25], 40, v91
    v_cmp_gt_i32_e64 s[26:27], 41, v91
    v_cmp_gt_i32_e64 s[28:29], 42, v91
    v_cmp_gt_i32_e64 s[30:31], 43, v91
    v_cndmask_b32_e32 v102, v102, v93, vcc
    v_cndmask_b32_e64 v103, v103, v93, s[2:3]
    v_cndmask_b32_e64 v104, v104, v93, s[4:5]
    v_cndmask_b32_e64 v105, v105, v93, s[6:7]
    v_cndmask_b32_e64 v10, v10, v93, s[8:9]
    v_cndmask_b32_e64 v11, v11, v93, s[10:11]
    v_cndmask_b32_e64 v12, v12, v93, s[12:13]
    v_cndmask_b32_e64 v13, v13, v93, s[14:15]
    v_cndmask_b32_e64 v106, v106, v93, s[16:17]
    v_cndmask_b32_e64 v107, v107, v93, s[18:19]
    v_cndmask_b32_e64 v108, v108, v93, s[20:21]
    v_cndmask_b32_e64 v109, v109, v93, s[22:23]
    v_cndmask_b32_e64 v94, v94, v93, s[24:25]
    v_cndmask_b32_e64 v95, v95, v93, s[26:27]
    v_cndmask_b32_e64 v96, v96, v93, s[28:29]
    v_cndmask_b32_e64 v97, v97, v93, s[30:31]
    v_cndmask_b32_e32 v2, v2, v93, vcc
    v_cndmask_b32_e64 v3, v3, v93, s[2:3]
    v_cndmask_b32_e64 v4, v4, v93, s[4:5]
    v_cndmask_b32_e64 v5, v5, v93, s[6:7]
    v_cndmask_b32_e64 v14, v14, v93, s[8:9]
    v_cndmask_b32_e64 v15, v15, v93, s[10:11]
    v_cndmask_b32_e64 v16, v16, v93, s[12:13]
    v_cndmask_b32_e64 v17, v17, v93, s[14:15]
    v_cndmask_b32_e64 v6, v6, v93, s[16:17]
    v_cndmask_b32_e64 v7, v7, v93, s[18:19]
    v_cndmask_b32_e64 v8, v8, v93, s[20:21]
    v_cndmask_b32_e64 v9, v9, v93, s[22:23]
    v_cndmask_b32_e64 v98, v98, v93, s[24:25]
    v_cndmask_b32_e64 v99, v99, v93, s[26:27]
    v_cndmask_b32_e64 v100, v100, v93, s[28:29]
    v_cndmask_b32_e64 v101, v101, v93, s[30:31]

.Lh3_wide_attention_block_34:
    s_add_i32 s2, s92, 1
    s_max_i32 s8, s2, s79
    s_branch .Lh3_wide_attention_block_16

.Lh3_wide_attention_block_35:
    s_lshl_b32 s7, s87, 14
    s_lshl_b32 s8, s88, 14
    s_add_i32 s8, s8, s7
    v_or_b32_e32 v90, s8, v118
    s_andn2_b64 vcc, exec, s[2:3]
    s_cbranch_vccz .Lh3_wide_attention_block_44

.Lh3_wide_attention_block_36:
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[86:89], v[104:107], v[14:17], v[86:89]
    ds_read_b128 v[108:111], v91 offset:3072
    ; Form the next K address ahead of the independent MFMA.
    v_add_u32_e32 v92, s89, v139
    v_mfma_f32_16x16x32_bf16 v[58:61], v[104:107], v[10:13], v[58:61]
    s_waitcnt lgkmcnt(2)
    buffer_load_dwordx4 v92, s[40:43], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[82:85], v[94:97], v[10:13], v[82:85]
    ds_read_b128 v[104:107], v91 offset:4096
    v_mfma_f32_16x16x32_bf16 v[78:81], v[94:97], v[14:17], v[78:81]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[74:77], v[100:103], v[14:17], v[74:77]
    ds_read_b128 v[92:95], v91 offset:5120
    v_mfma_f32_16x16x32_bf16 v[70:73], v[100:103], v[10:13], v[70:73]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[66:69], v[108:111], v[10:13], v[66:69]
    ds_read_b128 v[156:159], v91 offset:6144
    v_mfma_f32_16x16x32_bf16 v[62:65], v[108:111], v[14:17], v[62:65]
    s_mov_b32 m0, s5
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[96:99], v[104:107], v[14:17], v[54:57]
    ds_read_b128 v[160:163], v91 offset:7168
    v_mfma_f32_16x16x32_bf16 v[100:103], v[104:107], v[10:13], v[50:53]
    s_nop 2
    v_add_u32_e32 v50, s89, v169
    s_waitcnt lgkmcnt(2)
    buffer_load_dwordx4 v50, s[40:43], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[46:49], v[92:95], v[10:13], v[46:49]
    ds_read_b128 v[50:53], v91 offset:8192
    v_mfma_f32_16x16x32_bf16 v[42:45], v[92:95], v[14:17], v[42:45]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[92:95], v[156:159], v[14:17], v[38:41]
    ds_read_b128 v[54:57], v91 offset:9216
    v_mfma_f32_16x16x32_bf16 v[104:107], v[156:159], v[10:13], v[34:37]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[108:111], v[160:163], v[10:13], v[30:33]
    s_nop 2
    ds_read_b128 v[30:33], v91 offset:10240
    v_mfma_f32_16x16x32_bf16 v[156:159], v[160:163], v[14:17], v[22:25]
    s_mov_b32 m0, s6
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[10:13], v[50:53], v[6:9], v[86:89]
    ds_read_b128 v[34:37], v91 offset:11264
    v_mfma_f32_16x16x32_bf16 v[38:41], v[50:53], v[2:5], v[58:61]
    s_waitcnt lgkmcnt(2)
    buffer_load_dwordx4 v90, s[44:47], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[50:53], v[54:57], v[2:5], v[82:85]
    s_nop 2
    ds_read_b128 v[82:85], v91 offset:12288
    v_mfma_f32_16x16x32_bf16 v[14:17], v[54:57], v[6:9], v[78:81]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[22:25], v[30:33], v[6:9], v[74:77]
    s_nop 2
    ds_read_b128 v[74:77], v91 offset:13312
    v_mfma_f32_16x16x32_bf16 v[54:57], v[30:33], v[2:5], v[70:73]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[58:61], v[34:37], v[2:5], v[66:69]
    s_nop 0
    ds_read_b128 v[70:73], v91 offset:14336
    v_mfma_f32_16x16x32_bf16 v[30:33], v[34:37], v[6:9], v[62:65]
    s_mov_b32 m0, s4
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[34:37], v[82:85], v[6:9], v[96:99]
    ds_read_b128 v[78:81], v91 offset:15360
    v_mfma_f32_16x16x32_bf16 v[62:65], v[82:85], v[2:5], v[100:103]
    v_or_b32_e32 v66, 0x400, v90
    s_waitcnt lgkmcnt(2)
    buffer_load_dwordx4 v66, s[44:47], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[66:69], v[74:77], v[2:5], v[46:49]
    v_mfma_f32_16x16x32_bf16 v[42:45], v[74:77], v[6:9], v[42:45]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[46:49], v[70:73], v[6:9], v[92:95]
    v_mfma_f32_16x16x32_bf16 v[70:73], v[70:73], v[2:5], v[104:107]
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_16x16x32_bf16 v[74:77], v[78:81], v[2:5], v[108:111]
    v_mfma_f32_16x16x32_bf16 v[2:5], v[78:81], v[6:9], v[156:159]
    s_waitcnt vmcnt(0)
    s_barrier
    s_cmp_ge_i32 s88, s58
    s_mov_b32 s2, s88
    s_cbranch_scc1 .Lh3_wide_attention_block_38

.Lh3_wide_attention_block_37:
    s_add_i32 s2, s2, 1
    s_add_i32 s3, s2, s86
    s_mul_hi_u32 s6, s3, 0xaaaaaaab
    s_lshr_b32 s6, s6, 1
    s_mov_b64 s[4:5], src_shared_base
    s_add_i32 s10, s2, s87
    s_mul_i32 s6, s6, 3
    s_mul_i32 s4, s3, 0xa8000
    s_sub_i32 s3, s3, s6
    s_mul_hi_u32 s6, s10, 0xaaaaaaab
    s_lshr_b32 s6, s6, 1
    s_mul_i32 s6, s6, 3
    s_mulk_i32 s3, 0x4400
    s_sub_i32 s6, s10, s6
    s_add_i32 s3, s59, s3
    s_mul_i32 s11, s6, 0x4800
    s_add_u32 s6, s68, s11
    s_addc_u32 s7, s69, 0
    s_add_u32 s8, s3, 0x2200
    s_addc_u32 s9, s5, 0
    s_cmp_lg_u64 s[8:9], 0
    v_add_u32_e32 v6, s4, v139
    s_mov_b32 m0, s3
    s_nop 0
    buffer_load_dwordx4 v6, s[40:43], 0 offen lds
    s_cselect_b32 s3, s8, -1
    s_cmp_lg_u64 s[6:7], 0
    v_add_u32_e32 v6, s4, v169
    s_mov_b32 m0, s3
    s_nop 0
    buffer_load_dwordx4 v6, s[40:43], 0 offen lds
    s_cselect_b32 s3, s6, -1
    v_lshl_or_b32 v6, s10, 14, v118
    s_mov_b32 m0, s3
    s_nop 0
    buffer_load_dwordx4 v6, s[44:47], 0 offen lds
    s_add_i32 s3, s63, s11
    s_add_u32 s4, s3, 0xd000
    s_addc_u32 s5, s5, 0
    s_cmp_lg_u64 s[4:5], 0
    s_cselect_b32 s3, s4, -1
    v_or_b32_e32 v6, 0x400, v6
    s_mov_b32 m0, s3
    s_nop 0
    buffer_load_dwordx4 v6, s[44:47], 0 offen lds
    s_waitcnt vmcnt(0)
    s_barrier
    s_cmp_eq_u32 s2, s58
    s_cbranch_scc0 .Lh3_wide_attention_block_37

.Lh3_wide_attention_block_38:
    s_mov_b64 s[2:3], 0

.Lh3_wide_attention_block_39:
    s_and_b64 vcc, exec, s[2:3]
    s_cbranch_vccz .Lh3_wide_attention_block_49
    s_mov_b64 s[2:3], src_shared_base
    s_lshl_b64 s[70:71], s[54:55], 1
    s_add_u32 s2, 0x8800, s70
    s_addc_u32 s3, s3, s71
    s_add_i32 s5, s63, 0x11400
    s_add_i32 s6, s59, 0xaa00
    s_add_i32 s4, s63, 0x11800
    s_cmp_lg_u64 s[2:3], 0
    v_mov_b32_e32 v95, v120
    s_cselect_b32 s2, s2, -1
    s_mov_b32 m0, s2
    v_mov_b32_e32 v98, v183
    ds_read_b128 v[160:163], v95
    ds_read_b128 v[90:93], v95 offset:1024
    ds_read_b128 v[156:159], v95 offset:2048
    s_cmp_lt_i32 s88, 1
    zero_accumulators 25, 24, 23, 22, 33, 32, 31, 30, 41, 40, 39, 38
    zero_accumulators 37, 36, 35, 34, 45, 44, 43, 42, 49, 48, 47, 46
    zero_accumulators 57, 56, 55, 54, 53, 52, 51, 50, 61, 60, 59, 58
    zero_accumulators 65, 64, 63, 62, 73, 72, 71, 70, 69, 68, 67, 66
    zero_accumulators 77, 76, 75, 74, 81, 80, 79, 78, 89, 88, 87, 86
    zero_accumulators 85, 84, 83, 82, 21, 20, 19, 18
    s_cbranch_scc1 .Lh3_wide_attention_block_45
    s_lshl_b32 s2, s87, 14
    s_lshl_b32 s3, s88, 14
    v_mov_b32_e32 v6, 0
    s_add_i32 s90, s80, -1
    s_mul_i32 s89, s86, 0xa8000
    s_add_i32 s54, s3, s2
    s_mov_b64 s[2:3], 0
    s_mov_b32 s8, 0
    s_mov_b32 s9, 0x150000
    s_movk_i32 s7, 0x4000
    s_movk_i32 s91, 0x201
    zero_accumulators 18, 19, 20, 21, 82, 83, 84, 85, 86, 87, 88, 89
    zero_accumulators 78, 79, 80, 81, 74, 75, 76, 77, 66, 67, 68, 69
    zero_accumulators 70, 71, 72, 73, 62, 63, 64, 65, 58, 59, 60, 61
    zero_accumulators 50, 51, 52, 53, 54, 55, 56, 57, 46, 47, 48, 49
    zero_accumulators 42, 43, 44, 45, 34, 35, 36, 37, 38, 39, 40, 41
    zero_accumulators 30, 31, 32, 33, 22, 23, 24, 25
    v_mov_b32_e32 v94, 0xc29cd760
    v_mov_b32_e32 v96, 0xc29cd760
    v_mov_b32_e32 v106, v120
    s_mov_b32 s92, 0
    v_mov_b32_e32 v7, v6
    v_mov_b32_e32 v8, v6
    v_mov_b32_e32 v9, v6
    v_mov_b32_e32 v2, v6
    v_mov_b32_e32 v3, v6
    v_mov_b32_e32 v4, v6
    v_mov_b32_e32 v5, v6
    v_mov_b32_e32 v14, v6
    v_mov_b32_e32 v15, v6
    v_mov_b32_e32 v16, v6
    v_mov_b32_e32 v17, v6
    v_mov_b32_e32 v10, v6
    v_mov_b32_e32 v11, v6
    v_mov_b32_e32 v12, v6
    v_mov_b32_e32 v13, v6
    s_branch .Lh3_wide_attention_block_19

.Lh3_wide_attention_block_40:
    v_exp_f32_e64 v99, -v121
    v_exp_f32_e64 v97, -v97
    v_pk_mul_f32 v[18:19], v[98:99], v[18:19] op_sel:[1,0]
    v_pk_mul_f32 v[82:83], v[98:99], v[82:83] op_sel:[1,0]
    v_pk_mul_f32 v[84:85], v[98:99], v[84:85] op_sel:[1,0]
    v_pk_mul_f32 v[78:79], v[98:99], v[78:79] op_sel:[1,0]
    v_pk_mul_f32 v[80:81], v[98:99], v[80:81] op_sel:[1,0]
    v_pk_mul_f32 v[66:67], v[98:99], v[66:67] op_sel:[1,0]
    v_pk_mul_f32 v[68:69], v[98:99], v[68:69] op_sel:[1,0]
    v_pk_mul_f32 v[62:63], v[98:99], v[62:63] op_sel:[1,0]
    v_pk_mul_f32 v[64:65], v[98:99], v[64:65] op_sel:[1,0]
    v_pk_mul_f32 v[50:51], v[98:99], v[50:51] op_sel:[1,0]
    v_pk_mul_f32 v[52:53], v[98:99], v[52:53] op_sel:[1,0]
    v_pk_mul_f32 v[46:47], v[98:99], v[46:47] op_sel:[1,0]
    v_pk_mul_f32 v[48:49], v[98:99], v[48:49] op_sel:[1,0]
    v_pk_mul_f32 v[34:35], v[98:99], v[34:35] op_sel:[1,0]
    v_pk_mul_f32 v[36:37], v[98:99], v[36:37] op_sel:[1,0]
    v_pk_mul_f32 v[30:31], v[98:99], v[30:31] op_sel:[1,0]
    v_pk_mul_f32 v[32:33], v[98:99], v[32:33] op_sel:[1,0]
    v_pk_mul_f32 v[20:21], v[96:97], v[20:21] op_sel:[1,0]
    v_pk_mul_f32 v[86:87], v[96:97], v[86:87] op_sel:[1,0]
    v_pk_mul_f32 v[88:89], v[96:97], v[88:89] op_sel:[1,0]
    v_pk_mul_f32 v[74:75], v[96:97], v[74:75] op_sel:[1,0]
    v_pk_mul_f32 v[76:77], v[96:97], v[76:77] op_sel:[1,0]
    v_pk_mul_f32 v[70:71], v[96:97], v[70:71] op_sel:[1,0]
    v_pk_mul_f32 v[72:73], v[96:97], v[72:73] op_sel:[1,0]
    v_pk_mul_f32 v[58:59], v[96:97], v[58:59] op_sel:[1,0]
    v_pk_mul_f32 v[60:61], v[96:97], v[60:61] op_sel:[1,0]
    v_pk_mul_f32 v[54:55], v[96:97], v[54:55] op_sel:[1,0]
    v_pk_mul_f32 v[56:57], v[96:97], v[56:57] op_sel:[1,0]
    v_pk_mul_f32 v[42:43], v[96:97], v[42:43] op_sel:[1,0]
    v_pk_mul_f32 v[44:45], v[96:97], v[44:45] op_sel:[1,0]
    v_pk_mul_f32 v[38:39], v[96:97], v[38:39] op_sel:[1,0]
    v_pk_mul_f32 v[40:41], v[96:97], v[40:41] op_sel:[1,0]
    v_pk_mul_f32 v[22:23], v[96:97], v[22:23] op_sel:[1,0]
    v_pk_mul_f32 v[24:25], v[96:97], v[24:25] op_sel:[1,0]
    s_branch .Lh3_wide_attention_block_20

.Lh3_wide_attention_block_41:
    s_cmp_lt_i32 s92, s79
    s_cbranch_scc1 .Lh3_wide_attention_block_42
    v_mbcnt_lo_u32_b32 v95, -1, 0
    v_mbcnt_hi_u32_b32 v95, -1, v95
    s_nop 0
    v_ashrrev_i32_e32 v97, 4, v95
    v_lshrrev_b32_e32 v95, 31, v95
    v_add_u32_e32 v95, v97, v95
    v_lshrrev_b32_e32 v95, 1, v95
    v_sub_u32_e32 v95, 0, v95
    v_lshlrev_b32_e32 v95, 3, v95
    v_lshlrev_b32_e32 v97, 2, v97
    v_sub_u32_e32 v95, v95, v97
    v_add_u32_e32 v95, s90, v95
    v_cmp_gt_i32_e32 vcc, 0, v95
    v_mov_b32_e32 v97, 0xf2177617
    v_cmp_gt_i32_e64 s[2:3], 1, v95
    v_cmp_gt_i32_e64 s[4:5], 2, v95
    v_cmp_gt_i32_e64 s[6:7], 3, v95
    v_cmp_gt_i32_e64 s[8:9], 8, v95
    v_cmp_gt_i32_e64 s[10:11], 9, v95
    v_cmp_gt_i32_e64 s[12:13], 10, v95
    v_cmp_gt_i32_e64 s[14:15], 11, v95
    v_cmp_gt_i32_e64 s[16:17], 32, v95
    v_cmp_gt_i32_e64 s[18:19], 33, v95
    v_cmp_gt_i32_e64 s[20:21], 34, v95
    v_cmp_gt_i32_e64 s[22:23], 35, v95
    v_cmp_gt_i32_e64 s[24:25], 40, v95
    v_cmp_gt_i32_e64 s[26:27], 41, v95
    v_cmp_gt_i32_e64 s[28:29], 42, v95
    v_cmp_gt_i32_e64 s[30:31], 43, v95
    v_cndmask_b32_e32 v98, v98, v97, vcc
    v_cndmask_b32_e64 v99, v99, v97, s[2:3]
    v_cndmask_b32_e64 v100, v100, v97, s[4:5]
    v_cndmask_b32_e64 v101, v101, v97, s[6:7]
    v_cndmask_b32_e64 v6, v6, v97, s[8:9]
    v_cndmask_b32_e64 v7, v7, v97, s[10:11]
    v_cndmask_b32_e64 v8, v8, v97, s[12:13]
    v_cndmask_b32_e64 v9, v9, v97, s[14:15]
    v_cndmask_b32_e64 v110, v110, v97, s[16:17]
    v_cndmask_b32_e64 v111, v111, v97, s[18:19]
    v_cndmask_b32_e64 v112, v112, v97, s[20:21]
    v_cndmask_b32_e64 v113, v113, v97, s[22:23]
    v_cndmask_b32_e64 v102, v102, v97, s[24:25]
    v_cndmask_b32_e64 v103, v103, v97, s[26:27]
    v_cndmask_b32_e64 v104, v104, v97, s[28:29]
    v_cndmask_b32_e64 v105, v105, v97, s[30:31]
    v_cndmask_b32_e32 v2, v2, v97, vcc
    v_cndmask_b32_e64 v3, v3, v97, s[2:3]
    v_cndmask_b32_e64 v4, v4, v97, s[4:5]
    v_cndmask_b32_e64 v5, v5, v97, s[6:7]
    v_cndmask_b32_e64 v14, v14, v97, s[8:9]
    v_cndmask_b32_e64 v15, v15, v97, s[10:11]
    v_cndmask_b32_e64 v16, v16, v97, s[12:13]
    v_cndmask_b32_e64 v17, v17, v97, s[14:15]
    v_cndmask_b32_e64 v10, v10, v97, s[16:17]
    v_cndmask_b32_e64 v11, v11, v97, s[18:19]
    v_cndmask_b32_e64 v12, v12, v97, s[20:21]
    v_cndmask_b32_e64 v13, v13, v97, s[22:23]
    v_cndmask_b32_e64 v106, v106, v97, s[24:25]
    v_cndmask_b32_e64 v107, v107, v97, s[26:27]
    v_cndmask_b32_e64 v108, v108, v97, s[28:29]
    v_cndmask_b32_e64 v109, v109, v97, s[30:31]

.Lh3_wide_attention_block_42:
    s_add_i32 s2, s92, 1
    s_max_i32 s8, s2, s79
    s_branch .Lh3_wide_attention_block_21

.Lh3_wide_attention_block_43:
    v_mov_b32_e32 v13, 0
    v_mov_b32_e32 v18, v171
    v_mov_b32_e32 v19, v171
    v_mov_b32_e32 v20, v171
    v_mov_b32_e32 v21, v171
    s_mov_b32 s89, 0xa8000
    zero_accumulators 24, 23, 22, 33, 32, 31, 30, 41, 40, 39, 38, 37
    zero_accumulators 36, 35, 34, 45, 44, 43, 42, 49, 48, 47, 46, 57
    zero_accumulators 56, 55, 54, 53, 52, 51, 50, 65, 64, 63, 62, 69
    zero_accumulators 68, 67, 66, 77, 76, 75, 74, 73, 72, 71, 70, 81
    zero_accumulators 80, 79, 78, 85, 84, 83, 82, 89, 88, 87, 86, 61
    zero_accumulators 60, 59, 58
    v_mov_b32_e32 v12, v13
    v_mov_b32_e32 v11, v13
    v_mov_b32_e32 v10, v13
    v_mov_b32_e32 v17, v13
    v_mov_b32_e32 v16, v13
    v_mov_b32_e32 v15, v13
    v_mov_b32_e32 v14, v13
    v_mov_b32_e32 v5, v13
    v_mov_b32_e32 v4, v13
    v_mov_b32_e32 v3, v13
    v_mov_b32_e32 v2, v13
    v_mov_b32_e32 v9, v13
    v_mov_b32_e32 v8, v13
    v_mov_b32_e32 v7, v13
    v_mov_b32_e32 v6, v13
    s_mov_b32 s6, s68
    v_mov_b32_e32 v90, v118
    s_cbranch_execnz .Lh3_wide_attention_block_36

.Lh3_wide_attention_block_44:
    v_exp_f32_e64 v92, -v110
    v_exp_f32_e64 v93, -v93
    rescale_accumulator_pairs 92, 58, 60, 82, 84, 70, 72, 66, 68, 50, 52, 46, 48
    rescale_accumulator_pairs 92, 34, 36, 30, 32
    v_pk_mul_f32 v[86:87], v[92:93], v[86:87] op_sel:[1,0]
    v_pk_mul_f32 v[88:89], v[92:93], v[88:89] op_sel:[1,0]
    v_pk_mul_f32 v[78:79], v[92:93], v[78:79] op_sel:[1,0]
    v_pk_mul_f32 v[80:81], v[92:93], v[80:81] op_sel:[1,0]
    v_pk_mul_f32 v[74:75], v[92:93], v[74:75] op_sel:[1,0]
    v_pk_mul_f32 v[76:77], v[92:93], v[76:77] op_sel:[1,0]
    v_pk_mul_f32 v[62:63], v[92:93], v[62:63] op_sel:[1,0]
    v_pk_mul_f32 v[64:65], v[92:93], v[64:65] op_sel:[1,0]
    v_pk_mul_f32 v[54:55], v[92:93], v[54:55] op_sel:[1,0]
    v_pk_mul_f32 v[56:57], v[92:93], v[56:57] op_sel:[1,0]
    v_pk_mul_f32 v[42:43], v[92:93], v[42:43] op_sel:[1,0]
    v_pk_mul_f32 v[44:45], v[92:93], v[44:45] op_sel:[1,0]
    v_pk_mul_f32 v[38:39], v[92:93], v[38:39] op_sel:[1,0]
    v_pk_mul_f32 v[40:41], v[92:93], v[40:41] op_sel:[1,0]
    v_pk_mul_f32 v[22:23], v[92:93], v[22:23] op_sel:[1,0]
    v_pk_mul_f32 v[24:25], v[92:93], v[24:25] op_sel:[1,0]
    s_branch .Lh3_wide_attention_block_36

.Lh3_wide_attention_block_45:
    v_mov_b32_e32 v13, 0
    s_movk_i32 s54, 0x4000
    s_mov_b32 s89, 0x150000
    s_mov_b64 s[2:3], 0
    v_mov_b32_e32 v12, v13
    v_mov_b32_e32 v11, v13
    v_mov_b32_e32 v10, v13
    v_mov_b32_e32 v17, v13
    v_mov_b32_e32 v16, v13
    v_mov_b32_e32 v15, v13
    v_mov_b32_e32 v14, v13
    v_mov_b32_e32 v5, v13
    v_mov_b32_e32 v4, v13
    v_mov_b32_e32 v3, v13
    v_mov_b32_e32 v2, v13
    v_mov_b32_e32 v9, v13
    v_mov_b32_e32 v8, v13
    v_mov_b32_e32 v7, v13
    v_mov_b32_e32 v6, v13

.Lh3_wide_attention_block_46:
    s_andn2_b64 vcc, exec, s[2:3]
    s_cbranch_vccz .Lh3_wide_attention_block_99

.Lh3_wide_attention_block_47:
    s_waitcnt vmcnt(0)
    v_smfmac_f32_16x16x64_bf16 v[18:21], v[26:29], v[10:17], v181
    s_barrier
    v_smfmac_f32_16x16x64_bf16 v[18:21], v[26:29], v[2:9], v181
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[86:89], v[160:163], v[14:17], v[86:89]
    ds_read_b128 v[96:99], v95 offset:3072
    ; Form the next K address ahead of the independent MFMA.
    v_add_u32_e32 v94, s89, v139
    v_mfma_f32_16x16x32_bf16 v[82:85], v[160:163], v[10:13], v[82:85]
    s_waitcnt lgkmcnt(2)
    buffer_load_dwordx4 v94, s[40:43], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[78:81], v[90:93], v[10:13], v[78:81]
    ds_read_b128 v[104:107], v95 offset:4096
    v_mfma_f32_16x16x32_bf16 v[74:77], v[90:93], v[14:17], v[74:77]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[70:73], v[156:159], v[14:17], v[70:73]
    ds_read_b128 v[90:93], v95 offset:5120
    v_mfma_f32_16x16x32_bf16 v[66:69], v[156:159], v[10:13], v[66:69]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[62:65], v[96:99], v[10:13], v[62:65]
    ds_read_b128 v[108:111], v95 offset:6144
    v_mfma_f32_16x16x32_bf16 v[96:99], v[96:99], v[14:17], v[58:61]
    s_mov_b32 m0, s6
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[100:103], v[104:107], v[14:17], v[54:57]
    s_nop 0
    ds_read_b128 v[58:61], v95 offset:7168
    v_mfma_f32_16x16x32_bf16 v[104:107], v[104:107], v[10:13], v[50:53]
    s_nop 2
    v_add_u32_e32 v50, s89, v169
    s_waitcnt lgkmcnt(2)
    buffer_load_dwordx4 v50, s[40:43], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[46:49], v[90:93], v[10:13], v[46:49]
    ds_read_b128 v[50:53], v95 offset:8192
    v_mfma_f32_16x16x32_bf16 v[42:45], v[90:93], v[14:17], v[42:45]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[90:93], v[108:111], v[14:17], v[38:41]
    ds_read_b128 v[54:57], v95 offset:9216
    v_mfma_f32_16x16x32_bf16 v[108:111], v[108:111], v[10:13], v[34:37]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[120:123], v[58:61], v[10:13], v[30:33]
    s_nop 2
    ds_read_b128 v[30:33], v95 offset:10240
    v_mfma_f32_16x16x32_bf16 v[124:127], v[58:61], v[14:17], v[22:25]
    s_mov_b32 m0, s5
    s_waitcnt lgkmcnt(2)
    s_setprio 1
    v_mfma_f32_16x16x32_bf16 v[10:13], v[50:53], v[6:9], v[86:89]
    ds_read_b128 v[34:37], v95 offset:11264
    v_mfma_f32_16x16x32_bf16 v[38:41], v[50:53], v[2:5], v[82:85]
    v_add_u32_e32 v14, s54, v118
    s_waitcnt lgkmcnt(2)
    buffer_load_dwordx4 v14, s[44:47], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[50:53], v[54:57], v[2:5], v[78:81]
    s_nop 2
    ds_read_b128 v[78:81], v95 offset:12288
    v_mfma_f32_16x16x32_bf16 v[14:17], v[54:57], v[6:9], v[74:77]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[22:25], v[30:33], v[6:9], v[70:73]
    s_nop 2
    ds_read_b128 v[70:73], v95 offset:13312
    v_mfma_f32_16x16x32_bf16 v[54:57], v[30:33], v[2:5], v[66:69]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[58:61], v[34:37], v[2:5], v[62:65]
    ds_read_b128 v[74:77], v95 offset:14336
    v_mfma_f32_16x16x32_bf16 v[30:33], v[34:37], v[6:9], v[96:99]
    s_mov_b32 m0, s4
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_16x16x32_bf16 v[34:37], v[78:81], v[6:9], v[100:103]
    ds_read_b128 v[82:85], v95 offset:15360
    v_mfma_f32_16x16x32_bf16 v[62:65], v[78:81], v[2:5], v[104:107]
    v_add_u32_e32 v66, s54, v119
    s_waitcnt lgkmcnt(2)
    buffer_load_dwordx4 v66, s[44:47], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[66:69], v[70:73], v[2:5], v[46:49]
    v_mfma_f32_16x16x32_bf16 v[42:45], v[70:73], v[6:9], v[42:45]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[46:49], v[74:77], v[6:9], v[90:93]
    v_mfma_f32_16x16x32_bf16 v[70:73], v[74:77], v[2:5], v[108:111]
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_16x16x32_bf16 v[74:77], v[82:85], v[2:5], v[120:123]
    v_mfma_f32_16x16x32_bf16 v[2:5], v[82:85], v[6:9], v[124:127]
    s_setprio 0
    s_cmp_ge_i32 s88, s58
    s_cbranch_scc1 .Lh3_wide_attention_block_49

.Lh3_wide_attention_block_48:
    s_mov_b64 s[2:3], src_shared_base
    s_add_i32 s88, s88, 1
    s_add_i32 s2, s88, s86
    s_mul_hi_u32 s5, s2, 0xaaaaaaab
    s_lshr_b32 s5, s5, 1
    s_add_i32 s4, s88, s87
    s_mul_i32 s5, s5, 3
    s_mul_i32 s8, s2, 0xa8000
    s_sub_i32 s2, s2, s5
    s_mul_hi_u32 s5, s4, 0xaaaaaaab
    s_lshr_b32 s5, s5, 1
    s_mul_i32 s5, s5, 3
    s_lshl_b32 s9, s4, 14
    s_mulk_i32 s2, 0x4400
    s_sub_i32 s4, s4, s5
    s_add_i32 s2, s59, s2
    s_mul_i32 s10, s4, 0x4800
    s_add_u32 s4, s68, s10
    s_waitcnt vmcnt(0)
    s_addc_u32 s5, s69, 0
    s_add_u32 s6, s2, 0x2200
    s_addc_u32 s7, s3, 0
    s_barrier
    s_cmp_lg_u64 s[6:7], 0
    v_add_u32_e32 v6, s8, v139
    s_mov_b32 m0, s2
    s_nop 0
    buffer_load_dwordx4 v6, s[40:43], 0 offen lds
    s_cselect_b32 s2, s6, -1
    s_cmp_lg_u64 s[4:5], 0
    v_add_u32_e32 v6, s8, v169
    s_mov_b32 m0, s2
    s_nop 0
    buffer_load_dwordx4 v6, s[40:43], 0 offen lds
    s_cselect_b32 s2, s4, -1
    v_add_u32_e32 v6, s9, v118
    s_mov_b32 m0, s2
    s_nop 0
    buffer_load_dwordx4 v6, s[44:47], 0 offen lds
    s_add_i32 s2, s63, s10
    s_add_u32 s2, s2, 0xd000
    s_addc_u32 s3, s3, 0
    s_cmp_lg_u64 s[2:3], 0
    s_cselect_b32 s2, s2, -1
    v_add_u32_e32 v6, s9, v119
    s_mov_b32 m0, s2
    s_nop 0
    buffer_load_dwordx4 v6, s[44:47], 0 offen lds
    s_cmp_eq_u32 s88, s58
    s_cbranch_scc0 .Lh3_wide_attention_block_48

.Lh3_wide_attention_block_49:
    v_add_f32_e32 v6, v18, v19
    s_waitcnt vmcnt(0)
    v_mov_b32_e32 v7, v6
    s_barrier
    s_nop 0
    v_and_b32_e32 v7, 0x7fffffff, v7
    v_cmp_gt_u32_e32 vcc, s77, v7
    s_and_saveexec_b64 s[2:3], vcc
    s_xor_b64 s[2:3], exec, s[2:3]
    v_rcp_f32_e32 v7, v6
    v_cmp_lt_f32_e32 vcc, 0, v6
    s_nop 1
    v_cndmask_b32_e32 v7, 0, v7, vcc
    s_andn2_saveexec_b64 s[2:3], s[2:3]
    s_cbranch_execz .Lh3_wide_attention_block_50
    v_mov_b32_e32 v7, 0x7fc00000

.Lh3_wide_attention_block_50:
    s_or_b64 exec, exec, s[2:3]
    v_add_f32_e32 v8, v20, v21
    s_mulk_i32 s85, 0x3300
    v_mul_f32_e32 v9, v7, v38
    v_mul_f32_e32 v18, v7, v39
    v_mul_f32_e32 v19, v7, v40
    v_mul_f32_e32 v20, v7, v41
    v_or_b32_e32 v6, s85, v182
    v_cvt_pk_bf16_f32 v19, v19, v20
    v_cvt_pk_bf16_f32 v18, v9, v18
    v_mul_f32_e32 v9, v7, v50
    v_mul_f32_e32 v20, v7, v51
    v_mul_f32_e32 v21, v7, v52
    v_mul_f32_e32 v38, v7, v53
    v_add_u32_e32 v6, v6, v197
    v_cvt_pk_bf16_f32 v21, v21, v38
    v_cvt_pk_bf16_f32 v20, v9, v20
    ds_write2_b64 v6, v[18:19], v[20:21] offset1:4
    v_mul_f32_e32 v9, v7, v54
    v_mul_f32_e32 v18, v7, v55
    v_mul_f32_e32 v19, v7, v56
    v_mul_f32_e32 v20, v7, v57
    v_cvt_pk_bf16_f32 v19, v19, v20
    v_cvt_pk_bf16_f32 v18, v9, v18
    v_mul_f32_e32 v9, v7, v58
    v_mul_f32_e32 v20, v7, v59
    v_mul_f32_e32 v21, v7, v60
    v_mul_f32_e32 v38, v7, v61
    v_cvt_pk_bf16_f32 v21, v21, v38
    v_cvt_pk_bf16_f32 v20, v9, v20
    ds_write2_b64 v6, v[18:19], v[20:21] offset0:8 offset1:12
    v_mul_f32_e32 v9, v7, v62
    v_mul_f32_e32 v18, v7, v63
    v_mul_f32_e32 v19, v7, v64
    v_mul_f32_e32 v20, v7, v65
    v_cvt_pk_bf16_f32 v19, v19, v20
    v_cvt_pk_bf16_f32 v18, v9, v18
    v_mul_f32_e32 v9, v7, v66
    v_mul_f32_e32 v20, v7, v67
    v_mul_f32_e32 v21, v7, v68
    v_mul_f32_e32 v38, v7, v69
    v_cvt_pk_bf16_f32 v21, v21, v38
    v_cvt_pk_bf16_f32 v20, v9, v20
    ds_write2_b64 v6, v[18:19], v[20:21] offset0:16 offset1:20
    v_mul_f32_e32 v9, v7, v70
    v_mul_f32_e32 v18, v7, v71
    v_mul_f32_e32 v19, v7, v72
    v_mul_f32_e32 v20, v7, v73
    v_cvt_pk_bf16_f32 v19, v19, v20
    v_cvt_pk_bf16_f32 v18, v9, v18
    v_mul_f32_e32 v9, v7, v74
    v_mul_f32_e32 v20, v7, v75
    v_mul_f32_e32 v21, v7, v76
    v_mul_f32_e32 v7, v7, v77
    v_cvt_pk_bf16_f32 v21, v21, v7
    v_mov_b32_e32 v7, v8
    v_cvt_pk_bf16_f32 v20, v9, v20
    v_and_b32_e32 v7, 0x7fffffff, v7
    v_cmp_gt_u32_e32 vcc, s77, v7
    ds_write2_b64 v6, v[18:19], v[20:21] offset0:24 offset1:28
    s_and_saveexec_b64 s[2:3], vcc
    s_xor_b64 s[2:3], exec, s[2:3]
    v_rcp_f32_e32 v7, v8
    v_cmp_lt_f32_e32 vcc, 0, v8
    s_nop 1
    v_cndmask_b32_e32 v7, 0, v7, vcc
    s_andn2_saveexec_b64 s[2:3], s[2:3]
    s_cbranch_execz .Lh3_wide_attention_block_51
    v_mov_b32_e32 v7, 0x7fc00000

.Lh3_wide_attention_block_51:
    s_or_b64 exec, exec, s[2:3]
    v_mul_f32_e32 v8, v7, v10
    v_mul_f32_e32 v10, v7, v11
    v_mul_f32_e32 v9, v7, v12
    v_mul_f32_e32 v11, v7, v13
    v_cvt_pk_bf16_f32 v9, v9, v11
    v_cvt_pk_bf16_f32 v8, v8, v10
    v_mul_f32_e32 v10, v7, v14
    v_mul_f32_e32 v12, v7, v15
    v_mul_f32_e32 v11, v7, v16
    v_mul_f32_e32 v13, v7, v17
    v_cvt_pk_bf16_f32 v11, v11, v13
    v_cvt_pk_bf16_f32 v10, v10, v12
    v_add_u32_e32 v6, 0x1000, v6
    ds_write2_b64 v6, v[8:9], v[10:11] offset0:32 offset1:36
    v_mul_f32_e32 v8, v7, v22
    v_mul_f32_e32 v10, v7, v23
    v_mul_f32_e32 v9, v7, v24
    v_mul_f32_e32 v11, v7, v25
    v_cvt_pk_bf16_f32 v9, v9, v11
    v_cvt_pk_bf16_f32 v8, v8, v10
    v_mul_f32_e32 v10, v7, v30
    v_mul_f32_e32 v12, v7, v31
    v_mul_f32_e32 v11, v7, v32
    v_mul_f32_e32 v13, v7, v33
    v_cvt_pk_bf16_f32 v11, v11, v13
    v_cvt_pk_bf16_f32 v10, v10, v12
    ds_write2_b64 v6, v[8:9], v[10:11] offset0:40 offset1:44
    v_mul_f32_e32 v8, v7, v34
    v_mul_f32_e32 v10, v7, v35
    v_mul_f32_e32 v9, v7, v36
    v_mul_f32_e32 v11, v7, v37
    v_cvt_pk_bf16_f32 v9, v9, v11
    v_cvt_pk_bf16_f32 v8, v8, v10
    v_mul_f32_e32 v10, v7, v42
    v_mul_f32_e32 v12, v7, v43
    v_mul_f32_e32 v11, v7, v44
    v_mul_f32_e32 v13, v7, v45
    v_cvt_pk_bf16_f32 v11, v11, v13
    v_cvt_pk_bf16_f32 v10, v10, v12
    ds_write2_b64 v6, v[8:9], v[10:11] offset0:48 offset1:52
    v_mul_f32_e32 v8, v7, v46
    v_mul_f32_e32 v10, v7, v47
    v_mul_f32_e32 v9, v7, v48
    v_mul_f32_e32 v11, v7, v49
    v_cvt_pk_bf16_f32 v8, v8, v10
    v_mul_f32_e32 v2, v7, v2
    v_mul_f32_e32 v10, v7, v3
    v_mul_f32_e32 v3, v7, v4
    v_mul_f32_e32 v4, v7, v5
    v_cvt_pk_bf16_f32 v9, v9, v11
    v_cvt_pk_bf16_f32 v3, v3, v4
    v_cvt_pk_bf16_f32 v2, v2, v10
    ds_write2_b64 v6, v[8:9], v[2:3] offset0:56 offset1:60
    v_mbcnt_lo_u32_b32 v2, -1, 0
    v_mbcnt_hi_u32_b32 v2, -1, v2
    s_nop 0
    v_ashrrev_i32_e32 v3, 31, v2
    v_lshrrev_b32_e32 v3, 28, v3
    v_add_u32_e32 v3, v2, v3
    v_ashrrev_i32_e32 v5, 4, v3
    v_and_b32_e32 v3, -16, v3
    v_sub_u32_e32 v4, v2, v3
    v_lshlrev_b32_e32 v2, 3, v4
    v_ashrrev_i32_e32 v3, 31, v2
    v_add_u32_e32 v6, s84, v5
    v_lshl_add_u32 v4, v4, 4, s85
    v_lshl_add_u64 v[2:3], v[2:3], 1, v[116:117]
    v_cmp_gt_u32_e32 vcc, s78, v6
    s_and_saveexec_b64 s[2:3], vcc
    s_cbranch_execz .Lh3_wide_attention_block_52
    v_mad_u64_u32 v[8:9], s[4:5], v5, s74, v[4:5]
    ds_read_b128 v[10:13], v8
    v_mad_u64_u32 v[8:9], s[4:5], v6, s75, v[2:3]
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v[8:9], v[10:13], off

.Lh3_wide_attention_block_52:
    s_or_b64 exec, exec, s[2:3]
    v_add_u32_e32 v7, 4, v6
    v_cmp_gt_u32_e32 vcc, s78, v7
    s_and_saveexec_b64 s[2:3], vcc
    s_cbranch_execz .Lh3_wide_attention_block_53
    v_mad_u64_u32 v[8:9], s[4:5], v5, s74, v[4:5]
    ds_read_b128 v[10:13], v8 offset:1088
    v_mad_u64_u32 v[8:9], s[4:5], v7, s75, v[2:3]
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v[8:9], v[10:13], off

.Lh3_wide_attention_block_53:
    s_or_b64 exec, exec, s[2:3]
    v_add_u32_e32 v7, 8, v6
    v_cmp_gt_u32_e32 vcc, s78, v7
    s_and_saveexec_b64 s[2:3], vcc
    s_cbranch_execz .Lh3_wide_attention_block_54
    v_mad_u64_u32 v[8:9], s[4:5], v5, s74, v[4:5]
    ds_read_b128 v[10:13], v8 offset:2176
    v_mad_u64_u32 v[8:9], s[4:5], v7, s75, v[2:3]
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v[8:9], v[10:13], off

.Lh3_wide_attention_block_54:
    s_or_b64 exec, exec, s[2:3]
    v_add_u32_e32 v7, 12, v6
    v_cmp_gt_u32_e32 vcc, s78, v7
    s_and_saveexec_b64 s[2:3], vcc
    s_cbranch_execz .Lh3_wide_attention_block_55
    v_mad_u64_u32 v[8:9], s[4:5], v5, s74, v[4:5]
    ds_read_b128 v[10:13], v8 offset:3264
    v_mad_u64_u32 v[8:9], s[4:5], v7, s75, v[2:3]
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v[8:9], v[10:13], off

.Lh3_wide_attention_block_55:
    s_or_b64 exec, exec, s[2:3]
    v_add_u32_e32 v7, 16, v6
    v_cmp_gt_u32_e32 vcc, s78, v7
    s_and_saveexec_b64 s[2:3], vcc
    s_cbranch_execz .Lh3_wide_attention_block_56
    v_mad_u64_u32 v[8:9], s[4:5], v5, s74, v[4:5]
    ds_read_b128 v[10:13], v8 offset:4352
    v_mad_u64_u32 v[8:9], s[4:5], v7, s75, v[2:3]
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v[8:9], v[10:13], off

.Lh3_wide_attention_block_56:
    s_or_b64 exec, exec, s[2:3]
    v_add_u32_e32 v7, 20, v6
    v_cmp_gt_u32_e32 vcc, s78, v7
    s_and_saveexec_b64 s[2:3], vcc
    s_cbranch_execz .Lh3_wide_attention_block_57
    v_mad_u64_u32 v[8:9], s[4:5], v5, s74, v[4:5]
    ds_read_b128 v[10:13], v8 offset:5440
    v_mad_u64_u32 v[8:9], s[4:5], v7, s75, v[2:3]
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v[8:9], v[10:13], off

.Lh3_wide_attention_block_57:
    s_or_b64 exec, exec, s[2:3]
    v_add_u32_e32 v7, 24, v6
    v_cmp_gt_u32_e32 vcc, s78, v7
    s_and_saveexec_b64 s[2:3], vcc
    s_cbranch_execz .Lh3_wide_attention_block_58
    v_mad_u64_u32 v[8:9], s[4:5], v5, s74, v[4:5]
    ds_read_b128 v[10:13], v8 offset:6528
    v_mad_u64_u32 v[8:9], s[4:5], v7, s75, v[2:3]
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v[8:9], v[10:13], off

.Lh3_wide_attention_block_58:
    s_or_b64 exec, exec, s[2:3]
    v_add_u32_e32 v7, 28, v5
    v_add_u32_e32 v5, 28, v6
    v_cmp_gt_u32_e64 s[2:3], s78, v5

.Lh3_wide_attention_block_59:
    ; Use the comparison bits of the active lanes to vote on a frame update.
    s_and_b64 vcc, exec, s[64:65]
    s_cbranch_vccz .Lh3_wide_attention_block_97
    v_readfirstlane_b32 s28, v0
    s_and_b64 vcc, exec, s[66:67]
    s_cbranch_vccz .Lh3_wide_attention_block_97
    global_load_dword v4, v[114:115], off
    s_lshr_b32 s65, s28, 6
    s_and_b64 s[2:3], s[60:61], exec
    s_cselect_b32 s64, s83, 0xffffffd0
    v_add_u32_e32 v2, s64, v192
    v_cmp_gt_u32_e64 s[12:13], s78, v2
    v_add_u32_e32 v3, 4, v2
    v_add_u32_e32 v5, 8, v2
    v_cndmask_b32_e64 v6, 0, v2, s[12:13]
    v_add_u32_e32 v7, 12, v2
    v_add_u32_e32 v8, 16, v2
    v_add_u32_e32 v9, 20, v2
    v_add_u32_e32 v10, 24, v2
    v_add_u32_e32 v11, 28, v2
    v_add_u32_e32 v12, 32, v2
    v_add_u32_e32 v13, 36, v2
    v_add_u32_e32 v14, 40, v2
    v_add_u32_e32 v2, 44, v2
    v_cmp_gt_u32_e64 s[24:25], s78, v3
    v_cmp_gt_u32_e64 s[2:3], s78, v2
    s_mov_b32 s63, s55
    v_cndmask_b32_e64 v15, 0, v3, s[24:25]
    v_cndmask_b32_e64 v25, 0, v2, s[2:3]
    s_lshl_b64 s[26:27], s[62:63], 8
    v_cmp_gt_u32_e64 s[22:23], s78, v5
    v_cmp_gt_u32_e64 s[20:21], s78, v7
    v_cmp_gt_u32_e64 s[18:19], s78, v8
    v_cndmask_b32_e64 v16, 0, v5, s[22:23]
    v_cmp_gt_u32_e64 s[16:17], s78, v9
    v_cmp_gt_u32_e64 s[14:15], s78, v10
    v_cmp_gt_u32_e64 s[10:11], s78, v11
    v_cmp_gt_u32_e64 s[8:9], s78, v12
    v_cmp_gt_u32_e64 s[6:7], s78, v13
    v_cmp_gt_u32_e64 s[4:5], s78, v14
    v_cndmask_b32_e64 v17, 0, v7, s[20:21]
    v_cndmask_b32_e64 v18, 0, v8, s[18:19]
    v_cndmask_b32_e64 v19, 0, v9, s[16:17]
    v_cndmask_b32_e64 v20, 0, v10, s[14:15]
    v_cndmask_b32_e64 v21, 0, v11, s[10:11]
    v_cndmask_b32_e64 v22, 0, v12, s[8:9]
    v_cndmask_b32_e64 v23, 0, v13, s[6:7]
    v_cndmask_b32_e64 v24, 0, v14, s[4:5]
    s_lshl_b32 s54, s65, 3
    s_cmp_le_u32 s80, s54
    s_mov_b32 s42, 0
    s_waitcnt vmcnt(0)
    v_mul_hi_u32 v3, v4, s73
    v_mul_lo_u32 v2, v4, s73
    v_lshl_add_u64 v[2:3], s[36:37], 0, v[2:3]
    v_mul_hi_u32 v5, v4, s75
    v_mul_lo_u32 v4, v4, s75
    v_lshl_add_u64 v[2:3], v[2:3], 0, s[26:27]
    v_lshl_add_u64 v[4:5], s[34:35], 0, v[4:5]
    v_lshl_add_u64 v[2:3], v[2:3], 0, v[170:171]
    v_lshl_add_u64 v[254:255], v[4:5], 0, s[26:27]
    v_mad_u64_u32 v[4:5], s[26:27], v6, s73, v[2:3]
    v_mad_u64_u32 v[6:7], s[26:27], v15, s73, v[2:3]
    v_mad_u64_u32 v[8:9], s[26:27], v16, s73, v[2:3]
    v_mad_u64_u32 v[10:11], s[26:27], v17, s73, v[2:3]
    v_mad_u64_u32 v[12:13], s[26:27], v18, s73, v[2:3]
    v_mad_u64_u32 v[14:15], s[26:27], v19, s73, v[2:3]
    v_mad_u64_u32 v[16:17], s[26:27], v20, s73, v[2:3]
    v_mad_u64_u32 v[18:19], s[26:27], v21, s73, v[2:3]
    v_mad_u64_u32 v[54:55], s[26:27], v22, s73, v[2:3]
    v_mad_u64_u32 v[56:57], s[26:27], v23, s73, v[2:3]
    v_mad_u64_u32 v[58:59], s[26:27], v24, s73, v[2:3]
    v_mad_u64_u32 v[2:3], s[26:27], v25, s73, v[2:3]
    global_load_dwordx4 v[50:53], v[4:5], off
    global_load_dwordx4 v[46:49], v[6:7], off
    global_load_dwordx4 v[42:45], v[8:9], off
    global_load_dwordx4 v[38:41], v[10:11], off
    global_load_dwordx4 v[34:37], v[12:13], off
    global_load_dwordx4 v[30:33], v[14:15], off
    global_load_dwordx4 v[22:25], v[16:17], off
    s_nop 0
    global_load_dwordx4 v[18:21], v[18:19], off
    s_nop 0
    global_load_dwordx4 v[14:17], v[54:55], off
    global_load_dwordx4 v[10:13], v[56:57], off
    global_load_dwordx4 v[6:9], v[58:59], off
    s_nop 0
    global_load_dwordx4 v[2:5], v[2:3], off
    s_cbranch_scc1 .Lh3_wide_attention_block_60
    s_not_b64 s[26:27], s[54:55]
    s_add_u32 s26, s26, s80
    s_addc_u32 s27, s27, s82
    s_mulk_i32 s27, 0x2a00
    s_mul_hi_u32 s29, s26, 0x2a00
    s_add_i32 s27, s29, s27
    s_mulk_i32 s26, 0x2a00
    s_bitset1_b32 s26, 8
    s_cmp_eq_u32 s27, 0
    s_cselect_b32 s42, s26, -1

.Lh3_wide_attention_block_60:
    s_lshl_b64 s[26:27], s[62:63], 7
    s_mul_i32 s30, s81, 0x2a00
    s_mul_hi_u32 s29, s81, 0x2a00
    s_add_u32 s30, s38, s30
    s_addc_u32 s29, s39, s29
    s_lshl_b64 s[26:27], s[26:27], 1
    s_add_u32 s30, s30, s26
    s_addc_u32 s31, s29, s27
    s_cmpk_gt_u32 s28, 0xff
    s_cselect_b64 s[26:27], -1, 0
    s_cmpk_lt_u32 s28, 0x100
    s_mul_i32 s40, s54, 0x2a00
    s_cselect_b64 s[28:29], -1, 0
    s_mul_hi_u32 s41, s54, 0x2a00
    s_add_u32 s40, s30, s40
    s_addc_u32 s41, s31, s41
    s_ashr_i32 s59, s58, 31
    s_add_i32 s46, s80, 0xbe
    s_lshl_b64 s[30:31], s[58:59], 14
    s_add_u32 s30, s30, 0x4000
    s_addc_u32 s31, s31, 0
    v_mov_b64_e32 v[54:55], s[56:57]
    v_cmp_lt_u64_e32 vcc, s[30:31], v[54:55]
    s_and_b64 s[44:45], vcc, exec
    s_cselect_b32 s47, s30, -1
    s_cmp_eq_u32 s81, 0
    s_mul_i32 s30, s62, 0x251
    s_cselect_b32 s31, 0, 0x24f
    s_add_i32 s54, s30, s31
    s_lshl_b64 s[30:31], s[54:55], 14
    s_add_u32 s44, s52, s30
    s_addc_u32 s45, s53, s31
    s_cmp_gt_u32 s46, 63
    s_mul_i32 s54, s65, 0x220
    s_cselect_b32 s46, s47, 0
    s_add_i32 s30, s54, 0x1100
    s_mov_b32 s31, s55
    s_lshl_b32 s59, s54, 1
    s_lshl_b64 s[30:31], s[30:31], 1
    s_mov_b64 s[60:61], src_shared_base
    s_add_u32 s62, 0, s30
    s_addc_u32 s63, s61, s31
    s_cmp_lg_u64 s[62:63], 0
    s_mov_b32 m0, s59
    s_nop 0
    buffer_load_dwordx4 v139, s[40:43], 0 offen lds
    s_cselect_b32 s60, s62, -1
    s_mov_b32 m0, s60
    s_nop 0
    buffer_load_dwordx4 v169, s[40:43], 0 offen lds
    s_lshl_b32 s60, s65, 11
    s_add_i32 s66, s60, 0xcc00
    s_add_u32 s62, s66, 0x400
    s_addc_u32 s63, s61, 0
    s_mov_b32 s47, s43
    v_or_b32_e32 v188, s60, v1
    s_mov_b32 m0, s66
    s_nop 0
    buffer_load_dwordx4 v188, s[44:47], 0 offen lds
    s_cmp_lg_u64 s[62:63], 0
    s_cselect_b32 s62, s62, -1
    v_or_b32_e32 v189, 0x400, v188
    s_mov_b32 m0, s62
    s_nop 0
    buffer_load_dwordx4 v189, s[44:47], 0 offen lds
    s_and_b64 vcc, exec, s[28:29]
    s_cbranch_vccnz .Lh3_wide_attention_block_61
    s_lshl_b64 s[68:69], s[54:55], 1
    s_mov_b64 s[62:63], src_shared_base
    s_add_u32 s68, 0x4400, s68
    s_addc_u32 s69, s63, s69
    s_cmp_lg_u64 s[68:69], 0
    s_cselect_b32 s62, s68, -1
    s_add_u32 s30, 0x4400, s30
    s_addc_u32 s31, s63, s31
    s_mov_b32 m0, s62
    s_nop 0
    buffer_load_dwordx4 v180, s[40:43], 0 offen lds
    s_cmp_lg_u64 s[30:31], 0
    s_cselect_b32 s30, s30, -1
    v_add_u32_e32 v54, 0xa8010, v139
    s_mov_b32 m0, s30
    s_nop 0
    buffer_load_dwordx4 v54, s[40:43], 0 offen lds

.Lh3_wide_attention_block_61:
    s_mul_i32 s30, s65, 0x1200
    s_add_i32 s30, s30, 0x1a400
    v_or_b32_e32 v54, s30, v193
    v_lshl_or_b32 v55, v168, 1, s30
    v_add3_u32 v54, v54, v194, v195
    s_waitcnt vmcnt(11)
    v_cndmask_b32_e64 v53, 0, v53, s[12:13]
    v_cndmask_b32_e64 v52, 0, v52, s[12:13]
    v_cndmask_b32_e64 v51, 0, v51, s[12:13]
    v_cndmask_b32_e64 v50, 0, v50, s[12:13]
    v_add_u32_e32 v55, v55, v196
    s_waitcnt vmcnt(10)
    v_cndmask_b32_e64 v49, 0, v49, s[24:25]
    v_cndmask_b32_e64 v48, 0, v48, s[24:25]
    v_cndmask_b32_e64 v47, 0, v47, s[24:25]
    v_cndmask_b32_e64 v46, 0, v46, s[24:25]
    s_waitcnt vmcnt(9)
    v_cndmask_b32_e64 v45, 0, v45, s[22:23]
    v_cndmask_b32_e64 v44, 0, v44, s[22:23]
    v_cndmask_b32_e64 v43, 0, v43, s[22:23]
    v_cndmask_b32_e64 v42, 0, v42, s[22:23]
    s_waitcnt vmcnt(8)
    v_cndmask_b32_e64 v41, 0, v41, s[20:21]
    v_cndmask_b32_e64 v40, 0, v40, s[20:21]
    v_cndmask_b32_e64 v39, 0, v39, s[20:21]
    v_cndmask_b32_e64 v38, 0, v38, s[20:21]
    ds_write_b128 v55, v[50:53]
    ds_write_b128 v55, v[46:49] offset:1152
    ds_write_b128 v55, v[42:45] offset:2304
    ds_write_b128 v55, v[38:41] offset:3456
    s_waitcnt vmcnt(7)
    v_cndmask_b32_e64 v37, 0, v37, s[18:19]
    v_cndmask_b32_e64 v36, 0, v36, s[18:19]
    v_cndmask_b32_e64 v35, 0, v35, s[18:19]
    v_cndmask_b32_e64 v34, 0, v34, s[18:19]
    s_waitcnt vmcnt(6)
    v_cndmask_b32_e64 v33, 0, v33, s[16:17]
    v_cndmask_b32_e64 v32, 0, v32, s[16:17]
    v_cndmask_b32_e64 v31, 0, v31, s[16:17]
    v_cndmask_b32_e64 v30, 0, v30, s[16:17]
    s_waitcnt vmcnt(5)
    v_cndmask_b32_e64 v25, 0, v25, s[14:15]
    v_cndmask_b32_e64 v24, 0, v24, s[14:15]
    v_cndmask_b32_e64 v23, 0, v23, s[14:15]
    v_cndmask_b32_e64 v22, 0, v22, s[14:15]
    s_waitcnt vmcnt(4)
    v_cndmask_b32_e64 v21, 0, v21, s[10:11]
    v_cndmask_b32_e64 v20, 0, v20, s[10:11]
    v_cndmask_b32_e64 v19, 0, v19, s[10:11]
    v_cndmask_b32_e64 v18, 0, v18, s[10:11]
    v_mov_b32_e32 v182, v194
    v_mov_b32_e32 v1, v196
    ds_read_b128 v[194:197], v54
    ds_read_b128 v[198:201], v54 offset:64
    ds_read_b128 v[202:205], v54 offset:128
    ds_read_b128 v[206:209], v54 offset:192
    ds_write_b128 v55, v[34:37]
    ds_write_b128 v55, v[30:33] offset:1152
    ds_write_b128 v55, v[22:25] offset:2304
    ds_write_b128 v55, v[18:21] offset:3456
    s_waitcnt vmcnt(3)
    v_cndmask_b32_e64 v17, 0, v17, s[8:9]
    v_cndmask_b32_e64 v16, 0, v16, s[8:9]
    v_cndmask_b32_e64 v15, 0, v15, s[8:9]
    v_cndmask_b32_e64 v14, 0, v14, s[8:9]
    s_waitcnt vmcnt(2)
    v_cndmask_b32_e64 v13, 0, v13, s[6:7]
    v_cndmask_b32_e64 v12, 0, v12, s[6:7]
    v_cndmask_b32_e64 v11, 0, v11, s[6:7]
    v_cndmask_b32_e64 v10, 0, v10, s[6:7]
    s_waitcnt vmcnt(1)
    v_cndmask_b32_e64 v9, 0, v9, s[4:5]
    v_cndmask_b32_e64 v8, 0, v8, s[4:5]
    v_cndmask_b32_e64 v7, 0, v7, s[4:5]
    v_cndmask_b32_e64 v6, 0, v6, s[4:5]
    s_waitcnt vmcnt(0)
    v_cndmask_b32_e64 v5, 0, v5, s[2:3]
    v_cndmask_b32_e64 v4, 0, v4, s[2:3]
    v_cndmask_b32_e64 v3, 0, v3, s[2:3]
    v_cndmask_b32_e64 v2, 0, v2, s[2:3]
    ds_read_b128 v[210:213], v54
    ds_read_b128 v[214:217], v54 offset:64
    ds_read_b128 v[218:221], v54 offset:128
    ds_read_b128 v[222:225], v54 offset:192
    ds_write_b128 v55, v[14:17]
    ds_write_b128 v55, v[10:13] offset:1152
    ds_write_b128 v55, v[6:9] offset:2304
    ds_write_b128 v55, v[2:5] offset:3456
    ds_read_b128 v[226:229], v54
    ds_read_b128 v[230:233], v54 offset:64
    ds_read_b128 v[234:237], v54 offset:128
    ds_read_b128 v[238:241], v54 offset:192
    scratch_load_dword v190, off, off
    v_cndmask_b32_e64 v2, 0, 1, s[28:29]
    s_andn2_b64 vcc, exec, s[28:29]
    s_mov_b64 s[2:3], -1
    s_waitcnt vmcnt(0)
    s_cbranch_vccnz .Lh3_wide_attention_block_62
    s_waitcnt vmcnt(0)
    s_mov_b64 s[2:3], 0

.Lh3_wide_attention_block_62:
    s_andn2_b64 vcc, exec, s[2:3]
    s_cbranch_vccnz .Lh3_wide_attention_block_63
    s_waitcnt vmcnt(2)

.Lh3_wide_attention_block_63:
    s_and_b64 s[2:3], s[26:27], exec
    s_barrier
    s_cselect_b32 s67, 2, 1
    v_cndmask_b32_e64 v3, 0, 1, s[26:27]
    s_cmpk_gt_i32 s64, 0xffd0
    v_cmp_ne_u32_e32 vcc, 1, v2
    v_readfirstlane_b32 s68, v3
    s_cselect_b32 s69, s58, 0
    s_and_b64 vcc, exec, vcc
    s_mov_b64 s[2:3], -1
    s_cbranch_vccnz .Lh3_wide_attention_block_71
    s_mov_b64 s[2:3], src_shared_base
    s_lshl_b64 s[62:63], s[54:55], 1
    s_add_u32 s2, 0x4400, s62
    s_addc_u32 s3, s3, s63
    s_add_i32 s5, s59, 0x6600
    s_add_i32 s4, s60, 0xd000
    s_cmp_lg_u64 s[2:3], 0
    s_waitcnt lgkmcnt(14)
    v_mov_b32_e32 v135, v190
    s_cselect_b32 s2, s2, -1
    s_mov_b32 m0, s2
    v_mov_b32_e32 v144, v183
    ds_read_b128 v[146:149], v135
    ds_read_b128 v[140:143], v135 offset:1024
    s_mov_b32 s8, 0
    s_cmp_lt_i32 s69, 1
    v_mov_b32_e32 v41, 0
    s_cbranch_scc1 .Lh3_wide_attention_block_75
    v_mov_b32_e32 v6, 0
    s_add_i32 s71, s80, -1
    s_mov_b32 s9, 0xa8000
    s_mul_i32 s70, s67, 0xa8000
    s_mov_b64 s[2:3], 0
    s_movk_i32 s81, 0x201
    zero_accumulators 30, 31, 32, 33, 34, 35, 36, 37, 126, 127, 128, 129
    zero_accumulators 130, 131, 132, 133, 122, 123, 124, 125, 114, 115, 116, 117
    zero_accumulators 110, 111, 112, 113, 118, 119, 120, 121, 102, 103, 104, 105
    zero_accumulators 106, 107, 108, 109, 98, 99, 100, 101, 90, 91, 92, 93
    zero_accumulators 86, 87, 88, 89, 94, 95, 96, 97, 78, 79, 80, 81
    zero_accumulators 82, 83, 84, 85, 74, 75, 76, 77, 66, 67, 68, 69
    zero_accumulators 62, 63, 64, 65, 70, 71, 72, 73, 54, 55, 56, 57
    zero_accumulators 58, 59, 60, 61, 50, 51, 52, 53, 46, 47, 48, 49
    zero_accumulators 42, 43, 44, 45, 38, 39, 40, 41
    v_mov_b32_e32 v134, 0xc29cd760
    v_mov_b32_e32 v136, 0xc29cd760
    v_mov_b32_e32 v138, 0xc29cd760
    s_mov_b32 s6, s66
    s_mov_b32 s10, 0
    s_mov_b32 s82, 0
    v_mov_b32_e32 v7, v6
    v_mov_b32_e32 v8, v6
    v_mov_b32_e32 v9, v6
    v_mov_b32_e32 v14, v6
    v_mov_b32_e32 v15, v6
    v_mov_b32_e32 v16, v6
    v_mov_b32_e32 v17, v6
    v_mov_b32_e32 v10, v6
    v_mov_b32_e32 v11, v6
    v_mov_b32_e32 v12, v6
    v_mov_b32_e32 v13, v6
    v_mov_b32_e32 v2, v6
    v_mov_b32_e32 v3, v6
    v_mov_b32_e32 v4, v6
    v_mov_b32_e32 v5, v6
    v_mov_b32_e32 v22, v6
    v_mov_b32_e32 v23, v6
    v_mov_b32_e32 v24, v6
    v_mov_b32_e32 v25, v6
    v_mov_b32_e32 v18, v6
    v_mov_b32_e32 v19, v6
    v_mov_b32_e32 v20, v6
    v_mov_b32_e32 v21, v6
    s_branch .Lh3_wide_attention_block_2

.Lh3_wide_attention_block_64:
    v_exp_f32_e64 v145, -v165
    v_exp_f32_e64 v150, -v164
    v_exp_f32_e64 v137, -v137
    v_pk_mul_f32 v[126:127], v[144:145], v[126:127] op_sel:[1,0]
    v_pk_mul_f32 v[128:129], v[144:145], v[128:129] op_sel:[1,0]
    v_pk_mul_f32 v[114:115], v[144:145], v[114:115] op_sel:[1,0]
    v_pk_mul_f32 v[116:117], v[144:145], v[116:117] op_sel:[1,0]
    v_pk_mul_f32 v[102:103], v[144:145], v[102:103] op_sel:[1,0]
    v_pk_mul_f32 v[104:105], v[144:145], v[104:105] op_sel:[1,0]
    v_pk_mul_f32 v[90:91], v[144:145], v[90:91] op_sel:[1,0]
    v_pk_mul_f32 v[92:93], v[144:145], v[92:93] op_sel:[1,0]
    v_pk_mul_f32 v[78:79], v[144:145], v[78:79] op_sel:[1,0]
    v_pk_mul_f32 v[80:81], v[144:145], v[80:81] op_sel:[1,0]
    v_pk_mul_f32 v[66:67], v[144:145], v[66:67] op_sel:[1,0]
    v_pk_mul_f32 v[68:69], v[144:145], v[68:69] op_sel:[1,0]
    v_pk_mul_f32 v[54:55], v[144:145], v[54:55] op_sel:[1,0]
    v_pk_mul_f32 v[56:57], v[144:145], v[56:57] op_sel:[1,0]
    v_pk_mul_f32 v[46:47], v[144:145], v[46:47] op_sel:[1,0]
    v_pk_mul_f32 v[48:49], v[144:145], v[48:49] op_sel:[1,0]
    rescale_accumulator_pairs 150, 130, 132, 110, 112, 106, 108, 86, 88, 82, 84, 62, 64
    rescale_accumulator_pairs 150, 58, 60, 42, 44
    v_pk_mul_f32 v[122:123], v[136:137], v[122:123] op_sel:[1,0]
    v_pk_mul_f32 v[124:125], v[136:137], v[124:125] op_sel:[1,0]
    v_pk_mul_f32 v[118:119], v[136:137], v[118:119] op_sel:[1,0]
    v_pk_mul_f32 v[120:121], v[136:137], v[120:121] op_sel:[1,0]
    v_pk_mul_f32 v[98:99], v[136:137], v[98:99] op_sel:[1,0]
    v_pk_mul_f32 v[100:101], v[136:137], v[100:101] op_sel:[1,0]
    v_pk_mul_f32 v[94:95], v[136:137], v[94:95] op_sel:[1,0]
    v_pk_mul_f32 v[96:97], v[136:137], v[96:97] op_sel:[1,0]
    v_pk_mul_f32 v[74:75], v[136:137], v[74:75] op_sel:[1,0]
    v_pk_mul_f32 v[76:77], v[136:137], v[76:77] op_sel:[1,0]
    v_pk_mul_f32 v[70:71], v[136:137], v[70:71] op_sel:[1,0]
    v_pk_mul_f32 v[72:73], v[136:137], v[72:73] op_sel:[1,0]
    v_pk_mul_f32 v[50:51], v[136:137], v[50:51] op_sel:[1,0]
    v_pk_mul_f32 v[52:53], v[136:137], v[52:53] op_sel:[1,0]
    v_pk_mul_f32 v[38:39], v[136:137], v[38:39] op_sel:[1,0]
    v_pk_mul_f32 v[40:41], v[136:137], v[40:41] op_sel:[1,0]
    s_branch .Lh3_wide_attention_block_3

.Lh3_wide_attention_block_65:
    s_cmp_lt_i32 s82, s79
    s_cbranch_scc1 .Lh3_wide_attention_block_66
    v_mbcnt_lo_u32_b32 v135, -1, 0
    v_mbcnt_hi_u32_b32 v135, -1, v135
    s_nop 0
    v_ashrrev_i32_e32 v137, 4, v135
    v_lshrrev_b32_e32 v135, 31, v135
    v_add_u32_e32 v135, v137, v135
    v_lshrrev_b32_e32 v135, 1, v135
    v_sub_u32_e32 v135, 0, v135
    v_lshlrev_b32_e32 v135, 3, v135
    v_lshlrev_b32_e32 v137, 2, v137
    v_sub_u32_e32 v135, v135, v137
    v_add_u32_e32 v135, s71, v135
    v_cmp_gt_i32_e32 vcc, 0, v135
    v_mov_b32_e32 v137, 0xf2177617
    v_cmp_gt_i32_e64 s[2:3], 1, v135
    v_cmp_gt_i32_e64 s[4:5], 2, v135
    v_cmp_gt_i32_e64 s[6:7], 3, v135
    v_cmp_gt_i32_e64 s[8:9], 8, v135
    v_cmp_gt_i32_e64 s[10:11], 9, v135
    v_cmp_gt_i32_e64 s[12:13], 10, v135
    v_cmp_gt_i32_e64 s[14:15], 11, v135
    v_cmp_gt_i32_e64 s[16:17], 32, v135
    v_cmp_gt_i32_e64 s[18:19], 33, v135
    v_cmp_gt_i32_e64 s[20:21], 34, v135
    v_cmp_gt_i32_e64 s[22:23], 35, v135
    v_cmp_gt_i32_e64 s[24:25], 40, v135
    v_cmp_gt_i32_e64 s[26:27], 41, v135
    v_cmp_gt_i32_e64 s[28:29], 42, v135
    v_cmp_gt_i32_e64 s[30:31], 43, v135
    v_cndmask_b32_e32 v156, v156, v137, vcc
    v_cndmask_b32_e64 v157, v157, v137, s[2:3]
    v_cndmask_b32_e64 v158, v158, v137, s[4:5]
    v_cndmask_b32_e64 v159, v159, v137, s[6:7]
    v_cndmask_b32_e64 v18, v18, v137, s[8:9]
    v_cndmask_b32_e64 v19, v19, v137, s[10:11]
    v_cndmask_b32_e64 v20, v20, v137, s[12:13]
    v_cndmask_b32_e64 v21, v21, v137, s[14:15]
    v_cndmask_b32_e64 v160, v160, v137, s[16:17]
    v_cndmask_b32_e64 v161, v161, v137, s[18:19]
    v_cndmask_b32_e64 v162, v162, v137, s[20:21]
    v_cndmask_b32_e64 v163, v163, v137, s[22:23]
    v_cndmask_b32_e64 v148, v148, v137, s[24:25]
    v_cndmask_b32_e64 v149, v149, v137, s[26:27]
    v_cndmask_b32_e64 v150, v150, v137, s[28:29]
    v_cndmask_b32_e64 v151, v151, v137, s[30:31]
    v_cndmask_b32_e32 v10, v10, v137, vcc
    v_cndmask_b32_e64 v11, v11, v137, s[2:3]
    v_cndmask_b32_e64 v12, v12, v137, s[4:5]
    v_cndmask_b32_e64 v13, v13, v137, s[6:7]
    v_cndmask_b32_e64 v14, v14, v137, s[8:9]
    v_cndmask_b32_e64 v15, v15, v137, s[10:11]
    v_cndmask_b32_e64 v16, v16, v137, s[12:13]
    v_cndmask_b32_e64 v17, v17, v137, s[14:15]
    v_cndmask_b32_e64 v140, v140, v137, s[16:17]
    v_cndmask_b32_e64 v141, v141, v137, s[18:19]
    v_cndmask_b32_e64 v142, v142, v137, s[20:21]
    v_cndmask_b32_e64 v143, v143, v137, s[22:23]
    v_cndmask_b32_e64 v22, v22, v137, s[24:25]
    v_cndmask_b32_e64 v23, v23, v137, s[26:27]
    v_cndmask_b32_e64 v24, v24, v137, s[28:29]
    v_cndmask_b32_e64 v25, v25, v137, s[30:31]
    v_cndmask_b32_e32 v2, v2, v137, vcc
    v_cndmask_b32_e64 v3, v3, v137, s[2:3]
    v_cndmask_b32_e64 v4, v4, v137, s[4:5]
    v_cndmask_b32_e64 v5, v5, v137, s[6:7]
    v_cndmask_b32_e64 v152, v152, v137, s[8:9]
    v_cndmask_b32_e64 v153, v153, v137, s[10:11]
    v_cndmask_b32_e64 v154, v154, v137, s[12:13]
    v_cndmask_b32_e64 v155, v155, v137, s[14:15]
    v_cndmask_b32_e64 v6, v6, v137, s[16:17]
    v_cndmask_b32_e64 v7, v7, v137, s[18:19]
    v_cndmask_b32_e64 v8, v8, v137, s[20:21]
    v_cndmask_b32_e64 v9, v9, v137, s[22:23]
    v_cndmask_b32_e64 v144, v144, v137, s[24:25]
    v_cndmask_b32_e64 v145, v145, v137, s[26:27]
    v_cndmask_b32_e64 v146, v146, v137, s[28:29]
    v_cndmask_b32_e64 v147, v147, v137, s[30:31]

.Lh3_wide_attention_block_66:
    s_add_i32 s2, s82, 1
    s_max_i32 s10, s2, s79
    s_branch .Lh3_wide_attention_block_4

.Lh3_wide_attention_block_67:
    s_lshl_b32 s7, s68, 14
    s_lshl_b32 s8, s69, 14
    s_add_i32 s8, s8, s7
    v_or_b32_e32 v134, s8, v188
    s_andn2_b64 vcc, exec, s[2:3]
    s_cbranch_vccz .Lh3_wide_attention_block_76

.Lh3_wide_attention_block_68:
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[130:133], v[146:149], v[22:25], v[130:133]
    ds_read_b128 v[150:153], v135 offset:2048
    ; Form the next K address ahead of the independent MFMA.
    v_add_u32_e32 v136, s70, v139
    v_mfma_f32_16x16x32_bf16 v[126:129], v[146:149], v[18:21], v[126:129]
    v_mfma_f32_16x16x32_bf16 v[122:125], v[146:149], v[2:5], v[122:125]
    s_waitcnt lgkmcnt(1)
    buffer_load_dwordx4 v136, s[40:43], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[118:121], v[140:143], v[2:5], v[118:121]
    ds_read_b128 v[154:157], v135 offset:3072
    v_mfma_f32_16x16x32_bf16 v[114:117], v[140:143], v[18:21], v[114:117]
    v_mfma_f32_16x16x32_bf16 v[110:113], v[140:143], v[22:25], v[110:113]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[106:109], v[150:153], v[22:25], v[106:109]
    ds_read_b128 v[158:161], v135 offset:4096
    v_mfma_f32_16x16x32_bf16 v[102:105], v[150:153], v[18:21], v[102:105]
    v_mfma_f32_16x16x32_bf16 v[98:101], v[150:153], v[2:5], v[98:101]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[94:97], v[154:157], v[2:5], v[94:97]
    ds_read_b128 v[140:143], v135 offset:5120
    v_mfma_f32_16x16x32_bf16 v[144:147], v[154:157], v[18:21], v[90:93]
    v_mfma_f32_16x16x32_bf16 v[86:89], v[154:157], v[22:25], v[86:89]
    s_mov_b32 m0, s5
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[82:85], v[158:161], v[22:25], v[82:85]
    ds_read_b128 v[90:93], v135 offset:6144
    v_mfma_f32_16x16x32_bf16 v[148:151], v[158:161], v[18:21], v[78:81]
    v_mfma_f32_16x16x32_bf16 v[152:155], v[158:161], v[2:5], v[74:77]
    s_nop 2
    v_add_u32_e32 v74, s70, v169
    s_waitcnt lgkmcnt(1)
    buffer_load_dwordx4 v74, s[40:43], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[156:159], v[140:143], v[2:5], v[70:73]
    s_nop 2
    ds_read_b128 v[70:73], v135 offset:7168
    v_mfma_f32_16x16x32_bf16 v[160:163], v[140:143], v[18:21], v[66:69]
    v_mfma_f32_16x16x32_bf16 v[62:65], v[140:143], v[22:25], v[62:65]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[140:143], v[90:93], v[22:25], v[58:61]
    s_nop 2
    ds_read_b128 v[58:61], v135 offset:8192
    v_mfma_f32_16x16x32_bf16 v[164:167], v[90:93], v[18:21], v[54:57]
    v_mfma_f32_16x16x32_bf16 v[242:245], v[90:93], v[2:5], v[50:53]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[246:249], v[70:73], v[2:5], v[38:41]
    s_nop 2
    ds_read_b128 v[38:41], v135 offset:9216
    v_mfma_f32_16x16x32_bf16 v[250:253], v[70:73], v[18:21], v[46:49]
    v_mfma_f32_16x16x32_bf16 v[176:179], v[70:73], v[22:25], v[42:45]
    s_mov_b32 m0, s6
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[42:45], v[58:61], v[14:17], v[130:133]
    ds_read_b128 v[22:25], v135 offset:10240
    v_mfma_f32_16x16x32_bf16 v[74:77], v[58:61], v[10:13], v[126:129]
    v_mfma_f32_16x16x32_bf16 v[2:5], v[58:61], v[6:9], v[122:125]
    s_waitcnt lgkmcnt(1)
    buffer_load_dwordx4 v134, s[44:47], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[18:21], v[38:41], v[6:9], v[118:121]
    ds_read_b128 v[46:49], v135 offset:11264
    v_mfma_f32_16x16x32_bf16 v[78:81], v[38:41], v[10:13], v[114:117]
    v_mfma_f32_16x16x32_bf16 v[54:57], v[38:41], v[14:17], v[110:113]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[58:61], v[22:25], v[14:17], v[106:109]
    ds_read_b128 v[50:53], v135 offset:12288
    v_mfma_f32_16x16x32_bf16 v[90:93], v[22:25], v[10:13], v[102:105]
    v_mfma_f32_16x16x32_bf16 v[22:25], v[22:25], v[6:9], v[98:101]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[38:41], v[46:49], v[6:9], v[94:97]
    ds_read_b128 v[106:109], v135 offset:13312
    v_mfma_f32_16x16x32_bf16 v[94:97], v[46:49], v[10:13], v[144:147]
    v_mfma_f32_16x16x32_bf16 v[66:69], v[46:49], v[14:17], v[86:89]
    s_mov_b32 m0, s4
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[70:73], v[50:53], v[14:17], v[82:85]
    ds_read_b128 v[110:113], v135 offset:14336
    v_mfma_f32_16x16x32_bf16 v[98:101], v[50:53], v[10:13], v[148:151]
    v_mfma_f32_16x16x32_bf16 v[46:49], v[50:53], v[6:9], v[152:155]
    v_or_b32_e32 v50, 0x400, v134
    s_waitcnt lgkmcnt(1)
    buffer_load_dwordx4 v50, s[44:47], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[50:53], v[106:109], v[6:9], v[156:159]
    ds_read_b128 v[114:117], v135 offset:15360
    v_mfma_f32_16x16x32_bf16 v[102:105], v[106:109], v[10:13], v[160:163]
    v_mfma_f32_16x16x32_bf16 v[82:85], v[106:109], v[14:17], v[62:65]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[86:89], v[110:113], v[14:17], v[140:143]
    v_mfma_f32_16x16x32_bf16 v[106:109], v[110:113], v[10:13], v[164:167]
    v_mfma_f32_16x16x32_bf16 v[62:65], v[110:113], v[6:9], v[242:245]
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_16x16x32_bf16 v[6:9], v[114:117], v[6:9], v[246:249]
    v_mfma_f32_16x16x32_bf16 v[110:113], v[114:117], v[10:13], v[250:253]
    v_mfma_f32_16x16x32_bf16 v[10:13], v[114:117], v[14:17], v[176:179]
    s_waitcnt vmcnt(0)
    s_barrier
    s_cmp_ge_i32 s69, s58
    s_mov_b32 s2, s69
    s_cbranch_scc1 .Lh3_wide_attention_block_70

.Lh3_wide_attention_block_69:
    s_add_i32 s2, s2, 1
    s_add_i32 s3, s2, s67
    s_mul_hi_u32 s6, s3, 0xaaaaaaab
    s_lshr_b32 s6, s6, 1
    s_mov_b64 s[4:5], src_shared_base
    s_add_i32 s10, s2, s68
    s_mul_i32 s6, s6, 3
    s_mul_i32 s4, s3, 0xa8000
    s_sub_i32 s3, s3, s6
    s_mul_hi_u32 s6, s10, 0xaaaaaaab
    s_lshr_b32 s6, s6, 1
    s_mul_i32 s6, s6, 3
    s_mulk_i32 s3, 0x4400
    s_sub_i32 s6, s10, s6
    s_add_i32 s3, s59, s3
    s_mul_i32 s11, s6, 0x4800
    s_add_u32 s6, s66, s11
    s_addc_u32 s7, s61, 0
    s_add_u32 s8, s3, 0x2200
    s_addc_u32 s9, s5, 0
    s_cmp_lg_u64 s[8:9], 0
    v_add_u32_e32 v14, s4, v139
    s_mov_b32 m0, s3
    s_nop 0
    buffer_load_dwordx4 v14, s[40:43], 0 offen lds
    s_cselect_b32 s3, s8, -1
    s_cmp_lg_u64 s[6:7], 0
    v_add_u32_e32 v14, s4, v169
    s_mov_b32 m0, s3
    s_nop 0
    buffer_load_dwordx4 v14, s[40:43], 0 offen lds
    s_cselect_b32 s3, s6, -1
    v_lshl_or_b32 v14, s10, 14, v188
    s_mov_b32 m0, s3
    s_nop 0
    buffer_load_dwordx4 v14, s[44:47], 0 offen lds
    s_add_i32 s3, s60, s11
    s_add_u32 s4, s3, 0xd000
    s_addc_u32 s5, s5, 0
    s_cmp_lg_u64 s[4:5], 0
    s_cselect_b32 s3, s4, -1
    v_or_b32_e32 v14, 0x400, v14
    s_mov_b32 m0, s3
    s_nop 0
    buffer_load_dwordx4 v14, s[44:47], 0 offen lds
    s_waitcnt vmcnt(0)
    s_barrier
    s_cmp_eq_u32 s2, s58
    s_cbranch_scc0 .Lh3_wide_attention_block_69

.Lh3_wide_attention_block_70:
    s_mov_b64 s[2:3], 0

.Lh3_wide_attention_block_71:
    s_and_b64 vcc, exec, s[2:3]
    s_cbranch_vccz .Lh3_wide_attention_block_82
    s_mov_b64 s[2:3], src_shared_base
    s_lshl_b64 s[62:63], s[54:55], 1
    s_add_u32 s2, 0x8800, s62
    s_addc_u32 s3, s3, s63
    s_add_i32 s5, s60, 0x11400
    s_add_i32 s6, s59, 0xaa00
    s_add_i32 s4, s60, 0x11800
    s_cmp_lg_u64 s[2:3], 0
    s_waitcnt lgkmcnt(14)
    v_mov_b32_e32 v141, v190
    s_cselect_b32 s2, s2, -1
    s_mov_b32 m0, s2
    v_mov_b32_e32 v144, v183
    ds_read_b128 v[242:245], v141
    ds_read_b128 v[134:137], v141 offset:1024
    v_mov_b32_e32 v168, v180
    v_mov_b32_e32 v180, v192
    s_cmp_lt_i32 s69, 1
    zero_accumulators 49, 48, 47, 46, 41, 40, 39, 38, 45, 44, 43, 42
    zero_accumulators 53, 52, 51, 50, 61, 60, 59, 58, 57, 56, 55, 54
    zero_accumulators 73, 72, 71, 70, 65, 64, 63, 62, 69, 68, 67, 66
    zero_accumulators 77, 76, 75, 74, 85, 84, 83, 82, 81, 80, 79, 78
    zero_accumulators 97, 96, 95, 94, 89, 88, 87, 86, 93, 92, 91, 90
    zero_accumulators 101, 100, 99, 98, 109, 108, 107, 106, 105, 104, 103, 102
    zero_accumulators 121, 120, 119, 118, 113, 112, 111, 110, 117, 116, 115, 114
    zero_accumulators 125, 124, 123, 122, 133, 132, 131, 130, 129, 128, 127, 126
    zero_accumulators 37, 36, 35, 34, 33, 32, 31, 30
    s_cbranch_scc1 .Lh3_wide_attention_block_77
    s_lshl_b32 s2, s68, 14
    s_lshl_b32 s3, s69, 14
    v_mov_b32_e32 v6, 0
    s_add_i32 s71, s80, -1
    s_mul_i32 s70, s67, 0xa8000
    s_add_i32 s54, s3, s2
    s_mov_b64 s[2:3], 0
    s_mov_b32 s8, 0
    s_mov_b32 s9, 0x150000
    s_movk_i32 s7, 0x4000
    s_movk_i32 s80, 0x201
    zero_accumulators 30, 31, 32, 33, 34, 35, 36, 37, 126, 127, 128, 129
    zero_accumulators 130, 131, 132, 133, 122, 123, 124, 125, 114, 115, 116, 117
    zero_accumulators 110, 111, 112, 113, 118, 119, 120, 121, 102, 103, 104, 105
    zero_accumulators 106, 107, 108, 109, 98, 99, 100, 101, 90, 91, 92, 93
    zero_accumulators 86, 87, 88, 89, 94, 95, 96, 97, 78, 79, 80, 81
    zero_accumulators 82, 83, 84, 85, 74, 75, 76, 77, 66, 67, 68, 69
    zero_accumulators 62, 63, 64, 65, 70, 71, 72, 73, 54, 55, 56, 57
    zero_accumulators 58, 59, 60, 61, 50, 51, 52, 53, 42, 43, 44, 45
    zero_accumulators 38, 39, 40, 41, 46, 47, 48, 49
    v_mov_b32_e32 v138, 0xc29cd760
    v_mov_b32_e32 v140, 0xc29cd760
    v_mov_b32_e32 v142, 0xc29cd760
    v_mov_b32_e32 v148, v190
    s_mov_b32 s81, 0
    v_mov_b32_e32 v7, v6
    v_mov_b32_e32 v8, v6
    v_mov_b32_e32 v9, v6
    v_mov_b32_e32 v14, v6
    v_mov_b32_e32 v15, v6
    v_mov_b32_e32 v16, v6
    v_mov_b32_e32 v17, v6
    v_mov_b32_e32 v10, v6
    v_mov_b32_e32 v11, v6
    v_mov_b32_e32 v12, v6
    v_mov_b32_e32 v13, v6
    v_mov_b32_e32 v2, v6
    v_mov_b32_e32 v3, v6
    v_mov_b32_e32 v4, v6
    v_mov_b32_e32 v5, v6
    v_mov_b32_e32 v22, v6
    v_mov_b32_e32 v23, v6
    v_mov_b32_e32 v24, v6
    v_mov_b32_e32 v25, v6
    v_mov_b32_e32 v18, v6
    v_mov_b32_e32 v19, v6
    v_mov_b32_e32 v20, v6
    v_mov_b32_e32 v21, v6
    s_branch .Lh3_wide_attention_block_7

.Lh3_wide_attention_block_72:
    v_exp_f32_e64 v145, -v192
    v_exp_f32_e64 v146, -v191
    v_exp_f32_e64 v143, -v143
    v_pk_mul_f32 v[30:31], v[144:145], v[30:31] op_sel:[1,0]
    v_pk_mul_f32 v[126:127], v[144:145], v[126:127] op_sel:[1,0]
    v_pk_mul_f32 v[128:129], v[144:145], v[128:129] op_sel:[1,0]
    v_pk_mul_f32 v[114:115], v[144:145], v[114:115] op_sel:[1,0]
    v_pk_mul_f32 v[116:117], v[144:145], v[116:117] op_sel:[1,0]
    v_pk_mul_f32 v[102:103], v[144:145], v[102:103] op_sel:[1,0]
    v_pk_mul_f32 v[104:105], v[144:145], v[104:105] op_sel:[1,0]
    v_pk_mul_f32 v[90:91], v[144:145], v[90:91] op_sel:[1,0]
    v_pk_mul_f32 v[92:93], v[144:145], v[92:93] op_sel:[1,0]
    v_pk_mul_f32 v[78:79], v[144:145], v[78:79] op_sel:[1,0]
    v_pk_mul_f32 v[80:81], v[144:145], v[80:81] op_sel:[1,0]
    v_pk_mul_f32 v[66:67], v[144:145], v[66:67] op_sel:[1,0]
    v_pk_mul_f32 v[68:69], v[144:145], v[68:69] op_sel:[1,0]
    v_pk_mul_f32 v[54:55], v[144:145], v[54:55] op_sel:[1,0]
    v_pk_mul_f32 v[56:57], v[144:145], v[56:57] op_sel:[1,0]
    v_pk_mul_f32 v[42:43], v[144:145], v[42:43] op_sel:[1,0]
    v_pk_mul_f32 v[44:45], v[144:145], v[44:45] op_sel:[1,0]
    rescale_accumulator_pairs 146, 32, 130, 132, 110, 112, 106, 108, 86, 88, 82, 84, 62
    rescale_accumulator_pairs 146, 64, 58, 60, 38, 40
    v_pk_mul_f32 v[34:35], v[142:143], v[34:35] op_sel:[1,0]
    v_pk_mul_f32 v[36:37], v[142:143], v[36:37] op_sel:[1,0]
    v_pk_mul_f32 v[122:123], v[142:143], v[122:123] op_sel:[1,0]
    v_pk_mul_f32 v[124:125], v[142:143], v[124:125] op_sel:[1,0]
    v_pk_mul_f32 v[118:119], v[142:143], v[118:119] op_sel:[1,0]
    v_pk_mul_f32 v[120:121], v[142:143], v[120:121] op_sel:[1,0]
    v_pk_mul_f32 v[98:99], v[142:143], v[98:99] op_sel:[1,0]
    v_pk_mul_f32 v[100:101], v[142:143], v[100:101] op_sel:[1,0]
    v_pk_mul_f32 v[94:95], v[142:143], v[94:95] op_sel:[1,0]
    v_pk_mul_f32 v[96:97], v[142:143], v[96:97] op_sel:[1,0]
    v_pk_mul_f32 v[74:75], v[142:143], v[74:75] op_sel:[1,0]
    v_pk_mul_f32 v[76:77], v[142:143], v[76:77] op_sel:[1,0]
    v_pk_mul_f32 v[70:71], v[142:143], v[70:71] op_sel:[1,0]
    v_pk_mul_f32 v[72:73], v[142:143], v[72:73] op_sel:[1,0]
    v_pk_mul_f32 v[50:51], v[142:143], v[50:51] op_sel:[1,0]
    v_pk_mul_f32 v[52:53], v[142:143], v[52:53] op_sel:[1,0]
    v_pk_mul_f32 v[46:47], v[142:143], v[46:47] op_sel:[1,0]
    v_pk_mul_f32 v[48:49], v[142:143], v[48:49] op_sel:[1,0]
    s_branch .Lh3_wide_attention_block_8

.Lh3_wide_attention_block_73:
    s_cmp_lt_i32 s81, s79
    s_cbranch_scc1 .Lh3_wide_attention_block_74
    v_mbcnt_lo_u32_b32 v141, -1, 0
    v_mbcnt_hi_u32_b32 v141, -1, v141
    s_nop 0
    v_ashrrev_i32_e32 v143, 4, v141
    v_lshrrev_b32_e32 v141, 31, v141
    v_add_u32_e32 v141, v143, v141
    v_lshrrev_b32_e32 v141, 1, v141
    v_sub_u32_e32 v141, 0, v141
    v_lshlrev_b32_e32 v141, 3, v141
    v_lshlrev_b32_e32 v143, 2, v143
    v_sub_u32_e32 v141, v141, v143
    v_add_u32_e32 v141, s71, v141
    v_cmp_gt_i32_e32 vcc, 0, v141
    v_mov_b32_e32 v143, 0xf2177617
    v_cmp_gt_i32_e64 s[2:3], 1, v141
    v_cmp_gt_i32_e64 s[4:5], 2, v141
    v_cmp_gt_i32_e64 s[6:7], 3, v141
    v_cmp_gt_i32_e64 s[8:9], 8, v141
    v_cmp_gt_i32_e64 s[10:11], 9, v141
    v_cmp_gt_i32_e64 s[12:13], 10, v141
    v_cmp_gt_i32_e64 s[14:15], 11, v141
    v_cmp_gt_i32_e64 s[16:17], 32, v141
    v_cmp_gt_i32_e64 s[18:19], 33, v141
    v_cmp_gt_i32_e64 s[20:21], 34, v141
    v_cmp_gt_i32_e64 s[22:23], 35, v141
    v_cmp_gt_i32_e64 s[24:25], 40, v141
    v_cmp_gt_i32_e64 s[26:27], 41, v141
    v_cmp_gt_i32_e64 s[28:29], 42, v141
    v_cmp_gt_i32_e64 s[30:31], 43, v141
    v_cndmask_b32_e32 v160, v160, v143, vcc
    v_cndmask_b32_e64 v161, v161, v143, s[2:3]
    v_cndmask_b32_e64 v162, v162, v143, s[4:5]
    v_cndmask_b32_e64 v163, v163, v143, s[6:7]
    v_cndmask_b32_e64 v18, v18, v143, s[8:9]
    v_cndmask_b32_e64 v19, v19, v143, s[10:11]
    v_cndmask_b32_e64 v20, v20, v143, s[12:13]
    v_cndmask_b32_e64 v21, v21, v143, s[14:15]
    v_cndmask_b32_e64 v164, v164, v143, s[16:17]
    v_cndmask_b32_e64 v165, v165, v143, s[18:19]
    v_cndmask_b32_e64 v166, v166, v143, s[20:21]
    v_cndmask_b32_e64 v167, v167, v143, s[22:23]
    v_cndmask_b32_e64 v152, v152, v143, s[24:25]
    v_cndmask_b32_e64 v153, v153, v143, s[26:27]
    v_cndmask_b32_e64 v154, v154, v143, s[28:29]
    v_cndmask_b32_e64 v155, v155, v143, s[30:31]
    v_cndmask_b32_e32 v10, v10, v143, vcc
    v_cndmask_b32_e64 v11, v11, v143, s[2:3]
    v_cndmask_b32_e64 v12, v12, v143, s[4:5]
    v_cndmask_b32_e64 v13, v13, v143, s[6:7]
    v_cndmask_b32_e64 v14, v14, v143, s[8:9]
    v_cndmask_b32_e64 v15, v15, v143, s[10:11]
    v_cndmask_b32_e64 v16, v16, v143, s[12:13]
    v_cndmask_b32_e64 v17, v17, v143, s[14:15]
    v_cndmask_b32_e64 v22, v22, v143, s[16:17]
    v_cndmask_b32_e64 v23, v23, v143, s[18:19]
    v_cndmask_b32_e64 v24, v24, v143, s[20:21]
    v_cndmask_b32_e64 v25, v25, v143, s[22:23]
    v_cndmask_b32_e64 v148, v148, v143, s[24:25]
    v_cndmask_b32_e64 v149, v149, v143, s[26:27]
    v_cndmask_b32_e64 v150, v150, v143, s[28:29]
    v_cndmask_b32_e64 v151, v151, v143, s[30:31]
    v_cndmask_b32_e32 v2, v2, v143, vcc
    v_cndmask_b32_e64 v3, v3, v143, s[2:3]
    v_cndmask_b32_e64 v4, v4, v143, s[4:5]
    v_cndmask_b32_e64 v5, v5, v143, s[6:7]
    v_cndmask_b32_e64 v144, v144, v143, s[8:9]
    v_cndmask_b32_e64 v145, v145, v143, s[10:11]
    v_cndmask_b32_e64 v146, v146, v143, s[12:13]
    v_cndmask_b32_e64 v147, v147, v143, s[14:15]
    v_cndmask_b32_e64 v6, v6, v143, s[16:17]
    v_cndmask_b32_e64 v7, v7, v143, s[18:19]
    v_cndmask_b32_e64 v8, v8, v143, s[20:21]
    v_cndmask_b32_e64 v9, v9, v143, s[22:23]
    v_cndmask_b32_e64 v156, v156, v143, s[24:25]
    v_cndmask_b32_e64 v157, v157, v143, s[26:27]
    v_cndmask_b32_e64 v158, v158, v143, s[28:29]
    v_cndmask_b32_e64 v159, v159, v143, s[30:31]

.Lh3_wide_attention_block_74:
    s_add_i32 s2, s81, 1
    s_max_i32 s8, s2, s79
    s_branch .Lh3_wide_attention_block_9

.Lh3_wide_attention_block_75:
    v_mov_b32_e32 v32, v171
    v_mov_b32_e32 v33, v171
    v_mov_b32_e32 v30, v171
    v_mov_b32_e32 v31, v171
    v_mov_b32_e32 v21, 0
    v_mov_b64_e32 v[36:37], v[32:33]
    s_mov_b32 s70, 0xa8000
    zero_accumulators 40, 39, 38, 45, 44, 43, 42, 49, 48, 47, 46, 53
    zero_accumulators 52, 51, 50, 61, 60, 59, 58, 57, 56, 55, 54, 73
    zero_accumulators 72, 71, 70, 65, 64, 63, 62, 69, 68, 67, 66, 77
    zero_accumulators 76, 75, 74, 85, 84, 83, 82, 81, 80, 79, 78, 97
    zero_accumulators 96, 95, 94, 89, 88, 87, 86, 93, 92, 91, 90, 101
    zero_accumulators 100, 99, 98, 109, 108, 107, 106, 105, 104, 103, 102, 121
    zero_accumulators 120, 119, 118, 113, 112, 111, 110, 117, 116, 115, 114, 125
    zero_accumulators 124, 123, 122, 133, 132, 131, 130, 129, 128, 127, 126
    v_mov_b32_e32 v20, v21
    v_mov_b32_e32 v19, v21
    v_mov_b32_e32 v18, v21
    v_mov_b32_e32 v25, v21
    v_mov_b32_e32 v24, v21
    v_mov_b32_e32 v23, v21
    v_mov_b32_e32 v22, v21
    v_mov_b32_e32 v5, v21
    v_mov_b32_e32 v4, v21
    v_mov_b32_e32 v3, v21
    v_mov_b32_e32 v2, v21
    v_mov_b32_e32 v13, v21
    v_mov_b32_e32 v12, v21
    v_mov_b32_e32 v11, v21
    v_mov_b32_e32 v10, v21
    v_mov_b32_e32 v17, v21
    v_mov_b32_e32 v16, v21
    v_mov_b32_e32 v15, v21
    v_mov_b32_e32 v14, v21
    v_mov_b32_e32 v9, v21
    v_mov_b32_e32 v8, v21
    v_mov_b32_e32 v7, v21
    v_mov_b32_e32 v6, v21
    s_mov_b32 s6, s66
    v_mov_b32_e32 v134, v188
    v_mov_b64_e32 v[34:35], v[30:31]
    s_cbranch_execnz .Lh3_wide_attention_block_68

.Lh3_wide_attention_block_76:
    v_exp_f32_e64 v136, -v165
    v_exp_f32_e64 v138, -v164
    rescale_accumulator_pairs 136, 126, 128, 114, 116, 102, 104, 90, 92, 78, 80, 66, 68
    rescale_accumulator_pairs 136, 54, 56, 46, 48
    v_exp_f32_e64 v136, -v137
    rescale_accumulator_pairs 138, 130, 132, 110, 112, 106, 108, 86, 88, 82, 84, 62, 64
    rescale_accumulator_pairs 138, 58, 60, 42, 44
    rescale_accumulator_pairs 136, 122, 124, 118, 120, 98, 100, 94, 96, 74, 76, 70, 72
    rescale_accumulator_pairs 136, 50, 52, 38, 40
    s_branch .Lh3_wide_attention_block_68

.Lh3_wide_attention_block_77:
    v_mov_b32_e32 v21, 0
    s_movk_i32 s54, 0x4000
    s_mov_b32 s70, 0x150000
    s_mov_b64 s[2:3], 0
    v_mov_b32_e32 v20, v21
    v_mov_b32_e32 v19, v21
    v_mov_b32_e32 v18, v21
    v_mov_b32_e32 v25, v21
    v_mov_b32_e32 v24, v21
    v_mov_b32_e32 v23, v21
    v_mov_b32_e32 v22, v21
    v_mov_b32_e32 v5, v21
    v_mov_b32_e32 v4, v21
    v_mov_b32_e32 v3, v21
    v_mov_b32_e32 v2, v21
    v_mov_b32_e32 v13, v21
    v_mov_b32_e32 v12, v21
    v_mov_b32_e32 v11, v21
    v_mov_b32_e32 v10, v21
    v_mov_b32_e32 v17, v21
    v_mov_b32_e32 v16, v21
    v_mov_b32_e32 v15, v21
    v_mov_b32_e32 v14, v21
    v_mov_b32_e32 v9, v21
    v_mov_b32_e32 v8, v21
    v_mov_b32_e32 v7, v21
    v_mov_b32_e32 v6, v21

.Lh3_wide_attention_block_78:
    s_andn2_b64 vcc, exec, s[2:3]
    s_cbranch_vccz .Lh3_wide_attention_block_100

.Lh3_wide_attention_block_79:
    s_waitcnt vmcnt(0)
    v_smfmac_f32_16x16x64_bf16 v[30:33], v[26:29], v[18:25], v181
    s_barrier
    v_smfmac_f32_16x16x64_bf16 v[30:33], v[26:29], v[10:17], v181
    v_smfmac_f32_16x16x64_bf16 v[34:37], v[26:29], v[2:9], v181
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[130:133], v[242:245], v[22:25], v[130:133]
    ds_read_b128 v[142:145], v141 offset:2048
    ; Form the next K address ahead of the independent MFMA.
    v_add_u32_e32 v138, s70, v139
    v_mfma_f32_16x16x32_bf16 v[126:129], v[242:245], v[18:21], v[126:129]
    v_mfma_f32_16x16x32_bf16 v[122:125], v[242:245], v[2:5], v[122:125]
    s_waitcnt lgkmcnt(1)
    buffer_load_dwordx4 v138, s[40:43], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[118:121], v[134:137], v[2:5], v[118:121]
    ds_read_b128 v[146:149], v141 offset:3072
    v_mfma_f32_16x16x32_bf16 v[114:117], v[134:137], v[18:21], v[114:117]
    v_mfma_f32_16x16x32_bf16 v[110:113], v[134:137], v[22:25], v[110:113]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[106:109], v[142:145], v[22:25], v[106:109]
    ds_read_b128 v[150:153], v141 offset:4096
    v_mfma_f32_16x16x32_bf16 v[102:105], v[142:145], v[18:21], v[102:105]
    v_mfma_f32_16x16x32_bf16 v[98:101], v[142:145], v[2:5], v[98:101]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[94:97], v[146:149], v[2:5], v[94:97]
    ds_read_b128 v[134:137], v141 offset:5120
    v_mfma_f32_16x16x32_bf16 v[142:145], v[146:149], v[18:21], v[90:93]
    v_mfma_f32_16x16x32_bf16 v[86:89], v[146:149], v[22:25], v[86:89]
    s_mov_b32 m0, s6
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[82:85], v[150:153], v[22:25], v[82:85]
    ds_read_b128 v[90:93], v141 offset:6144
    v_mfma_f32_16x16x32_bf16 v[146:149], v[150:153], v[18:21], v[78:81]
    v_mfma_f32_16x16x32_bf16 v[150:153], v[150:153], v[2:5], v[74:77]
    s_nop 2
    v_add_u32_e32 v74, s70, v169
    s_waitcnt lgkmcnt(1)
    buffer_load_dwordx4 v74, s[40:43], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[154:157], v[134:137], v[2:5], v[70:73]
    s_nop 2
    ds_read_b128 v[70:73], v141 offset:7168
    v_mfma_f32_16x16x32_bf16 v[158:161], v[134:137], v[18:21], v[66:69]
    v_mfma_f32_16x16x32_bf16 v[62:65], v[134:137], v[22:25], v[62:65]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[134:137], v[90:93], v[22:25], v[58:61]
    s_nop 2
    ds_read_b128 v[58:61], v141 offset:8192
    v_mfma_f32_16x16x32_bf16 v[162:165], v[90:93], v[18:21], v[54:57]
    v_mfma_f32_16x16x32_bf16 v[176:179], v[90:93], v[2:5], v[50:53]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[184:187], v[70:73], v[2:5], v[46:49]
    s_nop 2
    ds_read_b128 v[46:49], v141 offset:9216
    v_mfma_f32_16x16x32_bf16 v[190:193], v[70:73], v[18:21], v[42:45]
    v_mfma_f32_16x16x32_bf16 v[194:197], v[70:73], v[22:25], v[38:41]
    s_mov_b32 m0, s5
    s_waitcnt lgkmcnt(1)
    s_setprio 1
    v_mfma_f32_16x16x32_bf16 v[42:45], v[58:61], v[14:17], v[130:133]
    ds_read_b128 v[22:25], v141 offset:10240
    v_mfma_f32_16x16x32_bf16 v[74:77], v[58:61], v[10:13], v[126:129]
    v_mfma_f32_16x16x32_bf16 v[2:5], v[58:61], v[6:9], v[122:125]
    v_add_u32_e32 v18, s54, v188
    s_waitcnt lgkmcnt(1)
    buffer_load_dwordx4 v18, s[44:47], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[18:21], v[46:49], v[6:9], v[118:121]
    ds_read_b128 v[50:53], v141 offset:11264
    v_mfma_f32_16x16x32_bf16 v[78:81], v[46:49], v[10:13], v[114:117]
    v_mfma_f32_16x16x32_bf16 v[54:57], v[46:49], v[14:17], v[110:113]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[58:61], v[22:25], v[14:17], v[106:109]
    ds_read_b128 v[46:49], v141 offset:12288
    v_mfma_f32_16x16x32_bf16 v[90:93], v[22:25], v[10:13], v[102:105]
    v_mfma_f32_16x16x32_bf16 v[22:25], v[22:25], v[6:9], v[98:101]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[38:41], v[50:53], v[6:9], v[94:97]
    ds_read_b128 v[106:109], v141 offset:13312
    v_mfma_f32_16x16x32_bf16 v[94:97], v[50:53], v[10:13], v[142:145]
    v_mfma_f32_16x16x32_bf16 v[66:69], v[50:53], v[14:17], v[86:89]
    s_mov_b32 m0, s4
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[70:73], v[46:49], v[14:17], v[82:85]
    ds_read_b128 v[110:113], v141 offset:14336
    v_mfma_f32_16x16x32_bf16 v[98:101], v[46:49], v[10:13], v[146:149]
    v_mfma_f32_16x16x32_bf16 v[46:49], v[46:49], v[6:9], v[150:153]
    v_add_u32_e32 v50, s54, v189
    s_waitcnt lgkmcnt(1)
    buffer_load_dwordx4 v50, s[44:47], 0 offen lds
    v_mfma_f32_16x16x32_bf16 v[50:53], v[106:109], v[6:9], v[154:157]
    ds_read_b128 v[114:117], v141 offset:15360
    v_mfma_f32_16x16x32_bf16 v[102:105], v[106:109], v[10:13], v[158:161]
    v_mfma_f32_16x16x32_bf16 v[82:85], v[106:109], v[14:17], v[62:65]
    s_waitcnt lgkmcnt(1)
    v_mfma_f32_16x16x32_bf16 v[86:89], v[110:113], v[14:17], v[134:137]
    v_mfma_f32_16x16x32_bf16 v[106:109], v[110:113], v[10:13], v[162:165]
    v_mfma_f32_16x16x32_bf16 v[62:65], v[110:113], v[6:9], v[176:179]
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_16x16x32_bf16 v[6:9], v[114:117], v[6:9], v[184:187]
    v_mfma_f32_16x16x32_bf16 v[110:113], v[114:117], v[10:13], v[190:193]
    v_mfma_f32_16x16x32_bf16 v[10:13], v[114:117], v[14:17], v[194:197]
    s_setprio 0
    s_cmp_ge_i32 s69, s58
    s_cbranch_scc1 .Lh3_wide_attention_block_81

.Lh3_wide_attention_block_80:
    s_mov_b64 s[2:3], src_shared_base
    s_add_i32 s69, s69, 1
    s_add_i32 s2, s69, s67
    s_mul_hi_u32 s5, s2, 0xaaaaaaab
    s_lshr_b32 s5, s5, 1
    s_add_i32 s4, s69, s68
    s_mul_i32 s5, s5, 3
    s_mul_i32 s8, s2, 0xa8000
    s_sub_i32 s2, s2, s5
    s_mul_hi_u32 s5, s4, 0xaaaaaaab
    s_lshr_b32 s5, s5, 1
    s_mul_i32 s5, s5, 3
    s_lshl_b32 s9, s4, 14
    s_mulk_i32 s2, 0x4400
    s_sub_i32 s4, s4, s5
    s_add_i32 s2, s59, s2
    s_mul_i32 s10, s4, 0x4800
    s_add_u32 s4, s66, s10
    s_waitcnt vmcnt(0)
    s_addc_u32 s5, s61, 0
    s_add_u32 s6, s2, 0x2200
    s_addc_u32 s7, s3, 0
    s_barrier
    s_cmp_lg_u64 s[6:7], 0
    v_add_u32_e32 v14, s8, v139
    s_mov_b32 m0, s2
    s_nop 0
    buffer_load_dwordx4 v14, s[40:43], 0 offen lds
    s_cselect_b32 s2, s6, -1
    s_cmp_lg_u64 s[4:5], 0
    v_add_u32_e32 v14, s8, v169
    s_mov_b32 m0, s2
    s_nop 0
    buffer_load_dwordx4 v14, s[40:43], 0 offen lds
    s_cselect_b32 s2, s4, -1
    v_add_u32_e32 v14, s9, v188
    s_mov_b32 m0, s2
    s_nop 0
    buffer_load_dwordx4 v14, s[44:47], 0 offen lds
    s_add_i32 s2, s60, s10
    s_add_u32 s2, s2, 0xd000
    s_addc_u32 s3, s3, 0
    s_cmp_lg_u64 s[2:3], 0
    s_cselect_b32 s2, s2, -1
    v_add_u32_e32 v14, s9, v189
    s_mov_b32 m0, s2
    s_nop 0
    buffer_load_dwordx4 v14, s[44:47], 0 offen lds
    s_cmp_eq_u32 s69, s58
    s_cbranch_scc0 .Lh3_wide_attention_block_80

.Lh3_wide_attention_block_81:
    v_mov_b32_e32 v192, v180
    v_mov_b32_e32 v180, v168

.Lh3_wide_attention_block_82:
    v_add_f32_e32 v14, v30, v31
    s_waitcnt vmcnt(0)
    s_barrier
    v_mov_b32_e32 v15, v14
    s_nop 0
    v_and_b32_e32 v15, 0x7fffffff, v15
    v_cmp_gt_u32_e32 vcc, s77, v15
    s_mov_b64 s[2:3], exec
    scratch_load_dwordx2 v[16:17], off, off offset:4
    s_waitcnt lgkmcnt(14)
    scratch_load_dword v195, off, off offset:16
    scratch_load_dword v197, off, off offset:24
    s_and_b64 s[4:5], s[2:3], vcc
    s_xor_b64 s[2:3], s[4:5], s[2:3]
    v_and_b32_e32 v193, 32, v0
    v_mov_b32_e32 v194, v182
    v_mov_b32_e32 v196, v1
    v_mov_b32_e32 v198, 0x23400
    s_waitcnt vmcnt(2)
    v_mov_b32_e32 v168, v16
    s_mov_b64 exec, s[4:5]
    v_rcp_f32_e32 v15, v14
    v_cmp_lt_f32_e32 vcc, 0, v14
    s_nop 1
    v_cndmask_b32_e32 v15, 0, v15, vcc
    s_or_saveexec_b64 s[2:3], s[2:3]
    scratch_load_dword v1, off, off offset:12
    scratch_load_dword v182, off, off offset:20
    s_xor_b64 exec, exec, s[2:3]
    s_cbranch_execz .Lh3_wide_attention_block_83
    v_mov_b32_e32 v15, 0x7fc00000

.Lh3_wide_attention_block_83:
    s_or_b64 exec, exec, s[2:3]
    v_add_f32_e32 v16, v32, v33
    s_mulk_i32 s65, 0x3300
    v_mul_f32_e32 v17, v15, v74
    v_mul_f32_e32 v30, v15, v75
    v_mul_f32_e32 v31, v15, v76
    v_mul_f32_e32 v32, v15, v77
    s_waitcnt vmcnt(0)
    v_or_b32_e32 v14, s65, v182
    v_cvt_pk_bf16_f32 v31, v31, v32
    v_cvt_pk_bf16_f32 v30, v17, v30
    v_mul_f32_e32 v17, v15, v78
    v_mul_f32_e32 v32, v15, v79
    v_mul_f32_e32 v33, v15, v80
    v_mul_f32_e32 v74, v15, v81
    v_add_u32_e32 v14, v14, v197
    v_cvt_pk_bf16_f32 v33, v33, v74
    v_cvt_pk_bf16_f32 v32, v17, v32
    ds_write2_b64 v14, v[30:31], v[32:33] offset1:4
    v_mul_f32_e32 v17, v15, v90
    v_mul_f32_e32 v30, v15, v91
    v_mul_f32_e32 v31, v15, v92
    v_mul_f32_e32 v32, v15, v93
    v_cvt_pk_bf16_f32 v31, v31, v32
    v_cvt_pk_bf16_f32 v30, v17, v30
    v_mul_f32_e32 v17, v15, v94
    v_mul_f32_e32 v32, v15, v95
    v_mul_f32_e32 v33, v15, v96
    v_mul_f32_e32 v74, v15, v97
    v_cvt_pk_bf16_f32 v33, v33, v74
    v_cvt_pk_bf16_f32 v32, v17, v32
    ds_write2_b64 v14, v[30:31], v[32:33] offset0:8 offset1:12
    v_mul_f32_e32 v17, v15, v98
    v_mul_f32_e32 v30, v15, v99
    v_mul_f32_e32 v31, v15, v100
    v_mul_f32_e32 v32, v15, v101
    v_cvt_pk_bf16_f32 v31, v31, v32
    v_cvt_pk_bf16_f32 v30, v17, v30
    v_mul_f32_e32 v17, v15, v102
    v_mul_f32_e32 v32, v15, v103
    v_mul_f32_e32 v33, v15, v104
    v_mul_f32_e32 v74, v15, v105
    v_cvt_pk_bf16_f32 v33, v33, v74
    v_cvt_pk_bf16_f32 v32, v17, v32
    ds_write2_b64 v14, v[30:31], v[32:33] offset0:16 offset1:20
    v_mul_f32_e32 v17, v15, v106
    v_mul_f32_e32 v30, v15, v107
    v_mul_f32_e32 v31, v15, v108
    v_mul_f32_e32 v32, v15, v109
    v_cvt_pk_bf16_f32 v31, v31, v32
    v_cvt_pk_bf16_f32 v30, v17, v30
    v_mul_f32_e32 v17, v15, v110
    v_mul_f32_e32 v32, v15, v111
    v_mul_f32_e32 v33, v15, v112
    v_mul_f32_e32 v15, v15, v113
    v_cvt_pk_bf16_f32 v33, v33, v15
    v_mov_b32_e32 v15, v16
    v_cvt_pk_bf16_f32 v32, v17, v32
    v_and_b32_e32 v15, 0x7fffffff, v15
    v_cmp_gt_u32_e32 vcc, s77, v15
    ds_write2_b64 v14, v[30:31], v[32:33] offset0:24 offset1:28
    s_and_saveexec_b64 s[2:3], vcc
    s_xor_b64 s[2:3], exec, s[2:3]
    v_rcp_f32_e32 v15, v16
    v_cmp_lt_f32_e32 vcc, 0, v16
    s_nop 1
    v_cndmask_b32_e32 v15, 0, v15, vcc
    s_andn2_saveexec_b64 s[2:3], s[2:3]
    s_cbranch_execz .Lh3_wide_attention_block_84
    v_mov_b32_e32 v15, 0x7fc00000

.Lh3_wide_attention_block_84:
    s_or_b64 exec, exec, s[2:3]
    v_mul_f32_e32 v17, v15, v42
    v_mul_f32_e32 v30, v15, v43
    v_mul_f32_e32 v31, v15, v44
    v_mul_f32_e32 v32, v15, v45
    v_add_f32_e32 v16, v34, v35
    v_cvt_pk_bf16_f32 v31, v31, v32
    v_cvt_pk_bf16_f32 v30, v17, v30
    v_mul_f32_e32 v17, v15, v54
    v_mul_f32_e32 v32, v15, v55
    v_mul_f32_e32 v33, v15, v56
    v_mul_f32_e32 v34, v15, v57
    v_cvt_pk_bf16_f32 v33, v33, v34
    v_cvt_pk_bf16_f32 v32, v17, v32
    v_add_u32_e32 v17, 0x1000, v14
    ds_write2_b64 v17, v[30:31], v[32:33] offset0:32 offset1:36
    v_mul_f32_e32 v30, v15, v58
    v_mul_f32_e32 v32, v15, v59
    v_mul_f32_e32 v31, v15, v60
    v_mul_f32_e32 v33, v15, v61
    v_cvt_pk_bf16_f32 v31, v31, v33
    v_cvt_pk_bf16_f32 v30, v30, v32
    v_mul_f32_e32 v32, v15, v66
    v_mul_f32_e32 v34, v15, v67
    v_mul_f32_e32 v33, v15, v68
    v_mul_f32_e32 v35, v15, v69
    v_cvt_pk_bf16_f32 v33, v33, v35
    v_cvt_pk_bf16_f32 v32, v32, v34
    ds_write2_b64 v17, v[30:31], v[32:33] offset0:40 offset1:44
    v_mul_f32_e32 v30, v15, v70
    v_mul_f32_e32 v32, v15, v71
    v_mul_f32_e32 v31, v15, v72
    v_mul_f32_e32 v33, v15, v73
    v_cvt_pk_bf16_f32 v31, v31, v33
    v_cvt_pk_bf16_f32 v30, v30, v32
    v_mul_f32_e32 v32, v15, v82
    v_mul_f32_e32 v34, v15, v83
    v_mul_f32_e32 v33, v15, v84
    v_mul_f32_e32 v35, v15, v85
    v_cvt_pk_bf16_f32 v33, v33, v35
    v_cvt_pk_bf16_f32 v32, v32, v34
    ds_write2_b64 v17, v[30:31], v[32:33] offset0:48 offset1:52
    v_mul_f32_e32 v30, v15, v86
    v_mul_f32_e32 v32, v15, v87
    v_add_f32_e32 v16, v16, v37
    v_mul_f32_e32 v31, v15, v88
    v_mul_f32_e32 v33, v15, v89
    v_cvt_pk_bf16_f32 v30, v30, v32
    v_mul_f32_e32 v10, v15, v10
    v_mul_f32_e32 v32, v15, v11
    v_mul_f32_e32 v11, v15, v12
    v_mul_f32_e32 v12, v15, v13
    v_add_f32_e32 v16, v16, v36
    v_cvt_pk_bf16_f32 v31, v31, v33
    v_cvt_pk_bf16_f32 v11, v11, v12
    v_cvt_pk_bf16_f32 v10, v10, v32
    ds_write2_b64 v17, v[30:31], v[10:11] offset0:56 offset1:60
    v_mov_b32_e32 v10, v16
    s_nop 0
    v_and_b32_e32 v10, 0x7fffffff, v10
    v_cmp_gt_u32_e32 vcc, s77, v10
    s_and_saveexec_b64 s[2:3], vcc
    s_xor_b64 s[2:3], exec, s[2:3]
    v_rcp_f32_e32 v10, v16
    v_cmp_lt_f32_e32 vcc, 0, v16
    s_nop 1
    v_cndmask_b32_e32 v10, 0, v10, vcc
    s_andn2_saveexec_b64 s[2:3], s[2:3]
    s_cbranch_execz .Lh3_wide_attention_block_85
    v_mov_b32_e32 v10, 0x7fc00000

.Lh3_wide_attention_block_85:
    s_or_b64 exec, exec, s[2:3]
    v_mul_f32_e32 v2, v10, v2
    v_mul_f32_e32 v11, v10, v3
    v_mul_f32_e32 v3, v10, v4
    v_mul_f32_e32 v4, v10, v5
    v_cvt_pk_bf16_f32 v3, v3, v4
    v_cvt_pk_bf16_f32 v2, v2, v11
    v_mul_f32_e32 v4, v10, v18
    v_mul_f32_e32 v11, v10, v19
    v_mul_f32_e32 v5, v10, v20
    v_mul_f32_e32 v12, v10, v21
    v_cvt_pk_bf16_f32 v5, v5, v12
    v_cvt_pk_bf16_f32 v4, v4, v11
    v_add_u32_e32 v11, 0x2000, v14
    ds_write2_b64 v11, v[2:3], v[4:5] offset0:64 offset1:68
    v_mul_f32_e32 v2, v10, v22
    v_mul_f32_e32 v4, v10, v23
    v_mul_f32_e32 v3, v10, v24
    v_mul_f32_e32 v5, v10, v25
    v_cvt_pk_bf16_f32 v3, v3, v5
    v_cvt_pk_bf16_f32 v2, v2, v4
    v_mul_f32_e32 v4, v10, v38
    v_mul_f32_e32 v12, v10, v39
    v_mul_f32_e32 v5, v10, v40
    v_mul_f32_e32 v13, v10, v41
    v_cvt_pk_bf16_f32 v5, v5, v13
    v_cvt_pk_bf16_f32 v4, v4, v12
    ds_write2_b64 v11, v[2:3], v[4:5] offset0:72 offset1:76
    v_mul_f32_e32 v2, v10, v46
    v_mul_f32_e32 v4, v10, v47
    v_mul_f32_e32 v3, v10, v48
    v_mul_f32_e32 v5, v10, v49
    v_cvt_pk_bf16_f32 v3, v3, v5
    v_cvt_pk_bf16_f32 v2, v2, v4
    v_mul_f32_e32 v4, v10, v50
    v_mul_f32_e32 v12, v10, v51
    v_mul_f32_e32 v5, v10, v52
    v_mul_f32_e32 v13, v10, v53
    v_cvt_pk_bf16_f32 v5, v5, v13
    v_cvt_pk_bf16_f32 v4, v4, v12
    ds_write2_b64 v11, v[2:3], v[4:5] offset0:80 offset1:84
    v_mul_f32_e32 v2, v10, v62
    v_mul_f32_e32 v4, v10, v63
    v_mul_f32_e32 v3, v10, v64
    v_mul_f32_e32 v5, v10, v65
    v_cvt_pk_bf16_f32 v3, v3, v5
    v_cvt_pk_bf16_f32 v2, v2, v4
    v_mul_f32_e32 v4, v10, v6
    v_mul_f32_e32 v6, v10, v7
    v_mul_f32_e32 v5, v10, v8
    v_mul_f32_e32 v7, v10, v9
    v_cvt_pk_bf16_f32 v5, v5, v7
    v_cvt_pk_bf16_f32 v4, v4, v6
    ds_write2_b64 v11, v[2:3], v[4:5] offset0:88 offset1:92
    v_mbcnt_lo_u32_b32 v2, -1, 0
    v_mbcnt_hi_u32_b32 v2, -1, v2
    s_nop 0
    v_ashrrev_i32_e32 v3, 31, v2
    v_lshrrev_b32_e32 v3, 28, v3
    v_add_u32_e32 v3, v2, v3
    v_ashrrev_i32_e32 v5, 4, v3
    v_and_b32_e32 v3, -16, v3
    v_sub_u32_e32 v4, v2, v3
    v_lshlrev_b32_e32 v2, 3, v4
    v_ashrrev_i32_e32 v3, 31, v2
    v_add_u32_e32 v6, s64, v5
    v_lshl_add_u32 v4, v4, 4, s65
    v_lshl_add_u64 v[2:3], v[2:3], 1, v[254:255]
    v_cmp_gt_u32_e32 vcc, s78, v6
    s_and_saveexec_b64 s[2:3], vcc
    s_cbranch_execz .Lh3_wide_attention_block_86
    v_mad_u64_u32 v[8:9], s[4:5], v5, s74, v[4:5]
    ds_read_b128 v[10:13], v8
    v_mad_u64_u32 v[8:9], s[4:5], v6, s75, v[2:3]
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v[8:9], v[10:13], off

.Lh3_wide_attention_block_86:
    s_or_b64 exec, exec, s[2:3]
    v_add_u32_e32 v7, 4, v6
    v_cmp_gt_u32_e32 vcc, s78, v7
    s_and_saveexec_b64 s[2:3], vcc
    s_cbranch_execz .Lh3_wide_attention_block_87
    v_mad_u64_u32 v[8:9], s[4:5], v5, s74, v[4:5]
    ds_read_b128 v[10:13], v8 offset:1088
    v_mad_u64_u32 v[8:9], s[4:5], v7, s75, v[2:3]
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v[8:9], v[10:13], off

.Lh3_wide_attention_block_87:
    s_or_b64 exec, exec, s[2:3]
    v_add_u32_e32 v7, 8, v6
    v_cmp_gt_u32_e32 vcc, s78, v7
    s_and_saveexec_b64 s[2:3], vcc
    s_cbranch_execz .Lh3_wide_attention_block_88
    v_mad_u64_u32 v[8:9], s[4:5], v5, s74, v[4:5]
    ds_read_b128 v[10:13], v8 offset:2176
    v_mad_u64_u32 v[8:9], s[4:5], v7, s75, v[2:3]
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v[8:9], v[10:13], off

.Lh3_wide_attention_block_88:
    s_or_b64 exec, exec, s[2:3]
    v_add_u32_e32 v7, 12, v6
    v_cmp_gt_u32_e32 vcc, s78, v7
    s_and_saveexec_b64 s[2:3], vcc
    s_cbranch_execz .Lh3_wide_attention_block_89
    v_mad_u64_u32 v[8:9], s[4:5], v5, s74, v[4:5]
    ds_read_b128 v[10:13], v8 offset:3264
    v_mad_u64_u32 v[8:9], s[4:5], v7, s75, v[2:3]
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v[8:9], v[10:13], off

.Lh3_wide_attention_block_89:
    s_or_b64 exec, exec, s[2:3]
    v_add_u32_e32 v7, 16, v6
    v_cmp_gt_u32_e32 vcc, s78, v7
    s_and_saveexec_b64 s[2:3], vcc
    s_cbranch_execz .Lh3_wide_attention_block_90
    v_mad_u64_u32 v[8:9], s[4:5], v5, s74, v[4:5]
    ds_read_b128 v[10:13], v8 offset:4352
    v_mad_u64_u32 v[8:9], s[4:5], v7, s75, v[2:3]
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v[8:9], v[10:13], off

.Lh3_wide_attention_block_90:
    s_or_b64 exec, exec, s[2:3]
    v_add_u32_e32 v7, 20, v6
    v_cmp_gt_u32_e32 vcc, s78, v7
    s_and_saveexec_b64 s[2:3], vcc
    s_cbranch_execz .Lh3_wide_attention_block_91
    v_mad_u64_u32 v[8:9], s[4:5], v5, s74, v[4:5]
    ds_read_b128 v[10:13], v8 offset:5440
    v_mad_u64_u32 v[8:9], s[4:5], v7, s75, v[2:3]
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v[8:9], v[10:13], off

.Lh3_wide_attention_block_91:
    s_or_b64 exec, exec, s[2:3]
    v_add_u32_e32 v7, 24, v6
    v_cmp_gt_u32_e32 vcc, s78, v7
    s_and_saveexec_b64 s[2:3], vcc
    s_cbranch_execz .Lh3_wide_attention_block_92
    v_mad_u64_u32 v[8:9], s[4:5], v5, s74, v[4:5]
    ds_read_b128 v[10:13], v8 offset:6528
    v_mad_u64_u32 v[8:9], s[4:5], v7, s75, v[2:3]
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v[8:9], v[10:13], off

.Lh3_wide_attention_block_92:
    s_or_b64 exec, exec, s[2:3]
    v_add_u32_e32 v7, 28, v6
    v_cmp_gt_u32_e32 vcc, s78, v7
    s_and_saveexec_b64 s[2:3], vcc
    s_cbranch_execz .Lh3_wide_attention_block_93
    v_mad_u64_u32 v[8:9], s[4:5], v5, s74, v[4:5]
    ds_read_b128 v[10:13], v8 offset:7616
    v_mad_u64_u32 v[8:9], s[4:5], v7, s75, v[2:3]
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v[8:9], v[10:13], off

.Lh3_wide_attention_block_93:
    s_or_b64 exec, exec, s[2:3]
    v_add_u32_e32 v7, 32, v6
    v_cmp_gt_u32_e32 vcc, s78, v7
    s_and_saveexec_b64 s[2:3], vcc
    s_cbranch_execz .Lh3_wide_attention_block_94
    v_mad_u64_u32 v[8:9], s[4:5], v5, s74, v[4:5]
    ds_read_b128 v[10:13], v8 offset:8704
    v_mad_u64_u32 v[8:9], s[4:5], v7, s75, v[2:3]
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v[8:9], v[10:13], off

.Lh3_wide_attention_block_94:
    s_or_b64 exec, exec, s[2:3]
    v_add_u32_e32 v7, 36, v6
    v_cmp_gt_u32_e32 vcc, s78, v7
    s_and_saveexec_b64 s[2:3], vcc
    s_cbranch_execz .Lh3_wide_attention_block_95
    v_mad_u64_u32 v[8:9], s[4:5], v5, s74, v[4:5]
    ds_read_b128 v[10:13], v8 offset:9792
    v_mad_u64_u32 v[8:9], s[4:5], v7, s75, v[2:3]
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v[8:9], v[10:13], off

.Lh3_wide_attention_block_95:
    s_or_b64 exec, exec, s[2:3]
    v_add_u32_e32 v7, 40, v6
    v_cmp_gt_u32_e32 vcc, s78, v7
    s_and_saveexec_b64 s[2:3], vcc
    s_cbranch_execz .Lh3_wide_attention_block_96
    v_mad_u64_u32 v[8:9], s[4:5], v5, s74, v[4:5]
    ds_read_b128 v[10:13], v8 offset:10880
    v_mad_u64_u32 v[8:9], s[4:5], v7, s75, v[2:3]
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v[8:9], v[10:13], off

.Lh3_wide_attention_block_96:
    s_or_b64 exec, exec, s[2:3]
    v_add_u32_e32 v7, 44, v5
    v_add_u32_e32 v5, 44, v6
    v_cmp_gt_u32_e64 s[2:3], s78, v5

.Lh3_wide_attention_block_97:
    s_and_saveexec_b64 s[4:5], s[2:3]
    s_cbranch_execz .Lh3_wide_attention_block_98
    v_mad_u64_u32 v[6:7], s[2:3], v7, s74, v[4:5]
    ds_read_b128 v[6:9], v6
    v_mad_u64_u32 v[2:3], s[2:3], v5, s75, v[2:3]
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v[2:3], v[6:9], off

.Lh3_wide_attention_block_98:
    s_or_b64 exec, exec, s[4:5]
    s_waitcnt lgkmcnt(0)
    s_barrier
    s_and_saveexec_b64 s[2:3], s[0:1]
    s_cbranch_execz .Lh3_wide_attention_block_26
    s_mov_b64 s[6:7], exec
    v_mbcnt_lo_u32_b32 v2, s6, 0
    v_mbcnt_hi_u32_b32 v2, s7, v2
    v_cmp_eq_u32_e32 vcc, 0, v2
    s_and_saveexec_b64 s[4:5], vcc
    s_cbranch_execz .Lh3_wide_attention_block_25
    s_bcnt1_i32_b64 s6, s[6:7]
    v_mov_b32_e32 v3, s6
    ; Claim the next descriptor once per workgroup and broadcast the index through LDS.
    global_atomic_add v3, v171, v3, s[48:49] sc0
    s_branch .Lh3_wide_attention_block_25

.Lh3_wide_attention_block_99:
    v_exp_f32_e64 v94, -v121
    v_exp_f32_e64 v96, -v97
    rescale_accumulator_pairs 94, 18, 82, 84, 78, 80, 66, 68, 62, 64, 50, 52, 46
    rescale_accumulator_pairs 94, 48, 34, 36, 30, 32
    rescale_accumulator_pairs 96, 20, 86, 88, 74, 76, 70, 72, 58, 60, 54, 56, 42
    rescale_accumulator_pairs 96, 44, 38, 40, 22, 24
    s_branch .Lh3_wide_attention_block_47

.Lh3_wide_attention_block_100:
    v_exp_f32_e64 v138, -v192
    v_exp_f32_e64 v140, -v191
    rescale_accumulator_pairs 138, 30, 126, 128, 114, 116, 102, 104, 90, 92, 78, 80, 66
    rescale_accumulator_pairs 138, 68, 54, 56, 42, 44
    v_exp_f32_e64 v138, -v143
    rescale_accumulator_pairs 140, 32, 130, 132, 110, 112, 106, 108, 86, 88, 82, 84, 62
    rescale_accumulator_pairs 140, 64, 58, 60, 38, 40
    rescale_accumulator_pairs 138, 34, 36, 122, 124, 118, 120, 98, 100, 94, 96, 74, 76
    rescale_accumulator_pairs 138, 70, 72, 50, 52, 46, 48
    s_branch .Lh3_wide_attention_block_79

.Lh3_wide_attention_block_101:
    s_endpgm
    s_nop 0
    s_nop 0
    s_nop 0
    s_nop 0
    s_nop 0
    s_nop 0
    s_nop 0
    s_nop 0
    s_nop 0
    s_nop 0
    s_nop 0
    s_nop 0
.size h3_wide_attention, .-h3_wide_attention

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel h3_wide_attention
    .amdhsa_group_segment_fixed_size 144388
    .amdhsa_private_segment_fixed_size 32
    .amdhsa_kernarg_size 496
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_dispatch_ptr 0
    .amdhsa_user_sgpr_queue_ptr 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_user_sgpr_dispatch_id 0
    .amdhsa_user_sgpr_kernarg_preload_length 0
    .amdhsa_user_sgpr_kernarg_preload_offset 0
    .amdhsa_user_sgpr_private_segment_size 0
    .amdhsa_uses_dynamic_stack 0
    .amdhsa_enable_private_segment 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 0
    .amdhsa_system_sgpr_workgroup_id_z 0
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 256
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 256
    .amdhsa_reserve_vcc 1
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_dx10_clamp 1
    .amdhsa_ieee_mode 1
    .amdhsa_fp16_overflow 0
    .amdhsa_tg_split 0
    .amdhsa_exception_fp_ieee_invalid_op 0
    .amdhsa_exception_fp_denorm_src 0
    .amdhsa_exception_fp_ieee_div_zero 0
    .amdhsa_exception_fp_ieee_overflow 0
    .amdhsa_exception_fp_ieee_underflow 0
    .amdhsa_exception_fp_ieee_inexact 0
    .amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel

; Each workgroup stages one key tile for one head. The short segment
; begins its own tile so no tile spans the sequence boundary.
.text
.p2align 8
.globl h3_pack_v
.protected h3_pack_v
.type h3_pack_v,@function
h3_pack_v:
    s_load_dword s6, s[0:1], 0x20
    s_load_dwordx2 s[4:5], s[0:1], 0x8
    s_lshl_b32 s7, s2, 6
    v_bfe_u32 v4, v0, 1, 2
    v_lshlrev_b32_e32 v3, 2, v0
    s_waitcnt lgkmcnt(0)
    s_add_i32 s8, s7, s6
    s_add_i32 s8, s8, 0xffff6c40
    s_cmpk_lt_i32 s2, 0x24f
    v_lshlrev_b32_e32 v5, 4, v0
    v_lshrrev_b32_e32 v1, 6, v0
    s_cselect_b32 s7, s7, s8
    v_mul_u32_u24_e32 v2, 0x2a00, v4
    v_and_b32_e32 v3, 0xe0, v3
    v_and_b32_e32 v5, 16, v5
    s_movk_i32 s8, 0x2a00
    v_lshl_add_u32 v11, v1, 3, s7
    v_or3_b32 v6, v5, v3, v2
    v_mov_b64_e32 v[2:3], s[4:5]
    v_mad_i64_i32 v[2:3], s[4:5], v11, s8, v[2:3]
    s_cselect_b32 s6, s6, 0x9380
    s_lshl_b32 s4, s3, 7
    s_ashr_i32 s5, s4, 31
    v_lshl_add_u64 v[8:9], s[4:5], 1, v[2:3]
    v_add_u32_e32 v2, v4, v11
    v_cmp_gt_i32_e32 vcc, s6, v2
    zero_accumulators 7, 2, 3, 4, 5
    s_and_saveexec_b64 s[4:5], vcc
    s_cbranch_execz .Lh3_pack_v_block_0
    v_lshl_add_u64 v[2:3], v[8:9], 0, v[6:7]
    global_load_dwordx4 v[2:5], v[2:3], off

.Lh3_pack_v_block_0:
    s_or_b64 exec, exec, s[4:5]
    v_and_b32_e32 v10, 63, v0
    v_lshlrev_b32_e32 v12, 4, v10
    s_movk_i32 s4, 0x900
    v_mad_u32_u24 v14, v1, s4, v12
    v_add_u32_e32 v6, 0xa800, v6
    s_waitcnt vmcnt(0)
    ds_write_b128 v14, v[2:5] offset:17408
    v_mul_hi_u32_u24_e32 v2, 0x61862, v6
    v_add_u32_e32 v2, v2, v11
    v_mul_u32_u24_e32 v13, 0x900, v1
    v_cmp_gt_i32_e32 vcc, s6, v2
    v_mov_b32_e32 v2, 0
    v_mov_b32_e32 v3, 0
    v_mov_b32_e32 v4, 0
    v_mov_b32_e32 v5, 0
    s_and_saveexec_b64 s[4:5], vcc
    s_cbranch_execz .Lh3_pack_v_block_1
    v_lshl_add_u64 v[2:3], v[8:9], 0, v[6:7]
    global_load_dwordx4 v[2:5], v[2:3], off

.Lh3_pack_v_block_1:
    s_or_b64 exec, exec, s[4:5]
    v_add_u32_e32 v6, v12, v13
    s_waitcnt vmcnt(0)
    ds_write_b128 v6, v[2:5] offset:18560
    v_bfe_i32 v2, v0, 4, 1
    s_mul_i32 s3, s3, 0x94400
    s_lshl_b32 s2, s2, 10
    v_and_b32_e32 v3, 0x480, v2
    v_or_b32_e32 v2, -8, v1
    s_add_i32 s3, s3, s2
    v_lshlrev_b32_e32 v1, 6, v1
    v_or3_b32 v6, s3, v1, v10
    v_lshlrev_b32_e32 v1, 1, v0
    s_load_dwordx2 s[0:1], s[0:1], 0x18
    v_lshrrev_b32_e32 v4, 5, v10
    v_and_b32_e32 v1, 0x380, v1
    s_movk_i32 s2, 0x1200
    v_mad_u32_u24 v1, v4, s2, v1
    v_lshlrev_b32_e32 v4, 3, v0
    v_and_b32_e32 v0, 3, v0
    v_and_b32_e32 v4, 0x60, v4
    v_lshlrev_b32_e32 v0, 3, v0
    v_or3_b32 v0, v1, v4, v0
    s_movk_i32 s2, 0x4400
    v_add3_u32 v0, v0, v3, s2
    s_mov_b64 s[2:3], 0
    s_waitcnt lgkmcnt(0)
    s_barrier

.Lh3_pack_v_block_2:
    ds_read_b64_tr_b16 v[8:9], v0
    ds_read_b64_tr_b16 v[10:11], v0 offset:2304
    v_add_co_u32_e32 v2, vcc, 8, v2
    s_xor_b64 s[4:5], vcc, -1
    s_and_b64 s[4:5], exec, s[4:5]
    v_lshl_add_u64 v[4:5], v[6:7], 4, s[0:1]
    v_add_u32_e32 v6, 0x200, v6
    v_add_u32_e32 v0, 0x2400, v0
    s_or_b64 s[2:3], s[4:5], s[2:3]
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v[4:5], v[8:11], off
    s_andn2_b64 exec, exec, s[2:3]
    s_cbranch_execnz .Lh3_pack_v_block_2
    s_endpgm
.size h3_pack_v, .-h3_pack_v

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel h3_pack_v
    .amdhsa_group_segment_fixed_size 35840
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 36
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_dispatch_ptr 0
    .amdhsa_user_sgpr_queue_ptr 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_user_sgpr_dispatch_id 0
    .amdhsa_user_sgpr_kernarg_preload_length 0
    .amdhsa_user_sgpr_kernarg_preload_offset 0
    .amdhsa_user_sgpr_private_segment_size 0
    .amdhsa_uses_dynamic_stack 0
    .amdhsa_enable_private_segment 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 0
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 15
    .amdhsa_next_free_sgpr 9
    .amdhsa_accum_offset 16
    .amdhsa_reserve_vcc 1
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_dx10_clamp 1
    .amdhsa_ieee_mode 1
    .amdhsa_fp16_overflow 0
    .amdhsa_tg_split 0
    .amdhsa_exception_fp_ieee_invalid_op 0
    .amdhsa_exception_fp_denorm_src 0
    .amdhsa_exception_fp_ieee_div_zero 0
    .amdhsa_exception_fp_ieee_overflow 0
    .amdhsa_exception_fp_ieee_underflow 0
    .amdhsa_exception_fp_ieee_inexact 0
    .amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           224
        .value_kind:     by_value
      - .offset:         224
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         232
        .size:           8
        .value_kind:     global_buffer
      - .offset:         240
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         244
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         248
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         252
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         254
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         256
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         258
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         260
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         262
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         280
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         288
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         296
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         304
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 144388
    .kernarg_segment_align: 8
    .kernarg_segment_size: 496
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           h3_wide_attention
    .private_segment_fixed_size: 32
    .sgpr_count:     99
    .sgpr_spill_count: 0
    .symbol:         h3_wide_attention.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 7
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 35840
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           h3_pack_v
    .private_segment_fixed_size: 0
    .sgpr_count:     15
    .sgpr_spill_count: 0
    .symbol:         h3_pack_v.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     15
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...
.end_amdgpu_metadata
