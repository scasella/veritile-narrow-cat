
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_cpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'ks3': 'i64', 'ks4': 'i64', 'ks5': 'i64', 'ks6': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cpu', index=None, multi_processor_count=8, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=1024, warp_size=None), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 1507328}, 'backend_hash': 'stub', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def Placeholder.KERNEL_NAME(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = (x0).to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (x0).to(tl.int64)
    tmp4 = (tmp3).to(tl.int64)
    tmp5 = (ks1 + ks2 + ks3).to(tl.int64)
    tmp6 = (tmp5).to(tl.int64)
    tmp7 = tmp4 < tmp6
    tmp8 = (x0).to(tl.int32)
    tmp9 = tl.full([1], 0, tl.int64)
    tmp10 = tmp8 >= tmp9
    tmp11 = (x0).to(tl.int64)
    tmp12 = (tmp11).to(tl.int64)
    tmp13 = (tl.broadcast_to(ks1, [XBLOCK])).to(tl.int64)
    tmp14 = (tmp13).to(tl.int64)
    tmp15 = tmp12 < tmp14
    tmp16 = tmp15 & tmp7
    tmp17 = tl.load(in_ptr0 + (ks1*x1 + (x0)), tmp16 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tmp12 >= tmp14
    tmp19 = (tl.broadcast_to(ks1 + ks3, [XBLOCK])).to(tl.int64)
    tmp20 = (tmp19).to(tl.int64)
    tmp21 = tmp12 < tmp20
    tmp22 = tmp18 & tmp21
    tmp23 = tmp22 & tmp7
    tmp24 = tl.load(in_ptr1 + (ks3*x1 + (((-1)*ks1) + (x0))), tmp23 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp25 = (tl.broadcast_to(ks1 + ks3, [XBLOCK])).to(tl.int32)
    tmp26 = tmp8 >= tmp25
    tmp27 = (tl.broadcast_to(ks1 + ks2 + ks3, [XBLOCK])).to(tl.int32)
    tmp28 = tmp8 < tmp27
    tmp29 = tmp26 & tmp7
    tmp30 = tl.load(in_ptr2 + (ks2*x1 + (((-1)*ks1) + ((-1)*ks3) + (x0))), tmp29 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tl.load(in_ptr3 + (ks2*x1 + (((-1)*ks1) + ((-1)*ks3) + (x0))), tmp29 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tmp30 + tmp31
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp29, tmp32, tmp33)
    tmp35 = tl.where(tmp22, tmp24, tmp34)
    tmp36 = tl.where(tmp15, tmp17, tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp7, tmp36, tmp37)
    tmp39 = (ks1 + ks2 + ks3).to(tl.int32)
    tmp40 = tmp0 >= tmp39
    tmp41 = (ks0).to(tl.int32)
    tmp42 = tmp0 < tmp41
    tmp43 = (x0 + ((-1)*ks1) + ((-1)*ks2) + ((-1)*ks3)).to(tl.int32)
    tmp44 = tl.full([1], 0, tl.int64)
    tmp45 = tmp43 >= tmp44
    tmp46 = (x0 + ((-1)*ks1) + ((-1)*ks2) + ((-1)*ks3)).to(tl.int64)
    tmp47 = (tmp46).to(tl.int64)
    tmp48 = (tl.broadcast_to(ks4, [XBLOCK])).to(tl.int64)
    tmp49 = (tmp48).to(tl.int64)
    tmp50 = tmp47 < tmp49
    tmp51 = tmp50 & tmp40
    tmp52 = tl.load(in_ptr4 + (ks4*x1 + (x0 + ((-1)*ks1) + ((-1)*ks2) + ((-1)*ks3))), tmp51 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp53 = tmp47 >= tmp49
    tmp54 = (tl.broadcast_to(ks4 + ks5, [XBLOCK])).to(tl.int64)
    tmp55 = (tmp54).to(tl.int64)
    tmp56 = tmp47 < tmp55
    tmp57 = tmp53 & tmp56
    tmp58 = tmp57 & tmp40
    tmp59 = tl.load(in_ptr5 + (ks5*x1 + (((-1)*ks4) + (x0 + ((-1)*ks1) + ((-1)*ks2) + ((-1)*ks3)))), tmp58 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp60 = (tl.broadcast_to(ks4 + ks5, [XBLOCK])).to(tl.int32)
    tmp61 = tmp43 >= tmp60
    tmp62 = (tl.broadcast_to(ks4 + ks5 + ks6, [XBLOCK])).to(tl.int32)
    tmp63 = tmp43 < tmp62
    tmp64 = tmp61 & tmp40
    tmp65 = tl.load(in_ptr6 + (ks6*x1 + (((-1)*ks4) + ((-1)*ks5) + (x0 + ((-1)*ks1) + ((-1)*ks2) + ((-1)*ks3)))), tmp64 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp66 = tl.load(in_ptr7 + (ks6*x1 + (((-1)*ks4) + ((-1)*ks5) + (x0 + ((-1)*ks1) + ((-1)*ks2) + ((-1)*ks3)))), tmp64 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp67 = tmp65 + tmp66
    tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
    tmp69 = tl.where(tmp64, tmp67, tmp68)
    tmp70 = tl.where(tmp57, tmp59, tmp69)
    tmp71 = tl.where(tmp50, tmp52, tmp70)
    tmp72 = tl.full(tmp71.shape, 0.0, tmp71.dtype)
    tmp73 = tl.where(tmp40, tmp71, tmp72)
    tmp74 = tl.where(tmp7, tmp38, tmp73)
    tl.store(out_ptr0 + (x2), tmp74, xmask)
