import VeriTile.Triton.Launch.Blocked1DConfig
open VeriTile.Triton
set_option pp.all false
set_option pp.proofs false
#print ElemDType
#print BufMeta
#print Blocked1DLaunch
#print Blocked1DLaunch.gridX
#print Blocked1DLaunch.bufs
#print Blocked1DLaunch.i32Limit
#print Blocked1DLaunch.RangesDisjoint
#print Blocked1DLaunch.GridRank
#print Blocked1DLaunch.BlockOk
#print Blocked1DLaunch.Covers
#print Blocked1DLaunch.LanesInBounds
#print Blocked1DLaunch.OffsetsFit
#print Blocked1DLaunch.NFits
#print Blocked1DLaunch.GridFits
#print Blocked1DLaunch.UnitStride
#print Blocked1DLaunch.DTypeOk
#print Blocked1DLaunch.OutDisjoint
#print Blocked1DLaunch.Pre
#check @Blocked1DLaunch.check_ok
#check @Blocked1DLaunch.check_complete
#check @Blocked1DLaunch.Pre.offset_injective
#check @Blocked1DLaunch.Pre.i32_offset_toInt
#check @Blocked1DLaunch.Pre.i32_mask_eq
