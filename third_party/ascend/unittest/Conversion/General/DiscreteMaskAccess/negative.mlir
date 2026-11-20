// RUN: triton-opt '--discrete-mask-access-conversion=compile-on-910-95=True compile-mode=simd_simt_template' --split-input-file %s | FileCheck %s --check-prefix=PASS
// RUN: triton-opt --triton-to-structured '--discrete-mask-access-conversion=compile-on-910-95=True compile-mode=simd_simt_template' '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt_template' --split-input-file %s | FileCheck %s --check-prefix=ROUTE

// Negative controls for the axis-discrete carve-out:
//   causal      - lower-triangular mask, not a rectangle
//   xori        - arith.xori is not understood, must stay conservative
//   c1          - per-row variable-length prefix, no constant axis
//   atomic      - gathered pointer, SIMT indirect atomic has no contMask

// PASS-LABEL: tt.func @causal_mask_negative
// PASS: tt.load {{.*}} {MixCompileDiscreteMask}
// PASS-LABEL: tt.func @xori_mask_negative
// PASS: tt.load {{.*}} {MixCompileDiscreteMask}
// PASS-LABEL: tt.func @c1_mask_negative
// PASS: tt.load {{.*}} {MixCompileDiscreteMask}
// PASS-LABEL: tt.func @atomic_gather_negative
// PASS: tt.atomic_rmw add, acq_rel, gpu, {{.*}} {ExplicitRouteToSIMD = false, MixCompileDiscreteMask}

// ROUTE-LABEL: tt.func @causal_mask_negative
// ROUTE: ascend.indirect_load
// ROUTE-LABEL: tt.func @xori_mask_negative
// ROUTE: ascend.indirect_load
// ROUTE-LABEL: tt.func @c1_mask_negative
// ROUTE: ascend.indirect_load
// ROUTE-LABEL: tt.func @atomic_gather_negative
// ROUTE: __builtin_indirect_atomic

tt.func @causal_mask_negative(%base: !tt.ptr<i32>) {
  %c0 = arith.constant dense<0> : tensor<8x16xi32>
  %c16 = arith.constant dense<16> : tensor<8x1xi32>

  %k = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %g = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>

  %k1 = tt.expand_dims %k {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %g1 = tt.expand_dims %g {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %kk = tt.broadcast %k1 : tensor<8x1xi32> -> tensor<8x16xi32>
  %gg = tt.broadcast %g1 : tensor<1x16xi32> -> tensor<8x16xi32>

  // mask = k >= g (lower triangular)
  %mask = arith.cmpi sge, %kk, %gg : tensor<8x16xi32>

  %k2 = arith.muli %k1, %c16 : tensor<8x1xi32>
  %k3 = tt.broadcast %k2 : tensor<8x1xi32> -> tensor<8x16xi32>
  %off = arith.addi %k3, %gg : tensor<8x16xi32>

  %splat = tt.splat %base : !tt.ptr<i32> -> tensor<8x16x!tt.ptr<i32>>
  %ptrs = tt.addptr %splat, %off : tensor<8x16x!tt.ptr<i32>>, tensor<8x16xi32>

  %v = tt.load %ptrs, %mask, %c0 : tensor<8x16x!tt.ptr<i32>>
  tt.store %ptrs, %v : tensor<8x16x!tt.ptr<i32>>
  tt.return
}

// -----

tt.func @xori_mask_negative(%base: !tt.ptr<i32>) {
  %c128 = arith.constant dense<128> : tensor<4x1xi32>
  %c16 = arith.constant dense<16> : tensor<1x8xi32>
  %c0i = arith.constant dense<0> : tensor<4x8x16xi32>
  %true = arith.constant dense<true> : tensor<4x8x16xi1>

  %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %c = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %k = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>

  %r1 = tt.expand_dims %r {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %r2 = arith.muli %r1, %c128 : tensor<4x1xi32>
  %r3 = tt.expand_dims %r2 {axis = 2 : i32} : tensor<4x1xi32> -> tensor<4x1x1xi32>
  %r4 = tt.broadcast %r3 : tensor<4x1x1xi32> -> tensor<4x8x16xi32>

  %c1 = tt.expand_dims %c {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %c2 = arith.muli %c1, %c16 : tensor<1x8xi32>
  %c3 = tt.expand_dims %c2 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
  %c4 = tt.broadcast %c3 : tensor<1x8x1xi32> -> tensor<4x8x16xi32>

  %k1 = tt.expand_dims %k {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %k2 = tt.expand_dims %k1 {axis = 0 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
  %k3 = tt.broadcast %k2 : tensor<1x1x16xi32> -> tensor<4x8x16xi32>

  %off_rc = arith.addi %r4, %c4 : tensor<4x8x16xi32>
  %off = arith.addi %off_rc, %k3 : tensor<4x8x16xi32>

  %splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x8x16x!tt.ptr<i32>>
  %ptrs = tt.addptr %splat, %off : tensor<4x8x16x!tt.ptr<i32>>, tensor<4x8x16xi32>

  %two = arith.constant dense<2> : tensor<8xi32>
  %zero2 = arith.constant dense<0> : tensor<8xi32>
  %mod = arith.remsi %c, %two : tensor<8xi32>
  %valid = arith.cmpi eq, %mod, %zero2 : tensor<8xi32>
  %v1 = tt.expand_dims %valid {axis = 0 : i32} : tensor<8xi1> -> tensor<1x8xi1>
  %v2 = tt.broadcast %v1 : tensor<1x8xi1> -> tensor<4x8xi1>
  %v3 = tt.expand_dims %v2 {axis = 2 : i32} : tensor<4x8xi1> -> tensor<4x8x1xi1>
  %base_mask = tt.broadcast %v3 : tensor<4x8x1xi1> -> tensor<4x8x16xi1>
  %mask = arith.xori %base_mask, %true : tensor<4x8x16xi1>

  %v = tt.load %ptrs, %mask, %c0i : tensor<4x8x16x!tt.ptr<i32>>
  tt.store %ptrs, %v : tensor<4x8x16x!tt.ptr<i32>>
  tt.return
}

// -----

tt.func @c1_mask_negative(%base: !tt.ptr<i32>, %boundp: !tt.ptr<i32>) {
  %c0 = arith.constant dense<0> : tensor<8x16xi32>
  %c16 = arith.constant dense<16> : tensor<8x1xi32>

  %k = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %g = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>

  %sb = tt.splat %boundp : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
  %bk = tt.addptr %sb, %k : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
  %bound = tt.load %bk : tensor<8x!tt.ptr<i32>>
  %b1 = tt.expand_dims %bound {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %b2 = tt.broadcast %b1 : tensor<8x1xi32> -> tensor<8x16xi32>

  %g1 = tt.expand_dims %g {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %g2 = tt.broadcast %g1 : tensor<1x16xi32> -> tensor<8x16xi32>
  %mask = arith.cmpi slt, %g2, %b2 : tensor<8x16xi32>

  %k1 = tt.expand_dims %k {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %k2 = arith.muli %k1, %c16 : tensor<8x1xi32>
  %k3 = tt.broadcast %k2 : tensor<8x1xi32> -> tensor<8x16xi32>
  %off = arith.addi %k3, %g2 : tensor<8x16xi32>

  %splat = tt.splat %base : !tt.ptr<i32> -> tensor<8x16x!tt.ptr<i32>>
  %ptrs = tt.addptr %splat, %off : tensor<8x16x!tt.ptr<i32>>, tensor<8x16xi32>

  %v = tt.load %ptrs, %mask, %c0 : tensor<8x16x!tt.ptr<i32>>
  tt.store %ptrs, %v : tensor<8x16x!tt.ptr<i32>>
  tt.return
}

// -----

tt.func @atomic_gather_negative(%base: !tt.ptr<i32>, %idxp: !tt.ptr<i32>) {
  %c16 = arith.constant dense<16> : tensor<8xi32>
  %zero = arith.constant dense<0> : tensor<8xi32>
  %val = arith.constant dense<1> : tensor<8x16xi32>

  %g = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %sidx = tt.splat %idxp : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
  %idx = tt.load %sidx : tensor<8x!tt.ptr<i32>>
  %row = arith.muli %idx, %c16 : tensor<8xi32>
  %row1 = tt.expand_dims %row {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %row2 = tt.broadcast %row1 : tensor<8x1xi32> -> tensor<8x16xi32>
  %g1 = tt.expand_dims %g {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %g2 = tt.broadcast %g1 : tensor<1x16xi32> -> tensor<8x16xi32>
  %off = arith.addi %row2, %g2 : tensor<8x16xi32>

  %splat = tt.splat %base : !tt.ptr<i32> -> tensor<8x16x!tt.ptr<i32>>
  %ptrs = tt.addptr %splat, %off : tensor<8x16x!tt.ptr<i32>>, tensor<8x16xi32>

  %valid = arith.cmpi sge, %idx, %zero : tensor<8xi32>
  %v1 = tt.expand_dims %valid {axis = 1 : i32} : tensor<8xi1> -> tensor<8x1xi1>
  %mask = tt.broadcast %v1 : tensor<8x1xi1> -> tensor<8x16xi1>

  %old = tt.atomic_rmw add, acq_rel, gpu, %ptrs, %val, %mask : (tensor<8x16x!tt.ptr<i32>>, tensor<8x16xi32>, tensor<8x16xi1>) -> tensor<8x16xi32>
  %outp = tt.addptr %splat, %off : tensor<8x16x!tt.ptr<i32>>, tensor<8x16xi32>
  tt.store %outp, %old : tensor<8x16x!tt.ptr<i32>>
  tt.return
}
