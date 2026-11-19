// RUN: triton-opt '--discrete-mask-access-conversion=compile-on-910-95=True compile-mode=simd_simt_template' %s | FileCheck %s --check-prefix=LABEL
// RUN: triton-opt --triton-to-structured '--discrete-mask-access-conversion=compile-on-910-95=True compile-mode=simd_simt_template' '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt_template' %s | FileCheck %s --check-prefix=ROUTE

// Causal (lower-triangular) masks are not rectangles and are not constant
// along either axis, so they must keep falling back to the SIMT template.

// LABEL-LABEL: tt.func @causal_mask_negative
// LABEL: tt.load {{.*}} {MixCompileDiscreteMask}

// ROUTE-LABEL: tt.func @causal_mask_negative
// ROUTE: ascend.indirect_load

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
