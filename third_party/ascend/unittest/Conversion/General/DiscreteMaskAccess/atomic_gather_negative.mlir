// RUN: triton-opt '--discrete-mask-access-conversion=compile-on-910-95=True compile-mode=simd_simt_template' %s | FileCheck %s --check-prefix=PASS
// RUN: triton-opt --triton-to-structured '--discrete-mask-access-conversion=compile-on-910-95=True compile-mode=simd_simt_template' '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt_template' %s | FileCheck %s --check-prefix=ROUTE

// Gather-row atomic: the pointer has two unstructured (gathered) axes, so
// routeToSIMD fails and the op must keep the original mask for the SIMT
// indirect atomic ABI (per-lane offsets + per-lane i8 mask, no contMask).

// PASS-LABEL: tt.func @atomic_gather_negative
// PASS: tt.atomic_rmw add, acq_rel, gpu, {{.*}} {ExplicitRouteToSIMD = false, MixCompileDiscreteMask}

// ROUTE-LABEL: tt.func @atomic_gather_negative
// ROUTE: __builtin_indirect_atomic

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
