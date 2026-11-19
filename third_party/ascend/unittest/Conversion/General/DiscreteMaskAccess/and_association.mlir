// RUN: triton-opt '--discrete-mask-access-conversion=compile-on-910-95=True compile-mode=simd_simt_template' %s | FileCheck %s --check-prefix=LABEL
// RUN: triton-opt --triton-to-structured '--discrete-mask-access-conversion=compile-on-910-95=True compile-mode=simd_simt_template' '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt_template' %s | FileCheck %s --check-prefix=ROUTE

// AND association order must not change the classification or the rewrite:
// both (disc & cont1) & cont2 and disc & (cont1 & cont2) must keep the
// discrete leaf as discMask and the two column prefixes as contMask.

// LABEL-LABEL: tt.func @and_left_assoc
// LABEL: tt.load {{.*}} {AxisDiscreteMask}
// LABEL-LABEL: tt.func @and_right_assoc
// LABEL: tt.load {{.*}} {AxisDiscreteMask}

// ROUTE-LABEL: tt.func @and_left_assoc
// ROUTE: tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<8x16x!tt.ptr<i32>>
// ROUTE: arith.select
// ROUTE-NOT: ascend.indirect_load
// ROUTE-LABEL: tt.func @and_right_assoc
// ROUTE: tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<8x16x!tt.ptr<i32>>
// ROUTE: arith.select
// ROUTE-NOT: ascend.indirect_load

tt.func @and_left_assoc(%base: !tt.ptr<i32>, %limit1: i32, %limit2: i32) {
  %c0i = arith.constant dense<0> : tensor<8x16xi32>
  %two = arith.constant dense<2> : tensor<8xi32>
  %zero = arith.constant dense<0> : tensor<8xi32>

  %k = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %g = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>

  %mod = arith.remsi %k, %two : tensor<8xi32>
  %row = arith.cmpi eq, %mod, %zero : tensor<8xi32>
  %row1 = tt.expand_dims %row {axis = 1 : i32} : tensor<8xi1> -> tensor<8x1xi1>
  %disc = tt.broadcast %row1 : tensor<8x1xi1> -> tensor<8x16xi1>

  %l1 = tt.splat %limit1 : i32 -> tensor<16xi32>
  %col1 = arith.cmpi slt, %g, %l1 : tensor<16xi32>
  %col1e = tt.expand_dims %col1 {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %cont1 = tt.broadcast %col1e : tensor<1x16xi1> -> tensor<8x16xi1>

  %l2 = tt.splat %limit2 : i32 -> tensor<16xi32>
  %col2 = arith.cmpi slt, %g, %l2 : tensor<16xi32>
  %col2e = tt.expand_dims %col2 {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %cont2 = tt.broadcast %col2e : tensor<1x16xi1> -> tensor<8x16xi1>

  %inner = arith.andi %disc, %cont1 : tensor<8x16xi1>
  %mask = arith.andi %inner, %cont2 : tensor<8x16xi1>

  %k1 = tt.expand_dims %k {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %c16 = arith.constant dense<16> : tensor<8x1xi32>
  %k2 = arith.muli %k1, %c16 : tensor<8x1xi32>
  %k3 = tt.broadcast %k2 : tensor<8x1xi32> -> tensor<8x16xi32>
  %g1 = tt.expand_dims %g {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %g2 = tt.broadcast %g1 : tensor<1x16xi32> -> tensor<8x16xi32>
  %off = arith.addi %k3, %g2 : tensor<8x16xi32>

  %splat = tt.splat %base : !tt.ptr<i32> -> tensor<8x16x!tt.ptr<i32>>
  %ptrs = tt.addptr %splat, %off : tensor<8x16x!tt.ptr<i32>>, tensor<8x16xi32>

  %v = tt.load %ptrs, %mask, %c0i : tensor<8x16x!tt.ptr<i32>>
  tt.store %ptrs, %v : tensor<8x16x!tt.ptr<i32>>
  tt.return
}

tt.func @and_right_assoc(%base: !tt.ptr<i32>, %limit1: i32, %limit2: i32) {
  %c0i = arith.constant dense<0> : tensor<8x16xi32>
  %two = arith.constant dense<2> : tensor<8xi32>
  %zero = arith.constant dense<0> : tensor<8xi32>

  %k = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %g = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>

  %mod = arith.remsi %k, %two : tensor<8xi32>
  %row = arith.cmpi eq, %mod, %zero : tensor<8xi32>
  %row1 = tt.expand_dims %row {axis = 1 : i32} : tensor<8xi1> -> tensor<8x1xi1>
  %disc = tt.broadcast %row1 : tensor<8x1xi1> -> tensor<8x16xi1>

  %l1 = tt.splat %limit1 : i32 -> tensor<16xi32>
  %col1 = arith.cmpi slt, %g, %l1 : tensor<16xi32>
  %col1e = tt.expand_dims %col1 {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %cont1 = tt.broadcast %col1e : tensor<1x16xi1> -> tensor<8x16xi1>

  %l2 = tt.splat %limit2 : i32 -> tensor<16xi32>
  %col2 = arith.cmpi slt, %g, %l2 : tensor<16xi32>
  %col2e = tt.expand_dims %col2 {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %cont2 = tt.broadcast %col2e : tensor<1x16xi1> -> tensor<8x16xi1>

  %inner = arith.andi %cont1, %cont2 : tensor<8x16xi1>
  %mask = arith.andi %disc, %inner : tensor<8x16xi1>

  %k1 = tt.expand_dims %k {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %c16 = arith.constant dense<16> : tensor<8x1xi32>
  %k2 = arith.muli %k1, %c16 : tensor<8x1xi32>
  %k3 = tt.broadcast %k2 : tensor<8x1xi32> -> tensor<8x16xi32>
  %g1 = tt.expand_dims %g {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %g2 = tt.broadcast %g1 : tensor<1x16xi32> -> tensor<8x16xi32>
  %off = arith.addi %k3, %g2 : tensor<8x16xi32>

  %splat = tt.splat %base : !tt.ptr<i32> -> tensor<8x16x!tt.ptr<i32>>
  %ptrs = tt.addptr %splat, %off : tensor<8x16x!tt.ptr<i32>>, tensor<8x16xi32>

  %v = tt.load %ptrs, %mask, %c0i : tensor<8x16x!tt.ptr<i32>>
  tt.store %ptrs, %v : tensor<8x16x!tt.ptr<i32>>
  tt.return
}
