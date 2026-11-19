// RUN: triton-opt '--discrete-mask-access-conversion=compile-on-910-95=True compile-mode=simd_simt_template' %s | FileCheck %s --check-prefix=PASS
// RUN: triton-opt --triton-to-structured '--discrete-mask-access-conversion=compile-on-910-95=True compile-mode=simd_simt_template' '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt_template' %s | FileCheck %s --check-prefix=ROUTE

// Store-side carve-out on the A5/A6 register pipeline.
// Case A: pure broadcast discrete mask -> full read-modify-write.
// Case B: discrete broadcast & continuous column prefix -> bounded
//         read-modify-write.
// Case C: both axes varying -> must keep bailing to the SIMT template.

// PASS-LABEL: tt.func @store_axis_discrete_pure
// PASS: tt.load {{.*}} : tensor<8x16x!tt.ptr<i32>>
// PASS: arith.select
// PASS: tt.store {{.*}} {DiscreteMask} : tensor<8x16x!tt.ptr<i32>>
// PASS-LABEL: tt.func @store_axis_discrete_and
// PASS: tt.load {{.*}}, %{{.*}} : tensor<8x16x!tt.ptr<i32>>
// PASS: arith.select
// PASS: tt.store {{.*}}, %{{.*}} {DiscreteMask} : tensor<8x16x!tt.ptr<i32>>
// PASS-LABEL: tt.func @store_axis_ambiguous_negative
// PASS: tt.store {{.*}} {MixCompileDiscreteMask} : tensor<8x16x!tt.ptr<i32>>

// ROUTE-LABEL: tt.func @store_axis_discrete_pure
// ROUTE: arith.select
// ROUTE-NOT: ascend.indirect_store
// ROUTE-LABEL: tt.func @store_axis_discrete_and
// ROUTE: arith.select
// ROUTE-NOT: ascend.indirect_store
// ROUTE-LABEL: tt.func @store_axis_ambiguous_negative
// ROUTE: ascend.indirect_store

tt.func @store_axis_discrete_pure(%base: !tt.ptr<i32>) {
  %two = arith.constant dense<2> : tensor<8xi32>
  %zero = arith.constant dense<0> : tensor<8xi32>
  %val = arith.constant dense<7> : tensor<8x16xi32>
  %c16 = arith.constant dense<16> : tensor<8x1xi32>

  %k = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %g = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>

  %mod = arith.remsi %k, %two : tensor<8xi32>
  %row = arith.cmpi eq, %mod, %zero : tensor<8xi32>
  %row1 = tt.expand_dims %row {axis = 1 : i32} : tensor<8xi1> -> tensor<8x1xi1>
  %mask = tt.broadcast %row1 : tensor<8x1xi1> -> tensor<8x16xi1>

  %k1 = tt.expand_dims %k {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %k2 = arith.muli %k1, %c16 : tensor<8x1xi32>
  %k3 = tt.broadcast %k2 : tensor<8x1xi32> -> tensor<8x16xi32>
  %g1 = tt.expand_dims %g {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %g2 = tt.broadcast %g1 : tensor<1x16xi32> -> tensor<8x16xi32>
  %off = arith.addi %k3, %g2 : tensor<8x16xi32>

  %splat = tt.splat %base : !tt.ptr<i32> -> tensor<8x16x!tt.ptr<i32>>
  %ptrs = tt.addptr %splat, %off : tensor<8x16x!tt.ptr<i32>>, tensor<8x16xi32>

  tt.store %ptrs, %val, %mask : tensor<8x16x!tt.ptr<i32>>
  tt.return
}

tt.func @store_axis_discrete_and(%base: !tt.ptr<i32>, %limit: i32) {
  %two = arith.constant dense<2> : tensor<8xi32>
  %zero = arith.constant dense<0> : tensor<8xi32>
  %val = arith.constant dense<7> : tensor<8x16xi32>
  %c16 = arith.constant dense<16> : tensor<8x1xi32>

  %k = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %g = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>

  %mod = arith.remsi %k, %two : tensor<8xi32>
  %row = arith.cmpi eq, %mod, %zero : tensor<8xi32>
  %row1 = tt.expand_dims %row {axis = 1 : i32} : tensor<8xi1> -> tensor<8x1xi1>
  %disc = tt.broadcast %row1 : tensor<8x1xi1> -> tensor<8x16xi1>

  %lim = tt.splat %limit : i32 -> tensor<16xi32>
  %col = arith.cmpi slt, %g, %lim : tensor<16xi32>
  %col1 = tt.expand_dims %col {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %cont = tt.broadcast %col1 : tensor<1x16xi1> -> tensor<8x16xi1>

  %mask = arith.andi %disc, %cont : tensor<8x16xi1>

  %k1 = tt.expand_dims %k {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %k2 = arith.muli %k1, %c16 : tensor<8x1xi32>
  %k3 = tt.broadcast %k2 : tensor<8x1xi32> -> tensor<8x16xi32>
  %g1 = tt.expand_dims %g {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %g2 = tt.broadcast %g1 : tensor<1x16xi32> -> tensor<8x16xi32>
  %off = arith.addi %k3, %g2 : tensor<8x16xi32>

  %splat = tt.splat %base : !tt.ptr<i32> -> tensor<8x16x!tt.ptr<i32>>
  %ptrs = tt.addptr %splat, %off : tensor<8x16x!tt.ptr<i32>>, tensor<8x16xi32>

  tt.store %ptrs, %val, %mask : tensor<8x16x!tt.ptr<i32>>
  tt.return
}

tt.func @store_axis_ambiguous_negative(%base: !tt.ptr<i32>, %limit: i32) {
  %val = arith.constant dense<7> : tensor<8x16xi32>
  %c16 = arith.constant dense<16> : tensor<8x1xi32>

  %k = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %g = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>

  %k1 = tt.expand_dims %k {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %g1 = tt.expand_dims %g {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %kk = tt.broadcast %k1 : tensor<8x1xi32> -> tensor<8x16xi32>
  %gg = tt.broadcast %g1 : tensor<1x16xi32> -> tensor<8x16xi32>
  %sum = arith.addi %kk, %gg : tensor<8x16xi32>
  %lim = arith.constant dense<12> : tensor<8x16xi32>
  %mask = arith.cmpi slt, %sum, %lim : tensor<8x16xi32>

  %k2 = arith.muli %k1, %c16 : tensor<8x1xi32>
  %k3 = tt.broadcast %k2 : tensor<8x1xi32> -> tensor<8x16xi32>
  %off = arith.addi %k3, %gg : tensor<8x16xi32>

  %splat = tt.splat %base : !tt.ptr<i32> -> tensor<8x16x!tt.ptr<i32>>
  %ptrs = tt.addptr %splat, %off : tensor<8x16x!tt.ptr<i32>>, tensor<8x16xi32>

  tt.store %ptrs, %val, %mask : tensor<8x16x!tt.ptr<i32>>
  tt.return
}
