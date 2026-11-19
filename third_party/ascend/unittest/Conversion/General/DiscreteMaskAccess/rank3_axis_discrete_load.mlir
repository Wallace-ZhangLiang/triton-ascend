// RUN: triton-opt '--discrete-mask-access-conversion=compile-on-910-95=True compile-mode=simd_simt_template' %s | FileCheck %s --check-prefix=LABEL
// RUN: triton-opt --triton-to-structured '--discrete-mask-access-conversion=compile-on-910-95=True compile-mode=simd_simt_template' '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt_template' %s | FileCheck %s --check-prefix=ROUTE

// Rank-3 axis-discrete mask load probes.
//
// Case A: pure broadcast discrete mask (discMask only, contMask == null).
// Case B: top-level AND of discrete broadcast + continuous column prefix.
// Case C: negative control -- arith.xori over a broadcast must stay rejected.
//
// LABEL checks that the routing predicate accepts rank 3 and stays
// conservative on the xori form; ROUTE checks the rewrite keeps the loads on
// the SIMD path (bounded load + select, no indirect access) and that the
// negative control still falls back to the SIMT indirect access.

// LABEL-LABEL: tt.func @rank3_axis_discrete_load_pure_broadcast
// LABEL: tt.load {{.*}} {AxisDiscreteMask}
// LABEL-LABEL: tt.func @rank3_axis_discrete_load_and_prefix
// LABEL: tt.load {{.*}} {AxisDiscreteMask}
// LABEL-LABEL: tt.func @rank3_axis_discrete_load_xori_negative
// LABEL: tt.load {{.*}} {MixCompileDiscreteMask}
// LABEL-LABEL: tt.func @rank3_axis_discrete_load_select
// LABEL: tt.load {{.*}} {AxisDiscreteMask}

// ROUTE-LABEL: tt.func @rank3_axis_discrete_load_pure_broadcast
// ROUTE: tt.load
// ROUTE: arith.select
// ROUTE-NOT: ascend.indirect_load
// ROUTE-LABEL: tt.func @rank3_axis_discrete_load_and_prefix
// ROUTE: tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<4x8x16x!tt.ptr<i32>>
// ROUTE: arith.select
// ROUTE-NOT: ascend.indirect_load
// ROUTE-LABEL: tt.func @rank3_axis_discrete_load_xori_negative
// ROUTE: ascend.indirect_load
// ROUTE-LABEL: tt.func @rank3_axis_discrete_load_select
// ROUTE: tt.load
// ROUTE: arith.select
// ROUTE-NOT: ascend.indirect_load

// -------- Case A: pure broadcast --------
tt.func @rank3_axis_discrete_load_pure_broadcast(
    %base: !tt.ptr<i32>, %limit: i32) {
  %c128 = arith.constant dense<128> : tensor<4x1xi32>
  %c16 = arith.constant dense<16> : tensor<1x8xi32>
  %c0i = arith.constant dense<0> : tensor<4x8x16xi32>

  %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %c = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %k = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>

  // offsets = r*128 + c*16 + k
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

  // Discrete leaf: (c % 2) == 0 cannot be parsed as a range, so this mask is
  // discrete and constant along axis 2 (and axis 0).
  %two = arith.constant dense<2> : tensor<8xi32>
  %zero2 = arith.constant dense<0> : tensor<8xi32>
  %mod = arith.remsi %c, %two : tensor<8xi32>
  %valid = arith.cmpi eq, %mod, %zero2 : tensor<8xi32>
  %v1 = tt.expand_dims %valid {axis = 0 : i32} : tensor<8xi1> -> tensor<1x8xi1>
  %v2 = tt.broadcast %v1 : tensor<1x8xi1> -> tensor<4x8xi1>
  %v3 = tt.expand_dims %v2 {axis = 2 : i32} : tensor<4x8xi1> -> tensor<4x8x1xi1>
  %mask = tt.broadcast %v3 : tensor<4x8x1xi1> -> tensor<4x8x16xi1>

  %v = tt.load %ptrs, %mask, %c0i : tensor<4x8x16x!tt.ptr<i32>>
  tt.store %ptrs, %v : tensor<4x8x16x!tt.ptr<i32>>
  tt.return
}

// -------- Case B: discrete broadcast AND continuous column prefix --------
tt.func @rank3_axis_discrete_load_and_prefix(
    %base: !tt.ptr<i32>, %limit: i32, %klimit: i32) {
  %c128 = arith.constant dense<128> : tensor<4x1xi32>
  %c16 = arith.constant dense<16> : tensor<1x8xi32>
  %c0i = arith.constant dense<0> : tensor<4x8x16xi32>

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

  // Discrete: broadcast((c % 2) == 0), invariant along axis 2.
  %two = arith.constant dense<2> : tensor<8xi32>
  %zero2 = arith.constant dense<0> : tensor<8xi32>
  %mod = arith.remsi %c, %two : tensor<8xi32>
  %valid = arith.cmpi eq, %mod, %zero2 : tensor<8xi32>
  %v1 = tt.expand_dims %valid {axis = 0 : i32} : tensor<8xi1> -> tensor<1x8xi1>
  %v2 = tt.broadcast %v1 : tensor<1x8xi1> -> tensor<4x8xi1>
  %v3 = tt.expand_dims %v2 {axis = 2 : i32} : tensor<4x8xi1> -> tensor<4x8x1xi1>
  %disc = tt.broadcast %v3 : tensor<4x8x1xi1> -> tensor<4x8x16xi1>

  // Continuous: column prefix k < klimit, invariant along axes 0 and 1.
  %klim = tt.splat %klimit : i32 -> tensor<16xi32>
  %col = arith.cmpi slt, %k, %klim : tensor<16xi32>
  %col1 = tt.expand_dims %col {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
  %col2 = tt.expand_dims %col1 {axis = 0 : i32} : tensor<1x16xi1> -> tensor<1x1x16xi1>
  %cont = tt.broadcast %col2 : tensor<1x1x16xi1> -> tensor<4x8x16xi1>

  %mask = arith.andi %disc, %cont : tensor<4x8x16xi1>

  %v = tt.load %ptrs, %mask, %c0i : tensor<4x8x16x!tt.ptr<i32>>
  tt.store %ptrs, %v : tensor<4x8x16x!tt.ptr<i32>>
  tt.return
}

// -------- Case C: negative control, xori is not understood --------
tt.func @rank3_axis_discrete_load_xori_negative(
    %base: !tt.ptr<i32>, %limit: i32) {
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

// -------- Case D: select form, applied at full rank --------
tt.func @rank3_axis_discrete_load_select(
    %base: !tt.ptr<i32>) {
  %c128 = arith.constant dense<128> : tensor<4x1xi32>
  %c16 = arith.constant dense<16> : tensor<1x8xi32>
  %c0i = arith.constant dense<0> : tensor<4x8x16xi32>
  %zero8 = arith.constant dense<0> : tensor<8xi32>
  %k4 = arith.constant dense<4> : tensor<8xi32>
  %k6 = arith.constant dense<6> : tensor<8xi32>

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

  // Three row-wise predicates combined by arith.select at full rank.  Every
  // operand is constant along axis 2, so the select is too.
  %cond = arith.cmpi slt, %c, %k6 : tensor<8xi32>
  %first = arith.cmpi slt, %c, %k4 : tensor<8xi32>
  %second = arith.cmpi sge, %c, %zero8 : tensor<8xi32>

  %cond1 = tt.expand_dims %cond {axis = 0 : i32} : tensor<8xi1> -> tensor<1x8xi1>
  %cond2 = tt.broadcast %cond1 : tensor<1x8xi1> -> tensor<4x8xi1>
  %cond3 = tt.expand_dims %cond2 {axis = 2 : i32} : tensor<4x8xi1> -> tensor<4x8x1xi1>
  %condF = tt.broadcast %cond3 : tensor<4x8x1xi1> -> tensor<4x8x16xi1>

  %first1 = tt.expand_dims %first {axis = 0 : i32} : tensor<8xi1> -> tensor<1x8xi1>
  %first2 = tt.broadcast %first1 : tensor<1x8xi1> -> tensor<4x8xi1>
  %first3 = tt.expand_dims %first2 {axis = 2 : i32} : tensor<4x8xi1> -> tensor<4x8x1xi1>
  %firstF = tt.broadcast %first3 : tensor<4x8x1xi1> -> tensor<4x8x16xi1>

  %second1 = tt.expand_dims %second {axis = 0 : i32} : tensor<8xi1> -> tensor<1x8xi1>
  %second2 = tt.broadcast %second1 : tensor<1x8xi1> -> tensor<4x8xi1>
  %second3 = tt.expand_dims %second2 {axis = 2 : i32} : tensor<4x8xi1> -> tensor<4x8x1xi1>
  %secondF = tt.broadcast %second3 : tensor<4x8x1xi1> -> tensor<4x8x16xi1>

  %mask = arith.select %condF, %firstF, %secondF : tensor<4x8x16xi1>, tensor<4x8x16xi1>

  %v = tt.load %ptrs, %mask, %c0i : tensor<4x8x16x!tt.ptr<i32>>
  tt.store %ptrs, %v : tensor<4x8x16x!tt.ptr<i32>>
  tt.return
}
