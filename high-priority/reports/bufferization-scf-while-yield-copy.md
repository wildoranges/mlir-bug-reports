# [fuzzer-found][mlir][bufferization][scf] scf.while does not copy tensors yielded from the "after" region back to the "before" region; in-place writes clobber outside tensors (segfault on constants)

With `-one-shot-bufferize="allow-return-allocs-from-loops"`, an `scf.while` whose **after** region yields a tensor defined *outside* the loop gets that buffer passed, uncopied, to the **before** region's block argument. `WhileOpInterface::isWritable` treats that argument as always writable:
https://github.com/llvm/llvm-project/blob/8445b527ed0da1f9156f27be8c193624d0292253/mlir/lib/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.cpp#L884-L893

So an in-place write in the before region clobbers the outside tensor, or segfaults when that tensor is a constant (a write into a read-only `memref.global`).

`WhileOpInterface::resolveConflicts` inserts copies only for the `scf.condition` operands (the "Update before region" loop). There is no corresponding copy for the `scf.yield` of the after region:
https://github.com/llvm/llvm-project/blob/8445b527ed0da1f9156f27be8c193624d0292253/mlir/lib/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.cpp#L895-L945

`scf.for` does copy every yielded value that may alias a value defined outside the loop (`doesNotAliasExternalValue`):
https://github.com/llvm/llvm-project/blob/8445b527ed0da1f9156f27be8c193624d0292253/mlir/lib/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.cpp#L668-L692

The equivalent copy for `scf.while` appears to have been removed in 6247988e0751 (D133477, 2022).

`while.mlir`:
```mlir
// The "after" region yields %ext (a tensor defined outside the loop) to the
// "before" region. The "before" region writes into its iter_arg %w.
// Value semantics: %ext is never modified -> must print 7 7 7 7.
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100.0 : f32
  %cstA = arith.constant dense<1.0> : tensor<4xf32>
  %cstE = arith.constant dense<7.0> : tensor<4xf32>
  %A = bufferization.alloc_tensor() copy(%cstA) : tensor<4xf32>
  %ext = bufferization.alloc_tensor() copy(%cstE) : tensor<4xf32>
  %r:2 = scf.while (%w = %A, %i = %c0) : (tensor<4xf32>, index) -> (tensor<4xf32>, index) {
    %w2 = tensor.insert %c100 into %w[%c0] : tensor<4xf32>
    %cond = arith.cmpi slt, %i, %c1 : index
    scf.condition(%cond) %w2, %i : tensor<4xf32>, index
  } do {
  ^bb0(%b: tensor<4xf32>, %j: index):
    %j1 = arith.addi %j, %c1 : index
    scf.yield %ext, %j1 : tensor<4xf32>, index
  }
  %pad = arith.constant 0.0 : f32
  %ve = vector.transfer_read %ext[%c0], %pad {in_bounds = [true]} : tensor<4xf32>, vector<4xf32>
  %vr = vector.transfer_read %r#0[%c0], %pad {in_bounds = [true]} : tensor<4xf32>, vector<4xf32>
  // expected: ( 7, 7, 7, 7 )
  vector.print %ve : vector<4xf32>
  // expected: ( 100, 7, 7, 7 )
  vector.print %vr : vector<4xf32>
  return
}
```

(`$LLVM_BUILD` is the LLVM/MLIR build directory.)

```console
$ LOWER="-buffer-deallocation-pipeline -convert-bufferization-to-memref -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm -expand-strided-metadata -finalize-memref-to-llvm -convert-func-to-llvm -convert-arith-to-llvm -convert-cf-to-llvm -convert-index-to-llvm -reconcile-unrealized-casts"
$ RUN="mlir-runner -e main -entry-point-result=void -O0 -shared-libs=$LLVM_BUILD/lib/libmlir_runner_utils.so,$LLVM_BUILD/lib/libmlir_c_runner_utils.so"
$ mlir-opt while.mlir -one-shot-bufferize="allow-return-allocs-from-loops copy-before-write" $LOWER | $RUN
( 7, 7, 7, 7 )
( 100, 7, 7, 7 )
$ mlir-opt while.mlir -one-shot-bufferize="allow-return-allocs-from-loops" $LOWER | $RUN
( 100, 7, 7, 7 )
( 100, 7, 7, 7 )
```

The first line printed is `%ext`. By value semantics it is never modified and must stay `( 7, 7, 7, 7 )`. With the default analysis it becomes `( 100, 7, 7, 7 )`. The `copy-before-write` run serves as the reference.

When the yielded outside tensor is a constant, the in-place write goes into a read-only global:

`while_const.mlir`:
```mlir
// The "after" region yields a constant tensor. Value semantics: the loop
// result is the constant with element 0 replaced by 100.
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100.0 : f32
  %init = arith.constant dense<1.0> : tensor<4xf32>
  %cst = arith.constant dense<7.0> : tensor<4xf32>
  %r:2 = scf.while (%w = %init, %i = %c0) : (tensor<4xf32>, index) -> (tensor<4xf32>, index) {
    %w2 = tensor.insert %c100 into %w[%c0] : tensor<4xf32>
    %cond = arith.cmpi slt, %i, %c1 : index
    scf.condition(%cond) %w2, %i : tensor<4xf32>, index
  } do {
  ^bb0(%b: tensor<4xf32>, %j: index):
    %j1 = arith.addi %j, %c1 : index
    scf.yield %cst, %j1 : tensor<4xf32>, index
  }
  %pad = arith.constant 0.0 : f32
  %vr = vector.transfer_read %r#0[%c0], %pad {in_bounds = [true]} : tensor<4xf32>, vector<4xf32>
  // expected: ( 100, 7, 7, 7 )
  vector.print %vr : vector<4xf32>
  return
}
```

```console
$ mlir-opt while_const.mlir -one-shot-bufferize="allow-return-allocs-from-loops copy-before-write" $LOWER | $RUN
( 100, 7, 7, 7 )
$ mlir-opt while_const.mlir -one-shot-bufferize="allow-return-allocs-from-loops" $LOWER | $RUN
Segmentation fault (core dumped)      # exit code 139
```

Both inputs verify and are UB-free, using only tensor value semantics.

**Version**: llvm-project main @ 8445b527ed0da1f9156f27be8c193624d0292253 (2026-11-16), Release + `LLVM_ENABLE_ASSERTIONS=ON`, x86_64 Linux.

This testcase was generated by an automated MLIR testing tool that I am developing; some of its steps use AI.

*Disclosure: I wrote this report with the help of Claude Code. I reproduced and reviewed everything in this report myself.*
