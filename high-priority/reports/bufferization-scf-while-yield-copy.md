# [fuzzer-found][mlir][bufferization][scf] scf.while does not copy tensors yielded from the "after" region back to the "before" region; in-place writes clobber outside tensors (segfault on constants)

## LLVM version and environment

- Commit: `0cae57720c3ca7cb09b5ba48d199f1ba04e3aae2` (2026-11-26)
- Build: `Release` with `LLVM_ENABLE_ASSERTIONS=ON`
- Platform: `x86_64` Linux

## Reproducer

**Case 1: clobbering an outside mutable tensor (`while.mlir`)**
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

**Case 2: writing into a read-only constant tensor (`while_const.mlir`)**
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

Commands (`$LLVM_BUILD` is the LLVM/MLIR build directory):
```console
$ LOWER="-buffer-deallocation-pipeline -convert-bufferization-to-memref -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm -expand-strided-metadata -finalize-memref-to-llvm -convert-func-to-llvm -convert-arith-to-llvm -convert-cf-to-llvm -convert-index-to-llvm -reconcile-unrealized-casts"
$ RUN="mlir-runner -e main -entry-point-result=void -O0 -shared-libs=$LLVM_BUILD/lib/libmlir_runner_utils.so,$LLVM_BUILD/lib/libmlir_c_runner_utils.so"
$ mlir-opt while.mlir -one-shot-bufferize="allow-return-allocs-from-loops copy-before-write" $LOWER | $RUN
$ mlir-opt while.mlir -one-shot-bufferize="allow-return-allocs-from-loops" $LOWER | $RUN
$ mlir-opt while_const.mlir -one-shot-bufferize="allow-return-allocs-from-loops copy-before-write" $LOWER | $RUN
$ mlir-opt while_const.mlir -one-shot-bufferize="allow-return-allocs-from-loops" $LOWER | $RUN
```


## Expected behaviour

Both inputs verify and are UB-free, using only tensor value semantics:
- In `while.mlir`, `%ext` is never modified, so it must stay `( 7, 7, 7, 7 )` and `%r#0` must be `( 100, 7, 7, 7 )` (matching the `copy-before-write` reference run).
- In `while_const.mlir`, the program should print `( 100, 7, 7, 7 )` without crashing.

## Actual behaviour

With `-one-shot-bufferize="allow-return-allocs-from-loops"`, the buffer yielded from the `after` region is passed uncopied to the `before` region and modified in place:

```console
$ mlir-opt while.mlir -one-shot-bufferize="allow-return-allocs-from-loops copy-before-write" $LOWER | $RUN
( 7, 7, 7, 7 )
( 100, 7, 7, 7 )
$ mlir-opt while.mlir -one-shot-bufferize="allow-return-allocs-from-loops" $LOWER | $RUN
( 100, 7, 7, 7 )
( 100, 7, 7, 7 )
$ mlir-opt while_const.mlir -one-shot-bufferize="allow-return-allocs-from-loops copy-before-write" $LOWER | $RUN
( 100, 7, 7, 7 )
$ mlir-opt while_const.mlir -one-shot-bufferize="allow-return-allocs-from-loops" $LOWER | $RUN
Segmentation fault (core dumped)      # exit code 139
```

## Your commitment regarding this effort

I am reporting this issue and do not currently plan to submit a fix PR. This is one of the high-priority bugs from my [MLIR fuzzing campaign](https://discourse.llvm.org/t/seeking-advice-on-reporting-100-mlir-bugs-found-by-an-automated-testing-tool/92920/8).

## Related issues / fuzzing campaign

- Fuzzing campaign summary: https://discourse.llvm.org/t/seeking-advice-on-reporting-100-mlir-bugs-found-by-an-automated-testing-tool/92920/8 (full list: https://github.com/wildoranges/mlir-bug-reports)
- Deduplication check: Searched open and closed `llvm/llvm-project` issues and PRs for `WhileOpInterface`, `resolveConflicts`, and `scf.while` + `one-shot-bufferize` on 2026-11-26; no duplicate found.

## AI Declaration

This testcase was generated by an automated MLIR testing tool that I am developing; some of its steps use AI. I ran the reproducers on the reported commit, verified the expected vs. actual output, checked the referenced code lines, and reviewed all text myself.

## Analysis and possible fix (optional)

<details>
<summary> Click to expand ... </summary>

**Verified observations:**

1. `WhileOpInterface::isWritable` treats `scf.while` block arguments as always writable:
   https://github.com/llvm/llvm-project/blob/0cae57720c3ca7cb09b5ba48d199f1ba04e3aae2/mlir/lib/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.cpp#L884-L893
2. `WhileOpInterface::resolveConflicts` inserts copies only for `scf.condition` operands (the "Update before region" loop); it does not copy `scf.yield` operands of the `after` region:
   https://github.com/llvm/llvm-project/blob/0cae57720c3ca7cb09b5ba48d199f1ba04e3aae2/mlir/lib/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.cpp#L895-L945
3. By comparison, `scf.for` copies every yielded value that may alias a value defined outside the loop (`doesNotAliasExternalValue`):
   https://github.com/llvm/llvm-project/blob/0cae57720c3ca7cb09b5ba48d199f1ba04e3aae2/mlir/lib/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.cpp#L668-L692
4. From git history, the equivalent copy for `scf.while` appears to have been removed in commit `6247988e0751` (D133477, 2022).

**Possible fix direction (hypothesis):**

In `WhileOpInterface::resolveConflicts`, also copy `after`-region `scf.yield` operands that may alias a value defined outside the `scf.while` (matching the `scf.for` / `scf.condition` handling).

</details>
