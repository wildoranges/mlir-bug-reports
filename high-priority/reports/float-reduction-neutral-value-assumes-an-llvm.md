# [fuzzer-found][mlir] Min/max reduction neutral values assume the float format has inf/NaN (crash on f8/f4 in --convert-vector-to-llvm and linalg split_reduction)

## LLVM version and environment

- Commit: `0cae57720c3ca7cb09b5ba48d199f1ba04e3aae2` (2026-11-26)
- Build: `Release` with `LLVM_ENABLE_ASSERTIONS=ON`
- Platform: `x86_64` Linux

## Reproducer

**Case 1: masked `vector.reduction` in `--convert-vector-to-llvm` / `--convert-to-llvm`**

`repro1.mlir`:
```mlir
func.func @f(%v: vector<4xf8E4M3FN>, %m: vector<4xi1>) -> f8E4M3FN {
  %0 = vector.mask %m { vector.reduction <maximumf>, %v : vector<4xf8E4M3FN> into f8E4M3FN } : vector<4xi1> -> f8E4M3FN
  return %0 : f8E4M3FN
}
```

`repro2.mlir`:
```mlir
func.func @f(%v: vector<4xf4E2M1FN>, %m: vector<4xi1>) -> f4E2M1FN {
  %r = vector.mask %m { vector.reduction <maxnumf>, %v : vector<4xf4E2M1FN> into f4E2M1FN } : vector<4xi1> -> f4E2M1FN
  return %r : f4E2M1FN
}
```

**Case 2: `linalg::splitReduction` via `arith::getIdentityValueAttr` on finite-only floats**

`repro3.mlir`:
```mlir
func.func @generic_split_maxnumf(%in: tensor<32xf4E2M1FN>, %out: tensor<f4E2M1FN>) -> tensor<f4E2M1FN> {
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                        affine_map<(d0) -> ()>],
        iterator_types = ["reduction"]}
  ins(%in : tensor<32xf4E2M1FN>)
  outs(%out : tensor<f4E2M1FN>) {
  ^bb0(%arg1: f4E2M1FN, %arg2: f4E2M1FN):
    %y = arith.maxnumf %arg1, %arg2 : f4E2M1FN
    linalg.yield %y : f4E2M1FN
  } -> tensor<f4E2M1FN>
  return %r : tensor<f4E2M1FN>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:4 = transform.structured.split_reduction %0 <split_factor = 4, insert_split_dimension = 0, inner_parallel>
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}
```

Commands:
```console
$ mlir-opt repro1.mlir -o /dev/null && echo verifies
$ mlir-opt repro1.mlir --convert-vector-to-llvm
$ mlir-opt repro2.mlir --convert-to-llvm
$ mlir-opt repro3.mlir -o /dev/null && echo verifies
$ mlir-opt repro3.mlir --transform-interpreter
```


## Expected behaviour

All three inputs verify and pass cleanly when `f32` is used instead of `f8E4M3FN`/`f4E2M1FN`. For narrow float formats that lack infinity or NaN (or lower to integer types in LLVM), these passes should either build a valid representable neutral value (falling back to finite extrema when inf/NaN do not exist, as XeGPU does since #233471) or fail cleanly without aborting.

## Actual behaviour

`repro1.mlir` and `repro2.mlir` hit a `cast<FloatType>` assertion failure in `MaskedReductionOpConversion` / `createReductionNeutralValue`, and `repro3.mlir` hits an `UNREACHABLE` in `APFloat::getNaN`:

```console
$ mlir-opt repro1.mlir -o /dev/null && echo verifies
verifies
$ mlir-opt repro1.mlir --convert-vector-to-llvm
mlir-opt: .../llvm/include/llvm/Support/Casting.h:566: decltype(auto) llvm::cast(From &) [To = mlir::FloatType, From = mlir::Type]: Assertion `isa<To>(Val) && "cast<Ty>() argument of incompatible type!"' failed.
...
#11 ... (anonymous namespace)::MaskedReductionOpConversion::matchAndRewriteMaskableOp(mlir::vector::MaskOp, mlir::vector::MaskableOpInterface, mlir::ConversionPatternRewriter&) const ConvertVectorToLLVM.cpp:0:0
#12 ... (anonymous namespace)::VectorMaskOpConversionBase<mlir::vector::ReductionOp>::matchAndRewrite(...) ConvertVectorToLLVM.cpp:0:0
#24 ... (anonymous namespace)::ConvertVectorToLLVMPass::runOnOperation() ConvertVectorToLLVMPass.cpp:0:0
$ mlir-opt repro2.mlir --convert-to-llvm
mlir-opt: .../llvm/include/llvm/Support/Casting.h:566: decltype(auto) llvm::cast(From &) [To = mlir::FloatType, From = mlir::Type]: Assertion `isa<To>(Val) && "cast<Ty>() argument of incompatible type!"' failed.
...
#11 ... (anonymous namespace)::MaskedReductionOpConversion::matchAndRewriteMaskableOp(...) ConvertVectorToLLVM.cpp:0:0
#25 ... (anonymous namespace)::ConvertToLLVMPass::runOnOperation() ConvertToLLVMPass.cpp:0:0
$ mlir-opt repro3.mlir -o /dev/null && echo verifies
verifies
$ mlir-opt repro3.mlir --transform-interpreter
This floating point format does not support NaN
UNREACHABLE executed at .../llvm/lib/Support/APFloat.cpp:1240!
...
#10 ... llvm::detail::IEEEFloat::makeNaN(bool, bool, llvm::APInt const*)
#11 ... mlir::arith::getIdentityValueAttr(mlir::arith::AtomicRMWKind, mlir::Type, mlir::OpBuilder&, mlir::Location, bool)
#12 ... mlir::arith::getNeutralElement(mlir::Operation*)
#13 ... mlir::linalg::splitReduction(...)
#14 ... mlir::transform::SplitReductionOp::applyToOne(...)
```

## Your commitment regarding this effort

I am reporting this issue and do not currently plan to submit a fix PR. This is one of the high-priority bugs from my [MLIR fuzzing campaign](https://discourse.llvm.org/t/seeking-advice-on-reporting-100-mlir-bugs-found-by-an-automated-testing-tool/92920/8).

## Related issues / fuzzing campaign

- Fuzzing campaign summary: https://discourse.llvm.org/t/seeking-advice-on-reporting-100-mlir-bugs-found-by-an-automated-testing-tool/92920/8 (full list: https://github.com/wildoranges/mlir-bug-reports)
- Deduplication check: Searched open and closed `llvm/llvm-project` issues and PRs for `createReductionNeutralValue`, `MaskedReductionOpConversion`, `getIdentityValueAttr`, and `f4E2M1FN`/`f8E4M3FN` reductions on 2026-11-26; no duplicate found.
- Related PRs: #233471 (fixed the same inf/NaN neutral-value assumption in XeGPU reduction lowering), #222275 (masked `fmaximum`/`fminimum` neutral values).

## AI Declaration

This testcase was generated by an automated MLIR testing tool that I am developing; some of its steps use AI. I ran the reproducers on the reported commit, verified the expected vs. actual output, checked the referenced code lines, and reviewed all text myself.

## Analysis and possible fix (optional)

<details>
<summary> Click to expand ... </summary>

**Verified observations:**

1. **VectorToLLVM `createReductionNeutralValue`:** takes the converted LLVM type and unconditionally does `cast<FloatType>(llvmType)`:
   https://github.com/llvm/llvm-project/blob/0cae57720c3ca7cb09b5ba48d199f1ba04e3aae2/mlir/lib/Conversion/VectorToLLVM/ConvertVectorToLLVM.cpp#L624-L634
   Because `LLVMTypeConverter::convertFloatType` lowers `f8E4M3FN`, `f4E2M1FN`, etc. to integer types (`i8`, `i4`), the cast aborts.
2. **`arith::getIdentityValueAttr` (called by `arith::getNeutralElement` / `linalg::splitReduction`):** calls `APFloat::getNaN` for `maxnumf`/`minnumf` (unless `useOnlyFiniteValue` is set) and `APFloat::getInf` for `maximumf`/`minimumf`, without checking whether the float semantics support NaN or infinity:
   https://github.com/llvm/llvm-project/blob/0cae57720c3ca7cb09b5ba48d199f1ba04e3aae2/mlir/lib/Dialect/Arith/IR/ArithOps.cpp#L3519-L3588
   For finite-only formats like `f4E2M1FN`, `APFloat::getNaN` hits `llvm_unreachable`.

**Possible fix direction (hypothesis):**

- In `arith::getIdentityValueAttr`, check `fltSemantics` (`hasInfinity` / `hasNaN`, or fall back to finite largest/smallest when unavailable, matching #233471).
- In `VectorToLLVM`'s `MaskedReductionOpConversion` / `createReductionNeutralValue`, bail out cleanly or handle converted integer types when lowering narrow float reductions.

</details>
