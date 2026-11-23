# [fuzzer-found][mlir] Min/max reduction neutral values assume the float format has inf/NaN (crash on f8/f4 in --convert-vector-to-llvm and linalg split_reduction)

Two places that build float min/max reduction neutral values assume every float format has infinity and NaN, and abort on the small float formats (f8, f4) that lack them. The XeGPU reduction lowering already handles this (#233471): it uses infinity or NaN only when the format has them and falls back to a finite value otherwise.

**1. VectorToLLVM `createReductionNeutralValue`** (masked `vector.reduction` lowering). The helper takes the converted LLVM type and immediately does `cast<FloatType>(llvmType)`. `LLVMTypeConverter::convertFloatType` lowers f8E4M3FN, f4E2M1FN and similar formats to integers (`i8`, `i4`), so the cast aborts before any neutral value is chosen.

https://github.com/llvm/llvm-project/blob/8445b527ed0da1f9156f27be8c193624d0292253/mlir/lib/Conversion/VectorToLLVM/ConvertVectorToLLVM.cpp#L625-L635

`repro.mlir`:
```mlir
func.func @f(%v: vector<4xf8E4M3FN>, %m: vector<4xi1>) -> f8E4M3FN {
  %0 = vector.mask %m { vector.reduction <maximumf>, %v : vector<4xf8E4M3FN> into f8E4M3FN } : vector<4xi1> -> f8E4M3FN
  return %0 : f8E4M3FN
}
```

```console
$ mlir-opt repro.mlir -o /dev/null && echo verifies
verifies
$ mlir-opt repro.mlir --convert-vector-to-llvm
mlir-opt: .../llvm/include/llvm/Support/Casting.h:566: decltype(auto) llvm::cast(From &) [To = mlir::FloatType, From = mlir::Type]: Assertion `isa<To>(Val) && "cast<Ty>() argument of incompatible type!"' failed.
...
#11 0x0000627bb497a86b (anonymous namespace)::MaskedReductionOpConversion::matchAndRewriteMaskableOp(mlir::vector::MaskOp, mlir::vector::MaskableOpInterface, mlir::ConversionPatternRewriter&) const ConvertVectorToLLVM.cpp:0:0
#12 0x0000627bb4979166 (anonymous namespace)::VectorMaskOpConversionBase<mlir::vector::ReductionOp>::matchAndRewrite(mlir::vector::MaskOp, mlir::vector::MaskOpAdaptor, mlir::ConversionPatternRewriter&) const ConvertVectorToLLVM.cpp:0:0
#13 ... llvm::LogicalResult mlir::ConversionPattern::dispatchTo1To1<mlir::ConvertOpToLLVMPattern<mlir::vector::MaskOp, false>, mlir::vector::MaskOp>(mlir::ConvertOpToLLVMPattern<mlir::vector::MaskOp, false> const&, mlir::vector::MaskOp, mlir::vector::MaskOp::GenericAdaptor<llvm::ArrayRef<mlir::ValueRange> >, mlir::ConversionPatternRewriter&)
#14 ... mlir::ConvertOpToLLVMPattern<mlir::vector::MaskOp, false>::matchAndRewrite(mlir::Operation*, llvm::ArrayRef<mlir::ValueRange>, mlir::ConversionPatternRewriter&) const
```

The same assertion fires for the NaN-ignoring kinds, e.g. `maxnumf` over `f4E2M1FN`:

`repro2.mlir`:
```mlir
func.func @f(%v: vector<4xf4E2M1FN>, %m: vector<4xi1>) -> f4E2M1FN {
  %r = vector.mask %m { vector.reduction <maxnumf>, %v : vector<4xf4E2M1FN> into f4E2M1FN } : vector<4xi1> -> f4E2M1FN
  return %r : f4E2M1FN
}
```

```console
$ mlir-opt repro2.mlir --convert-to-llvm
mlir-opt: .../llvm/include/llvm/Support/Casting.h:566: decltype(auto) llvm::cast(From &) [To = mlir::FloatType, From = mlir::Type]: Assertion `isa<To>(Val) && "cast<Ty>() argument of incompatible type!"' failed.
...
```

The same path lowers `vector<4xf32>` successfully.

**2. `arith::getIdentityValueAttr`** (used by `arith::getNeutralElement`). For `maxnumf`/`minnumf` it calls `APFloat::getNaN` unless `useOnlyFiniteValue` is set, and for `maximumf`/`minimumf` it calls `APFloat::getInf`. `linalg::splitReduction` asks for the neutral element without `useOnlyFiniteValue`, so a `maxnumf` reduction over the finite-only `f4E2M1FN` reaches `APFloat::getNaN` and hits an `UNREACHABLE`. The same input with `f32` passes.

https://github.com/llvm/llvm-project/blob/8445b527ed0da1f9156f27be8c193624d0292253/mlir/lib/Dialect/Arith/IR/ArithOps.cpp#L3507-L3576

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

```console
$ mlir-opt repro3.mlir -o /dev/null && echo verifies
verifies
$ mlir-opt repro3.mlir --transform-interpreter
This floating point format does not support NaN
UNREACHABLE executed at .../llvm/lib/Support/APFloat.cpp:965!
...
```

**Version**: llvm-project main @ 8445b527ed0da1f9156f27be8c193624d0292253 (2026-11-16), Release + `LLVM_ENABLE_ASSERTIONS=ON`, x86_64 Linux.

This testcase was generated by an automated MLIR testing tool that I am developing; some of its steps use AI.

*Disclosure: I wrote this report with the help of Claude Code. I reproduced and reviewed everything in this report myself.*
