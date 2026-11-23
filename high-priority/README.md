# High-priority bugs

Bugs that happen in passes real pipelines run on valid IR, with input that existing code produces, and with clearly wrong
behaviour. Each has a full report with a reproducer, the exact command and the llvm-project commit it was checked on.

| ID | Report | Area | Pass | Symptom | Input |
|---|---|---|---|---|---|
| F001 | [transfer_write lowering drops enclosing vector.mask (invalid IR)](reports/transfer-write-lowering-drops-enclosing-vector-mask.md) | Conversion/VectorToArmSME | convert-vector-to-arm-sme | invalid IR | produced by existing code |
| F002 | [Min/max reduction neutral values assume the float format has inf/NaN (crash on f8/f4 in --convert-vector-to-llvm and linalg split_reduction)](reports/float-reduction-neutral-value-assumes-an-llvm.md) | Conversion/VectorToLLVM | convert-vector-to-llvm, convert-to-llvm | crash | similar input in upstream tests |
| F003 | [scf.while does not copy tensors yielded from the "after" region back to the "before" region; in-place writes clobber outside tensors (segfault on constants)](reports/bufferization-scf-while-yield-copy.md) | SCF/Transforms | one-shot-bufferize{allow-return-allocs-from-loops} | wrong result | hand-written |
| F004 | [ConvertSelectionOpToSelect erases a selection with live results (assertion failure in eraseOp)](reports/convertselectionoptoselect-erases-a-selection-with-live-results.md) | SPIRV/IR | canonicalize | crash | produced by existing code |
| F005 | [sparse_tensor.lvl getSpeculatability asserts level index < dimension rank -> LICM crashes on the op's own documented BSR example](reports/sparse-lvl-speculatability-assert.md) | SparseTensor/IR | loop-invariant-code-motion | crash | produced by existing code |
| F006 | [insert_slice canonicalization of a tensor.cast source changes which unit dim a rank-reducing insert drops (miscompile)](reports/tensor-insert-slice-cast-folder-wrong-dropped-dim.md) | Tensor/IR | canonicalize | wrong result | similar input in upstream tests |
| F007 | [tensor.cast canonicalization (TensorCastExtractSlice) folds a dynamic extract_slice into a statically out-of-bounds one (invalid IR)](reports/tensor-cast-extract-slice-fold-out-of-bounds.md) | Tensor/IR | canonicalize | invalid IR | similar input in upstream tests |
| F008 | [tosa-convert-integer-type-to-signless drops unsignedness of tosa.cast inputs (input_unsigned is not set) -> wrong values](reports/tosa-signless-cast-input-unsigned.md) | Tosa/Transforms | tosa-convert-integer-type-to-signless, tosa-to-linalg | wrong result | similar input in upstream tests |
| F009 | [transpose(broadcast) fold lacks a scalable-vector guard (assertion failure in mlir-opt --canonicalize)](reports/transpose-broadcast-fold-lacks-a-scalable-vector.md) | Vector/IR | canonicalize | crash | similar input in upstream tests |
