#!/usr/bin/env python3
"""
Inspect cnnlstm_small TFLite model — dump CONCATENATION ops and tensor shapes.
Helps debug the concat fallback in TFLM.
"""
import os
import sys

# Add tflm schema path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TFLM_SCHEMA = os.path.join(SCRIPT_DIR, "..", "firmware", "tflm",
                            "tensorflow", "lite", "python", "schema_py_generated.py")
sys.path.insert(0, os.path.dirname(TFLM_SCHEMA))

import schema_py_generated as schema


def tensor_shape(subgraph, tensor_idx):
    """Get shape of tensor as tuple, or None if invalid."""
    if tensor_idx < 0:
        return None
    tensors = subgraph.Tensors(tensor_idx)
    if tensors is None:
        return None
    n = tensors.ShapeLength()
    if n == 0:
        return ()
    return tuple(tensors.Shape(j) for j in range(n))


def inspect_subgraph(model, sg_idx, prefix=""):
    """Inspect a subgraph for CONCATENATION ops."""
    subgraph = model.Subgraphs(sg_idx)
    if subgraph is None:
        return

    name = subgraph.Name() or f"subgraph_{sg_idx}"
    print(f"\n{prefix}=== {name} (subgraph {sg_idx}) ===")

    n_ops = subgraph.OperatorsLength()
    for op_idx in range(n_ops):
        op = subgraph.Operators(op_idx)
        if op is None:
            continue

        opcode_idx = op.OpcodeIndex()
        opcode = model.OperatorCodes(opcode_idx)
        builtin_code = opcode.BuiltinCode() if opcode else -1

        if builtin_code != schema.BuiltinOperator.CONCATENATION:
            continue

        # Get ConcatenationOptions
        axis = 0
        if op.BuiltinOptionsType() == schema.BuiltinOptions.ConcatenationOptions:
            opt_table = op.BuiltinOptions()
            if opt_table is not None:
                concat_opt = schema.ConcatenationOptions()
                concat_opt.Init(opt_table.Bytes, opt_table.Pos)
                axis = concat_opt.Axis()

        # Input shapes
        n_in = op.InputsLength()
        in_shapes = []
        for j in range(n_in):
            tidx = op.Inputs(j)
            sh = tensor_shape(subgraph, tidx)
            in_shapes.append((tidx, sh))

        # Output shape
        out_shapes = []
        n_out = op.OutputsLength()
        for j in range(n_out):
            tidx = op.Outputs(j)
            sh = tensor_shape(subgraph, tidx)
            out_shapes.append((tidx, sh))

        concat_axis_sum = sum(s[axis] for _, s in in_shapes if s and axis < len(s))
        out_axis_dim = out_shapes[0][1][axis] if out_shapes and out_shapes[0][1] and axis < len(out_shapes[0][1]) else -1

        print(f"  CONCAT op {op_idx}: axis={axis}")
        print(f"    inputs: {[(t, s) for t, s in in_shapes]}")
        print(f"    outputs: {[(t, s) for t, s in out_shapes]}")
        print(f"    concat_axis_sum={concat_axis_sum}, output.Dims(axis)={out_axis_dim}")
        if concat_axis_sum != out_axis_dim:
            print(f"    *** SHAPE MISMATCH -> triggers fallback ***")

        # Check if a different axis would match (model may have wrong axis)
        out_shape = out_shapes[0][1] if out_shapes and out_shapes[0][1] else None
        if out_shape and len(in_shapes) >= 2:
            for try_axis in range(len(out_shape)):
                if try_axis == axis:
                    continue
                try_sum = sum(
                    in_shapes[i][1][try_axis]
                    for i in range(n_in)
                    if in_shapes[i][1] and try_axis < len(in_shapes[i][1])
                )
                if try_axis < len(out_shape) and try_sum == out_shape[try_axis]:
                    print(f"    HINT: axis={try_axis} would match (sum={try_sum}) — model axis={axis} may be wrong!")


def main():
    tflite_path = os.path.join(SCRIPT_DIR, "..", "tflite_models", "cnnlstm_small_int8.tflite")
    if len(sys.argv) > 1:
        tflite_path = sys.argv[1]
    tflite_path = os.path.expanduser(tflite_path)

    if not os.path.isfile(tflite_path):
        print(f"ERROR: not found: {tflite_path}")
        print("  Generate with: uv run python quantise.py --models cnnlstm_small --data-dir <path>")
        sys.exit(1)

    with open(tflite_path, "rb") as f:
        buf = f.read()

    model = schema.Model.GetRootAs(buf, 0)
    n_sg = model.SubgraphsLength()
    print(f"Model: {tflite_path}")
    print(f"Subgraphs: {n_sg}")

    for sg_idx in range(n_sg):
        inspect_subgraph(model, sg_idx, "")


if __name__ == "__main__":
    main()
