#!/usr/bin/env python3
"""Replace dynamic SLICE+CONCAT scatter in cnnlstm_small while_body with DYNAMIC_UPDATE_SLICE.

TFLM requires static shapes; the default LSTM while_body uses dynamic SLICE which breaks it.
This patches each while_body to use op 151 (DYNAMIC_UPDATE_SLICE) with static shapes instead.
"""

import sys
import os
import struct
import numpy as np
import flatbuffers

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "firmware/tflm/tensorflow/lite/python"))
import schema_py_generated as schema  # noqa: E402

# BuiltinOperator.DYNAMIC_UPDATE_SLICE = 151
DUS_OP_CODE = 151
# BuiltinOptions union value for DynamicUpdateSliceOptions = 117
DUS_OPTIONS_TYPE = 117


# Helpers to extract raw data from the read-only FlatBuffer tables

def _buf_data(model, buf_idx):
    """Return raw bytes of a buffer, or None if empty."""
    b = model.Buffers(buf_idx)
    if b.DataLength() == 0:
        return None
    return bytes(b.Data(i) for i in range(b.DataLength()))


def _quant(t):
    """Extract quantization params dict from a Tensor (or None)."""
    q = t.Quantization()
    if q is None:
        return None
    return {
        "scale":       [q.Scale(i)     for i in range(q.ScaleLength())],
        "zero_point":  [q.ZeroPoint(i) for i in range(q.ZeroPointLength())],
        "min":         [q.Min(i)       for i in range(q.MinLength())],
        "max":         [q.Max(i)       for i in range(q.MaxLength())],
        "quantized_dimension": q.QuantizedDimension(),
    }


def _op_raw_builtin_options(op):
    """Return (options_type, raw_bytes) for an op's builtin options, or (0, None)."""
    opt_type = op.BuiltinOptionsType()
    if opt_type == 0:
        return 0, None
    table = op.BuiltinOptions()
    if table is None:
        return 0, None
    # The FlatBuffer Table stores the serialized bytes; extract them.
    # table.__o is the position of the table's vtable offset inside Bytes.
    # We need the raw bytes of the inline table.
    # Strategy: re-read the operator's bytes from the underlying buffer.
    # We can't easily get the size, so instead we keep the raw option_type
    # and rebuild the ConcatenationOptions specially (only one we need).
    return opt_type, table


def _concat_axis(op, model):
    """Return the axis for a CONCATENATION op."""
    table = op.BuiltinOptions()
    if table is None:
        return 0
    co = schema.ConcatenationOptions()
    co.Init(table.Bytes, table.Pos)
    return co.Axis()


# FlatBuffer builders for each table type

def _build_quantization(b, q):
    """Build a QuantizationParameters table; return offset."""
    if q is None:
        return None
    scale_off = None
    if q["scale"]:
        schema.QuantizationParametersStartScaleVector(b, len(q["scale"]))
        for v in reversed(q["scale"]):
            b.PrependFloat32(v)
        scale_off = b.EndVector()
    zp_off = None
    if q["zero_point"]:
        schema.QuantizationParametersStartZeroPointVector(b, len(q["zero_point"]))
        for v in reversed(q["zero_point"]):
            b.PrependInt64(v)
        zp_off = b.EndVector()
    min_off = None
    if q["min"]:
        schema.QuantizationParametersStartMinVector(b, len(q["min"]))
        for v in reversed(q["min"]):
            b.PrependFloat32(v)
        min_off = b.EndVector()
    max_off = None
    if q["max"]:
        schema.QuantizationParametersStartMaxVector(b, len(q["max"]))
        for v in reversed(q["max"]):
            b.PrependFloat32(v)
        max_off = b.EndVector()

    schema.QuantizationParametersStart(b)
    if min_off is not None:
        schema.QuantizationParametersAddMin(b, min_off)
    if max_off is not None:
        schema.QuantizationParametersAddMax(b, max_off)
    if scale_off is not None:
        schema.QuantizationParametersAddScale(b, scale_off)
    if zp_off is not None:
        schema.QuantizationParametersAddZeroPoint(b, zp_off)
    schema.QuantizationParametersAddQuantizedDimension(b, q["quantized_dimension"])
    return schema.QuantizationParametersEnd(b)


def _build_tensor(b, t_dict):
    """Build a Tensor table; return offset. Call BEFORE b.StartObject."""
    name_off = b.CreateString(t_dict["name"])
    shape_off = None
    if t_dict["shape"]:
        schema.TensorStartShapeVector(b, len(t_dict["shape"]))
        for v in reversed(t_dict["shape"]):
            b.PrependInt32(v)
        shape_off = b.EndVector()
    quant_off = _build_quantization(b, t_dict.get("quantization"))

    schema.TensorStart(b)
    if shape_off is not None:
        schema.TensorAddShape(b, shape_off)
    schema.TensorAddType(b, t_dict["type"])
    schema.TensorAddBuffer(b, t_dict["buffer"])
    schema.TensorAddName(b, name_off)
    if quant_off is not None:
        schema.TensorAddQuantization(b, quant_off)
    schema.TensorAddHasRank(b, t_dict.get("has_rank", True))
    return schema.TensorEnd(b)


def _build_concat_options(b, axis):
    """Build a ConcatenationOptions table; return offset."""
    schema.ConcatenationOptionsStart(b)
    schema.ConcatenationOptionsAddAxis(b, axis)
    return schema.ConcatenationOptionsEnd(b)


def _build_dus_options(b):
    """Build an empty DynamicUpdateSliceOptions table; return offset."""
    # Start/End a table with 0 fields
    b.StartObject(0)
    return b.EndObject()


def _build_operator(b, op_dict, model, orig_op=None):
    """
    Build an Operator table; return offset.

    op_dict keys:
      opcode_index, inputs, outputs
    Optional:
      builtin_options_type, concat_axis, is_dus
    """
    # Build vectors first
    schema.OperatorStartInputsVector(b, len(op_dict["inputs"]))
    for v in reversed(op_dict["inputs"]):
        b.PrependInt32(v)
    inputs_off = b.EndVector()

    schema.OperatorStartOutputsVector(b, len(op_dict["outputs"]))
    for v in reversed(op_dict["outputs"]):
        b.PrependInt32(v)
    outputs_off = b.EndVector()

    # Build options
    options_type = op_dict.get("builtin_options_type", 0)
    options_off = None

    if op_dict.get("is_dus"):
        options_off = _build_dus_options(b)
        options_type = DUS_OPTIONS_TYPE
    elif options_type == 10:  # ConcatenationOptions
        axis = op_dict.get("concat_axis", 0)
        options_off = _build_concat_options(b, axis)
    elif options_type != 0 and orig_op is not None:
        # For all other ops, we need to rebuild the options from the original.
        # We do this by using the T (object) API if available, or copying raw bytes.
        # For FC, SPLIT, LOGISTIC, TANH, MUL, ADD, GATHER, RESHAPE: options may be None
        # or have a few fields.  The safest approach: use the BuiltinOptionsCreator
        # to get the T object and rebuild via its Pack method.
        try:
            opt_table = orig_op.BuiltinOptions()
            if opt_table is not None:
                opt_t = schema.BuiltinOptionsCreator(options_type, opt_table)
                if opt_t is not None:
                    options_off = opt_t.Pack(b)
        except Exception:
            pass

    schema.OperatorStart(b)
    schema.OperatorAddOpcodeIndex(b, op_dict["opcode_index"])
    schema.OperatorAddInputs(b, inputs_off)
    schema.OperatorAddOutputs(b, outputs_off)
    if options_type != 0:
        schema.OperatorAddBuiltinOptionsType(b, options_type)
    if options_off is not None:
        schema.OperatorAddBuiltinOptions(b, options_off)
    return schema.OperatorEnd(b)


def _build_opcode(b, oc_dict):
    """Build an OperatorCode table; return offset."""
    schema.OperatorCodeStart(b)
    schema.OperatorCodeAddDeprecatedBuiltinCode(b, oc_dict["deprecated_builtin_code"])
    schema.OperatorCodeAddVersion(b, oc_dict["version"])
    schema.OperatorCodeAddBuiltinCode(b, oc_dict["builtin_code"])
    return schema.OperatorCodeEnd(b)


def _build_buffer(b, data):
    """Build a Buffer table; return offset."""
    data_off = None
    if data:
        data_bytes = bytes(data)
        schema.BufferStartDataVector(b, len(data_bytes))
        for byte in reversed(data_bytes):
            b.PrependByte(byte)
        data_off = b.EndVector()
    schema.BufferStart(b)
    if data_off is not None:
        schema.BufferAddData(b, data_off)
    return schema.BufferEnd(b)


# Main patch logic

def _extract_ops(sg, model):
    """Return list of op dicts with all fields needed to rebuild."""
    ops = []
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        bc = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        opt_type, _ = _op_raw_builtin_options(op)
        d = {
            "opcode_index":          op.OpcodeIndex(),
            "inputs":                [op.Inputs(j) for j in range(op.InputsLength())],
            "outputs":               [op.Outputs(j) for j in range(op.OutputsLength())],
            "builtin_options_type":  opt_type,
            "_orig_op":              op,  # keep reference for options copy
        }
        if bc == 2:  # CONCATENATION — also store axis
            d["concat_axis"] = _concat_axis(op, model)
        ops.append(d)
    return ops


def _extract_tensors(sg):
    """Return list of tensor dicts."""
    tensors = []
    for i in range(sg.TensorsLength()):
        t = sg.Tensors(i)
        tensors.append({
            "name":      t.Name().decode() if t.Name() else "",
            "type":      t.Type(),
            "buffer":    t.Buffer(),
            "shape":     [t.Shape(j) for j in range(t.ShapeLength())],
            "has_rank":  t.HasRank(),
            "quantization": _quant(t),
        })
    return tensors


def _patch_while_body_ops(ops, dus_opcode_idx):
    """
    Replace ops 18-25 (8 ops) with 4 ops using DYNAMIC_UPDATE_SLICE.

    Original ops (indices 18-25):
      18  RESHAPE(t1, t13)           -> t38   counter_1d
      19  CONCAT([t38, t9])          -> t39   [counter, -1, -1]  (size for before-slice)
      20  SLICE(t2, t6, t39)         -> t40   before  (DYNAMIC SHAPE)
      21  RESHAPE(t18, t13)          -> t41   counter_inc_1d
      22  CONCAT([t41, t10])         -> t42   [counter+1, 0, 0]  (begin for after-slice)
      23  SLICE(t2, t42, t11)        -> t43   after  (DYNAMIC SHAPE)
      24  RESHAPE(t37, t12)          -> t44   h_3d (1,1,64)
      25  CONCAT([t40, t44, t43])    -> t45   updated_buffer (64,1,64)

    Replacement (ops 18-21, indices 18-21):
      18  RESHAPE(t1, t13)           -> t38   counter_1d  (unchanged)
      19  CONCAT([t38, t10])         -> t39   [counter, 0, 0]  start_indices (3,)
                                              ^^^^^ change t9 -> t10 ^^^^^
      20  RESHAPE(t37, t12)          -> t44   h_3d (1,1,64)   (was op 24)
      21  DYNAMIC_UPDATE_SLICE(t2, t44, t39) -> t45   (new op)

    Notes:
      - t10 = [0, 0] (constant) vs t9 = [-1, -1]
      - ops 20-23 are removed, tensors 39-43 become unreferenced (OK for TFLM)
      - t45 is still the final output of the accumulation
    """
    assert len(ops) == 26, f"Expected 26 ops in while_body, got {len(ops)}"

    # Verify op 19 is CONCAT with expected inputs
    op19 = ops[19]
    assert op19["inputs"] == [38, 9], \
        f"Op 19 inputs unexpected: {op19['inputs']} (expected [38, 9])"

    # Verify op 24 is RESHAPE
    op24 = ops[24]
    assert op24["inputs"][0] == 37, \
        f"Op 24 inputs unexpected: {op24['inputs']} (expected [37, ...])"

    # Verify op 25 outputs t45
    op25 = ops[25]
    assert op25["outputs"] == [45], \
        f"Op 25 outputs unexpected: {op25['outputs']} (expected [45])"

    # Build new op sequence
    new_ops = list(ops[:18])  # ops 0-17 unchanged

    # Op 18: RESHAPE(t1, t13) -> t38  (unchanged)
    new_ops.append(ops[18])

    # Op 19 (modified): CONCAT([t38, t10]) -> t39  (change input[1]: t9 -> t10)
    op19_new = dict(op19)
    op19_new["inputs"] = [38, 10]  # t10 = [0, 0] instead of t9 = [-1, -1]
    new_ops.append(op19_new)

    # Op 20 (was op 24): RESHAPE(t37, t12) -> t44
    new_ops.append(ops[24])

    # Op 21 (new): DYNAMIC_UPDATE_SLICE(t2, t44, t39) -> t45
    new_ops.append({
        "opcode_index":         dus_opcode_idx,
        "inputs":               [2, 44, 39],
        "outputs":              [45],
        "builtin_options_type": DUS_OPTIONS_TYPE,
        "is_dus":               True,
        "_orig_op":             None,
    })

    assert len(new_ops) == 22, f"Expected 22 new ops, got {len(new_ops)}"
    return new_ops


def _rebuild_subgraph(b, sg, model, patch_fn=None, dus_opcode_idx=None):
    """Rebuild a subgraph.  If patch_fn is given, patch the operators list."""
    ops_dicts = _extract_ops(sg, model)
    if patch_fn is not None:
        ops_dicts = patch_fn(ops_dicts, dus_opcode_idx)
    tensors_dicts = _extract_tensors(sg)

    # --- Build all tensors ---
    tensor_offsets = []
    for t_dict in tensors_dicts:
        tensor_offsets.append(_build_tensor(b, t_dict))

    # --- Build all operators ---
    op_offsets = []
    for op_d in ops_dicts:
        orig = op_d.get("_orig_op")
        op_offsets.append(_build_operator(b, op_d, model, orig_op=orig))

    # --- Build the subgraph vectors ---
    schema.SubGraphStartTensorsVector(b, len(tensor_offsets))
    for off in reversed(tensor_offsets):
        b.PrependUOffsetTRelative(off)
    tensors_vec = b.EndVector()

    schema.SubGraphStartOperatorsVector(b, len(op_offsets))
    for off in reversed(op_offsets):
        b.PrependUOffsetTRelative(off)
    ops_vec = b.EndVector()

    inputs_list = [sg.Inputs(i) for i in range(sg.InputsLength())]
    schema.SubGraphStartInputsVector(b, len(inputs_list))
    for v in reversed(inputs_list):
        b.PrependInt32(v)
    inputs_vec = b.EndVector()

    outputs_list = [sg.Outputs(i) for i in range(sg.OutputsLength())]
    schema.SubGraphStartOutputsVector(b, len(outputs_list))
    for v in reversed(outputs_list):
        b.PrependInt32(v)
    outputs_vec = b.EndVector()

    name_off = b.CreateString(sg.Name().decode() if sg.Name() else "")

    schema.SubGraphStart(b)
    schema.SubGraphAddTensors(b, tensors_vec)
    schema.SubGraphAddInputs(b, inputs_vec)
    schema.SubGraphAddOutputs(b, outputs_vec)
    schema.SubGraphAddOperators(b, ops_vec)
    schema.SubGraphAddName(b, name_off)
    return schema.SubGraphEnd(b)


def patch_model(input_path, output_path):
    with open(input_path, "rb") as f:
        model_bytes = bytearray(f.read())

    model = schema.Model.GetRootAs(model_bytes, 0)

    # -----------------------------------------------------------------------
    # Collect all existing op codes + add DUS if not present
    # -----------------------------------------------------------------------
    op_codes = []
    dus_opcode_idx = None
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        bc = oc.BuiltinCode()
        if bc == 127:
            bc = oc.DeprecatedBuiltinCode()
        op_codes.append({
            "deprecated_builtin_code": min(oc.DeprecatedBuiltinCode(), 127),
            "builtin_code": oc.BuiltinCode(),
            "version": oc.Version(),
        })
        if bc == DUS_OP_CODE:
            dus_opcode_idx = i

    if dus_opcode_idx is None:
        op_codes.append({
            "deprecated_builtin_code": 127,  # PLACEHOLDER_FOR_GREATER_OP_CODES
            "builtin_code": DUS_OP_CODE,
            "version": 1,
        })
        dus_opcode_idx = len(op_codes) - 1
        print(f"  Added DUS op code at index {dus_opcode_idx}")

    # -----------------------------------------------------------------------
    # Build the model
    # -----------------------------------------------------------------------
    b = flatbuffers.Builder(len(model_bytes) + 4096)

    # Build buffers (bottom-up; build all first)
    buffer_offsets = []
    for i in range(model.BuffersLength()):
        data = _buf_data(model, i)
        buffer_offsets.append(_build_buffer(b, data))

    # Build subgraphs.  Subgraph indices 2 and 4 are while_body subgraphs.
    while_body_indices = set()
    for i in range(model.SubgraphsLength()):
        sg = model.Subgraphs(i)
        name = sg.Name().decode() if sg.Name() else ""
        if "while_body" in name:
            while_body_indices.add(i)
    print(f"  While-body subgraph indices: {sorted(while_body_indices)}")

    sg_offsets = []
    for i in range(model.SubgraphsLength()):
        sg = model.Subgraphs(i)
        if i in while_body_indices:
            sg_offsets.append(
                _rebuild_subgraph(b, sg, model,
                                  patch_fn=_patch_while_body_ops,
                                  dus_opcode_idx=dus_opcode_idx)
            )
            print(f"  Patched while_body subgraph [{i}]")
        else:
            sg_offsets.append(_rebuild_subgraph(b, sg, model))

    # Build op codes
    opcode_offsets = []
    for oc_dict in op_codes:
        opcode_offsets.append(_build_opcode(b, oc_dict))

    # Build description string
    desc_off = b.CreateString(model.Description().decode() if model.Description() else "")

    # Build metadata_buffer vector
    mb_len = model.MetadataBufferLength()
    if mb_len > 0:
        schema.ModelStartMetadataBufferVector(b, mb_len)
        for i in reversed(range(mb_len)):
            b.PrependInt32(model.MetadataBuffer(i))
        mb_vec = b.EndVector()
    else:
        mb_vec = None

    # ---- Assemble vectors ----
    schema.ModelStartBuffersVector(b, len(buffer_offsets))
    for off in reversed(buffer_offsets):
        b.PrependUOffsetTRelative(off)
    buffers_vec = b.EndVector()

    schema.ModelStartSubgraphsVector(b, len(sg_offsets))
    for off in reversed(sg_offsets):
        b.PrependUOffsetTRelative(off)
    subgraphs_vec = b.EndVector()

    schema.ModelStartOperatorCodesVector(b, len(opcode_offsets))
    for off in reversed(opcode_offsets):
        b.PrependUOffsetTRelative(off)
    opcodes_vec = b.EndVector()

    schema.ModelStart(b)
    schema.ModelAddVersion(b, model.Version())
    schema.ModelAddOperatorCodes(b, opcodes_vec)
    schema.ModelAddSubgraphs(b, subgraphs_vec)
    schema.ModelAddDescription(b, desc_off)
    schema.ModelAddBuffers(b, buffers_vec)
    if mb_vec is not None:
        schema.ModelAddMetadataBuffer(b, mb_vec)
    model_off = schema.ModelEnd(b)

    b.Finish(model_off)
    raw = bytes(b.Output())
    # Inject TFLite file identifier "TFL3" at bytes 4-7.
    # Without identifier: layout is [root_offset(4)] [data...]
    # With identifier:    layout is [root_offset(4)] ["TFL3"(4)] [data...]
    # The root offset must increase by 4 to account for the extra bytes.
    import struct as _struct
    old_root = _struct.unpack_from("<I", raw, 0)[0]
    new_root = old_root + 4  # identifier shifts data by 4 bytes
    result = _struct.pack("<I", new_root) + b"TFL3" + raw[4:]

    with open(output_path, "wb") as f:
        f.write(result)
    print(f"  Written {len(result) / 1024:.1f} KB → {output_path}")
    return result


# Verification: run a quick CPU TFLite inference to ensure patch didn't break

def verify_model(model_path, original_path, data_dir=None):
    try:
        import tensorflow as tf
        import sys as _sys
        sys_path_backup = list(_sys.path)

        interp = tf.lite.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        input_details = interp.get_input_details()
        output_details = interp.get_output_details()

        # Make a test input
        in_shape = input_details[0]["shape"]
        test_input = np.random.randn(*in_shape).astype(np.float32) * 0.1

        interp.set_tensor(input_details[0]["index"], test_input)
        interp.invoke()
        out = interp.get_tensor(output_details[0]["index"])
        print(f"  Patched model CPU inference OK, output shape={out.shape}, "
              f"sample={out.flat[:4]}")

        if original_path:
            interp2 = tf.lite.Interpreter(model_path=original_path)
            interp2.allocate_tensors()
            interp2.set_tensor(interp2.get_input_details()[0]["index"], test_input)
            interp2.invoke()
            out2 = interp2.get_tensor(interp2.get_output_details()[0]["index"])
            max_diff = float(np.max(np.abs(out - out2)))
            print(f"  Max diff vs original: {max_diff:.6f} "
                  f"({'OK' if max_diff < 1e-3 else 'LARGE DIFF — check patch!'})")
    except Exception as e:
        print(f"  Verification failed: {e}")


# Entry point

def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",  default="tflite_models/cnnlstm_small_int8.tflite")
    p.add_argument("--output", default=None,
                   help="Output path (defaults to overwriting input)")
    p.add_argument("--verify", action="store_true",
                   help="Run CPU TFLite inference to verify patch")
    args = p.parse_args()

    input_path = os.path.join(_REPO_ROOT, args.input) \
        if not os.path.isabs(args.input) else args.input
    output_path = args.output or input_path
    if not os.path.isabs(output_path):
        output_path = os.path.join(_REPO_ROOT, output_path)

    print(f"Patching {input_path} …")
    patch_model(input_path, output_path)

    if args.verify:
        print("Verifying patched model …")
        tmp = output_path + ".orig.tflite"
        import shutil
        shutil.copy2(input_path, tmp)
        verify_model(output_path, input_path if output_path != input_path else None)

    print("Done.")


if __name__ == "__main__":
    main()
