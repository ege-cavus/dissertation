#!/usr/bin/env python3
"""Fix wrong CONCAT output shapes in cnnlstm_small TFLite while_body subgraphs.

Changes the concat output shape from (64,1,64) to (1,1,129) so LSTM reads the right number of elements.
Usage: uv run python scripts/patch_concat_shapes.py [input.tflite [output.tflite]]
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TFLM_PYTHON = os.path.join(REPO_ROOT, "firmware", "tflm", "tensorflow", "lite", "python")
sys.path.insert(0, TFLM_PYTHON)

import schema_py_generated as schema
import flatbuffers

_TFLITE_FILE_IDENTIFIER = b"TFL3"


def _read_model(path: str) -> "schema.ModelT":
    with open(path, "rb") as f:
        data = bytearray(f.read())
    return _read_model_from_bytearray(data)


def _read_model_from_bytearray(data: bytearray) -> "schema.ModelT":
    model = schema.Model.GetRootAsModel(data, 0)
    model = schema.ModelT.InitFromObj(model)
    # Buffer offset handling (models > 2GB)
    for buf in model.buffers:
        if getattr(buf, "offset", 0):
            buf.data = bytes(data[buf.offset : buf.offset + buf.size])
            buf.offset = 0
            buf.size = 0
    for sg in model.subgraphs:
        for op in sg.operators:
            if getattr(op, "largeCustomOptionsOffset", 0):
                op.customOptions = bytes(
                    data[
                        op.largeCustomOptionsOffset : op.largeCustomOptionsOffset
                        + op.largeCustomOptionsSize
                    ]
                )
                op.largeCustomOptionsOffset = 0
                op.largeCustomOptionsSize = 0
    return model


def _write_model(model: "schema.ModelT", path: str) -> None:
    builder = flatbuffers.Builder(1024)
    model_offset = model.Pack(builder)
    builder.Finish(model_offset, file_identifier=_TFLITE_FILE_IDENTIFIER)
    with open(path, "wb") as f:
        f.write(builder.Output())


def _patch_reshape_for_concat(model: "schema.ModelT") -> int:
    """Fix RESHAPE ops 7 and 11: input (1,1,129) cannot reshape to (1,64,64).
    Patch output tensors 34, 43 and shape constant (tensor 15 buffer) to (1,1,129).
    """
    import struct
    n_patched = 0
    target_shape = [1, 1, 129]
    target_shape_bytes = list(struct.pack("<iii", *target_shape))
    for sg_idx, sg in enumerate(model.subgraphs):
        if sg_idx != 0:
            continue  # main graph only
        # Patch output tensors 34 and 43: (1,64,64) -> (1,1,129)
        for t_idx in (34, 43):
            if t_idx < len(sg.tensors):
                t = sg.tensors[t_idx]
                sh = list(t.shape) if hasattr(t.shape, "__iter__") else [t.shape]
                if sh == [1, 64, 64]:
                    t.shape = target_shape.copy()
                    n_patched += 1
                    print(f"  Patched tensor {t_idx}: (1,64,64) -> (1,1,129)")
        # Patch shape constant buffer: find buffer with [1,64,64] (tensor 15)
        t15 = sg.tensors[15] if len(sg.tensors) > 15 else None
        if t15 is not None and t15.buffer is not None:
            buf_idx = t15.buffer
            if buf_idx < len(model.buffers):
                buf = model.buffers[buf_idx]
                raw = buf.data
                if raw is not None:
                    raw = bytes(raw) if isinstance(raw, (list, bytearray)) else raw
                    if len(raw) >= 12:
                        arr = struct.unpack("<iii", raw[:12])
                        if arr == (1, 64, 64):
                            buf.data = target_shape_bytes + list(raw[12:]) if len(raw) > 12 else target_shape_bytes
                            n_patched += 1
                            print(f"  Patched buffer {buf_idx} (RESHAPE shape): [1,64,64] -> [1,1,129]")
    return n_patched


def _patch_slice_buffers(model: "schema.ModelT") -> int:
    """Patch SLICE size buffer 81: [-1,-1,-1] -> [1,1,64].
    Also ensure buffer 112 (begin for SLICE op 23) has [0,0,0] if empty.
    """
    import struct
    n_patched = 0
    new_size = list(struct.pack("<iii", 1, 1, 64))
    for buf_idx, buf in enumerate(model.buffers):
        raw = b""
        if buf.data is not None:
            raw = bytes(buf.data) if not isinstance(buf.data, bytes) else buf.data
        if len(raw) < 12:
            # Empty buffer 112 (begin for SLICE extracting last 64): add [0,0,65]
            if buf_idx == 112:
                buf.data = list(struct.pack("<iii", 0, 0, 65))
                n_patched += 1
                print(f"  Patched buffer {buf_idx}: added begin [0,0,65]")
            continue
        arr = struct.unpack("<iii", raw[:12])
        if arr == (-1, -1, -1):
            buf.data = new_size + list(raw[12:])
            n_patched += 1
            print(f"  Patched buffer {buf_idx}: size [-1,-1,-1] -> [1,1,64]")
    return n_patched


def _patch_concat_output_shapes(model: "schema.ModelT") -> int:
    """Patch tensors (64,1,64) -> (1,1,129).
    - Main graph: tensors 1, 30, 39 (WHILE loop vars) — required for
      CopySubgraphOutputsToOpOutputs size match.
    - while_body subgraphs: all (64,1,64) tensors (CONCAT output + loop inputs).
    - Skip while_cond: patching there can break SLICE ops.
    """
    n_patched = 0
    for sg_idx, sg in enumerate(model.subgraphs):
        sg_name = sg.name.decode() if isinstance(sg.name, bytes) else (sg.name or f"subgraph_{sg_idx}")
        # Skip while_cond subgraphs (indices 1 and 3)
        if "while_cond" in sg_name:
            continue
        for t_idx, tensor in enumerate(sg.tensors):
            if tensor.shape is None:
                continue
            sh = list(tensor.shape) if hasattr(tensor.shape, "__iter__") else [tensor.shape]
            if sh == [64, 1, 64]:
                tensor.shape = [1, 1, 129]
                n_patched += 1
                print(f"  Patched tensor {t_idx} in {sg_name}: (64,1,64) -> (1,1,129)")
    return n_patched


def main():
    default_in = os.path.join(REPO_ROOT, "tflite_models", "cnnlstm_small_int8.tflite")
    argv = sys.argv[1:]
    if len(argv) >= 2:
        in_path, out_path = argv[0], argv[1]
    elif len(argv) == 1:
        in_path = argv[0]
        out_path = in_path
    else:
        in_path = default_in
        out_path = in_path

    in_path = os.path.expanduser(in_path)
    out_path = os.path.expanduser(out_path)

    if not os.path.isfile(in_path):
        print(f"ERROR: not found: {in_path}")
        sys.exit(1)

    print(f"Reading: {in_path}")
    model = _read_model(in_path)
    n1 = _patch_concat_output_shapes(model)
    n2 = _patch_reshape_for_concat(model)
    n3 = _patch_slice_buffers(model)
    n = n1 + n2 + n3
    if n == 0:
        print("No tensors or buffers to patch.")
        sys.exit(0)
    print(f"Writing: {out_path} ({n1} concat + {n2} reshape + {n3} slice patched)")
    _write_model(model, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
