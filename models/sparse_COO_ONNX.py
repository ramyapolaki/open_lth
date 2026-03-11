import numpy as np
import onnx
from onnx import numpy_helper
from onnx.onnx_ml_pb2 import SparseTensorProto, TensorProto


DENSE_ONNX = "conv2_pruned_dense.onnx"
OUT_SPARSE_ONNX = "conv2_pruned_sparse_coo.onnx"

SPARSIFY = {
    "layers.0.conv1.weight",
    "layers.0.conv2.weight",
    "fc1.weight",
    "fc2.weight",
    "fc3.weight",
}


def make_tensorproto(name: str, arr: np.ndarray, dtype: int) -> TensorProto:
    t = TensorProto()
    t.name = name
    t.data_type = dtype
    t.dims.extend(list(arr.shape))
    t.raw_data = arr.tobytes(order="C")
    return t


def dense_to_sparse_coo(arr: np.ndarray, base_name: str) -> SparseTensorProto:
    arr = np.asarray(arr)
    dims = list(arr.shape)

    nz = np.argwhere(arr != 0)  # [nnz, rank]
    if nz.size == 0:
        nz = np.zeros((0, arr.ndim), dtype=np.int64)
        vals = np.zeros((0,), dtype=np.float32)
    else:
        nz = nz.astype(np.int64)
        vals = arr[tuple(nz.T)].astype(np.float32, copy=False)

    st = SparseTensorProto()
    st.dims.extend(dims)

    st.indices.CopyFrom(make_tensorproto(base_name + "_indices", nz, TensorProto.INT64))
    st.values.CopyFrom(make_tensorproto(base_name + "_values", vals, TensorProto.FLOAT))
    return st


model = onnx.load(DENSE_ONNX)
g = model.graph

new_dense_inits = []
new_sparse_inits = []
converted = []

for init in list(g.initializer):
    if init.name in SPARSIFY:
        arr = numpy_helper.to_array(init)
        new_sparse_inits.append(dense_to_sparse_coo(arr, init.name))
        converted.append(init.name)
    else:
        new_dense_inits.append(init)

del g.initializer[:]
g.initializer.extend(new_dense_inits)

del g.sparse_initializer[:]
g.sparse_initializer.extend(new_sparse_inits)

onnx.save(model, OUT_SPARSE_ONNX)

print("Saved:", OUT_SPARSE_ONNX)
print("Converted:", converted)
print("Dense initializers left:", [x.name for x in g.initializer])
print("Sparse initializers count:", len(g.sparse_initializer))
