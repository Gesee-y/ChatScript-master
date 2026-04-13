## tensor.nim
## ============
## A minimal dense-tensor / matrix library sufficient for MEIM.
## Tensors are stored in row-major order as a flat seq[float32].
##
## Supported shapes:
##   1-D  (vectors)
##   2-D  (matrices)
##   3-D  (Ce x Ce x Cr  core tensors per partition)
##   4-D  (K  x Ce x Ce x Cr stacked core tensor W)
##
## All gradient accumulation is done externally; this module is
## purely about storage and arithmetic.

import math, random, sequtils, strformat

type
  Tensor* = object
    data*: seq[float32]
    shape*: seq[int]   ## shape[0] x shape[1] x ... (row-major)

# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

proc newTensor*(shape: openArray[int]): Tensor =
  ## Allocate a zero-filled tensor.
  var total = 1
  for d in shape: total *= d
  Tensor(data: newSeq[float32](total), shape: @shape)

proc newTensorRand*(shape: openArray[int]; scale: float32 = 0.1'f32): Tensor =
  ## Allocate a tensor filled with uniform random values in [-scale, scale].
  result = newTensor(shape)
  for i in 0 ..< result.data.len:
    result.data[i] = (rand(2.0) - 1.0).float32 * scale

proc zeros*(shape: openArray[int]): Tensor = newTensor(shape)

proc ones*(shape: openArray[int]): Tensor =
  result = newTensor(shape)
  for i in 0 ..< result.data.len: result.data[i] = 1.0'f32

proc eye*(n: int): Tensor =
  ## n x n identity matrix.
  result = newTensor([n, n])
  for i in 0 ..< n: result.data[i * n + i] = 1.0'f32

proc copyTensor*(t: Tensor): Tensor =
  Tensor(data: t.data, shape: t.shape)

# ---------------------------------------------------------------------------
# Shape / indexing utilities
# ---------------------------------------------------------------------------

proc ndim*(t: Tensor): int = t.shape.len

proc numel*(t: Tensor): int =
  result = 1
  for d in t.shape: result *= d

proc flatIndex*(t: Tensor; idx: openArray[int]): int =
  ## Compute row-major flat index from multi-dimensional index.
  assert idx.len == t.shape.len
  result = 0
  var stride = 1
  for d in countdown(t.shape.len - 1, 0):
    result += idx[d] * stride
    stride *= t.shape[d]

proc `[]`*(t: Tensor; idx: openArray[int]): float32 =
  t.data[t.flatIndex(idx)]

proc `[]=`*(t: var Tensor; idx: openArray[int]; v: float32) =
  t.data[t.flatIndex(idx)] = v

# ---------------------------------------------------------------------------
# Arithmetic (element-wise)
# ---------------------------------------------------------------------------

proc `+`*(a, b: Tensor): Tensor =
  assert a.data.len == b.data.len
  result = newTensor(a.shape)
  for i in 0 ..< a.data.len: result.data[i] = a.data[i] + b.data[i]

proc `-`*(a, b: Tensor): Tensor =
  assert a.data.len == b.data.len
  result = newTensor(a.shape)
  for i in 0 ..< a.data.len: result.data[i] = a.data[i] - b.data[i]

proc `*`*(a: Tensor; s: float32): Tensor =
  result = newTensor(a.shape)
  for i in 0 ..< a.data.len: result.data[i] = a.data[i] * s

proc `*`*(s: float32; a: Tensor): Tensor = a * s

proc `/`*(a: Tensor; s: float32): Tensor = a * (1.0'f32 / s)

proc `+=`*(a: var Tensor; b: Tensor) =
  assert a.data.len == b.data.len
  for i in 0 ..< a.data.len: a.data[i] += b.data[i]

proc `-=`*(a: var Tensor; b: Tensor) =
  assert a.data.len == b.data.len
  for i in 0 ..< a.data.len: a.data[i] -= b.data[i]

proc `*=`*(a: var Tensor; s: float32) =
  for i in 0 ..< a.data.len: a.data[i] *= s

proc neg*(a: Tensor): Tensor = a * (-1.0'f32)

proc addInPlace*(a: var Tensor; b: Tensor; scale: float32 = 1.0'f32) =
  ## a += scale * b
  assert a.data.len == b.data.len
  for i in 0 ..< a.data.len: a.data[i] += scale * b.data[i]

proc elementwiseMul*(a, b: Tensor): Tensor =
  assert a.data.len == b.data.len
  result = newTensor(a.shape)
  for i in 0 ..< a.data.len: result.data[i] = a.data[i] * b.data[i]

# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------

proc sum*(t: Tensor): float32 =
  for v in t.data: result += v

proc sumSq*(t: Tensor): float32 =
  for v in t.data: result += v * v

proc norm*(t: Tensor): float32 = sqrt(sumSq(t))

# ---------------------------------------------------------------------------
# Matrix operations  (2-D only)
# ---------------------------------------------------------------------------

proc rows*(t: Tensor): int =
  assert t.shape.len >= 1; t.shape[0]
proc cols*(t: Tensor): int =
  assert t.shape.len >= 2; t.shape[1]

proc matmul*(A, B: Tensor): Tensor =
  ## C = A @ B   (M x K) @ (K x N) -> (M x N)
  let M = A.rows; let K = A.cols; let N = B.cols
  assert K == B.rows, &"matmul shape mismatch {A.shape} @ {B.shape}"
  result = newTensor([M, N])
  for i in 0 ..< M:
    for j in 0 ..< N:
      var s = 0.0'f32
      for k in 0 ..< K:
        s += A.data[i * K + k] * B.data[k * N + j]
      result.data[i * N + j] = s

proc matvec*(A: Tensor; x: Tensor): Tensor =
  ## y = A @ x   (M x N) @ (N,) -> (M,)
  let M = A.rows; let N = A.cols
  assert x.data.len == N
  result = newTensor([M])
  for i in 0 ..< M:
    var s = 0.0'f32
    for j in 0 ..< N:
      s += A.data[i * N + j] * x.data[j]
    result.data[i] = s

proc transpose*(A: Tensor): Tensor =
  ## Transpose a 2-D matrix.
  let M = A.rows; let N = A.cols
  result = newTensor([N, M])
  for i in 0 ..< M:
    for j in 0 ..< N:
      result.data[j * M + i] = A.data[i * N + j]

proc dot*(a, b: Tensor): float32 =
  ## Flat dot product of two tensors (treated as vectors).
  assert a.data.len == b.data.len
  for i in 0 ..< a.data.len: result += a.data[i] * b.data[i]

proc outerProduct*(a, b: Tensor): Tensor =
  ## (M,) x (N,) -> (M, N)
  let M = a.data.len; let N = b.data.len
  result = newTensor([M, N])
  for i in 0 ..< M:
    for j in 0 ..< N:
      result.data[i * N + j] = a.data[i] * b.data[j]

# ---------------------------------------------------------------------------
# Row / slice helpers
# ---------------------------------------------------------------------------

proc row*(A: Tensor; i: int): Tensor =
  ## Extract row i from a 2-D matrix as a 1-D tensor.
  let N = A.cols
  result = newTensor([N])
  let offset = i * N
  for j in 0 ..< N: result.data[j] = A.data[offset + j]

proc setRow*(A: var Tensor; i: int; v: Tensor) =
  let N = A.cols
  assert v.data.len == N
  let offset = i * N
  for j in 0 ..< N: A.data[offset + j] = v.data[j]

proc addToRow*(A: var Tensor; i: int; v: Tensor; scale: float32 = 1.0'f32) =
  let N = A.cols
  assert v.data.len == N
  let offset = i * N
  for j in 0 ..< N: A.data[offset + j] += scale * v.data[j]

# ---------------------------------------------------------------------------
# Reshape / view  (no copy — just change the shape metadata)
# ---------------------------------------------------------------------------

proc reshape*(t: Tensor; newShape: openArray[int]): Tensor =
  var total = 1
  for d in newShape: total *= d
  assert total == t.data.len, "reshape: total elements must match"
  Tensor(data: t.data, shape: @newShape)

# ---------------------------------------------------------------------------
# Softmax / log-sum-exp (over a 1-D tensor / seq of scores)
# ---------------------------------------------------------------------------

proc softmax*(scores: seq[float32]): seq[float32] =
  let maxS = scores.foldl(max(a, b), low(float32))
  result = scores.mapIt(exp(it - maxS))
  let s = result.foldl(a + b, 0.0'f32)
  for i in 0 ..< result.len: result[i] /= s

proc logSumExp*(scores: seq[float32]): float32 =
  let maxS = scores.foldl(max(a, b), low(float32))
  var s = 0.0'f32
  for v in scores: s += exp(v - maxS)
  maxS + ln(s)

# ---------------------------------------------------------------------------
# Dropout mask (applied in-place during training)
# ---------------------------------------------------------------------------

proc applyDropout*(t: var Tensor; rate: float32) =
  ## Zero out each element with probability `rate`, scale survivors.
  if rate <= 0.0'f32: return
  let scale = 1.0'f32 / (1.0'f32 - rate)
  for i in 0 ..< t.data.len:
    if rand(1.0) < rate.float64:
      t.data[i] = 0.0'f32
    else:
      t.data[i] *= scale

# ---------------------------------------------------------------------------
# Batch normalization (simple running-stats version, no learnable params)
# ---------------------------------------------------------------------------

proc batchNorm*(t: var Tensor; eps: float32 = 1e-5'f32) =
  ## Normalize each element of the flat vector (mean 0, std 1).
  var mean = 0.0'f32
  for v in t.data: mean += v
  mean /= t.data.len.float32
  var variance = 0.0'f32
  for v in t.data: variance += (v - mean) * (v - mean)
  variance /= t.data.len.float32
  let invStd = 1.0'f32 / sqrt(variance + eps)
  for i in 0 ..< t.data.len:
    t.data[i] = (t.data[i] - mean) * invStd

# ---------------------------------------------------------------------------
# Debug / display
# ---------------------------------------------------------------------------

proc `$`*(t: Tensor): string =
  result = &"Tensor(shape={t.shape}, data=[{t.data[0..min(7,t.data.len-1)]}...])"
