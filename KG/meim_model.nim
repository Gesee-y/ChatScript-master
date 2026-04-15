# meim_model.nim
# =============
# Model architecture for MEIM (Multi-partition Embedding Interaction Model).
# Implements the core tensor interaction: S(h,r,t) = sum_k h_k^T M_{W,r,k} t_k.

import math, strformat, strutils, random, tables, algorithm, sets
import tensor, kg_loader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

type
  MEIMConfig* = object
    ## All hyper-parameters for the MEIM model.
    numEntities*:  int        ## |E|
    numRelations*: int        ## |R|
    K*:            int        ## Number of partitions
    Ce*:           int        ## Embedding size per entity partition
    Cr*:           int        ## Embedding size per relation partition
    inputDropRate*:  float32  ## Dropout on entity input embeddings
    hiddenDropRate*: float32  ## Dropout on hidden layer (h^T M_{W,r,k})
    lambdaOrtho*:    float32  ## Weight for orthogonality regularisation
    lambdaUnitNorm*: float32  ## Weight for unit-norm regularisation
    orthoP*:         int      ## Exponent p for unit-norm penalty
    batchSize*:      int
    learningRate*:   float32
    lrDecay*:        float32  ## Multiplicative decay per epoch
    maxEpochs*:      int
    evalEvery*:      int      ## Evaluate on validation set every N epochs
    kVsAll*:         bool     ## Sample negatives instead of 1-vs-all
    kVsAllK*:        int      ## Number of negatives if kVsAll is true

proc defaultConfig*(nEntities, nRelations: int): MEIMConfig =
  MEIMConfig(
    numEntities: nEntities,
    numRelations: nRelations,
    K: 3, Ce: 32, Cr: 32,
    inputDropRate: 0.2'f32,
    hiddenDropRate: 0.3'f32,
    lambdaOrtho: 0.1'f32,
    lambdaUnitNorm: 0.1'f32,
    orthoP: 3,
    batchSize: 128,
    learningRate: 1e-3'f32,
    lrDecay: 1.0'f32,
    maxEpochs: 200,
    evalEvery: 10,
    kVsAll: false,
    kVsAllK: 10
  )

# ---------------------------------------------------------------------------
# Model Parameters and Optimization State
# ---------------------------------------------------------------------------

type
  MEIMParams* = object
    entityEmb*:   Tensor   ## [numEntities, K*Ce]
    relationEmb*: Tensor   ## [numRelations, K*Cr]
    W*:           Tensor   ## [K * Ce * Ce * Cr]  (Core tensor for interaction)
    wStrideK*:    int      ## Ce * Ce * Cr
    wStrideCe1*:  int      ## Ce * Cr
    wStrideCe2*:  int      ## Cr

  AdamState* = object
    m*: Tensor   ## First moment
    v*: Tensor   ## Second moment

proc newAdamState*(t: Tensor): AdamState =
  AdamState(m: newTensor(t.shape), v: newTensor(t.shape))

type
  Optimiser* = object
    lr*:      float32
    beta1*:   float32
    beta2*:   float32
    epsilon*: float32
    step*:    int64
    mE*:  AdamState
    mR*:  AdamState
    mW*:  AdamState

proc newOptimiser*(params: MEIMParams, lr: float32 = 3e-3'f32): Optimiser =
  Optimiser(
    lr: lr, beta1: 0.9'f32, beta2: 0.999'f32, epsilon: 1e-8'f32, step: 0,
    mE: newAdamState(params.entityEmb),
    mR: newAdamState(params.relationEmb),
    mW: newAdamState(params.W),
  )

proc initParams*(cfg: MEIMConfig): MEIMParams =
  let De = cfg.K * cfg.Ce
  let Dr = cfg.K * cfg.Cr
  let wTotal = cfg.K * cfg.Ce * cfg.Ce * cfg.Cr
  # Xavier-like initialization for stability
  let scaleE = sqrt(6.0 / (De.float + cfg.numEntities.float)).float32
  let scaleW = sqrt(1.0 / cfg.Ce.float).float32
  
  result.entityEmb   = newTensorRand([cfg.numEntities, De],  scale = scaleE)
  result.relationEmb = newTensorRand([cfg.numRelations, Dr], scale = 0.5'f32)
  result.W           = newTensorRand([wTotal],               scale = scaleW)
  result.wStrideK    = cfg.Ce * cfg.Ce * cfg.Cr
  result.wStrideCe1  = cfg.Ce * cfg.Cr
  result.wStrideCe2  = cfg.Cr

proc initFromEmbeddings*(params: var MEIMParams, embeddings: seq[seq[float]]) =
  ## Bootstraps entity embeddings using pre-trained vectors.
  ## Takes a sequence of embeddings aligned with entity IDs.
  let numEnts = params.entityEmb.shape[0]
  let dimMEIM = params.entityEmb.shape[1]
  
  for i in 0 ..< min(numEnts, embeddings.len):
    let emb = embeddings[i]
    if emb.len == 0: continue
    for j in 0 ..< min(dimMEIM, emb.len):
      params.entityEmb.data[i * dimMEIM + j] = emb[j].float32

# ---------------------------------------------------------------------------
# MEIM forward pass utilities
# ---------------------------------------------------------------------------

proc getEntityPartition*(params: MEIMParams; entityId, k, Ce: int): Tensor =
  result = newTensor([Ce])
  let rowOffset = entityId * (params.entityEmb.cols)
  let partOffset = k * Ce
  for i in 0 ..< Ce:
    result.data[i] = params.entityEmb.data[rowOffset + partOffset + i]

proc getRelationPartition*(params: MEIMParams; relId, k, Cr: int): Tensor =
  result = newTensor([Cr])
  let rowOffset = relId * (params.relationEmb.cols)
  let partOffset = k * Cr
  for i in 0 ..< Cr:
    result.data[i] = params.relationEmb.data[rowOffset + partOffset + i]

proc computeMappingMatrix*(params: MEIMParams; rk: Tensor; k, Ce, Cr: int): Tensor =
  result = newTensor([Ce, Ce])
  let kBase = k * params.wStrideK
  for i in 0 ..< Ce:
    for j in 0 ..< Ce:
      var s = 0.0'f32
      let ijBase = kBase + i * params.wStrideCe1 + j * params.wStrideCe2
      for c in 0 ..< Cr:
        s += params.W.data[ijBase + c] * rk.data[c]
      result.data[i * Ce + j] = s

proc getNormalizedEntityPartitions*(params: MEIMParams; cfg: MEIMConfig): seq[seq[Tensor]] =
  ## Pre-calculate all normalized partitions for all entities.
  ## Returns result[entityId][partitionK]
  result = newSeq[seq[Tensor]](cfg.numEntities)
  for eId in 0 ..< cfg.numEntities:
    result[eId] = newSeq[Tensor](cfg.K)
    for k in 0 ..< cfg.K:
      var p = getEntityPartition(params, eId, k, cfg.Ce)
      batchNorm(p)
      result[eId][k] = p

proc scoreTriple*(params: MEIMParams; cfg: MEIMConfig;
                  headId, tailId, relId: int): float32 =
  for k in 0 ..< cfg.K:
    var hk = getEntityPartition(params, headId, k, cfg.Ce)
    var tk = getEntityPartition(params, tailId, k, cfg.Ce)
    # Norm-stabilization (Implicit Batch/Layer Norm)
    batchNorm(hk)
    batchNorm(tk)
    
    let rk = getRelationPartition(params, relId, k, cfg.Cr)
    let Mk = computeMappingMatrix(params, rk, k, cfg.Ce, cfg.Cr)
    let Mkt = matvec(Mk, tk)
    result += dot(hk, Mkt)

proc scoreTripleWithDropout*(params: var MEIMParams; cfg: MEIMConfig;
                              headId, tailId, relId: int): float32 =
  for k in 0 ..< cfg.K:
    var hk = getEntityPartition(params, headId, k, cfg.Ce)
    var tk = getEntityPartition(params, tailId, k, cfg.Ce)
    batchNorm(hk)
    batchNorm(tk)
    applyDropout(hk, cfg.inputDropRate)
    let rk = getRelationPartition(params, relId, k, cfg.Cr)
    let Mk = computeMappingMatrix(params, rk, k, cfg.Ce, cfg.Cr)
    var hidden = matvec(Mk, tk)
    applyDropout(hidden, cfg.hiddenDropRate)
    result += dot(hk, hidden)

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

proc computeOrthoLoss*(params: MEIMParams; cfg: MEIMConfig;
                        relId: int): float32 =
  if cfg.lambdaOrtho == 0.0'f32 and cfg.lambdaUnitNorm == 0.0'f32:
    return 0.0'f32

  for k in 0 ..< cfg.K:
    let rk = getRelationPartition(params, relId, k, cfg.Cr)
    let Mk = computeMappingMatrix(params, rk, k, cfg.Ce, cfg.Cr)

    if cfg.lambdaOrtho != 0.0'f32:
      let Mt  = transpose(Mk)
      let MtM = matmul(Mt, Mk)
      let Id  = eye(cfg.Ce)
      let diff = MtM - Id
      result += cfg.lambdaOrtho * sumSq(diff)

    if cfg.lambdaUnitNorm != 0.0'f32:
      let normSq = sumSq(rk)
      let dev    = abs(normSq - 1.0'f32)
      result += cfg.lambdaUnitNorm * pow(dev.float64, cfg.orthoP.float64).float32

# ---------------------------------------------------------------------------
# Scorer for all entities
# ---------------------------------------------------------------------------

proc scoreAllTails*(params: MEIMParams; cfg: MEIMConfig;
                    headId, relId: int;
                    cache: seq[seq[Tensor]] = @[];
                    allowedIds: HashSet[int] = initHashSet[int]()): seq[float32] =
  result = newSeq[float32](cfg.numEntities)
  var hParts = newSeq[Tensor](cfg.K)
  var Mks    = newSeq[Tensor](cfg.K)
  for k in 0 ..< cfg.K:
    hParts[k] = getEntityPartition(params, headId, k, cfg.Ce)
    batchNorm(hParts[k])
    let rk    = getRelationPartition(params, relId, k, cfg.Cr)
    Mks[k]    = computeMappingMatrix(params, rk, k, cfg.Ce, cfg.Cr)

  # Use provided cache if available, otherwise calculate
  let allNormalized = if cache.len > 0: cache else: getNormalizedEntityPartitions(params, cfg)

  for tId in 0 ..< cfg.numEntities:
    if allowedIds.len > 0 and tId notin allowedIds:
      result[tId] = -1e30'f32
      continue

    var s = 0.0'f32
    for k in 0 ..< cfg.K:
      let tk = allNormalized[tId][k]
      let Mkt = matvec(Mks[k], tk)
      s += dot(hParts[k], Mkt)
    result[tId] = s

proc scoreAllHeads*(params: MEIMParams; cfg: MEIMConfig;
                    tailId, relId: int;
                    cache: seq[seq[Tensor]] = @[];
                    allowedIds: HashSet[int] = initHashSet[int]()): seq[float32] =
  result = newSeq[float32](cfg.numEntities)
  var tParts = newSeq[Tensor](cfg.K)
  var Mks    = newSeq[Tensor](cfg.K)
  for k in 0 ..< cfg.K:
    tParts[k] = getEntityPartition(params, tailId, k, cfg.Ce)
    batchNorm(tParts[k])
    let rk    = getRelationPartition(params, relId, k, cfg.Cr)
    Mks[k]    = computeMappingMatrix(params, rk, k, cfg.Ce, cfg.Cr)

  let allNormalized = if cache.len > 0: cache else: getNormalizedEntityPartitions(params, cfg)

  for hId in 0 ..< cfg.numEntities:
    if allowedIds.len > 0 and hId notin allowedIds:
      result[hId] = -1e30'f32
      continue

    var s = 0.0'f32
    for k in 0 ..< cfg.K:
      let hk = allNormalized[hId][k]
      let Mkt = matvec(Mks[k], tParts[k])
      s += dot(hk, Mkt)
    result[hId] = s

# ---------------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------------

type
  GradBuffer* = object
    entityEmb*:   Tensor
    relationEmb*: Tensor
    W*:           Tensor

proc newGradBuffer*(cfg: MEIMConfig): GradBuffer =
  let De = cfg.K * cfg.Ce
  let Dr = cfg.K * cfg.Cr
  let wTotal = cfg.K * cfg.Ce * cfg.Ce * cfg.Cr
  GradBuffer(
    entityEmb:   newTensor([cfg.numEntities, De]),
    relationEmb: newTensor([cfg.numRelations, Dr]),
    W:           newTensor([wTotal]),
  )

proc zeroGrads*(g: var GradBuffer) =
  for i in 0 ..< g.entityEmb.data.len:   g.entityEmb.data[i]   = 0.0'f32
  for i in 0 ..< g.relationEmb.data.len: g.relationEmb.data[i] = 0.0'f32
  for i in 0 ..< g.W.data.len:           g.W.data[i]           = 0.0'f32

proc accumulateScoreGrad*(params: MEIMParams; cfg: MEIMConfig;
                           headId, tailId, relId: int;
                           dScore: float32;
                           grads: var GradBuffer;
                           cache: seq[seq[Tensor]] = @[]) =
  for k in 0 ..< cfg.K:
    var hk = if cache.len > 0: cache[headId][k] else: (var p = getEntityPartition(params, headId, k, cfg.Ce); batchNorm(p); p)
    var tk = if cache.len > 0: cache[tailId][k] else: (var p = getEntityPartition(params, tailId, k, cfg.Ce); batchNorm(p); p)
    
    let rk = getRelationPartition(params, relId, k, cfg.Cr)
    let Mk = computeMappingMatrix(params, rk, k, cfg.Ce, cfg.Cr)

    let dHk = matvec(Mk, tk)
    let headRowOff  = headId * (cfg.K * cfg.Ce) + k * cfg.Ce
    for i in 0 ..< cfg.Ce:
      grads.entityEmb.data[headRowOff + i] += dScore * dHk.data[i]

    let MkT = transpose(Mk)
    let dTk = matvec(MkT, hk)
    let tailRowOff = tailId * (cfg.K * cfg.Ce) + k * cfg.Ce
    for i in 0 ..< cfg.Ce:
      grads.entityEmb.data[tailRowOff + i] += dScore * dTk.data[i]

    let kBase = k * params.wStrideK
    for i in 0 ..< cfg.Ce:
      for j in 0 ..< cfg.Ce:
        let ijBase = kBase + i * params.wStrideCe1 + j * params.wStrideCe2
        let hiti  = dScore * hk.data[i] * tk.data[j]
        for c in 0 ..< cfg.Cr:
          grads.W.data[ijBase + c] += hiti * rk.data[c]

    let relRowOff = relId * (cfg.K * cfg.Cr) + k * cfg.Cr
    for i in 0 ..< cfg.Ce:
      for j in 0 ..< cfg.Ce:
        let ijBase = kBase + i * params.wStrideCe1 + j * params.wStrideCe2
        let hiti   = dScore * hk.data[i] * tk.data[j]
        for c in 0 ..< cfg.Cr:
          grads.relationEmb.data[relRowOff + c] += hiti * params.W.data[ijBase + c]

proc accumulateOrthoGrad*(params: MEIMParams; cfg: MEIMConfig;
                           relId: int; grads: var GradBuffer) =
  if cfg.lambdaOrtho == 0.0'f32 and cfg.lambdaUnitNorm == 0.0'f32: return

  for k in 0 ..< cfg.K:
    let rk = getRelationPartition(params, relId, k, cfg.Cr)
    let Mk = computeMappingMatrix(params, rk, k, cfg.Ce, cfg.Cr)

    if cfg.lambdaOrtho != 0.0'f32:
      let MtM  = matmul(transpose(Mk), Mk)
      let Id   = eye(cfg.Ce)
      let diff = MtM - Id
      let dM   = matmul(Mk, diff) * (4.0'f32 * cfg.lambdaOrtho)

      let kBase = k * params.wStrideK
      let relRowOff = relId * (cfg.K * cfg.Cr) + k * cfg.Cr
      for i in 0 ..< cfg.Ce:
        for j in 0 ..< cfg.Ce:
          let dMij  = dM.data[i * cfg.Ce + j]
          let ijBase = kBase + i * params.wStrideCe1 + j * params.wStrideCe2
          for c in 0 ..< cfg.Cr:
            grads.W.data[ijBase + c] += dMij * rk.data[c]
            grads.relationEmb.data[relRowOff + c] += dMij * params.W.data[ijBase + c]

    if cfg.lambdaUnitNorm != 0.0'f32:
      let normSq = sumSq(rk)
      let dev    = normSq - 1.0'f32
      let absDev = abs(dev)
      if absDev > 1e-12'f32:
        let sign   = if dev > 0.0'f32: 1.0'f32 else: -1.0'f32
        let dCoeff = cfg.lambdaUnitNorm * sign * cfg.orthoP.float32 *
                     pow(absDev.float64, (cfg.orthoP - 1).float64).float32 * 2.0'f32
        let relRowOff = relId * (cfg.K * cfg.Cr) + k * cfg.Cr
        for c in 0 ..< cfg.Cr:
          grads.relationEmb.data[relRowOff + c] += dCoeff * rk.data[c]

# ---------------------------------------------------------------------------
# Optimisation rule
# ---------------------------------------------------------------------------

proc adamUpdate(param: var Tensor; grad: Tensor;
                state: var AdamState;
                lr, beta1, beta2, eps: float32; step: int64) =
  let b1t = 1.0'f32 - pow(beta1.float64, step.float64).float32
  let b2t = 1.0'f32 - pow(beta2.float64, step.float64).float32
  let lrt = lr * sqrt(b2t) / b1t
  for i in 0 ..< param.data.len:
    let g = grad.data[i]
    state.m.data[i] = beta1 * state.m.data[i] + (1.0'f32 - beta1) * g
    state.v.data[i] = beta2 * state.v.data[i] + (1.0'f32 - beta2) * g * g
    param.data[i] -= lrt * state.m.data[i] / (sqrt(state.v.data[i]) + eps)

proc adamStep*(opt: var Optimiser; params: var MEIMParams; grads: GradBuffer) =
  inc opt.step
  let s = opt.step
  adamUpdate(params.entityEmb,   grads.entityEmb,   opt.mE, opt.lr, opt.beta1, opt.beta2, opt.epsilon, s)
  adamUpdate(params.relationEmb, grads.relationEmb, opt.mR, opt.lr, opt.beta1, opt.beta2, opt.epsilon, s)
  adamUpdate(params.W,           grads.W,           opt.mW, opt.lr, opt.beta1, opt.beta2, opt.epsilon, s)

proc decayLR*(opt: var Optimiser; factor: float32) =
  opt.lr *= factor

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

proc saveParams*(params: MEIMParams, path: string) =
  var f: File
  if not open(f, path, fmWrite): return
  defer: close(f)
  proc writeTensor(f: File, t: Tensor) =
    var n = t.shape.len; discard f.writeBuffer(addr n, sizeof(int))
    for d in t.shape: discard f.writeBuffer(addr d, sizeof(int))
    var s = t.data.len; discard f.writeBuffer(addr s, sizeof(int))
    if s > 0: discard f.writeBuffer(addr t.data[0], s * sizeof(float32))
  f.writeTensor(params.entityEmb)
  f.writeTensor(params.relationEmb)
  f.writeTensor(params.W)
  discard f.writeBuffer(addr params.wStrideK, sizeof(int))
  discard f.writeBuffer(addr params.wStrideCe1, sizeof(int))
  discard f.writeBuffer(addr params.wStrideCe2, sizeof(int))

proc loadParams*(path: string): MEIMParams =
  var f: File
  if not open(f, path, fmRead): return
  defer: close(f)
  proc readTensor(f: File): Tensor =
    var n: int; discard f.readBuffer(addr n, sizeof(int))
    var sh = newSeq[int](n)
    for i in 0 ..< n: discard f.readBuffer(addr sh[i], sizeof(int))
    var s: int; discard f.readBuffer(addr s, sizeof(int))
    var d = newSeq[float32](s)
    if s > 0: discard f.readBuffer(addr d[0], s * sizeof(float32))
    Tensor(data: d, shape: sh)
  result.entityEmb = f.readTensor()
  result.relationEmb = f.readTensor()
  result.W = f.readTensor()
  discard f.readBuffer(addr result.wStrideK, sizeof(int))
  discard f.readBuffer(addr result.wStrideCe1, sizeof(int))
  discard f.readBuffer(addr result.wStrideCe2, sizeof(int))

proc saveConfig*(cfg: MEIMConfig, path: string) =
  var f: File; if not open(f, path, fmWrite): return
  defer: close(f)
  f.writeLine(&"{cfg.numEntities}"); f.writeLine(&"{cfg.numRelations}")
  f.writeLine(&"{cfg.K}"); f.writeLine(&"{cfg.Ce}"); f.writeLine(&"{cfg.Cr}")
  f.writeLine(&"{cfg.inputDropRate}"); f.writeLine(&"{cfg.hiddenDropRate}")

proc loadConfig*(path: string): MEIMConfig =
  let l = readFile(path).splitLines(); if l.len < 5: return
  result.numEntities = parseInt(l[0]); result.numRelations = parseInt(l[1])
  result.K = parseInt(l[2]); result.Ce = parseInt(l[3]); result.Cr = parseInt(l[4])
  if l.len > 5: result.inputDropRate = parseFloat(l[5]).float32
  if l.len > 6: result.hiddenDropRate = parseFloat(l[6]).float32
