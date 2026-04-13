## meim_model.nim
## ===============
## Implementation of MEIM: Multi-partition Embedding Interaction iMproved
## beyond Block Term Format for Knowledge Graph Link Prediction.
##
## Reference
## ---------
##   Hung-Nghiep Tran, Atsuhiro Takasu.
##   "MEIM: Multi-partition Embedding Interaction Beyond Block Term Format
##    for Efficient and Expressive Link Prediction."
##   IJCAI-22, 2022.
##
## Architecture summary
## --------------------
##   Embeddings:
##     H  in R^(|E| x De)   entity embeddings  (De = K * Ce)
##     T  in R^(|E| x De)
##     R  in R^(|R| x Dr)   relation embeddings (Dr = K * Cr)
##
##   Score function (Eq. 5-8 in paper):
##     S(h,t,r) = sum_{k=1}^{K}  h_k^T  M_{W,r,k}  t_k
##
##   where M_{W,r,k} = W_k x3 r_k  (bilinear mapping generated from core
##   tensor W_k in R^{Ce x Ce x Cr} and relation partition r_k in R^{Cr}).
##
##   Key novelties over MEI:
##     1. Independent core tensors  W_k  (one per partition, not shared).
##     2. Soft orthogonality loss on M_{W,r,k}  (Eq. 16).
##
##   Loss:
##     L = L_link_prediction + L_ortho
##   L_link_prediction uses 1-vs-all / k-vs-all softmax cross-entropy.
##   L_ortho = lambda_ortho * sum_k || M_W,r,k^T M_W,r,k - I ||_F^2
##           + lambda_unitnorm * sum_k |r_k^T r_k - 1|^p

import math
import tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

type
  MEIMConfig* = object
    ## All hyper-parameters for the MEIM model.
    numEntities*:  int        ## |E|
    numRelations*: int        ## |R|
    K*:            int        ## Number of partitions (paper: 3 for WN18RR/FB15K-237, 5 for YAGO3-10)
    Ce*:           int        ## Embedding size per entity partition
    Cr*:           int        ## Embedding size per relation partition
    inputDropRate*:  float32  ## Dropout on entity input embeddings
    hiddenDropRate*: float32  ## Dropout on hidden layer (h^T M_{W,r,k})
    lambdaOrtho*:    float32  ## Weight for orthogonality regularisation
    lambdaUnitNorm*: float32  ## Weight for unit-norm regularisation
    orthoP*:         int      ## Exponent p for unit-norm penalty (paper: 3)
    batchSize*:      int
    learningRate*:   float32
    lrDecay*:        float32  ## Multiplicative decay per epoch  (1.0 = no decay)
    maxEpochs*:      int
    evalEvery*:      int      ## Evaluate on validation set every N epochs
    kVsAll*:         bool     ## true -> k-vs-all;  false -> 1-vs-all
    kVsAllK*:        int      ## k for k-vs-all sampling

proc defaultConfig*(numEntities, numRelations: int): MEIMConfig =
  MEIMConfig(
    numEntities:  numEntities,
    numRelations: numRelations,
    K:            3,
    Ce:           100,
    Cr:           100,
    inputDropRate:  0.0'f32,
    hiddenDropRate: 0.0'f32,
    lambdaOrtho:    1e-1'f32,
    lambdaUnitNorm: 5e-4'f32,
    orthoP:         3,
    batchSize:      1024,
    learningRate:   3e-3'f32,
    lrDecay:        1.0'f32,
    maxEpochs:      1000,
    evalEvery:      10,
    kVsAll:         false,
    kVsAllK:        500,
  )

# ---------------------------------------------------------------------------
# Parameter block (one flat record used by the optimiser)
# ---------------------------------------------------------------------------

type
  MEIMParams* = object
    ## All learnable parameters of the MEIM model.
    ##
    ## entityEmb    : (|E|, K*Ce)   – entity embeddings  (shared for head & tail)
    ## relationEmb  : (|R|, K*Cr)   – relation embeddings
    ## W            : (K, Ce, Ce, Cr) – independent core tensors (stacked)
    entityEmb*:   Tensor   ## shape [numE, K*Ce]
    relationEmb*: Tensor   ## shape [numR, K*Cr]
    W*:           Tensor   ## shape [K*Ce*Ce*Cr]  (logically K x Ce x Ce x Cr)
    ## Strides for W indexing
    wStrideK*:    int      ## Ce * Ce * Cr
    wStrideCe1*:  int      ## Ce * Cr
    wStrideCe2*:  int      ## Cr

proc initParams*(cfg: MEIMConfig): MEIMParams =
  let De = cfg.K * cfg.Ce
  let Dr = cfg.K * cfg.Cr
  let wTotal = cfg.K * cfg.Ce * cfg.Ce * cfg.Cr
  result.entityEmb   = newTensorRand([cfg.numEntities, De],  scale = 1e-1'f32)
  result.relationEmb = newTensorRand([cfg.numRelations, Dr], scale = 1e-1'f32)
  result.W           = newTensorRand([wTotal],               scale = 1e-1'f32)
  result.wStrideK    = cfg.Ce * cfg.Ce * cfg.Cr
  result.wStrideCe1  = cfg.Ce * cfg.Cr
  result.wStrideCe2  = cfg.Cr

# ---------------------------------------------------------------------------
# MEIM forward pass utilities
# ---------------------------------------------------------------------------

proc getEntityPartition*(params: MEIMParams; entityId, k, Ce: int): Tensor =
  ## Extract partition k from an entity embedding row as a 1-D tensor.
  result = newTensor([Ce])
  let rowOffset = entityId * (params.entityEmb.cols)
  let partOffset = k * Ce
  for i in 0 ..< Ce:
    result.data[i] = params.entityEmb.data[rowOffset + partOffset + i]

proc getRelationPartition*(params: MEIMParams; relId, k, Cr: int): Tensor =
  ## Extract partition k from a relation embedding row as a 1-D tensor.
  result = newTensor([Cr])
  let rowOffset = relId * (params.relationEmb.cols)
  let partOffset = k * Cr
  for i in 0 ..< Cr:
    result.data[i] = params.relationEmb.data[rowOffset + partOffset + i]

proc computeMappingMatrix*(params: MEIMParams; rk: Tensor; k, Ce, Cr: int): Tensor =
  ## Compute M_{W,r,k} = W_k x3 r_k   (Ce x Ce matrix)
  ##
  ## W_k  is the Ce x Ce x Cr sub-tensor at index k.
  ## The mode-3 product with vector r_k in R^{Cr} collapses the Cr dimension:
  ##   M_{W,r,k}[i,j] = sum_{c=0}^{Cr-1}  W_k[i,j,c] * r_k[c]
  result = newTensor([Ce, Ce])
  let kBase = k * params.wStrideK
  for i in 0 ..< Ce:
    for j in 0 ..< Ce:
      var s = 0.0'f32
      let ijBase = kBase + i * params.wStrideCe1 + j * params.wStrideCe2
      for c in 0 ..< Cr:
        s += params.W.data[ijBase + c] * rk.data[c]
      result.data[i * Ce + j] = s

proc scoreTriple*(params: MEIMParams; cfg: MEIMConfig;
                  headId, tailId, relId: int): float32 =
  ## Compute S(h,t,r) = sum_{k=1}^{K}  h_k^T  M_{W,r,k}  t_k
  ## (Eq. 8 in the paper, without dropout – used during inference.)
  for k in 0 ..< cfg.K:
    let hk = getEntityPartition(params, headId, k, cfg.Ce)
    let tk = getEntityPartition(params, tailId, k, cfg.Ce)
    let rk = getRelationPartition(params, relId, k, cfg.Cr)
    let Mk = computeMappingMatrix(params, rk, k, cfg.Ce, cfg.Cr)
    # h_k^T M_k t_k  = dot(h_k, M_k @ t_k)
    let Mkt = matvec(Mk, tk)
    result += dot(hk, Mkt)

proc scoreTripleWithDropout*(params: var MEIMParams; cfg: MEIMConfig;
                              headId, tailId, relId: int): float32 =
  ## Same as scoreTriple but applies dropout in-place (training mode).
  for k in 0 ..< cfg.K:
    var hk = getEntityPartition(params, headId, k, cfg.Ce)
    var tk = getEntityPartition(params, tailId, k, cfg.Ce)
    let rk = getRelationPartition(params, relId, k, cfg.Cr)
    applyDropout(hk, cfg.inputDropRate)
    let Mk = computeMappingMatrix(params, rk, k, cfg.Ce, cfg.Cr)
    var hidden = matvec(Mk, tk)           ## h @ M_k = shape [Ce]
    applyDropout(hidden, cfg.hiddenDropRate)
    result += dot(hk, hidden)

# ---------------------------------------------------------------------------
# Soft orthogonality loss  (Eq. 16 in paper)
# ---------------------------------------------------------------------------

proc computeOrthoLoss*(params: MEIMParams; cfg: MEIMConfig;
                        relId: int): float32 =
  ## L_ortho for a single relation embedding.
  ## For each partition k:
  ##   ||M_{W,r,k}^T M_{W,r,k} - I||_F^2   * lambdaOrtho
  ##   + |r_k^T r_k - 1|^p                   * lambdaUnitNorm
  if cfg.lambdaOrtho == 0.0'f32 and cfg.lambdaUnitNorm == 0.0'f32:
    return 0.0'f32

  for k in 0 ..< cfg.K:
    let rk = getRelationPartition(params, relId, k, cfg.Cr)
    let Mk = computeMappingMatrix(params, rk, k, cfg.Ce, cfg.Cr)

    if cfg.lambdaOrtho != 0.0'f32:
      # MtM - I
      let Mt  = transpose(Mk)
      let MtM = matmul(Mt, Mk)
      let Id  = eye(cfg.Ce)
      let diff = MtM - Id
      result += cfg.lambdaOrtho * sumSq(diff)

    if cfg.lambdaUnitNorm != 0.0'f32:
      # |r_k^T r_k - 1|^p
      let normSq = sumSq(rk)
      let dev    = abs(normSq - 1.0'f32)
      result += cfg.lambdaUnitNorm * pow(dev.float64, cfg.orthoP.float64).float32

# ---------------------------------------------------------------------------
# Gradient computation (manual back-prop, dense SGD)
# ---------------------------------------------------------------------------

type
  GradBuffer* = object
    ## Accumulated gradients (same shapes as the corresponding params).
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

# ---------------------------------------------------------------------------
# Score all entities for (head, rel) -> tail prediction  (1-vs-all)
# Used both in training (softmax CE) and in evaluation (ranking).
# ---------------------------------------------------------------------------

proc scoreAllTails*(params: MEIMParams; cfg: MEIMConfig;
                    headId, relId: int): seq[float32] =
  ## Returns a score for every entity index (0 .. numEntities-1).
  result = newSeq[float32](cfg.numEntities)
  # Pre-compute h_k vectors and M_{W,r,k} matrices once per (h,r)
  var hParts = newSeq[Tensor](cfg.K)
  var Mks    = newSeq[Tensor](cfg.K)
  for k in 0 ..< cfg.K:
    hParts[k] = getEntityPartition(params, headId, k, cfg.Ce)
    let rk    = getRelationPartition(params, relId, k, cfg.Cr)
    Mks[k]    = computeMappingMatrix(params, rk, k, cfg.Ce, cfg.Cr)

  for tId in 0 ..< cfg.numEntities:
    var s = 0.0'f32
    for k in 0 ..< cfg.K:
      let tk  = getEntityPartition(params, tId, k, cfg.Ce)
      let Mkt = matvec(Mks[k], tk)
      s += dot(hParts[k], Mkt)
    result[tId] = s

proc scoreAllHeads*(params: MEIMParams; cfg: MEIMConfig;
                    tailId, relId: int): seq[float32] =
  ## Returns a score for every entity index as head entity for (?, rel, tail).
  result = newSeq[float32](cfg.numEntities)
  var tParts = newSeq[Tensor](cfg.K)
  var Mks    = newSeq[Tensor](cfg.K)
  for k in 0 ..< cfg.K:
    tParts[k] = getEntityPartition(params, tailId, k, cfg.Ce)
    let rk    = getRelationPartition(params, relId, k, cfg.Cr)
    Mks[k]    = computeMappingMatrix(params, rk, k, cfg.Ce, cfg.Cr)

  for hId in 0 ..< cfg.numEntities:
    var s = 0.0'f32
    for k in 0 ..< cfg.K:
      let hk  = getEntityPartition(params, hId, k, cfg.Ce)
      let Mkt = matvec(Mks[k], tParts[k])
      s += dot(hk, Mkt)
    result[hId] = s

# ---------------------------------------------------------------------------
# Gradient of the score w.r.t. each parameter, for a single (h,t,r) triple
# and a given loss gradient dL/dScore (scalar).
# ---------------------------------------------------------------------------

proc accumulateScoreGrad*(params: MEIMParams; cfg: MEIMConfig;
                           headId, tailId, relId: int;
                           dScore: float32;
                           grads: var GradBuffer) =
  ## Back-propagate `dScore` through the score function and accumulate
  ## gradients into `grads`.  This is called once per (triple, entity) pair
  ## during 1-vs-all training.
  ##
  ## Derivation (chain rule through Eq. 8):
  ##   S_k = h_k^T M_k t_k
  ##   dS_k/dh_k = M_k t_k
  ##   dS_k/dt_k = M_k^T h_k
  ##   dS_k/dM_k[i,j] = h_k[i] * t_k[j]
  ##   dM_k[i,j]/dW[k,i,j,c] = r_k[c]
  ##   dS_k/dW[k,i,j,c]  = h_k[i] * t_k[j] * r_k[c]
  ##   dM_k[i,j]/dr_k[c] = W_k[i,j,c]
  ##   dS_k/dr_k[c] = sum_{i,j} h_k[i] * t_k[j] * W_k[i,j,c]
  for k in 0 ..< cfg.K:
    let hk = getEntityPartition(params, headId, k, cfg.Ce)
    let tk = getEntityPartition(params, tailId, k, cfg.Ce)
    let rk = getRelationPartition(params, relId, k, cfg.Cr)
    let Mk = computeMappingMatrix(params, rk, k, cfg.Ce, cfg.Cr)

    # Gradient w.r.t. h_k :  dScore * M_k @ t_k
    let dHk = matvec(Mk, tk)
    let headRowOff  = headId * (cfg.K * cfg.Ce) + k * cfg.Ce
    for i in 0 ..< cfg.Ce:
      grads.entityEmb.data[headRowOff + i] += dScore * dHk.data[i]

    # Gradient w.r.t. t_k :  dScore * M_k^T @ h_k
    let MkT = transpose(Mk)
    let dTk = matvec(MkT, hk)
    let tailRowOff = tailId * (cfg.K * cfg.Ce) + k * cfg.Ce
    for i in 0 ..< cfg.Ce:
      grads.entityEmb.data[tailRowOff + i] += dScore * dTk.data[i]

    # Gradient w.r.t. W_k :  dScore * outer(h_k, t_k) * r_k  (rank-1 x Cr)
    let kBase = k * params.wStrideK
    for i in 0 ..< cfg.Ce:
      for j in 0 ..< cfg.Ce:
        let ijBase = kBase + i * params.wStrideCe1 + j * params.wStrideCe2
        let hiti  = dScore * hk.data[i] * tk.data[j]
        for c in 0 ..< cfg.Cr:
          grads.W.data[ijBase + c] += hiti * rk.data[c]

    # Gradient w.r.t. r_k :  dScore * sum_{i,j} W_k[i,j,:] h_k[i] t_k[j]
    let relRowOff = relId * (cfg.K * cfg.Cr) + k * cfg.Cr
    for i in 0 ..< cfg.Ce:
      for j in 0 ..< cfg.Ce:
        let ijBase = kBase + i * params.wStrideCe1 + j * params.wStrideCe2
        let hiti   = dScore * hk.data[i] * tk.data[j]
        for c in 0 ..< cfg.Cr:
          grads.relationEmb.data[relRowOff + c] += hiti * params.W.data[ijBase + c]

# ---------------------------------------------------------------------------
# Orthogonality loss gradient  (approximate: finite differences or analytic)
# Here we use the analytic gradient of  ||M^T M - I||_F^2  w.r.t. M,
# then back-propagate through M to W and r.
# ---------------------------------------------------------------------------

proc accumulateOrthoGrad*(params: MEIMParams; cfg: MEIMConfig;
                           relId: int; grads: var GradBuffer) =
  ## Add gradient of L_ortho for `relId` into `grads`.
  if cfg.lambdaOrtho == 0.0'f32 and cfg.lambdaUnitNorm == 0.0'f32: return

  for k in 0 ..< cfg.K:
    let rk = getRelationPartition(params, relId, k, cfg.Cr)
    let Mk = computeMappingMatrix(params, rk, k, cfg.Ce, cfg.Cr)

    if cfg.lambdaOrtho != 0.0'f32:
      # d/dM ||M^T M - I||_F^2 = 4 * M * (M^T M - I)
      let MtM  = matmul(transpose(Mk), Mk)
      let Id   = eye(cfg.Ce)
      let diff = MtM - Id                          # Ce x Ce
      let dM   = matmul(Mk, diff) * (4.0'f32 * cfg.lambdaOrtho)  # Ce x Ce

      # Back-prop dM to W and r.
      let kBase = k * params.wStrideK
      let relRowOff = relId * (cfg.K * cfg.Cr) + k * cfg.Cr
      for i in 0 ..< cfg.Ce:
        for j in 0 ..< cfg.Ce:
          let dMij  = dM.data[i * cfg.Ce + j]
          let ijBase = kBase + i * params.wStrideCe1 + j * params.wStrideCe2
          for c in 0 ..< cfg.Cr:
            # dL/dW[k,i,j,c] = dM[i,j] * r_k[c]
            grads.W.data[ijBase + c] += dMij * rk.data[c]
            # dL/dr_k[c] += dM[i,j] * W[k,i,j,c]
            grads.relationEmb.data[relRowOff + c] += dMij * params.W.data[ijBase + c]

    if cfg.lambdaUnitNorm != 0.0'f32:
      # d/dr_k |r_k^T r_k - 1|^p  =  p * sign(dev) * |dev|^{p-1} * 2 r_k
      let normSq = sumSq(rk)
      let dev    = normSq - 1.0'f32
      let absDev = abs(dev)
      if absDev > 1e-12'f32:
        let sign   = if dev > 0.0'f32: 1.0'f32 else: -1.0'f32
        let dCoeff = cfg.lambdaUnitNorm *
                     sign * cfg.orthoP.float32 *
                     pow(absDev.float64, (cfg.orthoP - 1).float64).float32 *
                     2.0'f32
        let relRowOff = relId * (cfg.K * cfg.Cr) + k * cfg.Cr
        for c in 0 ..< cfg.Cr:
          grads.relationEmb.data[relRowOff + c] += dCoeff * rk.data[c]

# ---------------------------------------------------------------------------
# Adam optimiser (per-parameter first and second moment estimates)
# ---------------------------------------------------------------------------

type
  AdamState* = object
    m*: Tensor   ## First moment
    v*: Tensor   ## Second moment (uncentred variance)

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
    vE*:  AdamState
    mR*:  AdamState
    vR*:  AdamState
    mW*:  AdamState
    vW*:  AdamState

proc newOptimiser*(params: MEIMParams; lr: float32 = 3e-3'f32): Optimiser =
  Optimiser(
    lr: lr, beta1: 0.9'f32, beta2: 0.999'f32, epsilon: 1e-8'f32, step: 0,
    mE: newAdamState(params.entityEmb),
    vE: newAdamState(params.entityEmb),
    mR: newAdamState(params.relationEmb),
    vR: newAdamState(params.relationEmb),
    mW: newAdamState(params.W),
    vW: newAdamState(params.W),
  )

proc adamUpdate(param: var Tensor; grad: Tensor;
                m, v: var AdamState;
                lr, beta1, beta2, eps: float32; step: int64) =
  ## Update `param` in-place using Adam rule.
  let b1t = 1.0'f32 - pow(beta1.float64, step.float64).float32
  let b2t = 1.0'f32 - pow(beta2.float64, step.float64).float32
  let lrt = lr * sqrt(b2t) / b1t
  for i in 0 ..< param.data.len:
    let g = grad.data[i]
    m.m.data[i] = beta1 * m.m.data[i] + (1.0'f32 - beta1) * g
    v.v.data[i] = beta2 * v.v.data[i] + (1.0'f32 - beta2) * g * g
    param.data[i] -= lrt * m.m.data[i] / (sqrt(v.v.data[i]) + eps)

proc adamStep*(opt: var Optimiser; params: var MEIMParams; grads: GradBuffer) =
  inc opt.step
  let s = opt.step
  adamUpdate(params.entityEmb,   grads.entityEmb,   opt.mE, opt.vE, opt.lr, opt.beta1, opt.beta2, opt.epsilon, s)
  adamUpdate(params.relationEmb, grads.relationEmb, opt.mR, opt.vR, opt.lr, opt.beta1, opt.beta2, opt.epsilon, s)
  adamUpdate(params.W,           grads.W,           opt.mW, opt.vW, opt.lr, opt.beta1, opt.beta2, opt.epsilon, s)

proc decayLR*(opt: var Optimiser; factor: float32) =
  opt.lr *= factor
