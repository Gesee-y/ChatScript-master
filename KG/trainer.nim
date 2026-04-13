## trainer.nim
## ============
## Training loop and evaluation metrics for MEIM.
##
## Loss: softmax cross-entropy (1-vs-all or k-vs-all) + soft orthogonality.
##
## Evaluation: filtered MRR, H@1, H@3, H@10 following the standard
## knowledge graph embedding protocol (Bordes et al., 2013).

import math, random, sequtils, strformat, tables, algorithm, times
import tensor, kg_loader, meim_model

# ---------------------------------------------------------------------------
# 1-vs-all softmax cross-entropy loss + gradients
# ---------------------------------------------------------------------------

type
  BatchLoss* = object
    loss*:     float32
    ortho*:    float32

proc trainBatch1vsAll*(params: var MEIMParams; cfg: MEIMConfig;
                        grads: var GradBuffer;
                        batch: seq[Triple]): BatchLoss =
  ## Process one mini-batch using 1-vs-all softmax cross-entropy.
  ##
  ## For each triple (h, t, r):
  ##   - Compute scores for all entities as candidate tail  ->  softmax  ->  CE
  ##   - Compute scores for all entities as candidate head  ->  softmax  ->  CE
  ##
  ## Gradient of CE loss w.r.t. score s_i:
  ##   dL/ds_i = p_i - y_i   (predicted prob minus one-hot label)

  zeroGrads(grads)
  var totalLoss   = 0.0'f32
  var totalOrtho  = 0.0'f32

  for triple in batch:
    let h = triple.head
    let t = triple.tail
    let r = triple.rel

    # --- Tail prediction (h, r) -> ? ---
    let tailScores = scoreAllTails(params, cfg, h, r)
    let tailProbs  = softmax(tailScores)
    # CE loss:  -log p[t]
    totalLoss -= ln(max(tailProbs[t], 1e-10'f32))
    # Gradient:  p_i - y_i
    for tId in 0 ..< cfg.numEntities:
      let dScore = tailProbs[tId] - (if tId == t: 1.0'f32 else: 0.0'f32)
      if abs(dScore) > 1e-7'f32:
        accumulateScoreGrad(params, cfg, h, tId, r, dScore / batch.len.float32, grads)

    # --- Head prediction ? <- (r, t) ---
    let headScores = scoreAllHeads(params, cfg, t, r)
    let headProbs  = softmax(headScores)
    totalLoss -= ln(max(headProbs[h], 1e-10'f32))
    for hId in 0 ..< cfg.numEntities:
      let dScore = headProbs[hId] - (if hId == h: 1.0'f32 else: 0.0'f32)
      if abs(dScore) > 1e-7'f32:
        accumulateScoreGrad(params, cfg, hId, t, r, dScore / batch.len.float32, grads)

    # --- Orthogonality loss & its gradient ---
    let ortho = computeOrthoLoss(params, cfg, r)
    totalOrtho += ortho
    accumulateOrthoGrad(params, cfg, r, grads)

  result.loss  = totalLoss  / batch.len.float32
  result.ortho = totalOrtho / batch.len.float32

# ---------------------------------------------------------------------------
# k-vs-all sampling  (negative sampling)
# ---------------------------------------------------------------------------

proc trainBatchKvsAll*(params: var MEIMParams; cfg: MEIMConfig;
                        grads: var GradBuffer;
                        batch: seq[Triple]): BatchLoss =
  ## k-vs-all: for each positive triple, sample k negative entities
  ## (for both tail and head), then apply softmax CE over (1 + k) candidates.
  zeroGrads(grads)
  var totalLoss  = 0.0'f32
  var totalOrtho = 0.0'f32

  for triple in batch:
    let h = triple.head; let t = triple.tail; let r = triple.rel

    # Sample k negative tails.
    var candTails = @[t]
    for _ in 0 ..< cfg.kVsAllK:
      candTails.add rand(cfg.numEntities - 1)
    # Compute scores for candidates.
    var tailCandScores: seq[float32]
    for tId in candTails:
      tailCandScores.add scoreTriple(params, cfg, h, tId, r)
    let tailProbs = softmax(tailCandScores)
    totalLoss -= ln(max(tailProbs[0], 1e-10'f32))   # position 0 = positive
    for ci, tId in candTails:
      let dScore = tailProbs[ci] - (if ci == 0: 1.0'f32 else: 0.0'f32)
      if abs(dScore) > 1e-7'f32:
        accumulateScoreGrad(params, cfg, h, tId, r, dScore / batch.len.float32, grads)

    # Sample k negative heads.
    var candHeads = @[h]
    for _ in 0 ..< cfg.kVsAllK:
      candHeads.add rand(cfg.numEntities - 1)
    var headCandScores: seq[float32]
    for hId in candHeads:
      headCandScores.add scoreTriple(params, cfg, hId, t, r)
    let headProbs = softmax(headCandScores)
    totalLoss -= ln(max(headProbs[0], 1e-10'f32))
    for ci, hId in candHeads:
      let dScore = headProbs[ci] - (if ci == 0: 1.0'f32 else: 0.0'f32)
      if abs(dScore) > 1e-7'f32:
        accumulateScoreGrad(params, cfg, hId, t, r, dScore / batch.len.float32, grads)

    let ortho = computeOrthoLoss(params, cfg, r)
    totalOrtho += ortho
    accumulateOrthoGrad(params, cfg, r, grads)

  result.loss  = totalLoss  / batch.len.float32
  result.ortho = totalOrtho / batch.len.float32

# ---------------------------------------------------------------------------
# Ranking evaluation (filtered MRR, H@1, H@3, H@10)
# ---------------------------------------------------------------------------

type
  EvalMetrics* = object
    mrr*:   float64
    hits1*: float64
    hits3*: float64
    hits10*: float64
    numTriples*: int

proc `$`*(m: EvalMetrics): string =
  &"MRR={m.mrr:.4f}  H@1={m.hits1:.4f}  H@3={m.hits3:.4f}  H@10={m.hits10:.4f}  (n={m.numTriples})"

proc filteredRank(scores: seq[float32]; trueIdx: int;
                  filterSet: seq[int]): int =
  ## Compute the filtered rank of `trueIdx`.
  ## All entities in `filterSet` (other known true answers) get their score
  ## reduced to -inf before ranking.
  let trueScore = scores[trueIdx]
  var rank = 1
  for i, s in scores:
    if i == trueIdx: continue
    if i in filterSet: continue   # filter out other true positives
    if s > trueScore: inc rank
  rank

proc evaluate*(params: MEIMParams; cfg: MEIMConfig;
               split: KGSplit;
               hrToTails: Table[(int,int), seq[int]];
               trToHeads: Table[(int,int), seq[int]];
               maxTriples: int = high(int)): EvalMetrics =
  ## Evaluate link prediction on `split`.
  ## Uses filtered metrics: other known true triples are excluded from ranking.
  var sumRR   = 0.0
  var h1, h3, h10 = 0
  var n = 0

  let nEval = min(split.triples.len, maxTriples)
  for i in 0 ..< nEval:
    let triple = split.triples[i]
    let h = triple.head; let t = triple.tail; let r = triple.rel

    # -- Tail prediction --
    let tailScores = scoreAllTails(params, cfg, h, r)
    let filterTails = hrToTails.getOrDefault((h, r), @[])
    let tailRank   = filteredRank(tailScores, t, filterTails)
    sumRR += 1.0 / tailRank.float64
    if tailRank == 1:  inc h1
    if tailRank <= 3:  inc h3
    if tailRank <= 10: inc h10
    inc n

    # -- Head prediction --
    let headScores = scoreAllHeads(params, cfg, t, r)
    let filterHeads = trToHeads.getOrDefault((t, r), @[])
    let headRank    = filteredRank(headScores, h, filterHeads)
    sumRR += 1.0 / headRank.float64
    if headRank == 1:  inc h1
    if headRank <= 3:  inc h3
    if headRank <= 10: inc h10
    inc n

  let nf = n.float64
  EvalMetrics(
    mrr:       sumRR / nf,
    hits1:     h1.float64 / nf,
    hits3:     h3.float64 / nf,
    hits10:    h10.float64 / nf,
    numTriples: n,
  )

# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

proc shuffle[T](s: var seq[T]) =
  for i in countdown(s.high, 1):
    let j = rand(i)
    swap(s[i], s[j])

proc train*(params: var MEIMParams; cfg: MEIMConfig; kg: KGDataset;
            verbose: bool = true): EvalMetrics =
  ## Full training procedure.
  ## Returns best validation metrics.

  let (hrToTails, trToHeads) = buildFilterIndex(kg)
  var opt   = newOptimiser(params, cfg.learningRate)
  var grads = newGradBuffer(cfg)

  var triples = kg.train.triples
  var bestMRR = 0.0
  var bestMetrics: EvalMetrics

  for epoch in 1 .. cfg.maxEpochs:
    let t0 = cpuTime()
    shuffle(triples)

    var epochLoss  = 0.0'f32
    var epochOrtho = 0.0'f32
    var nBatches   = 0

    var i = 0
    while i < triples.len:
      let bEnd  = min(i + cfg.batchSize, triples.len)
      let batch = triples[i ..< bEnd]

      let bl = if cfg.kVsAll:
        trainBatchKvsAll(params, cfg, grads, batch)
      else:
        trainBatch1vsAll(params, cfg, grads, batch)

      opt.adamStep(params, grads)
      epochLoss  += bl.loss
      epochOrtho += bl.ortho
      inc nBatches
      i = bEnd

    # Learning rate decay
    if cfg.lrDecay < 1.0'f32:
      opt.decayLR(cfg.lrDecay)

    let elapsed = cpuTime() - t0
    if verbose:
      echo &"[epoch {epoch:4d}/{cfg.maxEpochs}]  " &
           &"loss={epochLoss/nBatches.float32:.4f}  " &
           &"ortho={epochOrtho/nBatches.float32:.4f}  " &
           &"t={elapsed:.1f}s"

    # Validation evaluation
    if epoch mod cfg.evalEvery == 0 and kg.valid.triples.len > 0:
      let m = evaluate(params, cfg, kg.valid, hrToTails, trToHeads,
                       maxTriples = 500)   # cap for speed during training
      if verbose:
        echo &"  [valid] {m}"
      if m.mrr > bestMRR:
        bestMRR     = m.mrr
        bestMetrics = m

  result = bestMetrics

# ---------------------------------------------------------------------------
# Prediction API (top-K entities for a given (head, rel) or (tail, rel))
# ---------------------------------------------------------------------------

type
  Prediction* = object
    entityId*:   int
    entityName*: string
    score*:      float32

proc predictTails*(params: MEIMParams; cfg: MEIMConfig; kg: KGDataset;
                   headName, relName: string; topK: int = 10): seq[Prediction] =
  ## Predict the top-K tail entities for a given (head, relation) pair.
  if headName notin kg.entityToId:
    echo &"Unknown entity: {headName}"; return
  if relName  notin kg.relationToId:
    echo &"Unknown relation: {relName}"; return

  let headId = kg.entityToId[headName]
  let relId  = kg.relationToId[relName]
  let scores = scoreAllTails(params, cfg, headId, relId)

  # Rank by score descending.
  var ranked = toSeq(0 ..< cfg.numEntities)
  ranked.sort(proc(a, b: int): int = cmp(scores[b], scores[a]))

  for i in 0 ..< min(topK, ranked.len):
    let eid = ranked[i]
    result.add Prediction(entityId: eid,
                          entityName: kg.idToEntity[eid],
                          score: scores[eid])

proc predictHeads*(params: MEIMParams; cfg: MEIMConfig; kg: KGDataset;
                   tailName, relName: string; topK: int = 10): seq[Prediction] =
  ## Predict the top-K head entities for a given (relation, tail) pair.
  if tailName notin kg.entityToId:
    echo &"Unknown entity: {tailName}"; return
  if relName  notin kg.relationToId:
    echo &"Unknown relation: {relName}"; return

  let tailId = kg.entityToId[tailName]
  let relId  = kg.relationToId[relName]
  let scores = scoreAllHeads(params, cfg, tailId, relId)

  var ranked = toSeq(0 ..< cfg.numEntities)
  ranked.sort(proc(a, b: int): int = cmp(scores[b], scores[a]))

  for i in 0 ..< min(topK, ranked.len):
    let eid = ranked[i]
    result.add Prediction(entityId: eid,
                          entityName: kg.idToEntity[eid],
                          score: scores[eid])

proc scoreTripleByName*(params: MEIMParams; cfg: MEIMConfig; kg: KGDataset;
                         headName, tailName, relName: string): float32 =
  ## Score a single (head, tail, relation) triple given string names.
  if headName notin kg.entityToId: echo &"Unknown entity: {headName}"; return NaN
  if tailName notin kg.entityToId: echo &"Unknown entity: {tailName}"; return NaN
  if relName  notin kg.relationToId: echo &"Unknown relation: {relName}"; return NaN
  scoreTriple(params, cfg,
              kg.entityToId[headName],
              kg.entityToId[tailName],
              kg.relationToId[relName])
