import std/[tables, strutils, sets, deques, math, algorithm, sequtils, parseutils, os]
import kg_loader, meim_model, trainer

## Knowledge Graph — KGZS-SC Enhanced  (accuracy-safe edition)
##
## DESIGN INVARIANT
## ────────────────
## Exact (asserted) triplets and inferred (generalised) triplets live in
## completely separate stores.  A query on asserted facts NEVER touches the
## inference cache, so accuracy on known relations is always 100%.
##
## Every public proc that may generalise returns a `QueryResult` that carries
## a `wasInferred: bool` flag so the caller always knows the provenance of
## each answer.
##
## Inference cache
## ───────────────
## Results from `searchTailBFS`, `searchHeadBFS`, `zeroShotTail` and
## `zeroShotHead` are stored in `kg.inferredTriplets` (a separate table).
## They are NEVER written back to `kg.nodes[*].outgoing / incoming`.
## Subsequent calls consult the cache first, so repeated generalisation is
## O(1) after the first call.
##
## Improvements from KGZS-SC paper (Zhang et al., 2025)
## ─────────────────────────────────────────────────────
##  • Semantic embeddings per node (GloVe-compatible).
##  • GCN-style neighbourhood aggregation (`propagateEmbeddings`).
##  • Cosine-similarity-weighted BFS with pruning threshold.
##  • Zero-shot nearest-neighbour inference (`zeroShotTail/Head`).
##  • Hubness mitigation via L2-normalised scores before ranking.

# ───────────────────────────────────────────────────────────────────────────
# Types
# ───────────────────────────────────────────────────────────────────────────

type
  Embedding* = seq[float]

  ResultKind* = enum
    rkExact    ## Fact is explicitly asserted in the graph.
    rkCached   ## Fact was previously inferred and is now served from cache.
    rkInferred ## Fact is freshly inferred by generalisation (BFS or ZSL).

  QueryResult* = object
    ## Unified result type for every query proc.
    name*: string
    score*: float      ## 1.0 for exact hits; cosine sim for inferred.
    kind*: ResultKind
    wasInferred*: bool ## True iff kind != rkExact.

  KnowledgeNode* = object
    id*: int
    name*: string
    embedding*: Embedding
    outgoing*: Table[int, seq[int]] ## asserted only
    incoming*: Table[int, seq[int]] ## asserted only

  Triplet* = tuple[head: string, relation: string, tail: string]

  GeneralizationPolicy* = object
    allowedDown*: HashSet[int]
    allAllowedDown*: bool
    allowedUp*: HashSet[int]
    allAllowedUp*: bool

  ScoredResult* = object
    name*: string
    score*: float

  # Key for the inference cache: (headId, relId, tailId)
  InferredKey = tuple[head: int, rel: int, tail: int]

  KnowledgeGraph* = ref object
    nodes*: seq[KnowledgeNode]
    nodeMap*: Table[string, int]
    relMap*: Table[string, int]
    nextRelId*: int
    relPolicies*: Table[int, GeneralizationPolicy]
    embedDim*: int
    ## Inference cache — keyed by (headId, relId, tailId), value = score.
    ## Never merged into the asserted graph.
    inferredTriplets*: Table[InferredKey, float]
    ## Robustness: Sets of node IDs seen as head or tail for a given relation.
    relAllowedTails*: Table[int, HashSet[int]]
    relAllowedHeads*: Table[int, HashSet[int]]
    ## Taxonomic Schemas: (relId) -> (headerAncestorId, tailAncestorId)
    relSchemas*: Table[int, tuple[headType: int, tailType: int]]

# ───────────────────────────────────────────────────────────────────────────
# Construction
# ───────────────────────────────────────────────────────────────────────────

proc newKnowledgeGraph*(embedDim: int = 0): KnowledgeGraph =
  result = KnowledgeGraph(
    nodes: newSeq[KnowledgeNode](),
    nodeMap: initTable[string, int](),
    relMap: initTable[string, int](),
    nextRelId: 0,
    relPolicies: initTable[int, GeneralizationPolicy](),
    embedDim: embedDim,
    inferredTriplets: initTable[InferredKey, float](),
    relAllowedTails: initTable[int, HashSet[int]](),
    relAllowedHeads: initTable[int, HashSet[int]](),
    relSchemas: initTable[int, tuple[headType: int, tailType: int]]()
  )

proc getOrAddNode*(kg: KnowledgeGraph, name: string): int =
  if kg.nodeMap.hasKey(name): return kg.nodeMap[name]
  let newId = kg.nodes.len
  var node = KnowledgeNode(
    id: newId,
    name: name,
    outgoing: initTable[int, seq[int]](),
    incoming: initTable[int, seq[int]]()
  )
  if kg.embedDim > 0:
    node.embedding = newSeq[float](kg.embedDim)
  kg.nodes.add(node)
  kg.nodeMap[name] = newId
  return newId

proc getOrAddRelation*(kg: KnowledgeGraph, name: string): int =
  if kg.relMap.hasKey(name): return kg.relMap[name]
  let newRelId = kg.nextRelId
  kg.relMap[name] = newRelId
  inc kg.nextRelId
  return newRelId

# ───────────────────────────────────────────────────────────────────────────
# Generalisation policies
# ───────────────────────────────────────────────────────────────────────────

proc getPolicy(kg: KnowledgeGraph, relId: int): ptr GeneralizationPolicy =
  if not kg.relPolicies.hasKey(relId):
    kg.relPolicies[relId] = GeneralizationPolicy(
      allowedDown: initHashSet[int](),
      allAllowedDown: true,
      allowedUp: initHashSet[int](),
      allAllowedUp: true
    )
  return addr(kg.relPolicies[relId])

proc allowAllDown*(kg: KnowledgeGraph, targetRel: string) =
  let p = kg.getPolicy(kg.getOrAddRelation(targetRel))
  p.allAllowedDown = true; p.allowedDown.clear()

proc allowAllUp*(kg: KnowledgeGraph, targetRel: string) =
  let p = kg.getPolicy(kg.getOrAddRelation(targetRel))
  p.allAllowedUp = true; p.allowedUp.clear()

proc disallowAllDown*(kg: KnowledgeGraph, targetRel: string) =
  let p = kg.getPolicy(kg.getOrAddRelation(targetRel))
  p.allAllowedDown = false; p.allowedDown.clear()

proc disallowAllUp*(kg: KnowledgeGraph, targetRel: string) =
  let p = kg.getPolicy(kg.getOrAddRelation(targetRel))
  p.allAllowedUp = false; p.allowedUp.clear()

proc addRelevantDown*(kg: KnowledgeGraph, targetRel, relevantRel: string) =
  let tId = kg.getOrAddRelation(targetRel)
  let rId = kg.getOrAddRelation(relevantRel)
  let p = kg.getPolicy(tId)
  p.allAllowedDown = false; p.allowedDown.incl(rId)

proc addRelevantUp*(kg: KnowledgeGraph, targetRel, relevantRel: string) =
  let tId = kg.getOrAddRelation(targetRel)
  let rId = kg.getOrAddRelation(relevantRel)
  let p = kg.getPolicy(tId)
  p.allAllowedUp = false; p.allowedUp.incl(rId)

proc isDownAllowed(kg: KnowledgeGraph, targetRelId, currentRelId: int): bool =
  if not kg.relPolicies.hasKey(targetRelId): return true
  let p = addr(kg.relPolicies[targetRelId])
  return p.allAllowedDown or currentRelId in p.allowedDown

proc isUpAllowed(kg: KnowledgeGraph, targetRelId, currentRelId: int): bool =
  if not kg.relPolicies.hasKey(targetRelId): return true
  let p = addr(kg.relPolicies[targetRelId])
  return p.allAllowedUp or currentRelId in p.allowedUp

proc setSchema*(kg: KnowledgeGraph, relation, headTypeNode, tailTypeNode: string) =
  ## Defines taxonomic constraints for a relation.
  let rId = kg.getOrAddRelation(relation)
  let hId = kg.getOrAddNode(headTypeNode)
  let tId = kg.getOrAddNode(tailTypeNode)
  kg.relSchemas[rId] = (headType: hId, tailType: tId)

proc isA*(kg: KnowledgeGraph, nodeId, ancestorId: int): bool =
  ## Returns true if node has ancestorId via 'is_a' relation (BFS up).
  if nodeId == ancestorId: return true
  let isARelId = kg.relMap.getOrDefault("is_a", -1)
  if isARelId == -1: return false
  
  var queue = initDeque[int]()
  queue.addLast(nodeId)
  var visited = initHashSet[int]()
  visited.incl(nodeId)
  
  while queue.len > 0:
    let curr = queue.popFirst()
    if curr == ancestorId: return true
    
    # Remonter via 'is_a'
    # Dans notre KG, 'is_a' est stocké soit en sortant (A is_a B) soit entrant.
    # On veut remonter : A -> is_a -> B. Donc on suit 'outgoing[is_a]'.
    if kg.nodes[curr].outgoing.hasKey(isARelId):
      for parent in kg.nodes[curr].outgoing[isARelId]:
        if parent notin visited:
          visited.incl(parent)
          queue.addLast(parent)
  return false

# ───────────────────────────────────────────────────────────────────────────
# Data ingestion  (writes only to the asserted graph)
# ───────────────────────────────────────────────────────────────────────────

proc addTriplet*(kg: KnowledgeGraph, head, relation, tail: string) =
  ## Asserts a fact.  ONLY call for ground-truth triplets.
  ## Inferred results are stored via `cacheInference` (internal).
  let headId = kg.getOrAddNode(head)
  let tailId = kg.getOrAddNode(tail)
  let relId = kg.getOrAddRelation(relation)

  if not kg.nodes[headId].outgoing.hasKey(relId):
    kg.nodes[headId].outgoing[relId] = @[]
  if tailId notin kg.nodes[headId].outgoing[relId]:
    kg.nodes[headId].outgoing[relId].add(tailId)

  if not kg.nodes[tailId].incoming.hasKey(relId):
    kg.nodes[tailId].incoming[relId] = @[]
  if headId notin kg.nodes[tailId].incoming[relId]:
    kg.nodes[tailId].incoming[relId].add(headId)

  # Robustness: record observed head/tail patterns for type constraint enforcement
  kg.relAllowedTails.mgetOrPut(relId, initHashSet[int]()).incl(tailId)
  kg.relAllowedHeads.mgetOrPut(relId, initHashSet[int]()).incl(headId)

proc loadTriplets*(kg: KnowledgeGraph, triplets: openArray[Triplet]) =
  for t in triplets: kg.addTriplet(t.head, t.relation, t.tail)

proc loadFromFile*(kg: KnowledgeGraph, path: string, delimiter: char = '\t') =
  for line in lines(path):
    let parts = line.split(delimiter)
    if parts.len >= 3:
      kg.addTriplet(parts[0].strip(), parts[1].strip(), parts[2].strip())

# ───────────────────────────────────────────────────────────────────────────
# Inference cache helpers  (internal — never touches asserted graph)
# ───────────────────────────────────────────────────────────────────────────

proc cacheInference(kg: KnowledgeGraph,
                    headId, relId, tailId: int, score: float) =
  kg.inferredTriplets[(headId, relId, tailId)] = score

proc isCached(kg: KnowledgeGraph,
              headId, relId, tailId: int): bool =
  kg.inferredTriplets.hasKey((headId, relId, tailId))

proc cachedScore(kg: KnowledgeGraph,
                 headId, relId, tailId: int): float =
  kg.inferredTriplets.getOrDefault((headId, relId, tailId), 0.0)

proc clearInferenceCache*(kg: KnowledgeGraph) =
  ## Wipes all inferred results.  Asserted facts are untouched.
  kg.inferredTriplets.clear()

# ───────────────────────────────────────────────────────────────────────────
# Embedding API
# ───────────────────────────────────────────────────────────────────────────

proc setEmbedding*(kg: KnowledgeGraph, name: string, emb: Embedding) =
  let id = kg.getOrAddNode(name)
  if kg.embedDim == 0: kg.embedDim = emb.len
  kg.nodes[id].embedding = emb

proc loadEmbeddingsFromFile*(kg: KnowledgeGraph, path: string,
                              delimiter: char = '\t') =
  for line in lines(path):
    let parts = line.split(delimiter)
    if parts.len < 2: continue
    var emb = newSeq[float](parts.len - 1)
    for i in 1 ..< parts.len: emb[i-1] = parseFloat(parts[i].strip())
    kg.setEmbedding(parts[0].strip(), emb)

proc getEmbeddingsSeq*(kg: KnowledgeGraph): seq[seq[float]] =
  ## Returns embeddings of all nodes in a ID-indexed sequence.
  result = newSeq[seq[float]](kg.nodes.len)
  for i in 0 ..< kg.nodes.len:
    result[i] = kg.nodes[i].embedding

proc loadGloveEmbeddings*(kg: KnowledgeGraph, path: string) =
  ## Optimized GloVe loader for 800MB+ files.
  ## Minimizes allocations by only parsing relevant lines.
  if kg.nodeMap.len == 0: return

  # Create a reverse lookup for matching (lowercase node name -> list of node IDs)
  var lookup = initTable[string, seq[int]]()
  for name, id in kg.nodeMap:
    let key = name.toLowerAscii()
    if not lookup.hasKey(key): lookup[key] = @[]
    lookup[key].add(id)

  var f: File
  if not open(f, path): return
  defer: close(f)

  var line = ""
  while f.readLine(line):
    if line.len == 0: continue
    
    # Fast-scan for the first space to extract the word
    let firstSpace = line.find(' ')
    if firstSpace <= 0: continue
    
    # Check if the word is relevant before doing any heavy parsing
    let wordLower = line[0 ..< firstSpace].toLowerAscii()
    
    if lookup.hasKey(wordLower):
      let ids = lookup[wordLower]
      
      # Determine dimension if we don't know it yet
      if kg.embedDim == 0:
        var count = 0
        var i = firstSpace
        while i < line.len:
          # skip multiple spaces if any
          while i < line.len and line[i] == ' ': inc i
          if i < line.len:
            inc count
            # move to the end of the float
            while i < line.len and line[i] != ' ': inc i
        kg.embedDim = count

      if kg.embedDim == 0: continue
      
      var emb = newSeq[float](kg.embedDim)
      var pos = firstSpace
      var d = 0
      while d < kg.embedDim and pos < line.len:
        # Skip spaces
        while pos < line.len and line[pos] == ' ': inc pos
        if pos >= line.len: break
        
        var val: float
        let consumed = parseFloat(line, val, pos)
        if consumed == 0: break
        emb[d] = val
        pos += consumed
        inc d
      
      # Only assign if we successfully parsed exactly the expected number of dimensions
      if d == kg.embedDim:
        for id in ids:
          kg.nodes[id].embedding = emb

# ───────────────────────────────────────────────────────────────────────────
# Embedding math
# ───────────────────────────────────────────────────────────────────────────

proc l2Norm(v: Embedding): float =
  for x in v: result += x * x
  result = sqrt(result)

proc cosineSim*(a, b: Embedding): float =
  if a.len == 0 or b.len == 0 or a.len != b.len: return 0.0
  var dot = 0.0
  for i in 0 ..< a.len: dot += a[i] * b[i]
  let na = l2Norm(a); let nb = l2Norm(b)
  if na == 0.0 or nb == 0.0: return 0.0
  return dot / (na * nb)

proc propagateEmbeddings*(kg: KnowledgeGraph, layers: int = 2,
                           residual: bool = true) =
  ## GCN-style neighbourhood aggregation (eq. 4-5 from the paper).
  ## Operates on asserted edges only; never reads the inference cache.
  if kg.embedDim == 0: return
  for _ in 0 ..< layers:
    var newEmbs = newSeq[Embedding](kg.nodes.len)
    for i in 0 ..< kg.nodes.len:
      let node = kg.nodes[i]
      if node.embedding.len == 0:
        newEmbs[i] = newSeq[float](kg.embedDim); continue
      var agg = node.embedding
      var count = 1
      for _, targets in node.outgoing:
        for tId in targets:
          if kg.nodes[tId].embedding.len == kg.embedDim:
            for d in 0 ..< kg.embedDim: agg[d] += kg.nodes[tId].embedding[d]
            inc count
      for _, sources in node.incoming:
        for sId in sources:
          if kg.nodes[sId].embedding.len == kg.embedDim:
            for d in 0 ..< kg.embedDim: agg[d] += kg.nodes[sId].embedding[d]
            inc count
      for d in 0 ..< kg.embedDim: agg[d] /= float(count)
      if residual:
        for d in 0 ..< kg.embedDim: agg[d] = (agg[d] + node.embedding[d]) / 2.0
      newEmbs[i] = agg
    for i in 0 ..< kg.nodes.len: kg.nodes[i].embedding = newEmbs[i]

# ───────────────────────────────────────────────────────────────────────────
# Exact queries  — 100% accuracy guaranteed, never touches inference cache
# ───────────────────────────────────────────────────────────────────────────

proc queryTail*(kg: KnowledgeGraph, head, relation: string): seq[QueryResult] =
  ## Returns ONLY asserted tails.  `wasInferred` is always false.
  result = @[]
  if not kg.nodeMap.hasKey(head) or not kg.relMap.hasKey(relation): return
  let headId = kg.nodeMap[head]; let relId = kg.relMap[relation]
  if kg.nodes[headId].outgoing.hasKey(relId):
    for tId in kg.nodes[headId].outgoing[relId]:
      result.add(QueryResult(name: kg.nodes[tId].name,
                              score: 1.0,
                              kind: rkExact,
                              wasInferred: false))

proc queryHead*(kg: KnowledgeGraph, relation, tail: string): seq[QueryResult] =
  ## Returns ONLY asserted heads.  `wasInferred` is always false.
  result = @[]
  if not kg.relMap.hasKey(relation) or not kg.nodeMap.hasKey(tail): return
  let tailId = kg.nodeMap[tail]; let relId = kg.relMap[relation]
  if kg.nodes[tailId].incoming.hasKey(relId):
    for sId in kg.nodes[tailId].incoming[relId]:
      result.add(QueryResult(name: kg.nodes[sId].name,
                              score: 1.0,
                              kind: rkExact,
                              wasInferred: false))

# Convenience: plain string lists for callers that don't need provenance.
proc queryTailNames*(kg: KnowledgeGraph, head, relation: string): seq[string] =
  kg.queryTail(head, relation).mapIt(it.name)

proc queryHeadNames*(kg: KnowledgeGraph, relation, tail: string): seq[string] =
  kg.queryHead(relation, tail).mapIt(it.name)

# ───────────────────────────────────────────────────────────────────────────
# Generalised queries  — cache-first, clearly flagged
# ───────────────────────────────────────────────────────────────────────────

proc searchTailBFS*(kg: KnowledgeGraph, head, relation: string,
                    maxDepth: int = 5,
                    simThreshold: float = 0.0): seq[QueryResult] =
  ## Structural generalisation via BFS on asserted edges only.
  ##
  ## • Returns rkExact results first (direct hits found along traversal).
  ## • Returns rkCached for previously inferred results (O(1) lookup).
  ## • Returns rkInferred for fresh inferences, which are then cached.
  ## • NEVER writes to the asserted graph.
  ## • Cosine-similarity ordering when embeddings are available.
  result = @[]
  if not kg.nodeMap.hasKey(head) or not kg.relMap.hasKey(relation): return

  let startId = kg.nodeMap[head]
  let targetRelId = kg.relMap[relation]
  let headEmb = kg.nodes[startId].embedding
  let useEmb = kg.embedDim > 0 and headEmb.len == kg.embedDim

  # ── Check cache first ────────────────────────────────────────────────────
  var cachedResults: seq[QueryResult]
  for i in 0 ..< kg.nodes.len:
    if kg.isCached(startId, targetRelId, i):
      cachedResults.add(QueryResult(
        name: kg.nodes[i].name,
        score: kg.cachedScore(startId, targetRelId, i),
        kind: rkCached,
        wasInferred: true))
  if cachedResults.len > 0:
    return cachedResults

  # ── BFS on asserted edges ────────────────────────────────────────────────
  var queue = initDeque[(int, int)]()
  queue.addLast((startId, 0))
  var visited = initHashSet[int]()
  visited.incl(startId)
  var foundIds = initHashSet[int]()

  # Schema Check: Expected tail ancestor
  let tailAncestor = if kg.relSchemas.hasKey(targetRelId): kg.relSchemas[targetRelId].tailType else: -1

  while queue.len > 0:
    let (currId, depth) = queue.popFirst()
    if kg.nodes[currId].outgoing.hasKey(targetRelId):
      for tId in kg.nodes[currId].outgoing[targetRelId]:
        # Enforce Schema
        if tailAncestor == -1 or kg.isA(tId, tailAncestor):
          foundIds.incl(tId)
    if depth < maxDepth:
      var candidates: seq[(int, float)]
      for relId, targets in kg.nodes[currId].outgoing:
        if kg.isDownAllowed(targetRelId, relId):
          for tId in targets:
            if tId notin visited:
              var sim = 1.0
              if useEmb:
                sim = cosineSim(headEmb, kg.nodes[tId].embedding)
                if sim < simThreshold: continue
              candidates.add((tId, sim))
      if useEmb:
        candidates.sort(proc(a, b: (int, float)): int =
          if a[1] > b[1]: -1 elif a[1] < b[1]: 1 else: 0)
      for (tId, _) in candidates:
        if tId notin visited:
          visited.incl(tId)
          queue.addLast((tId, depth + 1))

  # ── Classify results and populate cache ─────────────────────────────────
  for id in foundIds:
    # Is this an asserted direct link from `head`?
    let isAsserted = kg.nodes[startId].outgoing.hasKey(targetRelId) and
                     id in kg.nodes[startId].outgoing[targetRelId]
    let score = if useEmb: cosineSim(headEmb, kg.nodes[id].embedding) else: 1.0
    let kind = if isAsserted: rkExact else: rkInferred

    if kind == rkInferred:
      kg.cacheInference(startId, targetRelId, id, score)

    result.add(QueryResult(name: kg.nodes[id].name,
                            score: score,
                            kind: kind,
                            wasInferred: kind != rkExact))

  result.sort(proc(a, b: QueryResult): int =
    if a.kind == rkExact and b.kind != rkExact: return -1
    if a.kind != rkExact and b.kind == rkExact: return 1
    if a.score > b.score: -1 elif a.score < b.score: 1 else: 0)

proc searchHeadBFS*(kg: KnowledgeGraph, relation, tail: string,
                    maxDepth: int = 3,
                    simThreshold: float = 0.0): seq[QueryResult] =
  ## Symmetric inductive BFS.  Same cache + provenance guarantees.
  result = @[]
  if not kg.relMap.hasKey(relation) or not kg.nodeMap.hasKey(tail): return

  let startId = kg.nodeMap[tail]
  let targetRelId = kg.relMap[relation]
  let tailEmb = kg.nodes[startId].embedding
  let useEmb = kg.embedDim > 0 and tailEmb.len == kg.embedDim

  # ── Check cache first ────────────────────────────────────────────────────
  var cachedResults: seq[QueryResult]
  for i in 0 ..< kg.nodes.len:
    if kg.isCached(i, targetRelId, startId):
      cachedResults.add(QueryResult(
        name: kg.nodes[i].name,
        score: kg.cachedScore(i, targetRelId, startId),
        kind: rkCached,
        wasInferred: true))
  if cachedResults.len > 0:
    return cachedResults

  # ── BFS on asserted edges ────────────────────────────────────────────────
  var queue = initDeque[(int, int)]()
  queue.addLast((startId, 0))
  var visited = initHashSet[int]()
  visited.incl(startId)
  var foundIds = initHashSet[int]()

  # Schema Check: Expected head ancestor
  let headAncestor = if kg.relSchemas.hasKey(targetRelId): kg.relSchemas[targetRelId].headType else: -1

  while queue.len > 0:
    let (currId, depth) = queue.popFirst()
    if kg.nodes[currId].incoming.hasKey(targetRelId):
      for sId in kg.nodes[currId].incoming[targetRelId]:
        # Enforce Schema
        if headAncestor == -1 or kg.isA(sId, headAncestor):
          foundIds.incl(sId)
    if depth < maxDepth:
      var candidates: seq[(int, float)]
      for relId, sources in kg.nodes[currId].incoming:
        if kg.isUpAllowed(targetRelId, relId):
          for sId in sources:
            if sId notin visited:
              var sim = 1.0
              if useEmb:
                sim = cosineSim(tailEmb, kg.nodes[sId].embedding)
                if sim < simThreshold: continue
              candidates.add((sId, sim))
      if useEmb:
        candidates.sort(proc(a, b: (int, float)): int =
          if a[1] > b[1]: -1 elif a[1] < b[1]: 1 else: 0)
      for (sId, _) in candidates:
        if sId notin visited:
          visited.incl(sId)
          queue.addLast((sId, depth + 1))

  # ── Classify and cache ───────────────────────────────────────────────────
  for id in foundIds:
    let isAsserted = kg.nodes[startId].incoming.hasKey(targetRelId) and
                     id in kg.nodes[startId].incoming[targetRelId]
    let score = if useEmb: cosineSim(tailEmb, kg.nodes[id].embedding) else: 1.0
    let kind = if isAsserted: rkExact else: rkInferred

    if kind == rkInferred:
      kg.cacheInference(id, targetRelId, startId, score)

    result.add(QueryResult(name: kg.nodes[id].name,
                            score: score,
                            kind: kind,
                            wasInferred: kind != rkExact))

  result.sort(proc(a, b: QueryResult): int =
    if a.kind == rkExact and b.kind != rkExact: return -1
    if a.kind != rkExact and b.kind == rkExact: return 1
    if a.score > b.score: -1 elif a.score < b.score: 1 else: 0)

# ───────────────────────────────────────────────────────────────────────────
# Zero-shot inference  (embedding nearest-neighbour, eq. 10 in paper)
# ───────────────────────────────────────────────────────────────────────────

proc zeroShotTail*(kg: KnowledgeGraph, head, relation: string,
                   topK: int = 5): seq[QueryResult] =
  ## Pure embedding-based inference for (head, relation, ?).
  ## Results are always flagged `wasInferred = true`.
  ## Cache-first: a repeated call is O(nodes) lookup only.
  result = @[]
  if kg.embedDim == 0 or not kg.nodeMap.hasKey(head): return
  let headId = kg.nodeMap[head]
  let relId = kg.getOrAddRelation(relation)
  let headEmb = kg.nodes[headId].embedding
  if headEmb.len == 0: return

  # Cache check
  var cached: seq[QueryResult]
  for i in 0 ..< kg.nodes.len:
    if kg.isCached(headId, relId, i):
      cached.add(QueryResult(name: kg.nodes[i].name,
                              score: kg.cachedScore(headId, relId, i),
                              kind: rkCached,
                              wasInferred: true))
  if cached.len > 0:
    cached.sort(proc(a, b: QueryResult): int =
      if a.score > b.score: -1 elif a.score < b.score: 1 else: 0)
    return cached[0 ..< min(topK, cached.len)]

  # Fresh inference: Link Prediction based on neighbors
  # Goal: Find nodes S similar to `head`, then return what S links to via `relation`.
  var tailScores = initTable[int, float]()
  for sId in 0 ..< kg.nodes.len:
    if sId == headId: continue
    let sim = cosineSim(headEmb, kg.nodes[sId].embedding)
    if sim > 0.5: # Similarity threshold for neighborhood
      # Check grounded outgoing edges for relation from this similar node S
      if kg.nodes[sId].outgoing.hasKey(relId):
        for tId in kg.nodes[sId].outgoing[relId]:
          # Score is the max similarity to any neighbor that points to this tail
          tailScores[tId] = max(tailScores.getOrDefault(tId, 0.0), sim)

  var scored: seq[(int, float)]
  for tId, score in tailScores:
    scored.add((tId, score))

  scored.sort(proc(a, b: (int, float)): int =
    if a[1] > b[1]: -1 elif a[1] < b[1]: 1 else: 0)

  let k = min(topK, scored.len)
  for i in 0 ..< k:
    let (tId, score) = scored[i]
    kg.cacheInference(headId, relId, tId, score)
    result.add(QueryResult(name: kg.nodes[tId].name,
                            score: score,
                            kind: rkInferred,
                            wasInferred: true))

proc zeroShotHead*(kg: KnowledgeGraph, relation, tail: string,
                   topK: int = 5): seq[QueryResult] =
  ## Symmetric zero-shot head inference.
  result = @[]
  if kg.embedDim == 0 or not kg.nodeMap.hasKey(tail): return
  let tailId = kg.nodeMap[tail]
  let relId = kg.getOrAddRelation(relation)
  let tailEmb = kg.nodes[tailId].embedding
  if tailEmb.len == 0: return

  var cached: seq[QueryResult]
  for i in 0 ..< kg.nodes.len:
    if kg.isCached(i, relId, tailId):
      cached.add(QueryResult(name: kg.nodes[i].name,
                              score: kg.cachedScore(i, relId, tailId),
                              kind: rkCached,
                              wasInferred: true))
  if cached.len > 0:
    cached.sort(proc(a, b: QueryResult): int =
      if a.score > b.score: -1 elif a.score < b.score: 1 else: 0)
    return cached[0 ..< min(topK, cached.len)]

  var scored: seq[(int, float)]
  for i in 0 ..< kg.nodes.len:
    if i == tailId: continue
    let sim = cosineSim(tailEmb, kg.nodes[i].embedding)
    if sim > 0.0: scored.add((i, sim))

  scored.sort(proc(a, b: (int, float)): int =
    if a[1] > b[1]: -1 elif a[1] < b[1]: 1 else: 0)

  let k = min(topK, scored.len)
  for i in 0 ..< k:
    let (hId, sim) = scored[i]
    kg.cacheInference(hId, relId, tailId, sim)
    result.add(QueryResult(name: kg.nodes[hId].name,
                            score: sim,
                            kind: rkInferred,
                            wasInferred: true))

# ───────────────────────────────────────────────────────────────────────────
# Utility
# ───────────────────────────────────────────────────────────────────────────

proc topSimilarNodes*(kg: KnowledgeGraph, name: string,
                      k: int = 5): seq[ScoredResult] =
  result = @[]
  if not kg.nodeMap.hasKey(name) or kg.embedDim == 0: return
  let qId = kg.nodeMap[name]
  let qEmb = kg.nodes[qId].embedding
  if qEmb.len == 0: return
  var scored: seq[ScoredResult]
  for i in 0 ..< kg.nodes.len:
    if i == qId: continue
    scored.add(ScoredResult(name: kg.nodes[i].name,
                             score: cosineSim(qEmb, kg.nodes[i].embedding)))
  scored.sort(proc(a, b: ScoredResult): int =
    if a.score > b.score: -1 elif a.score < b.score: 1 else: 0)
  return scored[0 ..< min(k, scored.len)]

# ───────────────────────────────────────────────────────────────────────────
# Tests
# ───────────────────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────────────────
# Fusion API: KG + MEIM (Neuro-Symbolic)
# ───────────────────────────────────────────────────────────────────────────

type
  HybridEngine* = object
    ## Combines Symbolic KG and Neural MEIM.
    kg*:           KnowledgeGraph
    meimParams*:   MEIMParams
    meimConfig*:   MEIMConfig
    dataset*:      KGDataset  ## Shared vocabulary mappings
    useNeural*:    bool       ## Enable/disable neural fallback

proc newHybridEngine*(kg: KnowledgeGraph): HybridEngine =
  result.kg = kg
  result.useNeural = false

proc syncMappings*(engine: var HybridEngine) =
  ## Rebuilds the dataset vocabulary from the symbolic graph nodes/relations.
  var eToId = initTable[string, int]()
  var rToId = initTable[string, int]()
  var idToE: seq[string]
  var idToR: seq[string]
  
  for name, id in engine.kg.nodeMap:
    eToId[name] = id
    if id >= idToE.len: idToE.setLen(id + 1)
    idToE[id] = name
    
  for name, id in engine.kg.relMap:
    rToId[name] = id
    if id >= idToR.len: idToR.setLen(id + 1)
    idToR[id] = name
    
  engine.dataset.entityToId = eToId
  engine.dataset.relationToId = rToId
  engine.dataset.idToEntity = idToE
  engine.dataset.idToRelation = idToR
  
  # Update config sizes
  engine.meimConfig.numEntities = idToE.len
  engine.meimConfig.numRelations = idToR.len

proc queryHybridTail*(engine: var HybridEngine, 
                      head, relation: string, 
                      threshold: float = 0.6,
                      robust: bool = true): seq[QueryResult] =
  ## Cascade query logic: Exact -> BFS -> ZSL -> MEIM.
  ## If 'robust' is true, candidates are filtered by observed semantic types.
  ## Taxonomic guardrail: If a schema exists for 'relation', results MUST match.

  let rId = engine.kg.relMap.getOrDefault(relation, -1)
  let hId = engine.kg.nodeMap.getOrDefault(head, -1)
  
  # Schema Check for Head
  if rId != -1 and hId != -1 and engine.kg.relSchemas.hasKey(rId):
    let schema = engine.kg.relSchemas[rId]
    if not engine.kg.isA(hId, schema.headType):
      # Head doesn't match schema (e.g., trying to treat with a Disease node)
      return @[]

  # 1. Exact query
  result = engine.kg.queryTail(head, relation)
  if result.len > 0: return

  # 2. BFS Inference
  result = engine.kg.searchTailBFS(head, relation)
  if result.len > 0:
    let maxScore = result.mapIt(it.score).foldl(max(a, b), 0.0)
    if maxScore >= threshold: return

  # 3. zero-shot Inference
  let allowedTails = if rId != -1: engine.kg.relAllowedTails.getOrDefault(rId, initHashSet[int]()) else: initHashSet[int]()
  let tailAncestor = if rId != -1 and engine.kg.relSchemas.hasKey(rId): engine.kg.relSchemas[rId].tailType else: -1

  let zsl = engine.kg.zeroShotTail(head, relation)
  var filteredZsl: seq[QueryResult] = @[]
  for res in zsl:
    let tId = engine.kg.nodeMap.getOrDefault(res.name, -1)
    if tId == -1: continue
    
    # Check Schema
    if tailAncestor != -1 and not engine.kg.isA(tId, tailAncestor): continue
    
    # Check Robust Filter (Historical usage)
    if not robust or allowedTails.len == 0 or tId in allowedTails:
      filteredZsl.add res

  if filteredZsl.len > 0:
    let maxScore = filteredZsl.mapIt(it.score).foldl(max(a, b), 0.0)
    if maxScore >= threshold: 
      result = filteredZsl
      return

  # 4. Neural Fallback (MEIM)
  if engine.useNeural and engine.meimParams.entityEmb.data.len > 0:
    let pred = predictTails(engine.meimParams, engine.meimConfig, engine.dataset, 
                            head, relation, topK = 30)
    for p in pred:
      let tId = engine.kg.nodeMap.getOrDefault(p.entityName, -1)
      if tId == -1: continue
      
      # Check Schema
      if tailAncestor != -1 and not engine.kg.isA(tId, tailAncestor): continue
      
      # Check Robust Filter
      if not robust or allowedTails.len == 0 or tId in allowedTails:
          result.add QueryResult(name: p.entityName, score: p.score.float, kind: rkInferred, wasInferred: true)
          if result.len >= 10: break

proc queryHybridHead*(engine: var HybridEngine, 
                      relation, tail: string, 
                      threshold: float = 0.6,
                      robust: bool = true): seq[QueryResult] =
  ## Cascade query logic: Exact -> BFS -> ZSL -> MEIM.
  
  let rId = engine.kg.relMap.getOrDefault(relation, -1)
  let tId = engine.kg.nodeMap.getOrDefault(tail, -1)

  # Schema Check for Tail
  if rId != -1 and tId != -1 and engine.kg.relSchemas.hasKey(rId):
    let schema = engine.kg.relSchemas[rId]
    if not engine.kg.isA(tId, schema.tailType):
      return @[]

  # 1. Exact query
  result = engine.kg.queryHead(relation, tail)
  if result.len > 0: return
  
  # 2. BFS Inference
  result = engine.kg.searchHeadBFS(relation, tail)
  if result.len > 0:
    let maxScore = result.mapIt(it.score).foldl(max(a, b), 0.0)
    if maxScore >= threshold: return
    
  # 3. Zero-Shot Inference
  let allowedHeads = if rId != -1: engine.kg.relAllowedHeads.getOrDefault(rId, initHashSet[int]()) else: initHashSet[int]()
  let headAncestor = if rId != -1 and engine.kg.relSchemas.hasKey(rId): engine.kg.relSchemas[rId].headType else: -1

  let zsl = engine.kg.zeroShotHead(relation, tail)
  var filteredZsl: seq[QueryResult] = @[]
  for res in zsl:
    let hId = engine.kg.nodeMap.getOrDefault(res.name, -1)
    if hId == -1: continue

    # Check Schema
    if headAncestor != -1 and not engine.kg.isA(hId, headAncestor): continue

    if not robust or allowedHeads.len == 0 or hId in allowedHeads:
      filteredZsl.add res

  if filteredZsl.len > 0:
    let maxScore = filteredZsl.mapIt(it.score).foldl(max(a, b), 0.0)
    if maxScore >= threshold: 
      result = filteredZsl
      return

  # 4. Neural Fallback (MEIM)
  if engine.useNeural and engine.meimParams.entityEmb.data.len > 0:
    let pred = predictHeads(engine.meimParams, engine.meimConfig, engine.dataset, 
                            tail, relation, topK = 30)
    for p in pred:
      let hId = engine.kg.nodeMap.getOrDefault(p.entityName, -1)
      if hId == -1: continue

      # Check Schema
      if headAncestor != -1 and not engine.kg.isA(hId, headAncestor): continue

      if not robust or allowedHeads.len == 0 or hId in allowedHeads:
          result.add QueryResult(name: p.entityName, score: p.score.float, kind: rkInferred, wasInferred: true)
          if result.len >= 10: break

proc queryCombinedTails*(engine: var HybridEngine,
                         heads: seq[string], 
                         relation: string): seq[QueryResult] =
  ## Combined interrogation: Interrogate multiple heads (e.g., 2 symptoms)
  ## and find the best candidate tail (disease) by intersecting results.
  if heads.len == 0: return
  
  var candidateScores = initTable[string, float]()
  var candidateCounts = initTable[string, int]()
  
  for h in heads:
    let res = engine.queryHybridTail(h, relation, robust = true)
    for r in res:
      candidateScores[r.name] = candidateScores.getOrDefault(r.name, 0.0) + r.score
      candidateCounts[r.name] = candidateCounts.getOrDefault(r.name, 0) + 1
      
  # Intersection + mean score
  for name, count in candidateCounts:
    if count == heads.len:
      result.add QueryResult(
        name: name,
        score: candidateScores[name] / heads.len.float,
        kind: rkInferred,
        wasInferred: true
      )
      
  result.sort(proc(a, b: QueryResult): int = cmp(b.score, a.score))

proc queryCombinedHeads*(engine: var HybridEngine,
                          relation: string,
                          tails: seq[string]): seq[QueryResult] =
  ## Interrogate multiple tails (symptoms) and find the best candidate head (disease).
  if tails.len == 0: return
  
  var candidateScores = initTable[string, float]()
  var candidateCounts = initTable[string, int]()
  
  for t in tails:
    let res = engine.queryHybridHead(relation, t, robust = true)
    for r in res:
      candidateScores[r.name] = candidateScores.getOrDefault(r.name, 0.0) + r.score
      candidateCounts[r.name] = candidateCounts.getOrDefault(r.name, 0) + 1
      
  for name, count in candidateCounts:
    if count == tails.len:
      result.add QueryResult(
        name: name,
        score: candidateScores[name] / tails.len.float,
        kind: rkInferred,
        wasInferred: true
      )
      
  result.sort(proc(a, b: QueryResult): int = cmp(b.score, a.score))

when isMainModule:
  import std/unittest

  let embAspirin = @[1.0, 0.0, 0.0, 0.0]
  let embMedicA = @[0.9, 0.1, 0.0, 0.0]
  let embIbuprofen = @[0.6, 0.4, 0.0, 0.0]
  let embParacea = @[0.5, 0.5, 0.0, 0.0]
  let embMedicine = @[0.5, 0.5, 0.1, 0.0]
  let embHeadache = @[0.0, 0.0, 1.0, 0.0]
  let embFever = @[0.0, 0.0, 0.8, 0.2]
  let embPillForm = @[0.8, 0.2, 0.0, 0.0]

  suite "Knowledge Graph — Accuracy-Safe KGZS-SC":

    setup:
      var kg = newKnowledgeGraph(embedDim = 4)
      kg.loadTriplets([
        ("Aspirin", "treats", "Headache"),
        ("Aspirin", "treats", "Fever"),
        ("Ibuprofen", "treats", "Headache"),
        ("Paracetamol", "is_a", "Medicine"),
        ("Paracetamol", "treats", "Fever"),
        ("MedicA", "is_a", "Aspirin"),
        ("PillForm", "is_shape_of", "Aspirin")
      ])
      kg.setEmbedding("Aspirin", embAspirin)
      kg.setEmbedding("MedicA", embMedicA)
      kg.setEmbedding("Ibuprofen", embIbuprofen)
      kg.setEmbedding("Paracetamol", embParacea)
      kg.setEmbedding("Medicine", embMedicine)
      kg.setEmbedding("Headache", embHeadache)
      kg.setEmbedding("Fever", embFever)
      kg.setEmbedding("PillForm", embPillForm)

    test "queryTail — always rkExact, wasInferred = false":
      let res = kg.queryTail("Aspirin", "treats")
      check res.len == 2
      for r in res:
        check r.kind == rkExact
        check r.wasInferred == false
        check r.score == 1.0

    test "queryHead — always rkExact, wasInferred = false":
      let res = kg.queryHead("treats", "Headache")
      check res.len == 2
      for r in res:
        check r.kind == rkExact
        check r.wasInferred == false

    test "BFS inference does not write to asserted graph":
      discard kg.searchTailBFS("MedicA", "treats", maxDepth = 2)
      let exact = kg.queryTail("MedicA", "treats")
      check exact.len == 0

    test "zeroShot inference does not write to asserted graph":
      discard kg.zeroShotTail("MedicA", "treats", topK = 3)
      let exact = kg.queryTail("MedicA", "treats")
      check exact.len == 0

    test "searchTailBFS — inferred results flagged wasInferred = true":
      let res = kg.searchTailBFS("MedicA", "treats", maxDepth = 2)
      let inferred = res.filterIt(it.wasInferred)
      check inferred.len > 0
      check "Headache" in inferred.mapIt(it.name) or
            "Fever" in inferred.mapIt(it.name)

    test "zeroShotTail — all results flagged wasInferred = true":
      let res = kg.zeroShotTail("MedicA", "treats", topK = 3)
      for r in res:
        check r.wasInferred == true
        check r.kind in {rkInferred, rkCached}

    test "searchTailBFS second call returns rkCached":
      discard kg.searchTailBFS("MedicA", "treats", maxDepth = 2)
      let res2 = kg.searchTailBFS("MedicA", "treats", maxDepth = 2)
      check res2.len > 0
      for r in res2:
        check r.kind == rkCached

    test "zeroShotTail second call returns rkCached":
      discard kg.zeroShotTail("MedicA", "treats", topK = 3)
      let res2 = kg.zeroShotTail("MedicA", "treats", topK = 3)
      for r in res2:
        check r.kind == rkCached

    test "clearInferenceCache — exact queries unaffected":
      discard kg.searchTailBFS("MedicA", "treats", maxDepth = 2)
      kg.clearInferenceCache()
      let exact = kg.queryTail("Aspirin", "treats")
      check exact.len == 2
      check exact[0].kind == rkExact
      let res = kg.searchTailBFS("MedicA", "treats", maxDepth = 2)
      check res.filterIt(it.kind == rkCached).len == 0

    test "rkExact results sort before rkInferred in mixed BFS":
      kg.addTriplet("MedicA", "treats", "Fever")
      let res = kg.searchTailBFS("MedicA", "treats", maxDepth = 2)
      if res.len > 1:
        check res[0].kind == rkExact

    test "Deduction / Induction policies still work":
      var kgPol = newKnowledgeGraph()
      kgPol.addTriplet("A", "part_of", "B")
      kgPol.addTriplet("B", "is_a", "C")
      kgPol.disallowAllDown("is_a")
      check kgPol.searchTailBFS("A", "is_a").len == 0
      kgPol.addRelevantDown("is_a", "part_of")
      check "C" in kgPol.searchTailBFS("A", "is_a").mapIt(it.name)
      kgPol.disallowAllUp("part_of")
      check kgPol.searchHeadBFS("part_of", "C").len == 0
      kgPol.addRelevantUp("part_of", "is_a")
      check "A" in kgPol.searchHeadBFS("part_of", "C").mapIt(it.name)

    test "propagateEmbeddings — asserted graph structure unchanged":
      let countBefore = kg.nodes[kg.nodeMap["Aspirin"]].outgoing.len
      kg.propagateEmbeddings(layers = 1)
      let countAfter = kg.nodes[kg.nodeMap["Aspirin"]].outgoing.len
      check countBefore == countAfter

    test "loadGloveEmbeddings — selective and case-insensitive":
      let glovePath = "test_glove.txt"
      # medicA is in KG (though capitalized differently), caffeine is NOT.
      let content = [
        "aspirin 1.0 0.0 0.0 0.0",
        "medica 0.9 0.1 0.0 0.0",
        "caffeine 0.5 0.5 0.5 0.5"
      ].join("\n")
      writeFile(glovePath, content)
      
      var kgG = newKnowledgeGraph()
      kgG.addTriplet("Aspirin", "treats", "Pain")
      kgG.addTriplet("MedicA", "is_a", "Aspirin")
      
      kgG.loadGloveEmbeddings(glovePath)
      
      check kgG.embedDim == 4
      check kgG.nodes[kgG.nodeMap["Aspirin"]].embedding[0] == 1.0
      check kgG.nodes[kgG.nodeMap["MedicA"]].embedding[1] == 0.1
      # "Pain" was not in GloVe, so it should be empty or default
      check kgG.nodes[kgG.nodeMap["Pain"]].embedding.allIt(it == 0.0)
      # Graph should NOT have caffeine
      check "caffeine" notin kgG.nodeMap
      
      removeFile(glovePath)

    test "HybridEngine cascade — Exact -> BFS":
      var kgH = newKnowledgeGraph()
      kgH.addTriplet("A", "is_a", "B")
      kgH.addTriplet("B", "is_a", "C")
      
      var engine = newHybridEngine(kgH)
      engine.syncMappings()
      
      # 1. Exact
      let resExact = engine.queryHybridTail("A", "is_a")
      check resExact.len == 1
      check resExact[0].name == "B"
      check resExact[0].kind == rkExact
      
      # 2. BFS Fallback
      let resBFS = engine.queryHybridTail("A", "target", threshold = 0.5) 
      # "target" doesn't exist, so it should be empty
      check resBFS.len == 0
      
      # Now add a path for BFS
      kgH.addTriplet("A", "part_of", "B")
      kgH.addTriplet("B", "treats", "D")
      kgH.addRelevantDown("treats", "part_of")
      let resBFS2 = engine.queryHybridTail("A", "treats")
      check resBFS2.len > 0
      check "D" in resBFS2.mapIt(it.name)
      check resBFS2[0].kind == rkInferred
      check resBFS2[0].wasInferred == true

    test "queryCombinedTails — Symptom intersection logic":
      var kgC = newKnowledgeGraph()
      kgC.loadTriplets([
        ("S1", "symptom_of", "D1"),
        ("S1", "symptom_of", "D2"),
        ("S2", "symptom_of", "D2"),
        ("S2", "symptom_of", "D3")
      ])
      var engine = newHybridEngine(kgC)
      engine.syncMappings()
      
      # Querying only S1
      let res1 = engine.queryCombinedTails(@["S1"], "symptom_of")
      check res1.len == 2 # D1, D2
      
      # Querying S1 AND S2 -> Should intersect on D2
      let resBoth = engine.queryCombinedTails(@["S1", "S2"], "symptom_of")
      check resBoth.len == 1
      check resBoth[0].name == "D2"
