import std/[os, strutils, strformat, random, sequtils, tables, sets]
import kg, kg_loader, meim_model, trainer

type
  ValidationMap* = Table[(string, string), HashSet[string]]

proc parseFlag*(args: seq[string]; flag: string; default: string): string =
  for i, a in args:
    if a == flag and i + 1 < args.len:
      return args[i + 1]
  default

proc parseIntFlag*(args: seq[string]; flag: string; default: int): int =
  let s = parseFlag(args, flag, "")
  if s.len == 0: default else: parseInt(s)

proc parseFloatFlag*(args: seq[string]; flag: string; default: float32): float32 =
  let s = parseFlag(args, flag, "")
  if s.len == 0: default else: parseFloat(s).float32

proc hasFlag*(args: seq[string]; flag: string): bool =
  for a in args:
    if a == flag:
      return true
  false

proc trainingLoadConfig*(): LoadConfig =
  result = defaultConfig()
  result.delimiter = ','
  result.headCol = 0
  result.relCol = 1
  result.tailCol = 2
  result.hasHeader = false

proc loadValidationData*(path: string): ValidationMap =
  result = initTable[(string, string), HashSet[string]]()
  if path.len == 0 or not fileExists(path):
    return

  for rawLine in readFile(path).splitLines():
    let line = rawLine.strip()
    if line.len == 0 or line.startsWith("#"):
      continue

    let parts = line.split(',')
    if parts.len < 5:
      continue
    if parts[0].strip().toLowerAscii() == "question_id":
      continue

    let head = parts[1].strip()
    let relation = parts[2].strip()
    let tails = parts[3].strip()
    if head.len == 0 or relation.len == 0 or tails.len == 0:
      continue

    let key = (head, relation)
    if key notin result:
      result[key] = initHashSet[string]()
    for tail in tails.split('|'):
      let cleanTail = tail.strip()
      if cleanTail.len > 0:
        result[key].incl(cleanTail)

proc configFromArgs*(args: seq[string]; numEntities, numRelations: int): MEIMConfig =
  result = defaultConfig(numEntities, numRelations)
  result.K = parseIntFlag(args, "--K", result.K)
  result.Ce = parseIntFlag(args, "--Ce", result.Ce)
  result.Cr = parseIntFlag(args, "--Cr", result.Cr)
  result.learningRate = parseFloatFlag(args, "--lr", result.learningRate)
  result.maxEpochs = parseIntFlag(args, "--epochs", result.maxEpochs)
  result.batchSize = parseIntFlag(args, "--batch", result.batchSize)
  result.lambdaOrtho = parseFloatFlag(args, "--ortho", result.lambdaOrtho)
  result.lambdaUnitNorm = parseFloatFlag(args, "--unorm", result.lambdaUnitNorm)
  result.inputDropRate = parseFloatFlag(args, "--idrop", result.inputDropRate)
  result.hiddenDropRate = parseFloatFlag(args, "--hdrop", result.hiddenDropRate)
  result.evalEvery = parseIntFlag(args, "--eval", result.evalEvery)
  result.kVsAll = hasFlag(args, "--kvsall")
  result.kVsAllK = parseIntFlag(args, "--kk", result.kVsAllK)
  result.lrDecay = parseFloatFlag(args, "--decay", result.lrDecay)

proc printConfig*(cfg: MEIMConfig) =
  echo ""
  echo "=== Training Configuration ==="
  echo &"  Entities:        {cfg.numEntities}"
  echo &"  Relations:       {cfg.numRelations}"
  echo &"  Partitions K:    {cfg.K}"
  echo &"  Ce:              {cfg.Ce}"
  echo &"  Cr:              {cfg.Cr}"
  echo &"  Learning rate:   {cfg.learningRate}"
  echo &"  LR decay:        {cfg.lrDecay}"
  echo &"  Batch size:      {cfg.batchSize}"
  echo &"  Epochs:          {cfg.maxEpochs}"
  echo &"  Eval every:      {cfg.evalEvery}"
  echo &"  lambda_ortho:    {cfg.lambdaOrtho}"
  echo &"  lambda_unitnorm: {cfg.lambdaUnitNorm}"
  echo &"  k-vs-all:        {cfg.kVsAll}  k={cfg.kVsAllK}"
  echo ""

proc buildSymbolicKG*(dataset: KGDataset; params: MEIMParams): KnowledgeGraph =
  let embedDim = if params.entityEmb.shape.len >= 2: params.entityEmb.shape[1] else: 0
  result = newKnowledgeGraph(embedDim = embedDim)

  for triple in dataset.train.triples:
    let head = dataset.idToEntity[triple.head]
    let relation = dataset.idToRelation[triple.rel]
    let tail = dataset.idToEntity[triple.tail]
    result.addTriplet(head, relation, tail)

  if embedDim > 0:
    for entityId, entityName in dataset.idToEntity:
      var embedding = newSeq[float](embedDim)
      let rowOffset = entityId * embedDim
      for j in 0 ..< embedDim:
        embedding[j] = params.entityEmb.data[rowOffset + j].float
      result.setEmbedding(entityName, embedding)

proc refineKnowledge*(kgInstance: KnowledgeGraph;
                      params: MEIMParams;
                      cfg: MEIMConfig;
                      dataset: KGDataset;
                      validMap: ValidationMap;
                      topK: int): int =
  var refinedCount = 0
  echo "\n[Refinement] Checking held-out answers against model predictions..."

  for key, expectedAnswers in validMap.pairs():
    let (headName, relName) = key
    let predictions = params.predictTails(cfg, dataset, headName, relName, topK = topK)
    let predictedNames = predictions.mapIt(it.entityName).toHashSet()

    var missing: seq[string]
    for expected in expectedAnswers:
      if expected notin predictedNames:
        kgInstance.addTriplet(headName, relName, expected)
        missing.add(expected)

    if missing.len > 0:
      inc refinedCount, missing.len
      echo &"  [+] Injected {missing.len} answer(s) for ({headName}, {relName})"

  echo &"[Refinement] Added {refinedCount} held-out triples to symbolic KG."
  refinedCount

proc printHelp*() =
  echo """
Training.nim - Train MEIM on KG data and save the resulting hybrid engine.

Usage:
  nim r Training.nim -- --train DATA/train_instance_1_train_pruned.txt [options]

Options:
  --train <path>        Training triples in head,relation,tail format.
  --valid <path>        Optional validation triples file in head,relation,tail format.
  --test <path>         Optional test triples file in head,relation,tail format.
  --answers <path>      Optional answers CSV generated from held-out questions.
  --outdir <dir>        Output directory for the saved hybrid engine.
  --seed <int>          Random seed (default 42).
  --refine-topk <int>   Top-K predictions inspected before injecting held-out answers.
  --skip-refine         Do not add held-out answers back into the symbolic KG.

Model options:
  --K --Ce --Cr --lr --epochs --batch --ortho --unorm --idrop --hdrop
  --decay --eval --kvsall --kk
"""

proc runTrainingPipeline*(args: seq[string]) =
  let trainPath = parseFlag(args, "--train", "")
  if trainPath.len == 0:
    echo "Error: --train <path> is required."
    quit(1)

  let validPath = parseFlag(args, "--valid", "")
  let testPath = parseFlag(args, "--test", "")
  let answersPath = parseFlag(args, "--answers", "")
  let explicitOutDir = parseFlag(args, "--outdir", "")
  let outDir = if explicitOutDir.len > 0: explicitOutDir else: "artifacts" / splitFile(trainPath).name
  let seed = parseIntFlag(args, "--seed", 42)
  let refineTopK = parseIntFlag(args, "--refine-topk", 20)
  let doRefine = not hasFlag(args, "--skip-refine")

  echo "[System] Loading dataset..."
  let loadCfg = trainingLoadConfig()
  let dataset = loadKGFromFiles(trainPath, validPath, testPath, loadCfg)
  if dataset.train.triples.len == 0:
    echo "Error: training split is empty."
    quit(1)

  var cfg = configFromArgs(args, dataset.numEntities, dataset.numRelations)
  printConfig(cfg)

  randomize(seed)
  echo &"[System] Random seed: {seed}"
  var params = initParams(cfg)

  echo "[Training] Starting MEIM optimization..."
  let bestValid = train(params, cfg, dataset, verbose = true)

  if dataset.valid.triples.len > 0:
    echo ""
    echo "=== Best Validation Result ==="
    echo $bestValid

  if dataset.test.triples.len > 0:
    let (hrToTails, trToHeads) = buildFilterIndex(dataset)
    let testMetrics = evaluate(params, cfg, dataset.test, hrToTails, trToHeads)
    echo ""
    echo "=== Test Evaluation ==="
    echo $testMetrics

  echo "\n[System] Building symbolic KG from training triples..."
  var symbolicKG = buildSymbolicKG(dataset, params)

  if doRefine and answersPath.len > 0:
    let validMap = loadValidationData(answersPath)
    discard refineKnowledge(symbolicKG, params, cfg, dataset, validMap, refineTopK)
  elif answersPath.len > 0:
    echo "[Refinement] Skipped by --skip-refine."
  else:
    echo "[Refinement] No --answers file supplied; symbolic KG contains the training graph only."

  var engine = newHybridEngine(symbolicKG)
  engine.meimParams = params
  engine.meimConfig = cfg
  engine.dataset = dataset
  engine.useNeural = true
  engine.syncMappings()

  echo &"\n[System] Saving hybrid engine to {outDir} ..."
  saveHybridEngine(engine, outDir, includeCache = false)
  echo &"[Done] Saved KG + embeddings + MEIM weights to {outDir}"

when isMainModule:
  let args = commandLineParams()
  if args.len == 0 or args[0] in ["-h", "--help", "help"]:
    printHelp()
    quit(0)
  runTrainingPipeline(args)
