## meim.nim
## =========
## Main entry point for the MEIM knowledge-graph link-prediction system.
##
## Usage
## -----
##   ./meim <command> [options]
##
## Commands:
##   train   --data <path> [options]   Train from a KG directory or single file.
##   eval    --data <path> [options]   Evaluate a trained model (not yet persisted).
##   predict --data <path> --head <entity> --rel <relation> [--topk 10]
##   demo                              Run on a tiny built-in toy KG.
##
## Options:
##   --K     <int>    Number of partitions                  (default: 3)
##   --Ce    <int>    Entity embedding size per partition    (default: 100)
##   --Cr    <int>    Relation embedding size per partition  (default: 100)
##   --lr    <float>  Learning rate                          (default: 3e-3)
##   --epochs<int>    Training epochs                        (default: 1000)
##   --batch <int>    Batch size                             (default: 1024)
##   --ortho <float>  Lambda for orthogonality loss          (default: 0.1)
##   --idrop <float>  Input dropout rate                     (default: 0.0)
##   --hdrop <float>  Hidden dropout rate                    (default: 0.0)
##   --eval  <int>    Evaluate every N epochs                (default: 10)
##   --topk  <int>    Top-K results for prediction           (default: 10)
##   --head  <str>    Head entity name (for predict command)
##   --rel   <str>    Relation name    (for predict command)
##   --tail  <str>    Tail entity name (for predict command)

import os, strutils, strformat, random, math, sequtils, times, tables
import tensor, kg_loader, meim_model, trainer

# ---------------------------------------------------------------------------
# Tiny built-in demo KG (nationality facts)
# ---------------------------------------------------------------------------

const demoTriples = [
  ("albert_einstein",    "nationality",     "germany"),
  ("albert_einstein",    "field",           "physics"),
  ("marie_curie",        "nationality",     "poland"),
  ("marie_curie",        "field",           "chemistry"),
  ("niels_bohr",         "nationality",     "denmark"),
  ("niels_bohr",         "field",           "physics"),
  ("max_planck",         "nationality",     "germany"),
  ("max_planck",         "field",           "physics"),
  ("lise_meitner",       "nationality",     "austria"),
  ("lise_meitner",       "field",           "physics"),
  ("richard_feynman",    "nationality",     "usa"),
  ("richard_feynman",    "field",           "physics"),
  ("werner_heisenberg",  "nationality",     "germany"),
  ("werner_heisenberg",  "field",           "physics"),
  ("erwin_schrodinger",  "nationality",     "austria"),
  ("erwin_schrodinger",  "field",           "physics"),
  ("enrico_fermi",       "nationality",     "italy"),
  ("enrico_fermi",       "field",           "physics"),
  ("paul_dirac",         "nationality",     "uk"),
  ("paul_dirac",         "field",           "physics"),
  ("germany",            "continent",       "europe"),
  ("poland",             "continent",       "europe"),
  ("denmark",            "continent",       "europe"),
  ("austria",            "continent",       "europe"),
  ("italy",              "continent",       "europe"),
  ("uk",                 "continent",       "europe"),
  ("usa",                "continent",       "north_america"),
  ("physics",            "domain",          "natural_science"),
  ("chemistry",          "domain",          "natural_science"),
]

proc buildDemoKG(): KGDataset =
  var entityMap   = initTable[string, int]()
  var relMap      = initTable[string, int]()
  var entities:   seq[string]
  var relations:  seq[string]

  proc intern(s: var Table[string, int]; names: var seq[string]; key: string): int =
    if key in s: return s[key]
    result = names.len; s[key] = result; names.add(key)

  var triples: seq[Triple]
  for (h, r, t) in demoTriples:
    let hId = intern(entityMap, entities, h)
    let tId = intern(entityMap, entities, t)
    let rId = intern(relMap, relations, r)
    triples.add (head: hId, tail: tId, rel: rId)

  # Use ~80% as train, rest as valid (no separate test for the demo).
  let nTrain = (triples.len * 8) div 10
  result.entityToId   = entityMap
  result.relationToId = relMap
  result.idToEntity   = entities
  result.idToRelation = relations
  result.train = KGSplit(triples: triples[0 ..< nTrain])
  result.valid = KGSplit(triples: triples[nTrain .. ^1])
  result.test  = KGSplit(triples: @[])

# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

proc parseFlag(args: seq[string]; flag: string; default: string): string =
  for i, a in args:
    if a == flag and i + 1 < args.len:
      return args[i + 1]
  default

proc parseIntFlag(args: seq[string]; flag: string; default: int): int =
  let s = parseFlag(args, flag, "")
  if s.len == 0: default else: parseInt(s)

proc parseFloatFlag(args: seq[string]; flag: string; default: float32): float32 =
  let s = parseFlag(args, flag, "")
  if s.len == 0: default else: parseFloat(s).float32

proc hasFlag(args: seq[string]; flag: string): bool =
  for a in args:
    if a == flag: return true

# ---------------------------------------------------------------------------
# Build MEIMConfig from CLI args + dataset sizes
# ---------------------------------------------------------------------------

proc configFromArgs(args: seq[string]; numE, numR: int): MEIMConfig =
  result = defaultConfig(numE, numR)
  result.K             = parseIntFlag(args, "--K",      result.K)
  result.Ce            = parseIntFlag(args, "--Ce",     result.Ce)
  result.Cr            = parseIntFlag(args, "--Cr",     result.Cr)
  result.learningRate  = parseFloatFlag(args, "--lr",     result.learningRate)
  result.maxEpochs     = parseIntFlag(args, "--epochs", result.maxEpochs)
  result.batchSize     = parseIntFlag(args, "--batch",  result.batchSize)
  result.lambdaOrtho   = parseFloatFlag(args, "--ortho",  result.lambdaOrtho)
  result.lambdaUnitNorm = parseFloatFlag(args, "--unorm", result.lambdaUnitNorm)
  result.inputDropRate  = parseFloatFlag(args, "--idrop", result.inputDropRate)
  result.hiddenDropRate = parseFloatFlag(args, "--hdrop", result.hiddenDropRate)
  result.evalEvery      = parseIntFlag(args, "--eval",  result.evalEvery)
  result.kVsAll         = hasFlag(args, "--kvsall")
  result.kVsAllK        = parseIntFlag(args, "--kk",   result.kVsAllK)
  result.lrDecay        = parseFloatFlag(args, "--decay", result.lrDecay)

# ---------------------------------------------------------------------------
# Print helper
# ---------------------------------------------------------------------------

proc printConfig(cfg: MEIMConfig) =
  echo ""
  echo "=== MEIM Configuration ==="
  echo &"  Entities:        {cfg.numEntities}"
  echo &"  Relations:       {cfg.numRelations}"
  echo &"  Partitions K:    {cfg.K}"
  echo &"  Ce (entity dim): {cfg.Ce}"
  echo &"  Cr (rel dim):    {cfg.Cr}"
  echo &"  Embedding size:  {cfg.K * cfg.Ce} (entity), {cfg.K * cfg.Cr} (relation)"
  echo &"  Learning rate:   {cfg.learningRate}"
  echo &"  LR decay:        {cfg.lrDecay}"
  echo &"  Batch size:      {cfg.batchSize}"
  echo &"  Epochs:          {cfg.maxEpochs}"
  echo &"  inputDrop:       {cfg.inputDropRate}"
  echo &"  hiddenDrop:      {cfg.hiddenDropRate}"
  echo &"  lambda_ortho:    {cfg.lambdaOrtho}"
  echo &"  lambda_unitnorm: {cfg.lambdaUnitNorm}"
  echo &"  k-vs-all:        {cfg.kVsAll}  k={cfg.kVsAllK}"
  echo ""

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

proc cmdDemo(args: seq[string]) =
  echo "=== MEIM Demo on built-in toy knowledge graph ==="
  let kg = buildDemoKG()
  echo &"Entities: {kg.numEntities}, Relations: {kg.numRelations}"
  echo &"Train triples: {kg.train.triples.len}, Valid: {kg.valid.triples.len}"

  var cfg = configFromArgs(args, kg.numEntities, kg.numRelations)
  # Override to reasonable settings for the tiny demo KG.
  cfg.K           = 2
  cfg.Ce          = 20
  cfg.Cr          = 20
  cfg.maxEpochs   = 200
  cfg.batchSize   = 16
  cfg.evalEvery   = 20
  cfg.lambdaOrtho = 1e-2'f32
  printConfig(cfg)

  randomize(42)
  var params = initParams(cfg)
  discard train(params, cfg, kg, verbose = true)

  echo ""
  echo "=== Predictions ==="
  echo "Top-5 tails for (albert_einstein, field):"
  let preds = predictTails(params, cfg, kg, "albert_einstein", "field", topK = 5)
  for i, p in preds:
    echo &"  {i+1}. {p.entityName:<30}  score={p.score:.4f}"

  echo ""
  echo "Top-5 heads for (field, physics):"
  let preds2 = predictHeads(params, cfg, kg, "physics", "field", topK = 5)
  for i, p in preds2:
    echo &"  {i+1}. {p.entityName:<30}  score={p.score:.4f}"

  echo ""
  echo "Triple scores:"
  for (h, r, t) in [("albert_einstein","field","physics"),
                     ("marie_curie","field","physics"),
                     ("albert_einstein","nationality","germany")]:
    let s = scoreTripleByName(params, cfg, kg, h, t, r)
    echo &"  ({h}, {r}, {t})  ->  {s:.4f}"

proc cmdTrain(args: seq[string]) =
  let dataPath = parseFlag(args, "--data", "")
  if dataPath.len == 0:
    echo "Error: --data <path> required"; quit(1)

  let trainPath = parseFlag(args, "--train", "")
  let validPath = parseFlag(args, "--valid", "")
  let testPath  = parseFlag(args, "--test",  "")

  let kg =
    if trainPath.len > 0:
      loadKGFromFiles(trainPath, validPath, testPath)
    else:
      loadKG(dataPath)

  let cfg = configFromArgs(args, kg.numEntities, kg.numRelations)
  printConfig(cfg)

  randomize()
  var params = initParams(cfg)
  let best = train(params, cfg, kg, verbose = true)

  echo ""
  echo "=== Best Validation Result ==="
  echo $best

  if kg.test.triples.len > 0:
    echo ""
    echo "=== Test Evaluation ==="
    let (hrToTails, trToHeads) = buildFilterIndex(kg)
    let testMetrics = evaluate(params, cfg, kg.test, hrToTails, trToHeads)
    echo $testMetrics

proc cmdEval(args: seq[string]) =
  let dataPath = parseFlag(args, "--data", "")
  if dataPath.len == 0:
    echo "Error: --data <path> required"; quit(1)

  let kg     = loadKG(dataPath)
  let cfg    = configFromArgs(args, kg.numEntities, kg.numRelations)
  randomize()
  var params = initParams(cfg)

  echo "[NOTE] No model persistence yet – evaluating with random params."
  let (hrToTails, trToHeads) = buildFilterIndex(kg)
  let m = evaluate(params, cfg, kg.valid, hrToTails, trToHeads, maxTriples = 200)
  echo &"Valid: {m}"

proc cmdPredict(args: seq[string]) =
  let dataPath = parseFlag(args, "--data", "")
  if dataPath.len == 0:
    echo "Error: --data <path> required"; quit(1)

  let headName = parseFlag(args, "--head", "")
  let tailName = parseFlag(args, "--tail", "")
  let relName  = parseFlag(args, "--rel",  "")
  let topK     = parseIntFlag(args, "--topk", 10)

  let kg     = loadKG(dataPath)
  let cfg    = configFromArgs(args, kg.numEntities, kg.numRelations)
  randomize()
  var params = initParams(cfg)

  echo "[NOTE] No model persistence yet – run `train` first and integrate saving."

  if headName.len > 0 and relName.len > 0:
    echo &"Top-{topK} tails for ({headName}, {relName}):"
    let preds = predictTails(params, cfg, kg, headName, relName, topK)
    for i, p in preds:
      echo &"  {i+1}. {p.entityName:<40}  score={p.score:.4f}"
  elif tailName.len > 0 and relName.len > 0:
    echo &"Top-{topK} heads for ({relName}, {tailName}):"
    let preds = predictHeads(params, cfg, kg, tailName, relName, topK)
    for i, p in preds:
      echo &"  {i+1}. {p.entityName:<40}  score={p.score:.4f}"
  elif headName.len > 0 and tailName.len > 0 and relName.len > 0:
    let s = scoreTripleByName(params, cfg, kg, headName, tailName, relName)
    echo &"Score({headName}, {relName}, {tailName}) = {s:.4f}"
  else:
    echo "Provide --head + --rel  or  --tail + --rel  or  all three."

proc printHelp() =
  echo """
MEIM – Multi-partition Embedding Interaction iMproved
Link prediction on knowledge graphs.

Commands:
  demo                       Run on a tiny built-in KG (no data needed).
  train  --data <path>       Train on a KG dataset (directory or file).
  eval   --data <path>       Evaluate with random params (placeholder).
  predict --data <path> ...  Predict top-K entities.

Data format:
  Directory: must contain train.{csv,tsv,txt} and optionally valid/test.
  Single file: all triples loaded as training data.
  Each line: head<delim>tail<delim>relation  (auto-detects CSV/TSV).

Key options:
  --K      <int>     Partitions (default 3)
  --Ce     <int>     Entity embedding dim per partition (default 100)
  --Cr     <int>     Relation embedding dim per partition (default 100)
  --lr     <float>   Learning rate (default 3e-3)
  --epochs <int>     Max training epochs (default 1000)
  --batch  <int>     Batch size (default 1024)
  --ortho  <float>   lambda_ortho for soft orthogonality (default 0.1)
  --unorm  <float>   lambda_unitnorm for unit-norm penalty (default 5e-4)
  --idrop  <float>   Input dropout rate (default 0.0)
  --hdrop  <float>   Hidden dropout rate (default 0.0)
  --decay  <float>   LR decay per epoch (default 1.0 = none)
  --eval   <int>     Evaluate every N epochs (default 10)
  --kvsall           Use k-vs-all sampling instead of 1-vs-all
  --kk     <int>     k for k-vs-all (default 500)
  --topk   <int>     Top-K predictions (predict command, default 10)
  --head   <str>     Head entity name (predict command)
  --tail   <str>     Tail entity name (predict command)
  --rel    <str>     Relation name    (predict command)
"""

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

when isMainModule:
  randomize()
  let args = commandLineParams()
  if args.len == 0 or args[0] in ["-h", "--help", "help"]:
    printHelp()
  elif args[0] == "demo":
    cmdDemo(args[1..^1])
  elif args[0] == "train":
    cmdTrain(args[1..^1])
  elif args[0] == "eval":
    cmdEval(args[1..^1])
  elif args[0] == "predict":
    cmdPredict(args[1..^1])
  else:
    echo &"Unknown command: {args[0]}"
    printHelp()
    quit(1)
