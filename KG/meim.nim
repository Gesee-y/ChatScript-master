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
  # --- Expertise & Domaines (Multi-expertise) ---
  ("albert_einstein",    "field",           "physics"),
  ("albert_einstein",    "field",           "philosophy"),
  ("marie_curie",        "field",           "chemistry"),
  ("marie_curie",        "field",           "physics"),
  ("linus_pauling",      "field",           "chemistry"),
  ("linus_pauling",      "field",           "biology"),
  
  # --- Nationalités & Géographie ---
  ("albert_einstein",    "nationality",     "germany"),
  ("albert_einstein",    "born_in",         "ulm"),
  ("ulm",                "located_in",      "germany"),
  ("marie_curie",        "nationality",     "poland"),
  ("marie_curie",        "nationality",     "france"),
  ("richard_feynman",    "nationality",     "usa"),
  
  # --- Hiérarchie Académique (Relations de Mentorat/Collaboration) ---
  ("max_planck",         "mentored",        "albert_einstein"),
  ("niels_bohr",         "collaborated_with","werner_heisenberg"),
  ("ernest_rutherford",  "mentored",        "niels_bohr"),
  ("j_j_thomson",        "mentored",        "ernest_rutherford"),
  ("robert_oppenheimer", "collaborated_with","richard_feynman"),

  # --- Distinctions (Propriétés partagées) ---
  ("albert_einstein",    "won_award",       "nobel_prize_physics"),
  ("marie_curie",        "won_award",       "nobel_prize_physics"),
  ("marie_curie",        "won_award",       "nobel_prize_chemistry"),
  ("richard_feynman",    "won_award",       "nobel_prize_physics"),
  ("linus_pauling",      "won_award",       "nobel_prize_chemistry"),
  ("nobel_prize_physics", "category",       "natural_science"),
  ("nobel_prize_chemistry","category",       "natural_science"),

  # --- Ontologie (Taxonomie) ---
  ("physics",            "sub_domain_of",   "natural_science"),
  ("chemistry",          "sub_domain_of",   "natural_science"),
  ("biology",            "sub_domain_of",   "natural_science"),
  ("natural_science",    "type",            "academic_discipline"),
  ("philosophy",         "type",            "humanities"),

  # --- Localisation & Continents ---
  ("germany",            "continent",       "europe"),
  ("france",             "continent",       "europe"),
  ("poland",             "continent",       "europe"),
  ("uk",                 "continent",       "europe"),
  ("denmark",            "continent",       "europe"),
  ("usa",                "continent",       "north_america"),
  
  # --- Relations de "Fait" pour tester l'inférence inverse ---
  ("quantum_mechanics",  "part_of",         "physics"),
  ("relativity",         "part_of",         "physics"),
  ("albert_einstein",    "developed",       "relativity"),
  ("werner_heisenberg",  "developed",       "quantum_mechanics"),
  ("niels_bohr",         "developed",       "quantum_mechanics")
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
  echo "=== Advanced Inferences ==="

  # 1. Test de Transitivité (Einstein -> ? -> North America)
  echo "Top-5 continents for Albert Einstein (Inference via USA/Germany):"
  let predsCont = predictTails(params, cfg, kg, "albert_einstein", "continent", topK = 5)
  for i, p in predsCont:
    echo &"  {i+1}. {p.entityName:<30}  score={p.score:.4f}"

  # 2. Test d'Asymétrie (Qui a été mentoré par Max Planck ?)
  echo ""
  echo "Top-5 students mentored by Max Planck (Directional test):"
  let predsMent = predictTails(params, cfg, kg, "max_planck", "mentored", topK = 5)
  for i, p in predsMent:
    echo &"  {i+1}. {p.entityName:<30}  score={p.score:.4f}"

  # 3. Test de Multi-expertise (Marie Curie)
  echo ""
  echo "Top-5 fields for Marie Curie (Should show Chemistry AND Physics):"
  let predsCurie = predictTails(params, cfg, kg, "marie_curie", "field", topK = 5)
  for i, p in predsCurie:
    echo &"  {i+1}. {p.entityName:<30}  score={p.score:.4f}"

  # 4. Test d'Inférence Inverse (Qui travaille dans les 'Natural Sciences' ?)
  echo ""
  echo "Top-5 entities linked to 'natural_science' (Inference via sub_domain_of):"
  let predsScience = predictHeads(params, cfg, kg, "natural_science", "sub_domain_of", topK = 5)
  for i, p in predsScience:
    echo &"  {i+1}. {p.entityName:<30}  score={p.score:.4f}"

  echo ""
  echo "=== Complex Triple Validation ==="
  let triplesToTest = [
    ("albert_einstein", "won_award", "nobel_prize_physics"),  # Direct (True)
    ("richard_feynman", "nationality", "usa"),               # Direct (True)
    ("richard_feynman", "nationality", "germany"),           # False
    ("linus_pauling", "field", "biology"),                   # Multi-label (True)
    ("albert_einstein", "mentored", "max_planck")            # Reversed Relation (Should be Low/False)
  ]

  for (h, r, t) in triplesToTest:
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
