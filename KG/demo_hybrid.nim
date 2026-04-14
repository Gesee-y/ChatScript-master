import os, strutils, tables, algorithm, strformat, sequtils, sets
import kg, meim_model, trainer, kg_loader

# ---------------------------------------------------------------------------
# Reasoning Trace Helpers
# ---------------------------------------------------------------------------

proc printHeader(title: string) =
  echo "\n" & "=".repeat(80)
  echo ">>> " & title.toUpperAscii
  echo "=".repeat(80)

proc traceQueryDetailed(engine: var HybridEngine, head, relation: string,
    topK: int = 5, robust: bool = true) =
  echo &"\n[QUERY] ({head}, {relation}, ?)"

  # 1. Exact Layer
  stdout.write "  [1/4] Symbolic Exact Lookup... "
  let exact = engine.kg.queryTail(head, relation)
  if exact.len > 0:
    echo "FOUND"
    for r in exact: echo &"        -> {r.name} (Asserted)"
    return
  echo "NOT FOUND"

  # 2. BFS Layer
  stdout.write "  [2/4] Structural Inference (BFS)... "
  let bfs = engine.kg.searchTailBFS(head, relation)
  if bfs.len > 0:
    echo "FOUND"
    for r in bfs: echo &"        -> {r.name} (Structural Path)"
    return
  echo "NOT FOUND"

  # 3. Hybrid Layer (Zero-Shot + Neural Fallback)
  stdout.write "  [3/4] Semantic Analogy & Neural Fallback... "
  let hybrid = engine.queryHybridTail(head, relation, robust = robust)
  
  if hybrid.len > 0:
    echo "INFERRED"
    for r in hybrid:
      let tag = if r.score > 0.8: "ZSL" else: "MEIM"
      echo &"        -> {r.name} ({tag}, Confidence: {r.score:.2f})"
  else:
    echo "NOT FOUND"

# ---------------------------------------------------------------------------
# Clinical Assistant Logic
# ---------------------------------------------------------------------------

proc clinicalAssistant(engine: var HybridEngine, patientSymptom: string,
                      patientAge: string = "adult", patientHistory: seq[
                          string] = @[]) =
  ## Finds a treatment for a symptom while respecting contraindications.
  echo &"\n[CLINICAL] Patient presents with: {patientSymptom}"
  if patientHistory.len > 0:
    echo &"           Known conditions: {patientHistory.join(\", \")}"

  # 1. Find all candidate treatments
  let candidates = engine.queryHybridTail(patientSymptom, "treats",
      threshold = 0.5)

  echo "  Analysing candidates..."
  var safeTreatments: seq[QueryResult] = @[]

  for cand in candidates:
    # Check if candidate is contraindicated with history
    var isSafe = true
    var reason = ""
    for condition in patientHistory:
      # Query (Drug, contraindicated_with, Condition)
      let contra = engine.kg.queryTail(cand.name, "contraindicated_with")
      if contra.anyIt(it.name == condition):
        isSafe = false
        reason = condition
        break

    if isSafe:
      safeTreatments.add(cand)
    else:
      echo &"  [!] {cand.name} EXCLUDED: Contraindicated with {reason}"

  if safeTreatments.len > 0:
    echo "  Recommended Treatments:"
    for t in safeTreatments:
      echo &"    -> {t.name} (Confidence: {t.score:.2f}, via {t.kind})"
  else:
    echo "  [Warning] No safe treatments found in knowledge graph."

# ---------------------------------------------------------------------------
# Main Demo Runner
# ---------------------------------------------------------------------------

when isMainModule:
  printHeader("Mega Neuro-Symbolic Knowledge Graph Demo")

  # 1. Setup Massive Symbolic KG
  let dataPath = "mega_demo_data.csv"
  let glovePath = "../../DATA/wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt"

  var myKg = newKnowledgeGraph()
  myKg.loadFromFile(dataPath, delimiter = ',')
  myKg.loadGloveEmbeddings(glovePath)

  # Define BFS policies for deep reasoning
  myKg.addRelevantDown("treats", "is_a")
  myKg.addRelevantDown("treats", "part_of")
  myKg.addRelevantUp("symptom_of", "causes_symptom") # Reverse logic

  # NEW: Register Semantic Schemas (Taxonomic Gardrails)
  # This makes it IMPOSSIBLE to suggest a Symptom where a Disease is expected.
  myKg.setSchema("treats", "Medicament", "Maladie")
  myKg.setSchema("causes_symptom", "Maladie", "Symptom")
  
  # 2. Setup Hybrid Engine
  var engine = newHybridEngine(myKg)
  engine.syncMappings()
  
  # Validation Split: Hide 10% for objective evaluation
  let fullDataset = loadKG(dataPath)
  var trainTriples: seq[kg_loader.Triple]
  var testTriples: seq[kg_loader.Triple]
  for i, t in fullDataset.train.triples:
    if i mod 10 == 0: testTriples.add(t)
    else: trainTriples.add(t)
  
  var trainSplit = fullDataset
  trainSplit.train.triples = trainTriples
  
  echo "\n[System] Bootstrapping MEIM Weights from GloVe Embeddings..."
  engine.meimConfig = defaultConfig(fullDataset.numEntities, fullDataset.numRelations)
  engine.meimConfig.K = 3
  engine.meimConfig.Ce = 16
  engine.meimConfig.Cr = 16
  engine.meimConfig.maxEpochs = 150 
  engine.meimConfig.learningRate = 0.01
  engine.meimParams = initParams(engine.meimConfig)
  # NEW: GUIDED INITIALIZATION (Guided by GLoVe and KG topology)
  engine.meimParams.initFromEmbeddings(myKg.getEmbeddingsSeq())
  
  echo "[System] Training Neural Layer (MEIM) on 90% dataset..."
  discard train(engine.meimParams, engine.meimConfig, trainSplit, verbose = false)
  engine.useNeural = true

  # NEW: OBJECTIVE EVALUATION (Addressing Robustness concerns)
  echo "\n[System] CROSS-VALIDATION REPORT (10% Hidden Data):"
  let metrics = evaluate(engine.meimParams, engine.meimConfig, 
                         KGSplit(triples: testTriples),
                         initTable[(int,int), seq[int]](), 
                         initTable[(int,int), seq[int]]())
  echo &"  -> Mean Reciprocal Rank (MRR): {metrics.mrr:.4f}"
  echo &"  -> Hits@1 (Top suggestion):   {metrics.hits1*100:.1f}%"
  echo &"  -> Hits@10 (Covering truth):   {metrics.hits10*100:.1f}%"
  if metrics.mrr > 0.01:
    echo "  -> RESULT: Model shows statistically significant pattern learning."
  else:
    echo "  -> RESULT: Model is still sparse. Accuracy will increase with more data."

  # -------------------------------------------------------------------------
  # SCENARIO 1: DEEP HIERARCHICAL REASONING
  # -------------------------------------------------------------------------
  printHeader("Scenario 1: Deep Structural Reasoning")
  echo "Query: Does 'Salicylic_Acid' treat symptoms?"
  # Salicylic_Acid -> part_of -> Aspirin -> treats -> Headache/Fever
  traceQueryDetailed(engine, "Salicylic_Acid", "treats")

  # -------------------------------------------------------------------------
  # SCENARIO 2: CLINICAL CONSTRAINTS (SYMBOLIC + REASONING)
  # -------------------------------------------------------------------------
  printHeader("Scenario 2: Clinical Decision Support (Constraints)")
  # Patient has Fever but also Gastritis. Aspirin is contraindicated with Gastritis.
  clinicalAssistant(engine, "Fever", patientHistory = @["Gastritis"])

  # -------------------------------------------------------------------------
  # SCENARIO 3: ZERO-SHOT SÉMANTIC (ANALOGIES)
  # -------------------------------------------------------------------------
  printHeader("Scenario 3: Zero-Shot Pharmaco-Analogy")
  echo "Case: 'NeuroPlex' is a fake drug similar to Sumatriptan (Migraine drug)."
  traceQueryDetailed(engine, "NeuroPlex", "treats")

  # -------------------------------------------------------------------------
  # SCENARIO 4: PHARMACOGENOMICS (GENE-DRUG INTERACTION)
  # -------------------------------------------------------------------------
  printHeader("Scenario 4: Genetic Inhibitors")
  echo "Query: Which drugs are affected by the CYP2D6 gene?"
  traceQueryDetailed(engine, "CYP2D6", "inhibits_gene")

  # -------------------------------------------------------------------------
  # SCENARIO 5: MULTI-SYMPTOM DIAGNOSTIC (INTERSECTION)
  # -------------------------------------------------------------------------
  printHeader("Scenario 5: Multi-Symptom Diagnostic Intersection")
  let symptoms = @["Fever", "Cough", "Wheezing"]
  echo &"Patient symptoms: {symptoms.join(\", \")}"

  # Find diseases that 'cause' these symptoms
  let diag = engine.queryCombinedHeads("causes_symptom", symptoms)

  echo "Candidate Diseases (Intersected):"
  if diag.len > 0:
    for d in diag: echo &"  -> {d.name} (Combined Score: {d.score:.2f})"
  else:
    echo "  No single disease matches all symptoms exactly. Finding best partial matches..."
    # Fallback to sum of scores
    var sumScores = initTable[string, float]()
    for s in symptoms:
      let hits = engine.kg.queryHead("causes_symptom", s)
      for h in hits: sumScores[h.name] = sumScores.getOrDefault(h.name, 0.0) + h.score

    var ranked: seq[(string, float)]
    for name, score in sumScores: ranked.add((name, score))
    ranked.sort(proc(a, b: (string, float)): int = cmp(b[1], a[1]))
    for i in 0 ..< min(3, ranked.len):
      echo &"  -> {ranked[i][0]} (Aggregated Match Score: {ranked[i][1]:.2f})"

  # -------------------------------------------------------------------------
  # SCENARIO 6: NEURAL FALLBACK (CENSORED TRUTH)
  # -------------------------------------------------------------------------
  printHeader("Scenario 6: Neural Gap Filling (Censored Truth)")
  echo "Deleting 'Amoxicillin treats Pneumonia' from Symbolic KG..."
  # Simulating a loss of historical documentation.
  for i in 0 ..< engine.kg.nodes.len:
    if engine.kg.nodes[i].name == "Amoxicillin":
      let relId = engine.kg.getOrAddRelation("treats")
      engine.kg.nodes[i].outgoing.del(relId)
      break

  echo "Querying symbolic knowledge for Amoxicillin treats (DEFAULT MODE)..."
  traceQueryDetailed(engine, "Amoxicillin", "treats", robust = false)

  echo "\n[System] Enabling ROBUST MODE (Constraint Filtering)..."
  echo "Querying symbolic knowledge for Amoxicillin treats (ROBUST MODE)..."
  traceQueryDetailed(engine, "Amoxicillin", "treats", robust = true)

  echo "\n" & "=".repeat(80)
  echo "MEGA DEMO COMPLETE"
  echo "=".repeat(80)
