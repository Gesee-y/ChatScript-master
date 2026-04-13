## kg_loader.nim
## ==============
## Loads a knowledge graph from various flat-file formats and maps all
## entity / relation strings to dense integer IDs.
##
## Supported input formats
## -----------------------
##   CSV   – delimiter-separated triples, one per line.
##           Columns can appear in any order; the loader auto-detects or
##           accepts an explicit (headCol, tailCol, relCol) spec.
##   TSV   – same as CSV but tab-delimited (extension ".tsv" or explicit).
##   NT    – N-Triples subset: lines of the form
##              <subject> <predicate> <object> .
##           URIs are stripped to their local name.
##
## Directory loading
## -----------------
##   Pass a directory path to `loadKG`.  The loader looks for files named
##   train.{csv,tsv,txt}, valid.{csv,tsv,txt}, test.{csv,tsv,txt} and
##   concatenates them.  The vocabulary (entity / relation maps) is built
##   from all triples so IDs are consistent across splits.

import os, strutils, tables, strformat

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

type
  Triple* = tuple[head, tail, rel: int]   ## All IDs are 0-based integers.

  KGSplit* = object
    ## One data split (train / valid / test).
    triples*: seq[Triple]

  KGDataset* = object
    ## The complete dataset with vocabulary and all three splits.
    entityToId*:   Table[string, int]
    relationToId*: Table[string, int]
    idToEntity*:   seq[string]
    idToRelation*: seq[string]
    train*:        KGSplit
    valid*:        KGSplit
    test*:         KGSplit

  LoadConfig* = object
    ## Fine-grained control over file parsing.
    delimiter*:  char    ## field separator (default: auto-detect)
    headCol*:    int     ## 0-based column index of head entity  (default 0)
    tailCol*:    int     ## 0-based column index of tail entity  (default 1)
    relCol*:     int     ## 0-based column index of relation     (default 2)
    hasHeader*:  bool    ## skip first line if true
    stripAngle*: bool    ## strip < > wrappers (for N-Triples / Turtle)

proc defaultConfig*(): LoadConfig =
  LoadConfig(delimiter: '\0', headCol: 0, tailCol: 1, relCol: 2,
             hasHeader: false, stripAngle: false)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

proc localName(uri: string): string =
  ## Extract the local name from a URI  e.g.  <http://ex.org/foo#bar>  ->  bar
  var s = uri
  if s.startsWith('<') and s.endsWith('>'): s = s[1..^2]
  let hashPos = s.rfind('#')
  if hashPos >= 0: return s[hashPos + 1 .. ^1]
  let slashPos = s.rfind('/')
  if slashPos >= 0: return s[slashPos + 1 .. ^1]
  s

proc detectDelimiter(line: string): char =
  ## Guess delimiter from the first non-empty line.
  if '\t' in line: return '\t'
  if ',' in line:  return ','
  if ' ' in line:  return ' '
  '\t'  # fallback

proc intern(s: var Table[string, int]; names: var seq[string]; key: string): int =
  if key in s: return s[key]
  result = names.len
  s[key] = result
  names.add(key)

proc parseTripleLine(fields: seq[string]; cfg: LoadConfig;
                     entityMap: var Table[string, int]; entities: var seq[string];
                     relMap: var Table[string, int]; relations: var seq[string]): Triple =
  ## Extract one triple from a seq of string fields.
  var h = fields[cfg.headCol].strip()
  var t = fields[cfg.tailCol].strip()
  var r = fields[cfg.relCol].strip()
  if cfg.stripAngle:
    h = localName(h); t = localName(t); r = localName(r)
  let hid = intern(entityMap, entities, h)
  let tid = intern(entityMap, entities, t)
  let rid = intern(relMap, relations, r)
  (head: hid, tail: tid, rel: rid)

proc splitLine(line: string; delim: char): seq[string] =
  ## Split a line by a single-character delimiter.
  line.split(delim)

proc loadFile(path: string; cfg: var LoadConfig;
              entityMap: var Table[string, int]; entities: var seq[string];
              relMap: var Table[string, int]; relations: var seq[string]): seq[Triple] =
  ## Parse one file into a list of triples, updating the global vocab.
  if not fileExists(path):
    return @[]

  let content = readFile(path)
  var lines = content.splitLines()

  # Auto-detect delimiter from first content line.
  var firstLine = ""
  for l in lines:
    let s = l.strip()
    if s.len > 0 and not s.startsWith('#'):
      firstLine = s; break
  if cfg.delimiter == '\0':
    cfg.delimiter = detectDelimiter(firstLine)

  var skipFirst = cfg.hasHeader
  for rawLine in lines:
    let line = rawLine.strip()
    if line.len == 0 or line.startsWith('#'): continue
    if skipFirst: skipFirst = false; continue

    let fields = splitLine(line, cfg.delimiter)
    let maxCol = max(cfg.headCol, max(cfg.tailCol, cfg.relCol))
    if fields.len <= maxCol: continue   # malformed row – skip

    result.add parseTripleLine(fields, cfg, entityMap, entities, relMap, relations)

# ---------------------------------------------------------------------------
# Candidate file names for each split
# ---------------------------------------------------------------------------

const splitNames = [
  ("train", @["train.csv","train.tsv","train.txt","training.csv","training.tsv"]),
  ("valid", @["valid.csv","valid.tsv","valid.txt","dev.csv","dev.tsv"]),
  ("test",  @["test.csv", "test.tsv", "test.txt"]),
]

proc findSplitFile(dir, splitKey: string; candidates: seq[string]): string =
  for name in candidates:
    let p = dir / name
    if fileExists(p): return p
  ""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

proc loadKG*(path: string; cfg: LoadConfig = defaultConfig()): KGDataset =
  ## Load a knowledge graph from `path`.
  ## `path` may be:
  ##   - a directory containing train/valid/test files
  ##   - a single file (all triples go into the train split)
  var mutableCfg = cfg
  var entityMap   = initTable[string, int]()
  var relMap      = initTable[string, int]()
  var entities:   seq[string]
  var relations:  seq[string]

  var trainTriples, validTriples, testTriples: seq[Triple]

  if dirExists(path):
    # Try to find each split by conventional name.
    for (key, candidates) in splitNames:
      let fpath = findSplitFile(path, key, candidates)
      if fpath.len == 0:
        echo &"[loader] No {key} file found in {path}"
        continue
      echo &"[loader] Loading {key}: {fpath}"
      let triples = loadFile(fpath, mutableCfg, entityMap, entities, relMap, relations)
      echo &"[loader]   -> {triples.len} triples"
      case key
      of "train": trainTriples = triples
      of "valid": validTriples = triples
      of "test":  testTriples  = triples
      else: discard
  else:
    # Single file – everything goes to train.
    echo &"[loader] Loading single file: {path}"
    trainTriples = loadFile(path, mutableCfg, entityMap, entities, relMap, relations)
    echo &"[loader]   -> {trainTriples.len} triples"

  result.entityToId   = entityMap
  result.relationToId = relMap
  result.idToEntity   = entities
  result.idToRelation = relations
  result.train  = KGSplit(triples: trainTriples)
  result.valid  = KGSplit(triples: validTriples)
  result.test   = KGSplit(triples: testTriples)

  echo &"[loader] Vocabulary: {entities.len} entities, {relations.len} relations"
  echo &"[loader] Splits: train={trainTriples.len}, valid={validTriples.len}, test={testTriples.len}"

proc numEntities*(kg: KGDataset): int = kg.idToEntity.len
proc numRelations*(kg: KGDataset): int = kg.idToRelation.len

# ---------------------------------------------------------------------------
# Build adjacency index: (h, r) -> seq[t]  and  (t, r) -> seq[h]
# Used for filtered evaluation (remove known true triples from ranking).
# ---------------------------------------------------------------------------

proc buildFilterIndex*(kg: KGDataset): (Table[(int,int), seq[int]], Table[(int,int), seq[int]]) =
  ## Returns:
  ##   hrToTails  – maps (head, rel) -> all known tail entities
  ##   trToHeads  – maps (tail, rel) -> all known head entities
  var hrToTails = initTable[(int,int), seq[int]]()
  var trToHeads = initTable[(int,int), seq[int]]()

  proc addTriple(triple: Triple) =
    let hrKey = (triple.head, triple.rel)
    let trKey = (triple.tail, triple.rel)
    if hrKey notin hrToTails: hrToTails[hrKey] = @[]
    hrToTails[hrKey].add triple.tail
    if trKey notin trToHeads: trToHeads[trKey] = @[]
    trToHeads[trKey].add triple.head

  for t in kg.train.triples: addTriple(t)
  for t in kg.valid.triples:  addTriple(t)
  for t in kg.test.triples:   addTriple(t)
  (hrToTails, trToHeads)

# ---------------------------------------------------------------------------
# Convenience: load from two/three separate file paths
# ---------------------------------------------------------------------------

proc loadKGFromFiles*(trainPath: string;
                      validPath: string = "";
                      testPath:  string = "";
                      cfg: LoadConfig = defaultConfig()): KGDataset =
  ## Load from explicit file paths.
  var mutableCfg = cfg
  var entityMap   = initTable[string, int]()
  var relMap      = initTable[string, int]()
  var entities:   seq[string]
  var relations:  seq[string]

  let tr = loadFile(trainPath, mutableCfg, entityMap, entities, relMap, relations)
  let va = if validPath.len > 0: loadFile(validPath, mutableCfg, entityMap, entities, relMap, relations) else: @[]
  let te = if testPath.len  > 0: loadFile(testPath,  mutableCfg, entityMap, entities, relMap, relations) else: @[]

  result.entityToId   = entityMap
  result.relationToId = relMap
  result.idToEntity   = entities
  result.idToRelation = relations
  result.train  = KGSplit(triples: tr)
  result.valid  = KGSplit(triples: va)
  result.test   = KGSplit(triples: te)

  echo &"[loader] Vocabulary: {entities.len} entities, {relations.len} relations"
  echo &"[loader] Splits: train={tr.len}, valid={va.len}, test={te.len}"
