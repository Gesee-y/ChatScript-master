## ChatScript Nim Wrapper (FFI)
## Facilitates use of ChatScript native DLL in Nim.
## 
## (c) 2024 - Integration Bridge

const 
  CS_OUTPUT_SIZE* = 40000
  CS_ID_SIZE* = 500
  CS_PENDING_RESTART* = -1
  
  # DLL Name
  ChatScriptDLL = "ChatScript_DLL.dll"

type
  # Function pointer types for callbacks
  CS_DebugAPI* = proc (buffer: cstring): cstring {.cdecl.}
  CS_DebugLoopAPI* = proc (buffer: cstring, entering: cint): cstring {.cdecl.}
  CS_DebugVarAPI* = proc (varName: cstring, value: cstring): cstring {.cdecl.}

  # User Filesystem structure
  CS_FILE* = pointer
  CS_UserCreate* = proc (name: cstring): CS_FILE {.cdecl.}
  CS_UserOpen* = proc (name: cstring): CS_FILE {.cdecl.}
  CS_UserClose* = proc (f: CS_FILE): cint {.cdecl.}
  CS_UserRead* = proc (buf: pointer, sz: csize_t, n: csize_t, f: CS_FILE): csize_t {.cdecl.}
  CS_UserWrite* = proc (buf: pointer, sz: csize_t, n: csize_t, f: CS_FILE): csize_t {.cdecl.}
  CS_UserDelete* = proc (name: cstring) {.cdecl.}

  CS_UserFileSystem* {.pure, final.} = object
    userCreate*: CS_UserCreate
    userOpen*: CS_UserOpen
    userClose*: CS_UserClose
    userRead*: CS_UserRead
    userWrite*: CS_UserWrite
    userDelete*: CS_UserDelete
    userEncrypt*: pointer
    userDecrypt*: pointer

# Core Functions
proc InitSystem*(argc: cint, argv: ptr cstring, unchangedPath: cstring = nil, 
                readonlyPath: cstring = nil, writablePath: cstring = nil, 
                userFiles: ptr CS_UserFileSystem = nil, 
                debugIn: CS_DebugAPI = nil, debugOut: CS_DebugAPI = nil): cuint 
                {.importc: "InitSystem", dynlib: ChatScriptDLL, cdecl.}

proc PerformChat*(user: cstring, usee: cstring, incoming: cstring, ip: cstring, output: cstring): cint 
                 {.importc: "PerformChat", dynlib: ChatScriptDLL, cdecl.}

proc PerformChatGivenTopic*(user: cstring, usee: cstring, incoming: cstring, ip: cstring, 
                           output: cstring, topicData: cstring): cint 
                           {.importc: "PerformChatGivenTopic", dynlib: ChatScriptDLL, cdecl.}

proc CloseSystem*() {.importc: "CloseSystem", dynlib: ChatScriptDLL, cdecl.}

proc GetUserVariable*(name: cstring, nojson: bool = false): cstring 
                     {.importc: "GetUserVariable", dynlib: ChatScriptDLL, cdecl.}

proc SetUserVariable*(name: cstring, value: cstring, assignment: bool = false, reuse: bool = false) 
                     {.importc: "SetUserVariable", dynlib: ChatScriptDLL, cdecl.}

# Optional Global Hooks (Debug)
var debugInput* {.importc: "debugInput", dynlib: ChatScriptDLL.}: CS_DebugAPI
var debugOutput* {.importc: "debugOutput", dynlib: ChatScriptDLL.}: CS_DebugAPI
var debugEndTurn* {.importc: "debugEndTurn", dynlib: ChatScriptDLL.}: CS_DebugAPI
var debugCall* {.importc: "debugCall", dynlib: ChatScriptDLL.}: CS_DebugLoopAPI
var debugVar* {.importc: "debugVar", dynlib: ChatScriptDLL.}: CS_DebugVarAPI
var debugMark* {.importc: "debugMark", dynlib: ChatScriptDLL.}: CS_DebugVarAPI
var debugMessage* {.importc: "debugMessage", dynlib: ChatScriptDLL.}: CS_DebugAPI
var debugAction* {.importc: "debugAction", dynlib: ChatScriptDLL.}: CS_DebugAPI

# --- Helper logic for easier Nim integration ---

proc initChatScript*(args: seq[string] = @[], rootPath: string = ""): bool =
  ## High-level initialization helper.
  # ChatScript expects argv[0] to be the program name.
  var full_args = @["chatscript_boot"]
  full_args.add(args)
  
  var c_args = newSeq[cstring](full_args.len)
  for i in 0..<full_args.len: c_args[i] = full_args[i].cstring
  
  let p_args = addr c_args[0]
  let res = InitSystem(full_args.len.cint, p_args, if rootPath == "": nil else: rootPath.cstring)
  return res == 0

proc chat*(user, bot, input: string, ip = "127.0.0.1"): (int, string) =
  ## High-level chat helper. Returns (volleyCount, responseString)
  # Allocate writable buffers for ALL inputs
  let uPtr = cast[cstring](alloc0(100))
  let bPtr = cast[cstring](alloc0(100))
  let iPtr = cast[cstring](alloc0(1024))
  let ipPtr = cast[cstring](alloc0(100))
  let oPtr = cast[cstring](alloc0(CS_OUTPUT_SIZE))
  
  copyMem(uPtr, user.cstring, min(user.len, 99))
  copyMem(bPtr, bot.cstring, min(bot.len, 99))
  copyMem(iPtr, input.cstring, min(input.len, 1023))
  copyMem(ipPtr, ip.cstring, min(ip.len, 99))
  
  defer:
    dealloc(uPtr)
    dealloc(bPtr)
    dealloc(iPtr)
    dealloc(ipPtr)
    dealloc(oPtr)
  
  # echo "[Debug] Calling PerformChat with all writable buffers"
  
  let count = PerformChat(uPtr, bPtr, iPtr, ipPtr, oPtr)
  
  let finalResponse = if oPtr[0] == '\0': "" else: $oPtr
  return (count.int, finalResponse)
