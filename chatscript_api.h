/**
 * @file   chatscript_api.h
 * @brief  ChatScript Public DLL API — Single include for host applications
 *
 * This header is the ONLY file your project needs to include.
 * It surfaces the full public interface of ChatScript when used as a
 * DLL (Windows) or shared object (.so on Linux).
 *
 * Copyright (C) 2011-2024 Bruce Wilcox — MIT License
 * Wrapper header created for integration use.
 */

#ifndef CHATSCRIPT_API_H
#define CHATSCRIPT_API_H

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * PLATFORM EXPORT / IMPORT MACROS
 * ============================================================ */
#ifdef _WIN32
  #ifdef CHATSCRIPT_BUILDING_DLL
    #define CS_API __declspec(dllexport)
  #else
    #define CS_API __declspec(dllimport)
  #endif
#else
  #define CS_API __attribute__((visibility("default")))
#endif

/* ============================================================
 * CONSTANTS
 * ============================================================ */

/** Maximum size of a ChatScript output buffer (bytes). */
#define CS_OUTPUT_SIZE     40000

/** Maximum size of a user ID or bot name string (bytes). */
#define CS_ID_SIZE         500

/** Returned by PerformChat when a system restart is pending. */
#define CS_PENDING_RESTART (-1)

/* ============================================================
 * CALLBACK TYPES
 * ============================================================
 *
 * These function pointer types allow you to hook into the CS
 * engine for debugging, logging, and introspection.
 *
 * Usage:  assign your function pointer to the matching global
 *         BEFORE calling InitSystem().
 */

/**
 * @brief Generic debug callback — receives a text buffer, may return modified text.
 *
 * Used for: debugInput, debugOutput, debugEndTurn, debugMessage, debugAction.
 *
 * @param buffer  Null-terminated string from the engine.
 * @return        Pointer to (optionally modified) buffer, or NULL to leave unchanged.
 */
typedef char* (*CS_DebugAPI)(char* buffer);

/**
 * @brief Loop-level debug callback — notified at each rule evaluation.
 *
 * @param buffer  Description of current rule being evaluated.
 * @param in      true = entering the rule, false = exiting the rule.
 * @return        Pointer to buffer (can be ignored).
 */
typedef char* (*CS_DebugLoopAPI)(char* buffer, int in);

/**
 * @brief Variable change callback — called whenever a user variable changes.
 *
 * @param var    Variable name (e.g. "$myVar").
 * @param value  New value as a string.
 * @return       Pointer to value (can be ignored).
 */
typedef char* (*CS_DebugVarAPI)(char* var, char* value);

/* ============================================================
 * USER FILESYSTEM OVERRIDE
 * ============================================================
 *
 * By default ChatScript stores user session state in flat files
 * under the USERS/ directory. Pass a populated USERFILESYSTEM
 * struct to InitSystem() to redirect all I/O to your own storage
 * (database, cloud, in-memory, etc.).
 *
 * CONTRACT:
 *   - create : open a new file for binary write ("wb" implied)
 *   - open   : open an existing file for binary read ("rb" implied)
 *   - close  : flush and release the handle
 *   - read   : standard fread semantics
 *   - write  : standard fwrite semantics
 *   - size   : fill buffer with file content & return byte count
 *   - del    : delete a user file by name
 *
 * Only one "file" is ever open at a time; you may manage the
 * actual storage via globals on your side.
 */
typedef void* CS_FILE;

typedef CS_FILE (*CS_UserCreate)  (const char* name);
typedef CS_FILE (*CS_UserOpen)    (const char* name);
typedef int     (*CS_UserClose)   (CS_FILE);
typedef size_t  (*CS_UserRead)    (void* buf, size_t sz, size_t n, CS_FILE);
typedef size_t  (*CS_UserWrite)   (const void* buf, size_t sz, size_t n, CS_FILE);
typedef int     (*CS_UserSize)    (CS_FILE, char* buf, size_t maxLen);
typedef void    (*CS_UserDelete)  (const char* name);

typedef struct CS_UserFileSystem {
    CS_UserCreate  userCreate;   /**< Create new user file (write mode). */
    CS_UserOpen    userOpen;     /**< Open existing user file (read mode). */
    CS_UserClose   userClose;    /**< Close file handle. */
    CS_UserRead    userRead;     /**< Read data from file. */
    CS_UserWrite   userWrite;    /**< Write data to file. */
    CS_UserDelete  userDelete;   /**< Delete user file. */
    void*          userEncrypt;  /**< Optional: encrypt on write (or NULL). */
    void*          userDecrypt;  /**< Optional: decrypt on read (or NULL). */
} CS_UserFileSystem;

/* ============================================================
 * CORE API FUNCTIONS
 * ============================================================ */

/**
 * @brief  Initialize the ChatScript engine. Call ONCE at startup.
 *
 * Loads the compiled bot (TOPIC/), dictionary (LIVEDATA/), runs
 * the ^csboot script, and prepares all subsystems.
 *
 * @param argc          Argument count (same as main's argc).
 * @param argv          Argument vector (same as main's argv).
 *                      Use this to pass configuration parameters
 *                      (see Configuration Parameters section).
 * @param unchangedPath Absolute path to the ChatScript root directory
 *                      (the folder containing LIVEDATA/, TOPIC/, etc.).
 *                      Pass NULL to auto-detect from CWD.
 * @param readonlyPath  Read-only resource path override, or NULL.
 * @param writablePath  Writable data path override (USERS/, LOGS/), or NULL.
 * @param userFiles     Pointer to a custom user filesystem, or NULL for
 *                      default flat-file storage.
 * @param debugIn       Callback invoked on every user input, or NULL.
 * @param debugOut      Callback invoked on every bot output, or NULL.
 *
 * @return 0 on success, non-zero on failure.
 *
 * @note  This function is NOT reentrant. Call from a single thread only.
 */
CS_API unsigned int InitSystem(
    int               argc,
    char*             argv[],
    char*             unchangedPath,
    char*             readonlyPath,
    char*             writablePath,
    CS_UserFileSystem* userFiles,
    CS_DebugAPI       debugIn,
    CS_DebugAPI       debugOut
);

/**
 * @brief  Process one conversational turn (volley).
 *
 * This is the main chat entry point. It:
 *   1. Loads the user's saved state (if any).
 *   2. Runs the NLP pipeline on `incoming`.
 *   3. Matches topics/rules and generates a response.
 *   4. Saves the user's updated state.
 *   5. Writes the bot's response into `output`.
 *
 * @param user      User identity string (login name / session ID).
 *                  Must be unique per user. Max CS_ID_SIZE bytes.
 *                  Pass "" to create an anonymous session.
 * @param usee      Bot name to talk to. Pass "" for the default bot
 *                  (set via defaultbot= config parameter).
 * @param incoming  The user's input text. Pass "" or NULL to start
 *                  a new conversation (triggers ^csboot greeting).
 * @param ip        Client IP address string for logging, or NULL.
 * @param output    Caller-allocated buffer of at least CS_OUTPUT_SIZE
 *                  bytes. Receives the bot's response.
 *
 * @return  > 0 : Current volley number (conversation turn count).
 *          = 0 : A debug command was executed, no chat response.
 *          CS_PENDING_RESTART (-1) : Engine needs restart. Call
 *                  CloseSystem() + InitSystem() then retry.
 *
 * @note  Thread safety: use one user ID per thread, or serialize
 *        calls with a mutex (ChatScript is NOT thread-safe internally).
 * @note  `output` is always null-terminated, even on error.
 */
CS_API int PerformChat(
    char* user,
    char* usee,
    char* incoming,
    char* ip,
    char* output
);

/**
 * @brief  Like PerformChat, but injects extra in-memory topic data.
 *
 * Allows you to dynamically provide script topic data that is not
 * compiled into the TOPIC/ binary. The `topicData` string must be
 * in ChatScript compiled topic format.
 *
 * @param user      Same as PerformChat.
 * @param usee      Same as PerformChat.
 * @param incoming  Same as PerformChat.
 * @param ip        Same as PerformChat.
 * @param output    Same as PerformChat.
 * @param topicData Null-terminated compiled topic script string.
 *
 * @return Same as PerformChat.
 */
CS_API int PerformChatGivenTopic(
    char* user,
    char* usee,
    char* incoming,
    char* ip,
    char* output,
    char* topicData
);

/**
 * @brief  Retrieves the string value of a user variable.
 *
 * @param var       The name of the variable (e.g., "$myvar").
 * @param nojson    Internal flag, set to false.
 * @return          Pointer to the variable's value string, or empty string if not found.
 */
CS_API char* GetUserVariable(const char* var, bool nojson = false);

/**
 * @brief  Sets or creates a user variable in the engine state.
 *
 * @param var          The name of the variable (e.g., "$myvar").
 * @param value        The string value to assign.
 * @param assignment   Internal flag, set to false.
 * @param reuse        Internal flag, set to false.
 */
CS_API void SetUserVariable(const char* var, char* value, bool assignment = false, bool reuse = false);

/**
 * @brief  Shut down the ChatScript engine. Call ONCE at exit.
 *
 * Saves all pending user data, closes database connections,
 * releases all allocated memory, and tears down the engine.
 *
 * @note  After CloseSystem(), you may call InitSystem() again
 *        to reload the engine (e.g. to hot-swap bots).
 */
CS_API void CloseSystem(void);

/* ============================================================
 * OPTIONAL DEBUG HOOKS
 * ============================================================
 *
 * Assign these function pointers BEFORE calling InitSystem()
 * to wire up debug/monitoring callbacks.
 */

/**
 * @brief  Called with the raw user input text before any processing.
 * Assign your CS_DebugAPI function to hook the input pipeline.
 */
extern CS_API CS_DebugAPI debugInput;

/**
 * @brief  Called with the bot's response text before it is returned.
 * Assign your CS_DebugAPI function to intercept/log output.
 */
extern CS_API CS_DebugAPI debugOutput;

/**
 * @brief  Called just before the user session state is saved to disk/DB.
 */
extern CS_API CS_DebugAPI debugEndTurn;

/**
 * @brief  Called at each rule entry/exit during pattern matching.
 * Passing a CS_DebugLoopAPI here enables per-rule tracing.
 */
extern CS_API CS_DebugLoopAPI debugCall;

/**
 * @brief  Called whenever a user variable ($var) is modified.
 * Useful for reactive data binding or introspection.
 */
extern CS_API CS_DebugVarAPI debugVar;

/**
 * @brief  Called whenever a concept/word mark is set during NLP.
 */
extern CS_API CS_DebugVarAPI debugMark;

/**
 * @brief  Called with internal engine messages (errors, warnings).
 */
extern CS_API CS_DebugAPI debugMessage;

/**
 * @brief  Called when a script action (^function call) is executed.
 */
extern CS_API CS_DebugAPI debugAction;

#ifdef __cplusplus
}
#endif
#endif /* CHATSCRIPT_API_H */
