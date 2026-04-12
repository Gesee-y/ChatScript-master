@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: build_dll.bat  —  Compile ChatScript as a Windows DLL (x64)
::                   using MinGW-W64 g++
::
:: Prerequisites:
::   - MinGW-W64 with g++ in PATH (C:\Program Files\mingw64\bin)
::   - Run from the ChatScript-master root directory.
::
:: Output:
::   BINARIES\ChatScript_DLL.dll
::   BINARIES\ChatScript_DLL.lib  (import library via dlltool)
:: ============================================================

set GPP=C:\Program Files\mingw64\bin\g++.exe
set DLLTOOL=C:\Program Files\mingw64\bin\dlltool.exe
set SRC=SRC
set OUT=BINARIES
set DLL_NAME=ChatScript_DLL
set OBJ_DIR=TMP\dll_obj

echo.
echo ====================================================
echo   ChatScript DLL Build  (MinGW-W64 x64)
echo ====================================================
echo.

:: --- Verify compiler ---
if not exist "%GPP%" (
    echo ERROR: g++ not found at "%GPP%"
    echo        Please install MinGW-W64 or adjust the GPP variable.
    exit /b 1
)

:: --- Create object directory ---
if not exist "%OBJ_DIR%" mkdir "%OBJ_DIR%"

:: ============================================================
:: PREPROCESSOR DEFINITIONS FOR DLL MODE
::
:: DLL=1               → activates extern "C" __declspec(dllexport) on API funcs
:: NOMAIN=1            → suppresses the main() function
:: WIN32               → Windows platform
:: DISCARDSERVER=1     → no standalone TCP server (not needed in DLL)
:: DISCARDMONGO=1      → no MongoDB
:: DISCARDPOSTGRES=1   → no PostgreSQL
:: DISCARDMYSQL=1      → no MySQL
:: DISCARDMICROSOFTSQL=1 → no MSSQL
:: DISCARD_PYTHON=1    → no Python embedding
:: DISCARD_TEXT_COMPRESSION=1 → no zlib compression
:: DISCARDWEBSOCKET=1  → no WebSocket server
:: DISCARDCLIENT=1     → no TCP client mode
:: ============================================================
set DEFINES=-DDLL=1 -DNOMAIN=1 -DWIN32=1 ^
 -DDISCARDSERVER=1 -DDISCARDMONGO=1 -DDISCARDPOSTGRES=1 ^
 -DDISCARDMYSQL=1 -DDISCARDMICROSOFTSQL=1 ^
 -DDISCARD_PYTHON=1 -DDISCARD_TEXT_COMPRESSION=1 ^
 -DDISCARDWEBSOCKET=1 -DDISCARDCLIENT=1 -DDISCARDJSONOPEN=1 -DDISCARD_JAPANESE=1

:: ============================================================
:: COMPILER FLAGS
:: -std=c++11          modern C++ standard
:: -O2                 release optimisation
:: -fPIC               position independent code (good practice even on Win)
:: -funsigned-char     chars are unsigned (required by CS source)
:: -Wno-*              suppress expected warnings in CS codebase
:: -I...               include paths
:: ============================================================
set CFLAGS=-std=c++11 -O2 -funsigned-char ^
 -Wno-write-strings -Wno-unused-variable -Wno-unknown-pragmas ^
 -Wno-char-subscripts -Wno-deprecated ^
 -I"%SRC%"

:: ============================================================
:: SOURCE FILES (all CS .cpp files needed for DLL)
:: Excludes: evserver.cpp (server only), cs_ev.cpp (evserver)
:: ============================================================
set SOURCES=^
 %SRC%\constructCode.cpp ^
 %SRC%\csocket.cpp ^
 %SRC%\cs_es.cpp ^
 %SRC%\cs_german.cpp ^
 %SRC%\cs_jp.cpp ^
 %SRC%\dictionarySystem.cpp ^
 %SRC%\duktape\duktape.cpp ^
 %SRC%\english.cpp ^
 %SRC%\englishTagger.cpp ^
 %SRC%\evserver.cpp ^
 %SRC%\factSystem.cpp ^
 %SRC%\functionExecute.cpp ^
 %SRC%\infer.cpp ^
 %SRC%\javascript.cpp ^
 %SRC%\jsmn.cpp ^
 %SRC%\json.cpp ^
 %SRC%\mainSystem.cpp ^
 %SRC%\markSystem.cpp ^
 %SRC%\mongodb.cpp ^
 %SRC%\mssql.cpp ^
 %SRC%\mysql.cpp ^
 %SRC%\os.cpp ^
 %SRC%\outputSystem.cpp ^
 %SRC%\patternSystem.cpp ^
 %SRC%\postgres.cpp ^
 %SRC%\privatesrc.cpp ^
 %SRC%\scriptCompile.cpp ^
 %SRC%\secure.cpp ^
 %SRC%\spellcheck.cpp ^
 %SRC%\systemVariables.cpp ^
 %SRC%\tagger.cpp ^
 %SRC%\testing.cpp ^
 %SRC%\textUtilities.cpp ^
 %SRC%\tokenSystem.cpp ^
 %SRC%\topicSystem.cpp ^
 %SRC%\userCache.cpp ^
 %SRC%\userSystem.cpp ^
 %SRC%\variableSystem.cpp ^
 %SRC%\zif.cpp

:: ============================================================
:: LINKER FLAGS
:: -shared              produce a DLL
:: -Wl,--out-implib     generate the import .lib alongside the .dll
:: -lws2_32             Windows sockets
:: -liphlpapi           IP helper API (GetAdaptersInfo used by CS)
:: ============================================================
set LDFLAGS=-shared ^
 -Wl,--out-implib,%OUT%\%DLL_NAME%.lib ^
 -L"%OUT%" ^
 -lws2_32 -liphlpapi

echo [1/2] Compiling sources...
echo.

"%GPP%" %CFLAGS% %DEFINES% %SOURCES% %LDFLAGS% -o "%OUT%\%DLL_NAME%.dll"

if errorlevel 1 (
    echo.
    echo =============================================
    echo   BUILD FAILED — see errors above
    echo =============================================
    exit /b 1
)

echo.
echo [2/2] Build successful!
echo.
echo Output files:
echo   %OUT%\%DLL_NAME%.dll     <- DLL to ship with your application
echo   %OUT%\%DLL_NAME%.lib     <- Import library to link against
echo   chatscript_api.h          <- Public header to include
echo.
echo ====================================================
echo   Required runtime files (copy next to your .exe):
echo   BINARIES\libcurl.dll
echo   BINARIES\zlib1.dll
echo   And the data directories: LIVEDATA\, TOPIC\, USERS\, LOGS\
echo ====================================================
echo.
