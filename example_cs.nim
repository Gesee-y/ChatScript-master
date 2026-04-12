import chatscript, os, strutils

proc main() =
  echo "--- ChatScript Nim Bridge Demo ---"
  
  # 1. Initialisation
  # On passe "local" pour charger cs_init.txt si présent, ou rien par défaut.
  # On spécifie le dossier racine actuel "."
  if not initChatScript(@["local"], getCurrentDir()):
    echo "Erreur: Impossible d'initialiser ChatScript."
    return

  echo "[Système] Moteur chargé avec succès."

  let username = "NimUser"
  let botName = "" # Utilise le bot par défaut
  
  # 2. Premier contact (Greeting)
  # Envoyer une chaîne vide déclenche le message de bienvenue du bot.
  var (volley, response) = chat(username, botName, "")
  echo "\n[Bot] ", response

  # 3. Boucle interactive
  while true:
    stdout.write("\n[" & username & "] > ")
    let input = stdin.readLine().strip()
    
    if input in ["quit", "exit", "q"]:
      break
      
    if input == "": continue

    # Appel au moteur via le wrapper
    (volley, response) = chat(username, botName, input)
    
    if volley == CS_PENDING_RESTART:
      echo "[Système] Le moteur demande un redémarrage (Hot swap / Reset)."
      break
      
    echo "[Bot] ", response

  # 4. Fermeture propre
  echo "\n[Système] Fermeture..."
  CloseSystem()
  echo "Terminé."

if isMainModule:
  main()
