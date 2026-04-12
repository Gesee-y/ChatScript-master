#include <iostream>
#include <windows.h>
#include "chatscript_api.h"

int main() {
    std::cout << "--- C++ DLL Test ---" << std::endl;
    
    char* args[] = { (char*)"chatscript", (char*)"local" };
    unsigned int res = InitSystem(2, args, NULL, NULL, NULL, NULL, NULL, NULL);
    
    if (res != 0) {
        std::cerr << "InitSystem failed: " << res << std::endl;
        return 1;
    }
    
    std::cout << "Moteur chargé." << std::endl;
    
    char output[CS_OUTPUT_SIZE];
    char user[100]; strcpy(user, "TestUser");
    char bot[100]; strcpy(bot, "");
    char input[100]; strcpy(input, "");
    char ip[100]; strcpy(ip, "127.0.0.1");
    
    int count = PerformChat(user, bot, input, ip, output);
    
    std::cout << "Volley: " << count << std::endl;
    std::cout << "Bot: " << output << std::endl;
    
    CloseSystem();
    return 0;
}
