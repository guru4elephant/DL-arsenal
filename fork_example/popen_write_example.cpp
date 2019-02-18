
#include <string>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <array>

int main() {

  std::string command("awk '{print $3}'");
  std::cout << "Opening writing pipe" << std::endl;
  FILE* pipe = popen(command.c_str(), "w");

  if (!pipe) {
    std::cerr << "Couldn't start command." << std::endl;
    return 0;
  }

  for (int i = 0; i < 10; ++i) {
    fprintf(pipe, "Count = %d\n", i);
  }

  pclose(pipe);
  return 0;
}
