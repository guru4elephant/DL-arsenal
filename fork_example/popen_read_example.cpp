
#include <string>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <array>

int main() {

  std::string command("ls /home/users/dongdaxiang/github_develop/ 2>&1");
  std::array<char, 128> buffer;
  std::string result;

  std::cout << "Opening reading pipe" << std::endl;
  FILE* pipe = popen(command.c_str(), "r");

  if (!pipe) {
    std::cerr << "Couldn't start command." << std::endl;
    return 0;
  }

  while (fgets(buffer.data(), 128, pipe) != NULL) {
    std::cout << "Reading..." << std::endl;
    result += buffer.data();
  }
  auto returnCode = pclose(pipe);

  std::cout << result << std::endl;
  std::cout << returnCode << std::endl;

  return 0;
}
