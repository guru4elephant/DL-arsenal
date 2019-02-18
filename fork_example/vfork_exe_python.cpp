
#include <stdio.h>
#include <unistd.h>
#include <thread>
#include <vector>
#include <iostream>
#include "common/shell.h"
#include "common/fs.h"

void read_file(int id) {
  pid_t pid;
  int thread_local_int = 0;
  printf("thread id: %d\n", id);
  int err_no = 0;
  char buf[1024];
  snprintf(buf, 1024, "%d.txt", id);
  std::string path = std::string(buf);
  std::shared_ptr<FILE> fp =
      paddle::ps::fs_open_read(path, &err_no, "python print_with_tail.py");
  std::array<char, 128> buffer;
  std::string result;
  while (fgets(buffer.data(), 128, fp.get()) != NULL) {
    result += buffer.data();
  }
  std::cout << result << std::endl;
}

int main() {
  std::vector<std::thread> threads;
  for (int i = 0; i < 10; ++i) {
    threads.push_back(std::thread(&read_file, i));
  }

  for (int i = 0; i < 10; ++i) {
    threads[i].join();
  }
}
