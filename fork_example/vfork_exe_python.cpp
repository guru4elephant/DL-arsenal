
#include <stdio.h>
#include <unistd.h>
#include <thread>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include "common/shell.h"
#include "common/fs.h"
#include "vfork_python.h"



void Fork::read_file() {
  int id = thread_id;
  std::ifstream fin("filelist.txt");
  std::vector<std::string> files;
  std::string line;
  while (getline(fin, line)) {
    files.push_back(line);
  }
  int err_no = 0;
  char buf[1024];
  for (int i = 0; i < files.size(); ++i) {
    if ((i + 1) % (id + 1) == 0) {
      snprintf(buf, 1024,
        "data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shu\
ffled/%s", files[i].c_str());
      std::string path = std::string(buf);
      /*
      std::shared_ptr<FILE> fp =
          paddle::ps::fs_open_read(path, &err_no, "/home/users/dongdaxiang/paddle_whls/refactor/paddle_release_home/python/bin/python word2vec_data_gen.py");
      */
      std::shared_ptr<FILE> fp =
          paddle::ps::fs_open_read(path, &err_no, "cat");
      std::string result;
      paddle::ps::LineFileReader reader;
      std::vector<std::string> lines;
      while (reader.getline(&*(fp.get()))) {
        lines.push_back(reader.get());
        if (lines.size() > 1000) {
          for (auto &ll : lines) {
            std::cout << ll << std::endl;
          }
        }
      }
      break;
    }
  }
}

int main() {
  std::vector<std::shared_ptr<Fork> > forks;
  for (int i = 0; i < 10; ++i) {
    std::shared_ptr<Fork> ff(new Fork());
    ff->set_thread_id(i);
    forks.push_back(ff);
  }
  std::vector<std::thread> threads;
  for (int i = 0; i < 10; ++i) {
    threads.push_back(std::thread(&Fork::read_file, forks[i].get()));
  }

  for (int i = 0; i < 10; ++i) {
    threads[i].detach();
  }
}
