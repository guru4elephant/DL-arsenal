
#include <stdio.h>

int main(int argc, char **argv) {
  FILE *fp = popen("awk '{print $1}'", "w");
  int i = 0;

  for (i = 0; i < 10; i++) {
    fprintf(fp, "Count = %d\n", i);
  }

  pclose(fp);

  return 0;
}
