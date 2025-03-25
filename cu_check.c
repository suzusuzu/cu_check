#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_CU(func)                                                \
  do {                                                                \
    CUresult res = (func);                                            \
    if (res != CUDA_SUCCESS) {                                        \
      const char *errName = NULL;                                     \
      const char *errDesc = NULL;                                     \
      cuGetErrorName(res, &errName);                                  \
      cuGetErrorString(res, &errDesc);                                \
      fprintf(stderr, "%s failed: %s %s\n", #func, errName, errDesc); \
      return -1;                                                      \
    }                                                                 \
  } while (0)

const char *getCUprocessState(CUprocessState state) {
  switch (state) {
    case CU_PROCESS_STATE_RUNNING:
      return "CU_PROCESS_STATE_RUNNING";
    case CU_PROCESS_STATE_LOCKED:
      return "CU_PROCESS_STATE_LOCKED";
    case CU_PROCESS_STATE_CHECKPOINTED:
      return "CU_PROCESS_STATE_CHECKPOINTED";
    case CU_PROCESS_STATE_FAILED:
      return "CU_PROCESS_STATE_FAILED";
    default:
      return "OTHER_STATE";
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s [state|lock|checkpoint|restore|unlock] <pid>\n",
            argv[0]);
    return -1;
  }

  const char *subcommand = argv[1];
  int pid = atoi(argv[2]);

  CHECK_CU(cuInit(0));

  if (strcmp(subcommand, "state") == 0) {
    CUprocessState state;
    CHECK_CU(cuCheckpointProcessGetState(pid, &state));
    printf("state: %s\n", getCUprocessState(state));
  } else if (strcmp(subcommand, "thread") == 0) {
    int threadId = 0;
    CHECK_CU(cuCheckpointProcessGetRestoreThreadId(pid, &threadId));
    printf("thread id: %d\n", threadId);
  } else if (strcmp(subcommand, "lock") == 0) {
    CUcheckpointLockArgs args = {
        .timeoutMs = 600000  // 10min timeout
    };
    CHECK_CU(cuCheckpointProcessLock(pid, &args));
    printf("locked successfully\n");
  } else if (strcmp(subcommand, "checkpoint") == 0) {
    CHECK_CU(cuCheckpointProcessCheckpoint(pid, NULL));
    printf("checkpointed successfully\n");
  } else if (strcmp(subcommand, "restore") == 0) {
    CHECK_CU(cuCheckpointProcessRestore(pid, NULL));
    printf("restored successfully\n");
  } else if (strcmp(subcommand, "unlock") == 0) {
    CHECK_CU(cuCheckpointProcessUnlock(pid, NULL));
    printf("unlocked successfully\n");
  } else {
    printf("unknown subcommand: %s\n", subcommand);
    return -1;
  }

  return 0;
}
