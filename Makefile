# Makefile

EXE=d2q9-bgk

# CC=mpiicc
# CFLAGS= -std=c99 -Wall -Ofast -qopt-report=1 -mtune=native -qopenmp

CC=mpicc
CFLAGS= -std=c99 -Wall -Ofast -mtune=native -fopenmp -g

# CC=mpicc
# CFLAGS= -std=c99 -Wall -Ofast -mtune=native -fopenmp -fno-tree-vectorize

# CC=mpicc
# CFLAGS= -std=c99 -Wall -Ofast -mtune=native -fopenmp -fopt-info-vec-optimized


FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
REF_FINAL_STATE_FILE=check/128x128.final_state.dat
REF_AV_VELS_FILE=check/128x128.av_vels.dat

all: $(EXE)

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

check:
	python3 check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

clean:
	rm -f $(EXE)
