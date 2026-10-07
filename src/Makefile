# Makefile for nasm-htm.core Spatial Pooler Core
# Compiles sp_core.asm to libspcore.so

NASM = nasm
GCC = gcc
NASMFLAGS = -f elf64 -g
GCCFLAGS = -shared -fPIC -mavx2 -O3

TARGET = libspcore.so
OBJS = sp_core.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(GCC) $(GCCFLAGS) -o $@ $(OBJS)

%.o: %.asm
	$(NASM) $(NASMFLAGS) -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
