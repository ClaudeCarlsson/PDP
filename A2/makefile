CC = mpicc
CFLAGS = -O3 -Wall -Wextra
TARGET = matrix_multiply
OBJS = matrix_multiply.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(OBJS) $(TARGET)
