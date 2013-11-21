CC = gcc
#CFLAGS = -O3 -c -Wall -I/home/monty/local/include
#LDLIBS = -L/home/monty/local/lib -lgsl -lgslcblas -lm
CFLAGS = -O3 -c -Wall -I/opt/local/include
LDLIBS = -L/opt/local/lib -lgsl -lgslcblas -lm

PROG = astable

all: $(PROG) test_cdf

$(PROG): main.o astable.o interpolation.o pqueue.o
	$(CC) main.o astable.o interpolation.o pqueue.o $(LDLIBS) -o $(PROG)

test_cdf: test_cdf.o astable.o
	$(CC) test_cdf.o astable.o $(LDLIBS) -o test_cdf

#test_interp: interpolation.o pqueue.o astable.o
#	gcc -g -I/opt/local/include -L/opt/local/lib interpolation.c astable.c pqueue.c -o interp  -lgsl -lm

test_autocdf: interpolation.c pqueue.c astable.c test_autocdf.c
	gcc -g -I/opt/local/include -L/opt/local/lib test_autocdf.c interpolation.c astable.c pqueue.c -o test_cdf -lgsl -lm

test-cdf:
	@echo "alpha=2 max error: "
	cat data/alpha200.txt | test_cdf 2
	@echo "alpha=1.99 max error: "
	cat data/alpha199.txt | test_cdf 1.99
	@echo "alpha=1.95 max error: "
	cat data/alpha195.txt | test_cdf 1.95
	@echo "alpha=1.90 max error: "
	cat data/alpha190.txt | test_cdf 1.90
	@echo "alpha=1.80 max error: "
	cat data/alpha180.txt | test_cdf 1.80
	@echo "alpha=1.70 max error: "
	cat data/alpha170.txt | test_cdf 1.70
	@echo "alpha=1.60 max error: "
	cat data/alpha160.txt | test_cdf 1.60
	@echo "alpha=1.50 max error: "
	cat data/alpha150.txt | test_cdf 1.50
	@echo "alpha=1.40 max error: "
	cat data/alpha140.txt | test_cdf 1.40
	@echo "alpha=1.30 max error: "
	cat data/alpha130.txt | test_cdf 1.30
	@echo "alpha=1.20 max error: "
	cat data/alpha120.txt | test_cdf 1.20
	@echo "alpha=1.10 max error: "
	cat data/alpha110.txt | test_cdf 1.10
	@echo "alpha=1 max error: "
	cat data/alpha100.txt | test_cdf 1
	@echo "alpha=0.90 max error: "
	cat data/alpha090.txt | test_cdf 0.90
	@echo "alpha=0.80 max error: "
	cat data/alpha080.txt | test_cdf 0.80
	@echo "alpha=0.70 max error: "
	cat data/alpha070.txt | test_cdf 0.70
	@echo "alpha=0.60 max error: "
	cat data/alpha060.txt | test_cdf 0.60
	@echo "alpha=0.50 max error: "
	cat data/alpha050.txt | test_cdf 0.50
	@echo "alpha=0.40 max error: "
	cat data/alpha040.txt | test_cdf 0.40
	@echo "alpha=0.30 max error: "
	cat data/alpha030.txt | test_cdf 0.30
	@echo "alpha=0.20 max error: "
	cat data/alpha020.txt | test_cdf 0.20
	@echo "alpha=0.10 max error: "
	cat data/alpha010.txt | test_cdf 0.10

main.o: main.c
	$(CC) $(CFLAGS) main.c

astable.o: astable.c
	$(CC) $(CFLAGS) astable.c

test_cdf.o: test_cdf.c
	$(CC) $(CFLAGS) test_cdf.c

interpolation.o: interpolation.c
	$(CC) $(CFLAGS) interpolation.c

pqueue.o: pqueue.c pqueue.h
	$(CC) $(CFLAGS) pqueue.c

clean:
	rm -f $(PROG) main.o astable.o test_cdf test_cdf.o test_interp
