all: fbm.so lm.so

fbm.so: fbm.c
	gcc -o fbm.so -shared fbm.c -L/opt/local/lib -lgsl

lm.so: lm.c astable/astable.c astable/pqueue.c astable/interpolation.c
	gcc -o lm.so -shared lm.c astable/astable.c astable/pqueue.c astable/interpolation.c -L/opt/local/lib -lgsl

clean:
	rm fbm.so lm.so
