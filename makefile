fbm.so: fbm.c
	gcc -o fbm.so -shared fbm.c -L/opt/local/lib -lgsl

clean:
	rm fbm.so
