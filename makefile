CFLAGS = -std=c++20 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

VulkanTest: main.cc
	g++ $(CFLAGS) -o VulkanTest main.cc $(LDFLAGS) -D NDEBUG

debug: main.cc
	g++ $(CFLAGS) -o VulkanTest main.cc $(LDFLAGS) -g

.PHONY: test clean

test: VulkanTest
	./VulkanTest

run: VulkanTest
	./VulkanTest

clean:
	rm -f VulkanTest
