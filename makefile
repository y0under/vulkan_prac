CFLAGS = -std=c++20 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi
LIB_LINKERS := ~/VulkanSDK/1.3.275.0/macOS/lib

VulkanTest: main.cc
	g++ $(CFLAGS) -o VulkanTest main.cc $(LDFLAGS) -rpath $(LIB_LINKERS) -D NDEBUG

debug: main.cc
	# g++ $(CFLAGS) -o VulkanTest main.cc $(LDFLAGS) -g
	g++ $(CFLAGS) -o VulkanTest main.cc $(LDFLAGS) -rpath $(LIB_LINKERS)

.PHONY: test clean

test: VulkanTest
	./VulkanTest

run: VulkanTest
	./VulkanTest

clean:
	rm -f VulkanTest
