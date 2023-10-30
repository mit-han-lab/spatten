CXXFLAGS := -O2 -g -std=c++20 -I$(shell dirname $(shell which verilator))/../share/verilator/include/vltstd/ -I../third_party/ramulator2/src/ -I../third_party/ramulator2/ext/spdlog/include/ -I../third_party/ramulator2/ext/yaml-cpp/include/ 
LDFLAGS := -L../third_party/ramulator2/ -lramulator

DRAMSimDPIDriverRamulator2.so: DRAMSimDPIDriverRamulator2.cpp
	$(CXX) -fPIC -shared DRAMSimDPIDriverRamulator2.cpp -o DRAMSimDPIDriverRamulator2.so $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f DRAMSimDPIDriverRamulator2.so