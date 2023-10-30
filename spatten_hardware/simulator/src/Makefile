SRCS := \
		bert.cpp

OBJDIR := objs
OBJS := $(patsubst %,$(OBJDIR)/%.o,$(basename $(SRCS)))
CXX := g++-10
CXXFLAGS := -std=c++20 -g -O3 -I../third_party/fast-cpp-csv-parser
LDFLAGS := -g -lfmt -pthread

bert: $(OBJS)
	$(CXX) $(OBJS) -o bert $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

clean:
	rm -f objs/*.o bert