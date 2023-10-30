#include <map>
#include <random>
#include <svdpi.h>

#include "base/base.h"
#include "base/request.h"
#include "base/config.h"
#include "frontend/frontend.h"
#include "memory_system/memory_system.h"


class DRAMSimDPIDriver {
public:
    static constexpr size_t MAX_PENDING_REQS = 16;
    using addr_t = unsigned long long;

    DRAMSimDPIDriver(int dataWidth, int seed) : dataWidth(dataWidth), frontend(nullptr), memorysystem(nullptr), numPendingReqs(0), rng(seed) {}
    DRAMSimDPIDriver(const DRAMSimDPIDriver &) = delete;
    ~DRAMSimDPIDriver() {
        if (frontend) {
            frontend->finalize();
        }
        if (memorysystem) {
            memorysystem->finalize();
        }
    }

    void init() {
        if (dataWidth % 8 != 0) {
            throw std::invalid_argument("dataWidth must be multiplies of 8");
        }
        YAML::Node config = Ramulator::Config::parse_config_file("ramulator_config.yaml", {});
        frontend = Ramulator::Factory::create_frontend(config);
        memorysystem = Ramulator::Factory::create_memory_system(config);
        frontend->connect_memory_system(memorysystem);
        memorysystem->connect_frontend(frontend);
        numPendingReqs = 0;
    }

    bool pushRequest(addr_t addr, const svBitVecVal* data, svBit is_write) {
        if (addr % (dataWidth / 8) != 0) {
            throw std::invalid_argument("Unaligned access is not allowed");
        }

        if (numPendingReqs >= MAX_PENDING_REQS) {
            return false;
        }

        RequestContext ctx;
        ctx.addr = addr;
        std::copy(data, data + SV_PACKED_DATA_NELEMS(dataWidth), std::back_inserter(ctx.data));
        ctx.is_write = is_write;

        if (!frontend->receive_external_requests(0, addr, 0, [reqctx = std::move(ctx), this](Ramulator::Request &req) {
            auto ctx = reqctx;
            if (ctx.is_write) {
                writeMemory(ctx.addr, ctx.data);
            } else {
                ctx.data = readMemory(ctx.addr);
            }
            completedReqs.push(ctx);
        })) {
            return false;
        }

        numPendingReqs++;
        return true;
    }

    bool popResponse(addr_t *addr, svBitVecVal* data, svBit* is_write) {
        if (completedReqs.empty()) {
            return false;
        }

        numPendingReqs--;

        RequestContext ctx = std::move(completedReqs.front());
        completedReqs.pop();

        *addr = ctx.addr;
        std::copy(ctx.data.begin(), ctx.data.end(), data);
        *is_write = ctx.is_write;

        return true;
    }

    void tick() {
        memorysystem->tick();
    }

    std::vector<svBitVecVal> readMemory(addr_t addr) {
        if (!memContent.contains(addr)) {
            memContent[addr].resize(SV_PACKED_DATA_NELEMS(dataWidth));
            std::uniform_int_distribution<svBitVecVal> dist(std::numeric_limits<svBitVecVal>::min(), std::numeric_limits<svBitVecVal>::max());
            std::generate(memContent[addr].begin(), memContent[addr].end(), [this, &dist]() { return dist(rng); });
        }
        return memContent.at(addr);
    }
    void writeMemory(addr_t addr, std::vector<svBitVecVal> data) {
        assert(data.size() == SV_PACKED_DATA_NELEMS(dataWidth));
        memContent[addr] = std::move(data);
    }

private:
    struct RequestContext {
        addr_t addr;
        std::vector<svBitVecVal> data;
        svBit is_write;
    };

private:
    const int dataWidth;
    Ramulator::IFrontEnd *frontend;
    Ramulator::IMemorySystem *memorysystem;
    size_t numPendingReqs;
    std::queue<RequestContext> completedReqs;

    std::map<addr_t, std::vector<svBitVecVal>> memContent;
    std::mt19937 rng;
};

static long long nextHandle = 0x12340000;
static std::map<long long, std::unique_ptr<DRAMSimDPIDriver>> objs;

extern "C" {

int DRAMSimDPIDriverCreate(int dataWidth, int seed, long long *obj) {
    *obj = nextHandle++;
    objs[*obj] = std::make_unique<DRAMSimDPIDriver>(dataWidth, seed);
    objs.at(*obj)->init();
    return 0;
}

void DRAMSimDPIDriverDestroy(long long obj) {
    objs.erase(obj);
}

int DRAMSimDPIDriverPopResponse(long long obj, unsigned long long *addr, svBitVecVal* data, svBit* is_write) {
    if (!objs.at(obj)->popResponse(addr, data, is_write)) {
        return -1;
    }
    return 0;
}
int DRAMSimDPIDriverPushRequest(long long obj, unsigned long long  addr, const svBitVecVal* data, svBit is_write) {
    if (!objs.at(obj)->pushRequest(addr, data, is_write)) {
        return -1;
    }
    return 0;
}
int DRAMSimDPIDriverTick(long long obj) {
    objs.at(obj)->tick();
    return 0;
}

}