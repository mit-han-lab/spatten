#include <iostream>
#include <fstream>
#include <vector>
#include <random>

#include <version>
#ifdef __cpp_lib_format
#include <format>
using std::format;
#else
#include <fmt/format.h>
using fmt::format;
#endif

#include <csv.h>


struct HWParameters {
    size_t numDRAMChannels = 16;
    size_t numMultipliers = 512;
    size_t sizeZE = 4;

    bool useRamulator = false;
};

HWParameters hwParam;

struct SimProfile {
    size_t sizeD;
    size_t sentenceLength;
    size_t fetchKey;
    // size_t fetchVal;
    size_t keyWidth;
    size_t valueWidth;
    bool topK;
    size_t topKNum;

    size_t parallelism() const {
        return hwParam.numMultipliers / sizeD;
    }
    size_t batchSize() const {
        return (fetchKey + parallelism() - 1) / parallelism();
    }
};

size_t simDRAMImpl(size_t nTransaction) {
    if (!hwParam.useRamulator) {
        return nTransaction;
    }

    std::ofstream fdram("dram.trace");
    for (size_t i = 0; i < nTransaction; i++)
        fdram << "0x" << std::hex << i * 32 << " R" << std::endl;
    fdram.flush();
    fdram.close();

    static size_t cntTimes = 0;
    std::ofstream fdram_all("dram_all.trace", std::ios::app);
    for (size_t i = 0; i < nTransaction; i++)
        fdram_all << "0x" << std::hex << cntTimes * 0x1000000 + i * 32 << " R" << std::endl;
    fdram_all.close();
    cntTimes++;

    system("./ramulator HBM-config.cfg --mode=dram dram.trace > /dev/null");
    system("grep 'DRAM cycles' HBM.stats | awk '{print $2}' > dram-result.txt");

    std::ifstream fresult("dram-result.txt");
    size_t cycle;
    fresult >> cycle;
    return cycle;
}

size_t simDRAM(const std::vector<SimProfile> &profiles) {
    size_t nTransPerChannel = 0;
    for (auto &&profile : profiles) {
        size_t keyNumTransaction = profile.batchSize() * profile.parallelism() * profile.sizeD * profile.keyWidth / 8 / 32;
        size_t valNumTransaction = profile.batchSize() * profile.parallelism() * profile.sizeD * profile.valueWidth / 8 / 32;
        nTransPerChannel += (keyNumTransaction + hwParam.numDRAMChannels - 1) / hwParam.numDRAMChannels;
        nTransPerChannel += (valNumTransaction + hwParam.numDRAMChannels - 1) / hwParam.numDRAMChannels;
    }

    return simDRAMImpl(nTransPerChannel);
}

size_t simFillPipeline(const SimProfile &profile) {
    return profile.sentenceLength * profile.batchSize();
}

size_t simDrainPipeline(const SimProfile &profile) {
    size_t latencyKeyMat = 1 + 1 + 9;   // SRAM + Multiplier + Reduction Tree
    size_t latencySoftMax =
        7 +     // Dequantize
        21 +    // e^x
        23 +    // accumulator
        29 +    // divider
        7 +     // quantize
        profile.batchSize();    // FIFO
    size_t latencyRequantize = 1;   // skip requantize
    size_t latencyTopK = 1;         // skip TopK
    size_t latencyValMat = 1 + 1 + 1;   // SRAM + Multiplier + Accumulator
    return latencyKeyMat + latencySoftMax + latencyRequantize + latencyTopK + latencyValMat;
}

size_t simTopK(const SimProfile &profile, int seed) {
    size_t cycle = 0;
    cycle += profile.batchSize();   // fill buffer

    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> distScores;

    std::vector<int> seq(profile.fetchKey);
    for (int &val : seq)
        val = distScores(gen);

    size_t K = profile.topKNum;
    std::vector<int> left, right;
    while (!seq.empty()) {
        std::uniform_int_distribution<int> distPivot(0, seq.size());
        int pivot = seq[distPivot(gen)];

        left.clear();
        right.clear();
        size_t nEqual = 0;

        for (int val : seq) {
            if (val < pivot)
                left.push_back(val);
            else if (val > pivot)
                right.push_back(val);
            else
                nEqual++;
        }

        cycle += seq.size() / hwParam.sizeZE + 1 + 1 + 1;   // latency of Comparators, ZE and FIFO
        cycle += 1; // stateAfterRun

        if (left.size() < K) {
            if (left.size() + nEqual >= K)
                break;
            seq = std::move(left);
        } else {
            seq = std::move(right);
            K -= left.size() + nEqual;
        }
    }

    cycle += (profile.topKNum + profile.parallelism() - 1) / profile.parallelism(); // drain buffer
    return cycle;
}

// size_t sumCycle = 0;

size_t run(const std::vector<SimProfile> &profiles, size_t layerId) {
    size_t cycle = 0;

    size_t cycleDRAM  = simDRAM(profiles);
    size_t cycleFill  = 0;
    size_t cycleDrain = simDrainPipeline(profiles.back());
    size_t cycleTopK = 0;

    for (auto &&profile : profiles) {
        cycleFill += simFillPipeline(profile);
    }
    
    for (int seed = 0; auto &&profile : profiles) {
        if (profile.topK) {
            cycleTopK += simTopK(profile, seed++);
        }
    }

    cycle = cycleDRAM + cycleFill + cycleDrain + cycleTopK;

    std::cout << format("{}\t{}\n", layerId, cycle);

    return cycle;
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        hwParam.numDRAMChannels = std::stoi(argv[1]);
    }
    if (argc > 2) {
        hwParam.numMultipliers = std::stoi(argv[2]);
    }
    if (argc > 3) {
        hwParam.sizeZE = std::stoi(argv[3]);
    }
    if (argc > 4) {
        hwParam.useRamulator = std::stoi(argv[4]) > 0;
    }

    std::cout << format("#DRAM channels = {}\n", hwParam.numDRAMChannels);
    std::cout << format("#Multipliers = {} * 2\n", hwParam.numMultipliers);
    std::cout << format("TopK parallelism = {}\n", hwParam.sizeZE);
    std::cout << format("Use Ramulator = {}\n", hwParam.useRamulator);

    system("rm -f dram_all.trace");

    io::CSVReader<15> in("input.csv");
    in.read_header(io::ignore_extra_column, "layer_id", "head_id", "embedding_length_D", "sentence_length_L", "key_value_query_fetch_num", "quant_key_bit", "quant_value_bit", "quant_query_bit", "auto_requant_thres", "if_requant", "auto_requant_incre", "auto_requant_num", "if_accumulate_importance", "if_topk", "topk");

    std::string layer_id, head_id, embedding_length_D, sentence_length_L, key_value_query_fetch_num, quant_key_bit, quant_value_bit, quant_query_bit, auto_requant_thres, if_requant, auto_requant_incre, auto_requant_num, if_accumulate_importance, if_topk, topk;

    int lastLayerId = 0;
    std::vector<SimProfile> profiles;

    size_t sumCycle = 0;

    while (in.read_row(layer_id, head_id, embedding_length_D, sentence_length_L, key_value_query_fetch_num, quant_key_bit, quant_value_bit, quant_query_bit, auto_requant_thres, if_requant, auto_requant_incre, auto_requant_num, if_accumulate_importance, if_topk, topk)) {
        if (layer_id[0] < '0' || layer_id[0] > '9')
            break;

        int layerId = std::stoi(layer_id);
        if (layerId != lastLayerId) {
            sumCycle += run(profiles, lastLayerId);
            profiles.clear();
            lastLayerId = layerId;
        }

        SimProfile profile;
        profile.sizeD = std::stoi(embedding_length_D);
        profile.sentenceLength = std::stoi(sentence_length_L);
        profile.fetchKey = std::stoi(key_value_query_fetch_num);
        profile.keyWidth = std::stoi(quant_key_bit);
        profile.valueWidth = std::stoi(quant_value_bit);
        profile.topK = if_topk.find("True") != std::string::npos;
        profile.topKNum = std::stoi(topk);

        if (profile.keyWidth == 6 || profile.keyWidth > 8)
            profile.keyWidth = 8;
        if (profile.valueWidth == 6 || profile.valueWidth > 8)
            profile.valueWidth = 8;

        profiles.push_back(profile);

    }
    sumCycle += run(profiles, lastLayerId);

    std::cout << format("SUM\t{}\n", sumCycle);

    return 0;
}