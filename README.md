
# SpAtten: Sparse Attention with Token Pruning and Head Pruning in Large Language Models


[[paper](https://arxiv.org/abs/2012.09852)] [[slides](https://www.dropbox.com/s/z189gu92h7uy7yt/SpAtten-for-long-video-no-animation.pdf?dl=0)] [[video](https://www.youtube.com/watch?v=Cln8hFxM9Do)] [[website](https://hanlab.mit.edu/projects/spatten)] 

<!-- ![schemes](figures/schemes.png) -->


## TL;DR
We propose sparse attention (SpAtten) with **KV token pruning, local V pruning, head pruning, and KV progressive quantization** to improve LLM efficiency.

## News
- SpAtten and SpAtten-Chip won the 1st Place Award at 2023 DAC University Demo.
- SpAtten is spotlighted on [MIT Homepage](http://mit.edu/spotlight/streamlining-sentence-analysis).
- SpAtten is covered by [MIT News](https://news.mit.edu/2021/language-learning-efficiency-0210).
- [2023/10] SpAtten-LLM and SpAtten hardware released.


## Abstract
We present SpAtten, an efficient algorithm-architecture co-design that leverages token sparsity, head sparsity, and quantization opportunities to reduce the attention computation and memory access. Inspired by the high redundancy of human languages, we propose the novel KV token pruning to prune away unimportant tokens in the sentence. We also propose head pruning to remove unessential heads. Cascade pruning is fundamentally different from weight pruning since there is no trainable weight in the attention mechanism, and the pruned tokens and heads are selected on the fly. To efficiently support them on hardware, we design a novel top-k engine to rank token and head importance scores with high throughput. Furthermore, we propose KV progressive quantization that first fetches MSBs only and performs the computation; if the confidence is low, it fetches LSBs and recomputes the attention outputs, trading computation for memory reduction.

### Token pruning for classification task:
![basic_idea](assets/corrected-teaser.png)

### Token pruning for generation task:
![schemes](assets/fig_gpt.jpeg)



## SpAtten Usage

### Environment Setup

```bash
conda create -yn spatten python=3.8
conda activate spatten

pip install torch torchvision torchaudio
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece

python setup.py develop
```

### Run SpAtten Llama Chatbot

```bash
CUDA_VISIBLE_DEVICES=0 python run_spatten_llama.py  --enable_spatten
```

## SpAtten Hardware Usage
This repo also contains the RTL-level simulation model of SpAtten in `spatten_hardware/hardware/` for accurate performance evaluation on generative models like GPT-2 and a fast behavior model in `spatten_hardware/simulator` for quick evaluation on BERT.

### Running RTL simulation for SpAtten
#### Prerequisites
- [Verilator](https://www.veripool.org/verilator/) version [v4.218](https://github.com/verilator/verilator/releases/tag/v4.218)

  Note that there is a known [issue](https://github.com/verilator/verilator/issues/4424) with the latest Verilator that may cause random assertion failure on startup of simulation. Use v4.218 as a workaround.
- [SBT](https://www.scala-sbt.org/)
- C/C++ build tools for verilator and ramulator. `gcc,g++>=12`, `cmake`
- Workload information in CSV format. There are some examples in hardware/workloads

#### Quick Start
Build the ramulator2
```
$ cd spatten_hardware/hardware/third_party/ramulator2
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
$ make
$ cd ../../../..
```
Build the Verilog (DPI) interface for ramulator
```
$ cd hardware/dpi
$ make
$ cd ../../..
```
Use the python script to run SpAtten simulation with a workload file
```
python3 run_spatten_hardware.py hardware/workloads/summary-gpt2-small-wikitext2-per8.csv
```
The evaluation results is located in the working directory `spatten.workdir/summary.txt`

### SpAtten Hardware Architecture
![spatten arch](https://assets-global.website-files.com/64f4e81394e25710d22d042e/6515ab835deaead9f35609ac_spatten_arch.jpeg)

SpAtten uses a specialized pipeline to support efficient attention and focus on memory traffic optimizations for decoding models like GPT2 and LLMs. 

This repo contains the following major modules in SpAtten, and the main pipeline implementation is in [SpAttenController.scala](./spatten_hardware/hardware/src/main/scala/spatten/SpAttenController.scala).

- A parallelized top-k unit (10) that dynamically decides the values to fetch: [TopK.scala](./spatten_hardware/hardware/src/main/scala/spatten/TopK.scala), which uses [QuickSelect.scala](./spatten_hardware/hardware/src/main/scala/spatten/utils/QuickSelect.scala) to choose the k-th largest element from attention prob
- A matrix fetcher ((3) and (6) in the figure) that loads the key/value matrix from DRAM and convert the bitwidth when necessary: [MatrixFetcher.scala](./spatten_hardware/hardware/src/main/scala/spatten/MatrixFetcher.scala)
- The Q\*K (7) and Prob\*V (11) unit and the corresponding key / value buffers: [DotProduct.scala](./spatten_hardware/hardware/src/main/scala/spatten/DotProduct.scala), [MultiplyValue.scala](./spatten_hardware/hardware/src/main/scala/spatten/MultiplyValue.scala), [Buffer.scala](./spatten_hardware/hardware/src/main/scala/spatten/Buffer.scala), [BufferManager.scala](./spatten_hardware/hardware/src/main/scala/spatten/BufferManager.scala)
- A progressive quantization module (9) to decide whether or not to load the LSBs of keys: [RequantDecision.scala](./spatten_hardware/hardware/src/main/scala/spatten/RequantDecision.scala)


## TODOs
We will release the code and data soon, please stay tuned.

- [ ] Release core code of SpAtten, including Llama-2, MPT, Falcon, and Pythia.
- [ ] Release SpAtten perplexity evaluation code
- [ ] Release SpAtten Llama Chatbot demo.
- [ ] Release a docker image for hardware simulation.


## Citation

If you find SpAtten useful or relevant to your project and research, please kindly cite our paper:

```bibtex
@article{wang2021spatten,
        title={SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning},
        author={Wang, Hanrui and Zhang, Zhekai and Han, Song},
        journal={HPCA},
        year={2021}
        }
```
<!-- 
```bibtex
@article{wang2021spattenllm,
        title={SpAtten-LLM: Sparse Attention with Token Pruning and Head Pruning in Large Language Models},
        author={Wang, Hanrui and Xiao, Guangxuan and Yang, Shang and Tang, Haotian, and Zhang, Zhekai and Han, Song},
        journal={Technical Report},
        year={2023}
        }
``` -->

