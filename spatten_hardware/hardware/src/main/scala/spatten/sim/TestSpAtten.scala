package spatten.sim


import com.github.tototoshi.csv._

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinal.core.sim._

import spatten._

import java.io._
import scala.util.Try

object TestSpAtten {
    import TopKLatencyModel._

    def main(args: Array[String]): Unit = {
        val axiConfig = Axi4Config(32, 256, 
                useId     = false,
                useRegion = false,
                useBurst  = false,
                useLock   = false,
                useCache  = false,
                useSize   = false,
                useQos    = false,
                useLen    = false,
                useLast   = true,
                useResp   = false,
                useProt   = false,
                useStrb   = false)
        
        val reader = CSVReader.open(new File(args(0)))
        val numBufferLines  = args(1).toInt
        val topKParallelism = args(2).toInt
        val numMultipliers  = args(3).toInt
        val bandwidthDownsample = if (args.size >= 5) args(4).toInt else 1

        println(s"Task file: ${args(0)}")
        println(s"numBufferLines=${numBufferLines}, topKParallelism=${topKParallelism}, numMultipliers=${numMultipliers}")
        println(s"bandwidth downsample=${bandwidthDownsample}")

        val requestLimit = Try(System.getProperty("spatten.requestLimit").toInt).getOrElse(Int.MaxValue)
        if (requestLimit < Int.MaxValue) {
            println(s"Limit to ${requestLimit} requests")
        }

        val selectIteration = Try(System.getProperty("spatten.selectIteration").toInt).getOrElse(-1)
        if (selectIteration >= 0) {
            println(s"Select ${selectIteration}-th iteration")
        }

        val reqs = reader.allWithHeaders().filter(x => 
            selectIteration < 0 || Try(x("iteration_id").toInt == selectIteration).getOrElse(false)
        ).map(map => new {
            val FORCE_NO_PRUNING       = false
            val FORCE_LOCAL_PRUNING    = false
            val FORCE_8B_QUANTIZATION  = false
            val FORCE_NO_QUANTIZATION  = false
            
            var quant_key_bit = map("quant_key_bit").toInt
            var quant_val_bit = map("quant_value_bit").toInt
            var key_fetch_num = if (FORCE_NO_PRUNING) map("sentence_length_L").toInt else map("key_fetch_num").toInt
            var val_fetch_num = if (FORCE_NO_PRUNING) map("sentence_length_L").toInt else map("value_fetch_num").toInt
            var if_requant    = map("if_requant").toBoolean

            if (FORCE_LOCAL_PRUNING) {
                key_fetch_num = (val_fetch_num * map("sentence_length_L").toLong / key_fetch_num).toInt
                val_fetch_num = key_fetch_num
            }

            // TODO: support quant_key_bit = 12
            if (quant_key_bit == -1) {
                quant_key_bit = 8
                if_requant = true
            } else if (quant_key_bit == 10 || quant_key_bit == 12) {
                quant_key_bit = 8
                if_requant = true
            }

            if (quant_val_bit == -1) {
                quant_val_bit = 8
            } else if (quant_val_bit == 10 || quant_val_bit == 12) {
                quant_val_bit = 8
            }

            if (FORCE_8B_QUANTIZATION) {
                quant_key_bit = 8
                quant_val_bit = 8
                if_requant = false
            }

            if (FORCE_NO_QUANTIZATION) {
                quant_key_bit = 32
                quant_val_bit = 32
                if_requant = false
            }
        }).slice(0, requestLimit)

        val config = SpAttenConfigScaledownDRAMOnly(
            SpAttenConfig(
                dramBusConfig = axiConfig, 
                numBufferLines = numBufferLines,
                numMultipliers = numMultipliers
            ), 
            ratio = bandwidthDownsample
        )

        // val drivers = Array.tabulate(config.numDRAMChannel) { i => new DRAMSimDriver(s"chan${i}") }

        // drivers.foreach { _.init() }

        val cwd = System.getProperty("user.dir")
        val withWaveform = System.getProperty("spatten.withWaveform") == "1"

        if (withWaveform) {
            println("Waveform enabled")
        }

        val simCompiled = (if (withWaveform) SimConfig.withFstWave else SimConfig)
            .withConfig(SpinalConfig(
                // verbose = true, 
                bitVectorWidthMax = 32768, 
                defaultConfigForClockDomains = ClockDomainConfig(resetKind = SYNC)
            ))
            .allOptimisation
            .addSimulatorFlag("-Wno-UNSIGNED")
            .addSimulatorFlag("--threads 8")
            .addSimulatorFlag("--trace-threads 2")
            .addRtl("dpi/DRAMSimDPIDriver.sv")
            .addSimulatorFlag(s"-LDFLAGS -L${cwd}/dpi/ -LDFLAGS -l:DRAMSimDPIDriverRamulator2.so")
            .compile({
                val dut = new SpAttenSim(config)
                dut.spatten_inst.controller_inst.buffer_key.data_ram_insts.foreach { _.ram_insts.foreach { inst => 
                    inst.areaCounter.count_read.value.simPublic()
                    inst.areaCounter.count_write.value.simPublic()
                }}
                dut.spatten_inst.controller_inst.buffer_val.data_ram_insts.foreach { _.ram_insts.foreach { inst => 
                    inst.areaCounter.count_read.value.simPublic()
                    inst.areaCounter.count_write.value.simPublic()
                }}
                dut.spatten_inst.controller_inst.mat_fetcher_inst.stageLoadData.subareaRespFifoCounters.foreach { area => 
                    area.count_read.value.simPublic()
                    area.count_write.value.simPublic()
                }
                dut
            })

        simCompiled.doSimUntilVoid { dut => 

            println("Before Simulation")

            val CLOCK_PERIOD = 1000

            dut.clockDomain.forkStimulus(CLOCK_PERIOD)
            // (drivers, dut.io.dram).zipped.foreach { _.drive(_, dut.clockDomain) }

            println("Start Simulation")
            
            // SimTimeout(10000)

            var batch_part_id = 0

            def nextCSVReq(id: Int) = {
                val req = reqs(id)

                dut.io.req.valid #= true
                dut.io.req.query.foreach { _.randomize() }

                val metadata = dut.io.req.metadata
                metadata.profile_key.bit_count #= req.quant_key_bit
                metadata.profile_key.fused_mat #= (if (req.quant_key_bit == 6) 2 else 1)
                metadata.profile_val.bit_count #= req.quant_val_bit
                metadata.profile_val.fused_mat #= (if (req.quant_val_bit == 6) 2 else 1)
                metadata.size_d #= 64
                // metadata.key_fetch_num         #= (req.key_fetch_num + 31) / 32 * 32
                // metadata.val_fetch_num         #= (req.val_fetch_num + 31) / 32 * 32
                metadata.key_fetch_num         #= (req.key_fetch_num + 7) / 8 * 8
                metadata.val_fetch_num         #= (req.val_fetch_num + 7) / 8 * 8
                metadata.key_base_addr         #= 0xAB10000 + 0x10000 * id
                metadata.key_requant_base_addr #= 0xAB20000 + 0x10000 * id
                metadata.val_base_addr         #= 0xCD10000 + 0x10000 * id
                metadata.universal_id #= id

                
                if (req.if_requant) {
                    metadata.thres_requantize #= metadata.thres_requantize.maxValue
                } else {
                    metadata.thres_requantize #= metadata.thres_requantize.minValue
                }

                if (req.quant_key_bit == 6) {
                    dut.io.req.last #= batch_part_id % 2 == 1
                    metadata.topk_latency #= (if (batch_part_id % 2 == 0) topKLatency(req.key_fetch_num, req.val_fetch_num, topKParallelism) else 0)
                    batch_part_id += 1
                } else {
                    dut.io.req.last #= true
                    metadata.topk_latency #= topKLatency(req.key_fetch_num, req.val_fetch_num, topKParallelism)
                    batch_part_id = 0
                }
            }

            dut.io.req.valid  #= false
            dut.io.resp.ready #= true

            fork {
                var cntPop = 0
                while (cntPop < reqs.size) {
                    dut.clockDomain.waitSampling()
                    if (dut.io.resp.valid.toBoolean) {
                        println(s"${simTime()}: Got response ${dut.io.resp.metadata.universal_id.toBigInt} (${cntPop}) of ${reqs.size} reqs")
                        cntPop += 1
                    }
                }
                val fsummary = new FileWriter("summary.txt", true)
                try {
                    fsummary.write(s"${args(0)},${simTime()/CLOCK_PERIOD},${config.numBufferLines},${topKParallelism}\n")
                } finally {
                    fsummary.close()
                }

                val fsramstats = new FileWriter("sram-stats.txt")
                try {
                    dut.spatten_inst.controller_inst.buffer_key.data_ram_insts.foreach { _.ram_insts.foreach { inst => 
                        val cntRead  = inst.areaCounter.count_read.value.toBigInt
                        val cntWrite = inst.areaCounter.count_write.value.toBigInt
                        val depth    = inst.areaCounter.depth
                        val width    = inst.areaCounter.width
                        fsramstats.write(s"KEYBUF 0 0 ${width} ${depth*width} 1 ${cntRead * width * 1000000000 / (simTime() / CLOCK_PERIOD)} ${cntWrite * width * 1000000000 / (simTime() / CLOCK_PERIOD)}\n")
                    }}
                    dut.spatten_inst.controller_inst.buffer_val.data_ram_insts.foreach { _.ram_insts.foreach { inst => 
                        val cntRead  = inst.areaCounter.count_read.value.toBigInt
                        val cntWrite = inst.areaCounter.count_write.value.toBigInt
                        val depth    = inst.areaCounter.depth
                        val width    = inst.areaCounter.width
                        fsramstats.write(s"VALBUF 0 0 ${width} ${depth*width} 1 ${cntRead * width * 1000000000 / (simTime() / CLOCK_PERIOD)} ${cntWrite * width * 1000000000 / (simTime() / CLOCK_PERIOD)}\n")
                    }}
                    dut.spatten_inst.controller_inst.mat_fetcher_inst.stageLoadData.subareaRespFifoCounters.foreach { area => 
                        val cntRead  = area.count_read.value.toBigInt
                        val cntWrite = area.count_write.value.toBigInt
                        val depth    = area.depth
                        val width    = area.width
                        fsramstats.write(s"RESPFIFO 0 0 ${width} ${depth*width} 1 ${cntRead * width * 1000000000 / (simTime() / CLOCK_PERIOD)} ${cntWrite * width * 1000000000 / (simTime() / CLOCK_PERIOD)}\n")
                    } 
                } finally {
                    fsramstats.close()
                }
                
                simSuccess()
            }

            for (i <- 0 until reqs.size) {
                do {
                    dut.clockDomain.waitSampling()
                } while(!dut.io.req.ready.toBoolean && dut.io.req.valid.toBoolean)

                nextCSVReq(i)
            }

            do {
                dut.clockDomain.waitSampling()
            } while(!dut.io.req.ready.toBoolean && dut.io.req.valid.toBoolean)

            dut.io.req.valid #= false
        }

        // drivers.foreach { _.exit() }
    }
}