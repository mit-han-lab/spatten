package spatten

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._

object MatrixFetcher {
    case class Request[TContext <: Data](config: SpAttenConfig, genContext: HardType[TContext]) extends Bundle {
        val addr     = Vec(UInt(config.dramBusConfig.addressWidth bits), config.numMatrixFetcherChannel)
        val num_addr = UInt(log2Up(config.numMatrixFetcherChannel + 1) bits)

        // the following fields must remain the same for all fragments of a request
        val high_bits = Bool    // indicates that the fetched data should be put to higher bits in buffer
        val size_d    = UInt(log2Up(config.sizeD + 1) bits)
        val profile   = QuantProfile(config)
        val context   = genContext()
    }
    case class Response[TContext <: Data](config: SpAttenConfig, genContext: HardType[TContext]) extends Bundle {
        val data      = Vec(Vec(config.genFix, config.numMultipliers), config.maxFusedMatrix)

        val high_bits = Bool
        val profile   = QuantProfile(config)
        val context   = genContext()
    }
}

class MatrixFetcher[TContext <: Data](genContext: HardType[TContext])(implicit config: SpAttenConfig) extends Component with Stageable {
    import MatrixFetcher._

    val io = new Bundle {
        val bus  = Array.fill(config.numMatrixFetcherChannel) { master(Axi4ReadOnly(config.dramBusConfig)) }
        val req  = slave  Stream(Fragment(Request(config, genContext)))
        val resp = master Stream(Fragment(Response(config, genContext)))
    }

    case class QuantProfileLiteral(bit_count: Int, fused_mat: Int) {
        def asBits = {
            val profile = QuantProfile(config)
            profile.bit_count := bit_count
            profile.fused_mat := fused_mat
            profile.asBits
        }
    }

    type Stage = StageComponent

    // (bit_count, fused_mat)
    val supportedProfiles = Array(
        // (32, 1)
        (4, 1), (6, 2), (8, 1)
        ).map(QuantProfileLiteral.tupled(_))
    val maxSplittedFragments = 8

    val transactionLength = 32

    // split one request into multiple requests when bit_count * size_d > transaction_length (32 bytes)
    //     A0   A1   A2   A3   
    //  => A0   A0+1 A1   A1+1
    //     A2   A2+1 A3   A3+1
    // 
    // bit_count * fused_number * size_d == transaction_length * num_fragments
    // size_d == transaction_length * num_fragments / bit_count / fused_number
    val stageSplitRequest = new Stage("StageSplitRequest") {
        case class RequestWithFragmentInfo() extends Bundle {
            val req           = Request(config, genContext)
            val num_fragments = UInt(log2Up(maxSplittedFragments + 1) bits)
            val fragment_id   = UInt(log2Up(maxSplittedFragments) bits)
            val original_last  = Bool
        }

        val input_req = stageIn(io.req)

        val subareaPrepare = new Area {
            val input = input_req

            when (input.valid) {
                assert(input.num_addr =/= 0)
                assert(input.last || input.num_addr === config.numMatrixFetcherChannel)
            }

            val num_fragments = UInt(log2Up(maxSplittedFragments + 1) bits)
            val fragment_id   = Reg(UInt(log2Up(maxSplittedFragments) bits)) init(0)

            switch (input.profile.asBits) {
                for (profile <- supportedProfiles) {
                    is (profile.asBits) {
                        num_fragments := 0
                        for (i <- 1 to maxSplittedFragments) {
                            val size = transactionLength * 8 * i / profile.bit_count / profile.fused_mat
                            if (size <= input.size_d.maxValue) {
                                when (size === input.size_d) {
                                    num_fragments := i
                                }
                            }
                        }
                        assert(!input.valid || num_fragments =/= 0, "No available num_fragment")
                    }
                }
                default {
                    assert(!input.valid, Seq("Quantization Profile (", input.profile.bit_count, ", ", input.profile.fused_mat, ") is not supported"))
                    num_fragments.assignDontCare()
                }
            }

            val fragmented = Stream(RequestWithFragmentInfo())
            fragmented.valid         := input.valid
            fragmented.req           := input.fragment
            fragmented.num_fragments := num_fragments
            fragmented.fragment_id   := fragment_id
            fragmented.original_last  := input.last
            input.ready := fragmented.ready && (fragment_id === num_fragments - 1)

            when (fragmented.fire) {
                fragment_id := fragment_id + 1
                when (input.fire) {
                    fragment_id := 0
                }
            }

            val output = fragmented.stage
        }

        val subareaSplit = new Area {
            val input = subareaPrepare.output

            val output = input.translateInto(Stream(Fragment(Request(config, genContext)))) { (to, from) => 
                to.last     := from.original_last && from.fragment_id === from.num_fragments - 1
                to.fragment := from.req

                switch (from.num_fragments ## from.fragment_id) {
                    for (i <- 1 to maxSplittedFragments; j <- 0 until i) {
                        is (B(i, widthOf(from.num_fragments) bits) ## B(j, widthOf(from.fragment_id) bits)) {
                            // from.num_fragments === i && from.fragment_id === j

                            to.fragment.addr := Vec.tabulate(config.numMatrixFetcherChannel) { k => 
                                // index in current fragment = k
                                // index in all fragments    = k + j * config.numMatrixFetcherChannel
                                // corresponding index in input = (k + j * config.numMatrixFetcherChannel) / i
                                from.req.addr((k + j * config.numMatrixFetcherChannel) / i) + 
                                    (transactionLength * ((k + j * config.numMatrixFetcherChannel) % i))
                            }

                            val num_addr_splitted         = from.req.num_addr * i
                            val previously_processed_addr = j * config.numMatrixFetcherChannel

                            when (num_addr_splitted < previously_processed_addr) {
                                to.fragment.num_addr := 0
                            } elsewhen (num_addr_splitted > previously_processed_addr + config.numMatrixFetcherChannel) {
                                to.fragment.num_addr := config.numMatrixFetcherChannel
                            } otherwise {
                                when (num_addr_splitted < previously_processed_addr + config.numMatrixFetcherChannel) { 
                                    assert(!input.valid || from.original_last)
                                }
                                to.fragment.num_addr := (from.req.num_addr * i - j * config.numMatrixFetcherChannel).resized
                            }
                        
                        }
                    }
                    default {
                        assert(!input.valid)
                        to.fragment.addr.assignDontCare()
                        to.fragment.num_addr.assignDontCare()
                    }
                }

            }.stage()

            // val output = translated.throwWhen(translated.num_addr === 0).stage
        }
        
        val output = stageOut(subareaSplit.output)
    }

    case class DataLoaderContext() extends Bundle {
        val is_last = Bool
    }
    case class RequestContext() extends Bundle {
        val size_d    = UInt(log2Up(config.sizeD + 1) bits)
        val high_bits = Bool
        val profile   = QuantProfile(config)
        val context   = genContext()
    }
    case class LoadDataResults() extends Bundle {
        val data       = Vec(Bits(transactionLength * 8 bits), config.numMatrixFetcherChannel)
        val is_padding = Vec(Bool, config.numMatrixFetcherChannel)
        val is_last    = Bool
        // val req        = RequestContext()
    }

    val context_fifo = StreamFifo(RequestContext(), 16)

    val stageLoadData = new Stage("StageLoadData") {
        val input = stageIn(stageSplitRequest.output)

        val context_fifo_push = stageDrive(context_fifo.io.push)

        val axiloaders = Array.fill(config.numMatrixFetcherChannel) { new AXIDataLoader(config.dramBusConfig, DataLoaderContext()) }

        for (i <- 0 until config.numMatrixFetcherChannel) {
            axiloaders(i).io.bus >> stageDrive(io.bus(i))
        }

        val addr_fifos   = Array.fill(config.numMatrixFetcherChannel) { StreamFifo(cloneOf(axiloaders(0).io.req.payload), 16) }
        val resp_fifos   = Array.fill(config.numMatrixFetcherChannel) { StreamFifo(cloneOf(axiloaders(0).io.resp.payload), 64) }

        // for simulation
        val subareaRespFifoCounters = Array.tabulate(config.numMatrixFetcherChannel) { i => new Area {
            val depth       = 64
            val width       = resp_fifos(i).io.push.data.getBitsWidth
            val count_read  = Counter(32 bits, resp_fifos(i).io.pop.fire)
            val count_write = Counter(32 bits, resp_fifos(i).io.push.fire)
        }}
        // 
        
        input.ready := Vec(addr_fifos.map(_.io.push.ready)).reduce(_ && _) && context_fifo_push.ready

        for (fifo <- addr_fifos) {
            fifo.io.push.valid := False
            fifo.io.push.payload.assignDontCare()
        }
        context_fifo_push.valid := False
        context_fifo_push.payload.assignDontCare()

        when (input.fire) {
            for (i <- 0 until config.numMatrixFetcherChannel) {
                val fifo = addr_fifos(i)
                fifo.io.push.valid           := True
                fifo.io.push.addr            := input.addr(i)
                fifo.io.push.len             := transactionLength
                fifo.io.push.read_enable     := input.num_addr > i
                fifo.io.push.context.is_last := input.last
            }

            when (input.isFirst) {
                context_fifo_push.valid   := True
                context_fifo_push.payload.assignAllByName(input.fragment)
            }
        }

        for ((loader, addr_fifo, resp_fifo) <- (axiloaders, addr_fifos, resp_fifos).zipped) {
            loader.io.req     << addr_fifo.io.pop
            resp_fifo.io.push << loader.io.resp
        }

        val results = Stream(LoadDataResults())
        results.valid := Vec(resp_fifos.map(_.io.pop.valid)).reduce(_ && _)
        
        for (i <- 0 until config.numMatrixFetcherChannel) {
            results.data(i)       := resp_fifos(i).io.pop.data
            results.is_padding(i) := !resp_fifos(i).io.pop.read_enable
        }
        // results.req     := context_fifo.io.pop.payload
        results.is_last := resp_fifos(resp_fifos.size - 1).io.pop.context.is_last

        for (fifo <- resp_fifos) {
            fifo.io.pop.ready := results.fire
        }

        val output = stageOut(results.stage())
    }

    val possibleReadWidth = supportedProfiles.map(profile => profile.bit_count * config.numMultipliers * config.maxFusedMatrix)
    val writeWidth = transactionLength * config.numMatrixFetcherChannel * 8
    val gcdRWWidth = possibleReadWidth.foldLeft(BigInt(writeWidth))(_.gcd(_)).toInt
    assert(gcdRWWidth % (transactionLength * 8) == 0)

    println(s"MatrixFetcher: maxReadWidth=${possibleReadWidth.max}")

    case class RealignElement() extends Bundle {
        val data       = Bits(gcdRWWidth bits)
        val is_padding = Bool   // all bits of `data' are paddings
        val is_last    = Bool
    }
    case class RealignedData() extends Bundle {
        // val data     = Vec(Bool, possibleReadWidth.max)
        val data        = Bits(possibleReadWidth.max bits)
        val req_context = RequestContext()
    }

    val stageRealign = new Stage("StageRealign") {
        val input = stageIn(stageLoadData.output)

        val context_fifo_pop = stageIn(context_fifo.io.pop)

        val sizePush = transactionLength * 8 * config.numMatrixFetcherChannel / gcdRWWidth
        val sizePop  = possibleReadWidth.max / gcdRWWidth

        val ufifo_inst = new FIFOUnalignedPop(RealignElement(), sizePush, sizePop)

        ufifo_inst.io.push << input.translateInto(Stream(Vec(RealignElement(), sizePush))) { (to, from) =>
            for (i <- 0 until sizePush) {
                val startIdx = i * gcdRWWidth / (transactionLength * 8)
                val endIdx   = (i + 1) * gcdRWWidth / (transactionLength * 8)
                // corresponding to data[startIdx, endIdx) from input
                to(i).data       := from.data.slice(startIdx, endIdx).asBits
                to(i).is_padding := from.is_padding(startIdx)
            }
            for (i <- 0 until sizePush - 1) {
                to(i).is_last    := !to(i).is_padding && to(i + 1).is_padding
            }
            to(sizePush - 1).is_last := !to(sizePush - 1).is_padding && from.is_last
        }

        val pop = ufifo_inst.io.pop.haltWhen(!context_fifo_pop.valid)

        switch (context_fifo_pop.profile.asBits) {
            for (profile <- supportedProfiles) {
                is (profile.asBits) {
                    ufifo_inst.io.num_pop := profile.bit_count * config.numMultipliers * config.maxFusedMatrix / gcdRWWidth
                }
            }
            default {
                assert(!context_fifo_pop.valid)
                ufifo_inst.io.num_pop := 0
            }
        }

        val translated = pop.continueWhen(pop.num_valid_data >= ufifo_inst.io.num_pop)
            .throwWhen(pop.data(0).is_padding)
            .translateInto(Stream(Fragment(RealignedData()))) { (to, from) => 

                // to.fragment.data        := Vec(from.data.map(_.data.asBools).foldLeft(Array[Bool]())(_ ++ _))
                to.fragment.data        := from.data.map(_.data).asBits
                to.fragment.req_context := context_fifo_pop.payload
                to.last := from.data((ufifo_inst.io.num_pop - 1).resized).is_last || from.data((ufifo_inst.io.num_pop - 1).resized).is_padding
            }

        context_fifo_pop.ready := translated.fire && translated.last

        val output = stageOut(translated.stage())
    }
    
    val stageBitwidthConvert = new Stage("StageBitwidthConvert") {
        val input = stageIn(stageRealign.output)

        val output = stageOut(input.translateInto(cloneOf(io.resp)) { (to, from) => 
            switch (from.fragment.req_context.profile.asBits) {
                for (profile <- supportedProfiles) {
                    is (profile.asBits) {
                        for (i <- 0 until config.maxFusedMatrix; j <- 0 until config.numMultipliers) {
                            val idxFrom  = j * profile.fused_mat + i / profile.fused_mat * (config.numMultipliers * profile.fused_mat) + i % profile.fused_mat
                            // val dataFrom = B(from.fragment.data.slice(idxFrom * profile.bit_count, (idxFrom + 1) * profile.bit_count))
                            val dataFrom = from.fragment.data(idxFrom * profile.bit_count, profile.bit_count bits)
                            when (from.fragment.req_context.high_bits) {
                                to.fragment.data(i)(j).assignFromBits(dataFrom.resizeLeft(config.widthQuantValue))
                            } otherwise {
                                to.fragment.data(i)(j).assignFromBits(dataFrom.resize(config.widthQuantValue))  // TODO: Not correct when first read < 8bit
                            }
                        }
                    }
                }
                default {
                    to.fragment.data.assignDontCare()
                }
            }
            to.fragment.high_bits := from.fragment.req_context.high_bits
            to.fragment.profile   := from.fragment.req_context.profile
            to.fragment.context   := from.fragment.req_context.context
            to.last := from.last
        }.stage())
    }

    io.resp << stageBitwidthConvert.output
}