package spatten

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._

import scala.collection.mutable.ArrayBuffer
import scala.util.Random


class SpAttenController(implicit config: SpAttenConfig) extends Component with Stageable {
    import config._

    val io = new Bundle {
        val bus  = Vec(master(Axi4ReadOnly(dramBusConfig)), numMatrixFetcherChannel)
        val req  = slave Stream(Fragment(SpAttenRequest(config)))
        val resp = master Stream(SpAttenResponse(config))
    }

    case class ScoreBufferLine() extends Bundle {
        val indexes   = Vec(UInt(log2Up(maxNumKey) bits), numMatrixFetcherChannel)
        val num_index = UInt(log2Up(numMatrixFetcherChannel + 1) bits)
    }
    val score_buf = new MultiPortRAMFactory(ScoreBufferLine(), numBufferLines)

    val rng = new Random
    rng.setSeed(1)
    score_buf.init(for (i <- 0 until numBufferLines) yield {
        val ret = ScoreBufferLine()
        ret.num_index := numMatrixFetcherChannel
        ret.indexes := Vec(rng.shuffle(Seq.range(0, numMatrixFetcherChannel * 2)).slice(0, numMatrixFetcherChannel).sorted.map(x => U((x + i * numMatrixFetcherChannel) % maxNumKey)))
        ret
    })

    val requantBitCount = 4

    val requant_profile = QuantProfile(config)
    requant_profile.bit_count := requantBitCount
    requant_profile.fused_mat := 1

    case class MatrixFetcherContext() extends Bundle {
        val is_value_req = Bool
    }

    val keybuf_alloc_inst = new BufferManager(BufferManager.Config(numBufferLines, 16), Fragment(SpAttenRequest(config)), NoData)
    val mat_fetcher_inst  = new MatrixFetcher(MatrixFetcherContext())(config)
    (mat_fetcher_inst.io.bus, io.bus).zipped.foreach { _ >> _ }


    // batch_ids are guaranteed not to be conflicted with each other in the buffer
    //  because one batch will take at least one line in the buffer
    //  and the BufferManager follows a first-in-first-out manner
    private def genBatchId() = UInt(log2Up(numBufferLines + 1) bits)
    private def genPartId()  = UInt(log2Up(maxBatchSize) bits)
    private def keymatNumLines(metadata: SpAttenRequestMetadata) = {
        val num_vector_per_buf_line = numMultipliers / sizeD    // TODO: should be numMultipliers / metadata.size_d
        val num_lines_raw = (metadata.key_fetch_num + num_vector_per_buf_line - 1) / num_vector_per_buf_line
        (num_lines_raw + maxFusedMatrix - 1) / maxFusedMatrix * maxFusedMatrix
    }
    private def valmatNumLines(metadata: SpAttenRequestMetadata) = {
        val num_vector_per_buf_line = numMultipliers / sizeD    // TODO: should be numMultipliers / metadata.size_d
        val num_lines_raw = (metadata.val_fetch_num + num_vector_per_buf_line - 1) / num_vector_per_buf_line
        (num_lines_raw + maxFusedMatrix - 1) / maxFusedMatrix * maxFusedMatrix
    }

    case class RequestEx() extends Bundle {
        val req               = SpAttenRequest(config)
        val is_last_in_batch  = Bool
        val is_requantize     = Bool
        val buf_addr          = genBufAddr()
        val alloc_handle      = cloneOf(keybuf_alloc_inst.io.alloc_resp.handle)
        val batch_id          = genBatchId()
        val part_id           = genPartId()
    }

    type Stage = StageComponent

    // allocate from the key buffer and get the start address
    val stageAllocKeyBuffer = new Stage("StageAllocKeyBuffer") {
        val input = stageIn(io.req)

        // TODO: flexible size_d
        assert(!input.valid || input.metadata.size_d === sizeD)

        input.translateInto(stageDrive(keybuf_alloc_inst.io.alloc_req)) { (to, from) => 
            when (input.isFirst) {
                to.size := (keymatNumLines(from.fragment.metadata) * from.fragment.metadata.profile_key.fused_mat).resized
            } otherwise {
                to.size := 0
            }
            to.context  := from
        }

        val alloc_resp = stageIn(keybuf_alloc_inst.io.alloc_resp)

        val batch_id = Counter(widthOf(genBatchId()) bits)
        when (alloc_resp.fire && alloc_resp.context.last) {
            batch_id.increment()
        }
        val part_id = Counter(widthOf(genPartId()) bits)
        when (alloc_resp.fire) {
            when (alloc_resp.context.last) {
                val fused_mat = alloc_resp.context.fragment.metadata.profile_key.fused_mat
                assert(fused_mat === 1 || fused_mat === part_id.value + 1)
                part_id.clear()
            } otherwise {
                assert(!part_id.willOverflowIfInc)
                part_id.increment()
            }
        }

        val last_start_addr   = RegNextWhen(alloc_resp.start_addr, alloc_resp.fire && alloc_resp.size =/= 0)
        val last_alloc_handle = RegNextWhen(alloc_resp.handle,     alloc_resp.fire && alloc_resp.size =/= 0)

        val output = stageOut(alloc_resp.translateInto(Stream(RequestEx())) { (to, from) => 
            when (from.size === 0) {
                to.buf_addr     := last_start_addr
                to.alloc_handle := last_alloc_handle
            } otherwise {
                to.buf_addr     := from.start_addr
                to.alloc_handle := from.handle
            }
            to.req               := from.context.fragment
            to.is_last_in_batch  := from.context.last
            to.part_id           := part_id
            to.is_requantize     := False
            to.batch_id          := batch_id
        })
    }
    

    case class RequantizeReport() extends Bundle {
        val need_requantize  = Bool
        // val batch_id         = Bool
    }

    val requantize_report = Stream(RequantizeReport())

    // put all requests to a FIFO and deal with requantization
    val stageRequantizeOrRelease = new Stage("StageRequantizeOrRelease") {
        val depthReqFIFO = 16
        val req_fifo = StreamFifo(RequestEx(), depthReqFIFO)

        val input_new        = stageIn(stageAllocKeyBuffer.output).haltWhen(req_fifo.io.availability < depthReqFIFO / 2)  // halt the stream so that requantize requests are always able to be written to FIFO
        val input_requantize = Stream(RequestEx())

        val input = StreamArbiterFactory.lowerFirst.transactionLock.onArgs(input_requantize, input_new)

        val requantize_report = stageIn(SpAttenController.this.requantize_report)

        val requantize_or_release_req = StreamDemux(
            StreamJoin.arg(req_fifo.io.pop, requantize_report).translateWith(req_fifo.io.pop.payload), 
            requantize_report.need_requantize.asUInt, 2)
        val release_req    = requantize_or_release_req(0)
        val requantize_req = requantize_or_release_req(1)     // need_requantize === 1

        stageDrive(keybuf_alloc_inst.io.release_req) <-< release_req
            .throwWhen(!release_req.is_last_in_batch)
            .translateInto(cloneOf(keybuf_alloc_inst.io.release_req)) { (to, from) => 
                to.handle := from.alloc_handle
            }
        stageIn(keybuf_alloc_inst.io.release_resp).freeRun()

        input_requantize <-/< requantize_req.translateInto(cloneOf(input_requantize)) { (to, from) =>
            to := from
            to.is_requantize.allowOverride := True
        }

        val input_forked = StreamFork2(input) 

        req_fifo.io.push << input_forked._1

        val output = stageOut(input_forked._2.stage())
        // output <-< input_forked._2
    }

    case class BufferTag() extends Bundle {
        val batch_id      = genBatchId()
        val is_requantize = Bool
    }

    val buffer_config = Buffer.Config(genFix(), BufferTag(), NoData, NoData, 
        sizeLine      = numMultipliers,
        numLines      = numBufferLines,
        numWritePorts = maxFusedMatrix,
        numReadPorts  = 1,
        numBanks      = maxFusedMatrix,
        initT         = Some({() => 
            genFix().setAll()
        }),
        initTag       = Some({() => 
            val ret = BufferTag()
            ret.batch_id.setAll()
            ret.is_requantize := False
            ret
        }))
    val buffer_key = new Buffer((x: BufferTag, y: BufferTag) => x === y)(buffer_config)

    val output_alloc_key_forked = StreamFork(stageRequantizeOrRelease.output, 4)

    private def prepareWriteBuffer(
        self: StageImplementation,
        input: Stream[RequestEx], 
        buffer: Buffer[SInt, BufferTag, NoData, NoData],
        numLines: (SpAttenRequestMetadata) => UInt,
        getProfile: (SpAttenRequestMetadata) => QuantProfile) = new Area {

        val input_forked = StreamFork(input, maxFusedMatrix)

        for (portId <- 0 until maxFusedMatrix) {
            self.stageDrive(buffer.io.write_req(portId)) <-< input_forked(portId)
                .translateInto(cloneOf(buffer.io.write_req(portId))) { (to, from) => 
                    
                    when (from.is_requantize) {
                        to.start_addr := (from.buf_addr + numLines(from.req.metadata) * from.part_id + portId).resized
                        to.addr_inc   := maxFusedMatrix
                        to.size       := (numLines(from.req.metadata) / maxFusedMatrix).resized

                    } elsewhen (getProfile(from.req.metadata).fused_mat =/= 1) {
                        to.start_addr := (from.buf_addr + numLines(from.req.metadata) * portId).resized
                        to.addr_inc   := 1
                        to.size       := numLines(from.req.metadata).resized

                    } otherwise {
                        to.start_addr := from.buf_addr + portId
                        to.addr_inc   := maxFusedMatrix
                        to.size       := (numLines(from.req.metadata) / maxFusedMatrix).resized
                    }

                    to.mask              := Mux(from.is_requantize, 
                        B(widthQuantValue bits, (requantBitCount - 1 downto 0) -> true, default -> false), 
                        B(widthQuantValue bits, default -> true))
                    to.tag.batch_id      := from.batch_id
                    to.tag.is_requantize := from.is_requantize
                    
                }
        }

        for (resp <- buffer.io.write_resp) {
            self.stageIn(resp).freeRun()
        }
    }

    // fork(0): tell key buffer the start address and size of the next write
    val stagePrepareWriteKeyBuffer = new Stage("StagePrepareWriteKeyBuffer") {
        val input_raw = stageIn(output_alloc_key_forked(0))
        val input = input_raw.throwWhen(!input_raw.is_requantize && input_raw.part_id =/= 0)

        val impl = prepareWriteBuffer(this, input, buffer_key, keymatNumLines, _.profile_key)
    }

    val stageMatFetcherManager = new Stage("StageMatFetcherManager") {
        val req_key = stageInDef(mat_fetcher_inst.io.req)
        val req_val = stageInDef(mat_fetcher_inst.io.req)

        stageDrive(mat_fetcher_inst.io.req) << StreamArbiterFactory.roundRobin.fragmentLock.onArgs(req_key, req_val)

        val mat_fetcher_resp = stageIn(mat_fetcher_inst.io.resp)

        val resp_demux = StreamDemux(mat_fetcher_resp, mat_fetcher_resp.context.is_value_req.asUInt, 2)

        val resp_key = stageOut(resp_demux(0))
        val resp_val = stageOut(resp_demux(1))
        // resp_key << resp_demux(0)
        // resp_val << resp_demux(1)
    }
    
    // fork(1): read pruned indexes from score buffer and feed addresses to MatrixFetcher
    val areaFetchKeyMat = new Area {
        val input_raw = output_alloc_key_forked(1)
        val input = input_raw.throwWhen(!input_raw.is_requantize && input_raw.part_id =/= 0)

        val score_buf_index = Counter(0 until numBufferLines)

        val input_deser = Stream(Fragment(cloneOf(input.payload)))
        input_deser.fragment := input.payload
        input_deser.last     := score_buf_index === (input.req.metadata.key_fetch_num + numMatrixFetcherChannel - 1) / numMatrixFetcherChannel - 1
        input_deser.valid    := input.valid

        input.ready := input_deser.fire && input_deser.last

        when (input_deser.fire) {
            when (input_deser.last) {
                score_buf_index.clear()
            } otherwise {
                score_buf_index.increment()
            }
        }

        val score_buf_line  = score_buf.readSync(score_buf_index, input_deser.fire)

        stageMatFetcherManager.req_key <-< input_deser.stage().translateInto(cloneOf(mat_fetcher_inst.io.req)) { (to, from) => 
            val metadata  = from.fragment.req.metadata
            
            
            val profile   = Mux(from.fragment.is_requantize, requant_profile, metadata.profile_key)
            val base_addr = Mux(from.fragment.is_requantize, metadata.key_requant_base_addr, metadata.key_base_addr)
            val key_vector_size = metadata.size_d * profile.bit_count * profile.fused_mat / 8
            
            to.last               := from.last
            to.fragment.addr      := Vec(score_buf_line.indexes.map(_ * key_vector_size + base_addr))
            to.fragment.profile   := profile
            to.fragment.high_bits := !from.fragment.is_requantize
            to.fragment.size_d    := metadata.size_d
            to.fragment.num_addr  := Mux(from.last, (metadata.key_fetch_num - 1) % numMatrixFetcherChannel + 1, U(numMatrixFetcherChannel)).resized
            to.fragment.context.is_value_req := False
        }
 
        // write fetched data to key buffer
        val fetch_resp = StreamFork(stageMatFetcherManager.resp_key, maxFusedMatrix)
        for (portId <- 0 until maxFusedMatrix) {
            buffer_key.io.write_data(portId) <-< fetch_resp(portId)
                .translateInto(cloneOf(buffer_key.io.write_data(portId))) { (to, from) => 
                    to.fragment.data := from.fragment.data(portId)
                    to.last := from.last
                }
        }
    }

    private def prepareReadBuffer(
        self: StageImplementation,
        input: Stream[RequestEx], 
        buffer: Buffer[SInt, BufferTag, NoData, NoData],
        numLines: (SpAttenRequestMetadata) => UInt,
        getProfile: (SpAttenRequestMetadata) => QuantProfile) = new Area {

        self.stageDrive(buffer.io.read_req(0)) <-< input.translateInto(cloneOf(buffer.io.read_req(0))) { (to, from) => 
            when (getProfile(from.req.metadata).fused_mat =/= 1) {
                to.start_addr := (from.buf_addr + numLines(from.req.metadata) * from.part_id).resized
            } otherwise {
                to.start_addr := from.buf_addr
            }
            to.size              := numLines(from.req.metadata).resized
            to.tag.batch_id      := from.batch_id
            to.tag.is_requantize := from.is_requantize
        }
    }

    // fork(2): tell key buffer the start address to read data
    val stagePrepareReadKeyBuffer = new Stage("StagePrepareReadKeyBuffer") {
        val input = stageIn(output_alloc_key_forked(2))

        val impl = prepareReadBuffer(this, input, buffer_key, keymatNumLines, _.profile_key)
    }

    // fork(3): read data from buffer and perform the dot product
    val stageDotProduct = new Stage("StageDotProduct") {
        val input        = stageIn(output_alloc_key_forked(3))
        val input_buffer = stageIn(buffer_key.io.read_data(0))

        input.ready := input_buffer.fire && input_buffer.last

        val dotproduct_inst = new DotProduct(RequestEx())

        dotproduct_inst.io.req << input_buffer
            .haltWhen(!input.valid)
            .translateInto(cloneOf(dotproduct_inst.io.req)) { (to, from) => 
                to.fragment.query   := input.req.query
                to.fragment.key     := from.fragment.data
                to.fragment.size_d  := input.req.metadata.size_d
                to.fragment.context := input.payload
                to.last             := from.last
            }
        
        val output = stageOut(dotproduct_inst.io.resp)
    }

    val stageSoftmax = new Stage("StageSoftmax") {
        val input = stageIn(stageDotProduct.output)

        val softmax_inst = new Softmax(RequestEx())

        input.translateInto(softmax_inst.io.req) { (to, from) => 
            to.fragment.data    := from.fragment.results
            to.fragment.context := from.fragment.context
            to.last := from.last
        }

        val output = stageOut(softmax_inst.io.resp)
        // output << softmax_inst.io.resp
    }

    val stageRequantDecision = new Stage("StageRequantDecision") {
        val input = stageIn(stageSoftmax.output)

        val requant_inst = new RequantDecision(RequestEx())

        input.translateInto(requant_inst.io.req) { (to, from) => 
            to.fragment.data      := from.fragment.results
            to.fragment.threshold := from.fragment.context.req.metadata.thres_requantize
            to.fragment.context   := from.fragment.context
            to.last               := from.last
        }

        val requant_result_forked = StreamFork2(requant_inst.io.resp)

        // val output_requantize_report = stageOutDef(Stream(RequantizeReport()))
        
        requant_result_forked._1
            .throwWhen(requant_result_forked._1.isTail)
            .translateInto(stageDrive(requantize_report)) { (to, from) => 
                to.need_requantize := from.fragment.need_requant && !from.fragment.context.is_requantize
            }

        val output = stageOut(requant_result_forked._2)
    }

    // buffer the following requests after a request have to perform requantization
    val stageRequantBuffer = new Stage("StageRequantBuffer") {
        val input = stageIn(stageRequantDecision.output)
        // input << stageRequantDecision.output

        val sizeRequantBuffer = numBufferLines * 16

        case class BufferedRequest() extends Bundle {
            val data                    = Vec(genFix(), numSoftMaxUnit)
            val reqex                   = RequestEx()
            val num_requant_before_this = UInt(log2Up(sizeRequantBuffer) bits)
        }
        val requant_fifo = StreamFifo(Fragment(BufferedRequest()), sizeRequantBuffer)

        val need_requant_counter   = Counter(0 until sizeRequantBuffer)
        val finish_requant_counter = Counter(0 until sizeRequantBuffer)

        assert(!need_requant_counter.willOverflow)
        assert(!finish_requant_counter.willOverflow)

        val input_need_requantize = input.fragment.need_requant && !input.fragment.context.is_requantize
        val input_is_requantize   = input.fragment.context.is_requantize
        val input_is_normal       = !input_need_requantize && !input_is_requantize

        when (input.fire && input.last) {
            when (input_need_requantize) {
                need_requant_counter.increment()
            } elsewhen (input_is_normal) {
                need_requant_counter.clear()
            }
        }

        val input_demuxed = StreamDemux(input.throwWhen(input_need_requantize).translateInto(Stream(Fragment(BufferedRequest()))) { (to, from) => 
            to.fragment.data                    := from.fragment.data
            to.fragment.reqex                   := from.fragment.context
            to.fragment.num_requant_before_this := need_requant_counter
            to.last := from.last
        }, input_is_requantize.asUInt, 2)

        requant_fifo.io.push << input_demuxed(0)

        val output_raw = StreamArbiterFactory.lowerFirst.fragmentLock.onArgs(
            requant_fifo.io.pop.haltWhen(requant_fifo.io.pop.num_requant_before_this > finish_requant_counter),
            input_demuxed(1))

        when (output_raw.fire && output_raw.last) {
            when (output_raw.fragment.reqex.is_requantize) {
                finish_requant_counter.increment()
            } otherwise {
                finish_requant_counter.clear()
            }
        }

        val output = stageOut(output_raw.stage())
    }

    // TODO: When size_d > 64, not all scores are valid from the softmax module, we need a FIFO here
    case class TopKElement() extends Bundle {
        val score = genFix()
        val index = UInt(log2Up(maxNumKey) bits)
    }

    // Unbatch when quant_key === 6 bits and quant_val =/= 6 bits
    // Rebatch when quant_val === 6 bits
    val stageRebatch = new Stage("StageRebatch") {
        val input = stageIn(stageRequantBuffer.output)

        val part_id = Counter(0 until maxFusedMatrix)

        assert(!part_id.willOverflow)

        val metadata = input.reqex.req.metadata

        when (input.fire && input.last) {
            when (part_id >= metadata.profile_val.fused_mat - 1) {
                part_id.clear()
            } otherwise {
                part_id.increment()
            }
        }

        val output = stageOut(input.translateInto(cloneOf(input)) { (to, from) => 
            when (metadata.profile_key.fused_mat === 1 && metadata.profile_val.fused_mat === 1) { 
                // don't touch the batch info in BERT mode
                to.fragment.reqex.part_id          := from.fragment.reqex.part_id
                to.fragment.reqex.is_last_in_batch := from.fragment.reqex.is_last_in_batch
            } otherwise {
                to.fragment.reqex.part_id          := part_id.resized
                to.fragment.reqex.is_last_in_batch := part_id === metadata.profile_val.fused_mat - 1
            }
            to.fragment.reqex.alloc_handle      := 0
            to.fragment.reqex.batch_id          := 0
            to.fragment.reqex.buf_addr          := 0
            to.fragment.reqex.is_requantize     := False
            to.fragment.reqex.req               := from.fragment.reqex.req
            to.fragment.data                    := from.fragment.data
            to.fragment.num_requant_before_this := 0
            to.last := from.last
        }.stage())
    }

    // val useDummyTopK = true

    val topk_inst = if (useDummyTopK) { 
        new TopKDummy(TopK.Config(
            TopKElement(), RequestEx(), 
            numSoftMaxUnit, maxNumKey, 
            (x: TopKElement, y: TopKElement) => x.score > y.score,
            (x: TopKElement, y: TopKElement) => x.score === y.score), (x: RequestEx) => x.req.metadata.topk_latency)
    } else {
        new TopK(TopK.Config(
            TopKElement(), RequestEx(), 
            numSoftMaxUnit, maxNumKey, 
            (x: TopKElement, y: TopKElement) => x.score > y.score,
            (x: TopKElement, y: TopKElement) => x.score === y.score))
    }

    // We lost the index information during the matrixfetcher stage :(, load them back before doing TopK
    val areaReloadIndexes = new Area {
        val input = stageRebatch.output

        val counter         = Counter(0 until numMatrixFetcherChannel / numSoftMaxUnit)
        val score_buf_index = Counter(0 until numBufferLines)

        when (input.fire) {
            when (input.last) {
                counter.clear()
                score_buf_index.clear()
            } otherwise {
                counter.increment()
                when (counter.willOverflow) {
                    score_buf_index.increment()
                }
            }
        }

        val score_buf_line = score_buf.readSync(score_buf_index, input.fire)
        
        val output = input.stage().translateInto(cloneOf(topk_inst.io.req)) { (to, from) => 
            for (i <- 0 until numSoftMaxUnit) {
                to.fragment.data(i).score := from.fragment.data(i)
                to.fragment.data(i).index := 
                    Vec.tabulate(numMatrixFetcherChannel / numSoftMaxUnit) { j => 
                        score_buf_line.indexes(j * numSoftMaxUnit + i) 
                    }.access(counter)
            }
            to.fragment.target  := (from.fragment.reqex.req.metadata.val_fetch_num - 1).resized
            to.fragment.len     := numSoftMaxUnit
            to.fragment.context := from.fragment.reqex
            to.last := from.last
        }.stage()
    }

    val stageAdjustFragmentLen = new Stage("StageAdjustFragmentLen") {
        val input = stageIn(areaReloadIndexes.output)

        val sum_len = Reg(UInt(log2Up(maxNumKey + 1) bits)) init(0)

        when (input.fire) {
            when (input.last) {
                sum_len := 0
            } otherwise {
                sum_len := sum_len + numSoftMaxUnit
            }
        }

        val output = stageOut(
            input
            .throwWhen(sum_len >= input.context.req.metadata.key_fetch_num)
            .translateInto(cloneOf(input)) { (to, from) => 
                to.assignAllByName(from)
                when (from.fragment.context.req.metadata.key_fetch_num - sum_len <= numSoftMaxUnit) {
                    to.fragment.len := (from.fragment.context.req.metadata.key_fetch_num - sum_len).resized
                    to.last := True
                }
            }
        )
    }

    // TODO: Add attention score on 6bit
    val stageTopK = new Stage("StageTopK") {
        val input = stageIn(stageAdjustFragmentLen.output)

        val input_should_skip_topk = input.context.req.metadata.val_fetch_num >= input.context.req.metadata.key_fetch_num
        val input_demuxed = StreamDemux(input, input_should_skip_topk.asUInt, 2)

        val input_continue = input_demuxed(0)
        val input_skip     = input_demuxed(1).translateInto(cloneOf(topk_inst.io.resp)) { (to, from) => 
            to.assignAllByName(from)
        }

        val topk_resp = stageIn(topk_inst.io.resp)

        val num_onfly_transaction     = Reg(UInt(16 bits)) init(0)
        val num_onfly_transaction_inc = input.fire && input.isFirst && !input_should_skip_topk
        val num_onfly_transaction_dec = topk_resp.fire && topk_resp.last
        num_onfly_transaction := num_onfly_transaction + num_onfly_transaction_inc.asUInt - num_onfly_transaction_dec.asUInt
        assert(num_onfly_transaction =/= num_onfly_transaction.maxValue)    // overflow/underflow should never happen

        stageDrive(topk_inst.io.req) << input_continue

        val output = stageOutDef(topk_inst.io.resp)
        output << StreamArbiterFactory.lowerFirst.fragmentLock.onArgs(
            topk_resp, 
            input_skip.haltWhen(num_onfly_transaction =/= 0))
    }

    val valbuf_release_request = Stream(RequestEx())

    val stageAllocValBuffer = new Stage("StageAllocValBuffer") {
        val input = stageIn(stageTopK.output)
        val input_release = stageIn(valbuf_release_request)

        val valbuf_alloc_inst = new BufferManager(BufferManager.Config(numBufferLines, 16), cloneOf(input.payload), NoData)

        input.translateInto(valbuf_alloc_inst.io.alloc_req) { (to, from) => 
            to.size    := Mux(input.isFirst && from.fragment.context.part_id === 0, 
                valmatNumLines(from.fragment.context.req.metadata) * from.fragment.context.req.metadata.profile_val.fused_mat,
                U(0)).resized
            to.context := from
        }

        val batch_id = Counter(widthOf(genBatchId()) bits)

        val alloc_resp = valbuf_alloc_inst.io.alloc_resp
        when (alloc_resp.fire && alloc_resp.context.last && alloc_resp.context.fragment.context.is_last_in_batch) {
            batch_id.increment()
        }

        val last_start_addr   = RegNextWhen(alloc_resp.start_addr, alloc_resp.fire && alloc_resp.size =/= 0)
        val last_alloc_handle = RegNextWhen(alloc_resp.handle,     alloc_resp.fire && alloc_resp.size =/= 0)


        val output = stageOut(valbuf_alloc_inst.io.alloc_resp.translateInto(cloneOf(input)) { (to, from) => 
            to := from.context
            to.fragment.context.buf_addr.allowOverride := Mux(from.size === 0, last_start_addr, from.start_addr)
            to.fragment.context.alloc_handle.allowOverride := Mux(from.size === 0, last_alloc_handle, from.handle)              
            to.fragment.context.batch_id.allowOverride := batch_id
        })

        input_release.throwWhen(!input_release.is_last_in_batch).translateInto(valbuf_alloc_inst.io.release_req) { (to, from) => 
            to.handle := from.alloc_handle
        }
        valbuf_alloc_inst.io.release_resp.freeRun()
    }

    val output_alloc_val_forked = StreamFork(stageAllocValBuffer.output, 4)
    val buffer_val = new Buffer((x: BufferTag, y: BufferTag) => x.batch_id === y.batch_id)(buffer_config)

    val stagePrepareWriteValBuffer = new Stage("StagePrepareWriteValBuffer") {
        val input_raw = stageIn(output_alloc_val_forked(0))
        val input = input_raw
            .throwWhen(!input_raw.isFirst || input_raw.context.part_id =/= 0)
            .translateInto(Stream(RequestEx())) { (to, from) => 
                to := from.fragment.context
            }

        val impl = prepareWriteBuffer(this, input, buffer_val, valmatNumLines, _.profile_val)
    }

    val stageFetchValMat = new Stage("StageFetchValMat") {
        val input_raw = stageIn(output_alloc_val_forked(1))
        val input = input_raw.throwWhen(input_raw.context.part_id =/= 0)

        val slowdownRatio = numMatrixFetcherChannel / numSoftMaxUnit

        case class SlowdownResult() extends Bundle {
            val indexes = Vec(Vec(UInt(log2Up(maxNumKey) bits), numSoftMaxUnit), slowdownRatio)
            val last    = Bool
            val len     = UInt(log2Up(numMatrixFetcherChannel + 1) bits)
            val reqex   = RequestEx()
        }

        val slowdown = Stream(SlowdownResult())

        val slowdown_valid    = Reg(Bool) init(False)
        val slowdown_payload  = Reg(SlowdownResult())

        slowdown.valid   := slowdown_valid
        slowdown.payload := slowdown_payload

        input.ready := slowdown.ready || !slowdown.valid
    
        val counter = Counter(0 until slowdownRatio)

        when (slowdown.fire) {
            slowdown_valid := False
        }
        when (input.fire) {
            for (i <- 0 until slowdownRatio) {
                when (counter === i) {
                    slowdown_payload.indexes(i) := Vec(input.data.map(_.index))
                }
            }
            slowdown_payload.reqex            := input.context
            counter.increment()

            when (input.last || counter.willOverflow) {
                counter.clear()
                slowdown_valid := True
                slowdown_payload.len  := counter * numSoftMaxUnit + input.len
                slowdown_payload.last := input.last
            }
        } 

        stageDrive(stageMatFetcherManager.req_val) <-< slowdown.translateInto(cloneOf(mat_fetcher_inst.io.req)) { (to, from) => 
            val metadata        = from.reqex.req.metadata
            val profile         = metadata.profile_val
            val val_vector_size = metadata.size_d * profile.bit_count * profile.fused_mat / 8

            to.fragment.addr := Vec(
                (from.indexes.foldLeft(new ArrayBuffer[UInt])(_ ++ _)).map(_ * val_vector_size + from.reqex.req.metadata.val_base_addr))
            to.fragment.num_addr             := from.len
            to.fragment.size_d               := metadata.size_d
            to.fragment.profile              := metadata.profile_val
            to.fragment.high_bits            := True
            to.fragment.context.is_value_req := True
            to.last := from.last
        }

        val fetch_resp = StreamFork(stageIn(stageMatFetcherManager.resp_val), maxFusedMatrix)
        for (portId <- 0 until maxFusedMatrix) {
            stageDrive(buffer_val.io.write_data(portId)) <-< fetch_resp(portId)
                .translateInto(cloneOf(buffer_val.io.write_data(portId))) { (to, from) => 
                    to.fragment.data := from.fragment.data(portId)
                    to.last := from.last
                }
        }
    }

    val stagePrepareReadValBuffer = new Stage("StagePrepareReadValBuffer") {
        val input_raw = stageIn(output_alloc_val_forked(2))
        val input = input_raw
            .throwWhen(!input_raw.isFirst)
            .translateInto(Stream(RequestEx())) { (to, from) => 
                to := from.fragment.context
            }

        val impl = prepareReadBuffer(this, input, buffer_val, valmatNumLines, _.profile_val)
    }

    val stageMultiplyValue = new Stage("StageMultiplyValue") {
        val input        = stageIn(output_alloc_val_forked(3)).queue(numMatrixFetcherChannel / numSoftMaxUnit * maxFusedMatrix * 8)
        val input_buffer = stageIn(buffer_val.io.read_data(0))

        // val input_finished        = RegNextWhen(input.last, input.fire, False)
        // val input_buffer_finished = RegNextWhen(input_buffer.last, input.fire, False)

        // TODO: input needs to be slowed down when size_d > 64

        when (input.valid && input_buffer.valid) {
            assert (input.last === input_buffer.last)
        }

        // val input_sync        = input       .haltWhen(input_finished && !input_buffer_finished).throwWhen(!input_finished && input_buffer_finished)
        // val input_buffer_sync = input_buffer.haltWhen(!input_finished && input_buffer_finished).throwWhen(input_finished && !input_buffer_finished)

        val multvalue_inst = new MultiplyValue(RequestEx())

        StreamJoin.arg(input, input_buffer).translateInto(multvalue_inst.io.req) { (to, from) => 
            to.fragment.scores  := Vec(input.data.map(_.score))
            to.fragment.values  := input_buffer.data
            to.fragment.size_d  := input.context.req.metadata.size_d
            to.fragment.context := input.context
            to.last             := input.last
        }

        val resp_forked = StreamFork2(multvalue_inst.io.resp)

        stageDrive(valbuf_release_request) << resp_forked._1.translateInto(Stream(RequestEx())) { (to, from) => 
            to := from.context
        }

        val output = stageOut(resp_forked._2.translateInto(cloneOf(io.resp)) { (to, from) => 
            to.results  := from.results
            to.metadata := from.context.req.metadata
        })
    }

    io.resp << stageMultiplyValue.output
    
    score_buf.build()
}