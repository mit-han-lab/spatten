package spatten

import spinal.core._
import spinal.lib._

object Buffer {
    case class Config[T <: Data, TTag <: Data, TWriteContext <: Data, TReadContext <: Data](
        genT: HardType[T], 
        genTag: HardType[TTag], 
        genWriteContext: HardType[TWriteContext],
        genReadContext: HardType[TReadContext],
        sizeLine: Int, 
        numLines: Int,
        numWritePorts: Int,
        numReadPorts: Int,
        numBanks: Int,
        initT: Option[() => T] = None,
        initTag: Option[() => TTag] = None) {
        
        def genAddr() = UInt(log2Up(numLines) bits)
        def genSize() = UInt(log2Up(numLines + 1) bits)
        def genMask() = Bits(widthOf(genT()) bits)
    }

    case class WriteRequest[TTag <: Data, TContext <: Data](config: Config[_ <: Data, TTag, TContext, _ <: Data]) extends Bundle {
        val start_addr = config.genAddr()
        val size       = config.genSize()
        val addr_inc   = config.genSize()
        val mask       = config.genMask()
        val tag        = config.genTag()
        val context    = config.genWriteContext()
    }
    case class WriteResponse[TContext <: Data](config: Config[_ <: Data, _ <: Data, TContext, _ <: Data]) extends Bundle {
        val context    = config.genWriteContext()
    }
    case class WriteData[T <: Data](config: Config[T, _ <: Data, _ <: Data, _ <: Data]) extends Bundle {
        val data       = Vec(config.genT(), config.sizeLine)
    }

    case class ReadRequest[T <: Data, TTag <: Data, TContext <: Data](config: Config[T, TTag, _ <: Data, TContext]) extends Bundle {
        val start_addr = config.genAddr()
        val size       = config.genSize()
        val tag        = config.genTag()
        val context    = config.genReadContext()
    }
    case class ReadData[T <: Data, TContext <: Data](config: Config[T, _ <: Data, _ <: Data, TContext]) extends Bundle {
        val data       = Vec(config.genT(), config.sizeLine)
        val context    = config.genReadContext()
    }
}

class Buffer[T <: Data, TTag <: Data, TWriteContext <: Data, TReadContext <: Data](funcMatchTag: (TTag, TTag) => Bool)(
    config: Buffer.Config[T, TTag, TWriteContext, TReadContext]) extends Component with Stageable {

    import Buffer._
    import config._

    val io = new Bundle {
        val write_req  = Vec(slave  Stream(WriteRequest(config)),        numWritePorts)
        val write_resp = Vec(master Stream(WriteResponse(config)),       numWritePorts)
        val write_data = Vec(slave  Stream(Fragment(WriteData(config))), numWritePorts)

        val read_req   = Vec(slave  Stream(ReadRequest(config)),         numReadPorts)
        val read_data  = Vec(master Stream(Fragment(ReadData(config))),  numReadPorts)
    }

    case class RAMReadContext() extends Bundle {
        val rollback_flag = Bool
        val tag           = genTag()
        val context       = genReadContext()
        val last          = Bool
    }

    type Stage = StageComponent

    require(initT.isEmpty == initTag.isEmpty) // to make sure all rams are in lockstep mode

    val data_ram_insts = Array.tabulate(widthOf(genT())) { bitId => 
        new MultiBankRAM(MultiBankRAM.Config(
            Bits(sizeLine bits), NoData, NoData, 
            numLines, 1, numBanks, numWritePorts, numReadPorts,
            initValue = if (!initT.isEmpty) { Some(() => B(sizeLine bits, default -> initT.get().asBits(bitId))) } else None))
    }

    val tag_ram_inst = new MultiBankRAM(MultiBankRAM.Config(
        genTag, NoData, RAMReadContext(),
        numLines, 0, numBanks, numWritePorts, numReadPorts,
        initValue = initTag))

    val areaWrite = Array.tabulate(numWritePorts) { portId => new Area { 
        val subareaReq = new Area {
            val input      = io.write_req(portId)
            val input_data = io.write_data(portId)

            val offset = Reg(genAddr()) init(0)
            val size   = Counter(widthOf(genSize()) bits)

            assert(!size.willOverflow)

            when (input_data.fire) {
                when (input_data.last) {
                    offset := 0
                    size.clear()
                } otherwise {
                    offset := (offset + input.addr_inc).resized
                    size.increment()
                }
            }

            input.ready := input_data.fire && input_data.last

            val input_data_forked = StreamFork(input_data.throwWhen(size >= input.size), widthOf(genT()) + 1, true)

            for (bitId <- 0 until widthOf(genT())) {
                data_ram_insts(bitId).io.write_req(portId) <-/< input_data_forked(bitId)
                    .continueWhen(input.valid)
                    .translateInto(cloneOf(data_ram_insts(bitId).io.write_req(portId))) { (to, from) => 
                        to.addr := (input.start_addr + offset).resized
                        to.data := B(from.fragment.data.map(_.asBits(bitId)).seq)
                        // to.data := Vec(from.fragment.data.map(_.asBits(bitId))).asBits   // <-- This leads to many iterations in PhaseInferWidth
                        to.mask := input.mask(bitId).asBits
                    }
                
                // all rams should work in lockstep mode
                if (bitId < widthOf(genT()) - 1) {
                    assert(data_ram_insts(bitId).io.write_req(portId).ready === data_ram_insts(bitId + 1).io.write_req(portId).ready)
                } else {
                    assert(data_ram_insts(bitId).io.write_req(portId).ready === tag_ram_inst.io.write_req(portId).ready)
                }
            }
            tag_ram_inst.io.write_req(portId) <-/< input_data_forked(widthOf(genT()))
                .continueWhen(input.valid)
                .translateInto(cloneOf(tag_ram_inst.io.write_req(portId))) { (to, from) => 
                    to.addr := (input.start_addr + offset).resized
                    to.data := input.tag
                }
        }

        val subareaResp = new Area {
            // TODO: write_resp
            io.write_resp(portId).valid := False
            io.write_resp(portId).payload.assignDontCare()
            for (ram <- data_ram_insts :+ tag_ram_inst) {
                ram.io.write_resp(portId).freeRun()
            }
        }
    }}

    val areaRead = Array.tabulate(numReadPorts) { portId => new Area {
        val stageGenReq = new Stage(s"StageReadGenReq${portId}") {
            val input = stageIn(io.read_req(portId))

            val req = cloneOf(tag_ram_inst.io.read_req(0))

            val size = Counter(widthOf(genSize()) bits)

            assert(!size.willOverflow)

            when (req.fire) {
                assert(size < input.size)
                when (size === input.size - 1) {
                    size.clear()
                } otherwise {
                    size.increment()
                }
            }

            input.ready               := req.fire && size === input.size - 1

            req.valid                 := input.valid
            req.addr                  := (input.start_addr + size).resized
            req.context.tag           := input.tag
            req.context.rollback_flag := False
            req.context.context       := input.context
            req.context.last          := size === input.size - 1
            
            val output = stageOut(req.stage())
        }

        val stageSendReq = new Stage(s"StageReadSendReq${portId}") {
            val input = stageIn(stageGenReq.output)
            val input_need_rollback = stageInDef(Flow(Bool))

            val output = cloneOf(input)

            val fifoDepth = 16

            val head_commit   = Reg(UInt(log2Up(fifoDepth) bits)) init(0)
            val head_issue    = Reg(UInt(log2Up(fifoDepth) bits)) init(0)
            val tail          = Reg(UInt(log2Up(fifoDepth) bits)) init(0)
            val rollback_flag = Reg(Bool) init(False)

            val head_issue_next = UInt(log2Up(fifoDepth) bits)
            head_issue := head_issue_next

            val fifo = Mem(cloneOf(input.payload), fifoDepth)

            input.ready := (tail + 1).resize(log2Up(fifoDepth)) =/= head_commit

            when (input.fire) {
                fifo.write(tail, input.payload)
                tail := (tail + 1).resized
            }

            head_issue_next := head_issue
            when (output.fire) {
                head_issue_next := (head_issue + 1).resized
            }
            when (input_need_rollback.fire) {
                when (!input_need_rollback.payload) {
                    head_commit   := (head_commit + 1).resized
                } otherwise {
                    head_issue_next := head_commit
                    rollback_flag   := !rollback_flag
                }
            }

            val fifo_head   = fifo.readSync(head_issue_next)

            output.valid   := head_issue =/= tail && !(RegNext(head_issue_next === tail && input.fire) init(True)) // RegNext to avoid write conflict
            output.payload := fifo_head
            output.context.rollback_flag.allowOverride := rollback_flag

            val forked = StreamFork(output, widthOf(genT()) + 1, true)

            val req_tag  = stageDrive(tag_ram_inst.io.read_req(portId))
            val req_data = data_ram_insts.map(inst => stageDrive(inst.io.read_req(portId)))

            for (bitId <- 0 until widthOf(genT())) {
                req_data(bitId) <-< forked(bitId)
                    .translateInto(cloneOf(req_data(bitId))) { (to, from) => 
                        to.addr := from.addr
                    }

                // lockstep check
                // if (bitId < widthOf(genT()) - 1) {
                //     assert(req_data(bitId).ready === req_data(bitId + 1).ready)
                // } else {
                //     assert(req_data(bitId).ready === req_tag.ready)
                // }
            }
            req_tag <-< forked(widthOf(genT()))
        }

        val stageResp = new Stage(s"StageReadResp${portId}") {
            val rollback_flag = Reg(Bool) init(False)

            val resp_tag  = stageIn(tag_ram_inst.io.read_resp(portId))
            val resp_data = data_ram_insts.map(inst => stageIn(inst.io.read_resp(portId)))

            // lockstep check
            for (bitId <- 0 until widthOf(genT()) - 1) {
                assert(resp_data(bitId).valid === resp_data(bitId + 1).valid)
            }
            assert(resp_tag.valid === resp_data(0).valid)

            val match_flag    = rollback_flag === resp_tag.context.rollback_flag
            val match_tag     = funcMatchTag(resp_tag.data, resp_tag.context.tag)
            val need_rollback = match_flag && !match_tag

            val input_need_rollback = stageDrive(stageSendReq.input_need_rollback)

            val resp = StreamJoin(resp_data :+ resp_tag)
            when (resp.fire && match_flag) {
                input_need_rollback.valid   := True
                input_need_rollback.payload := need_rollback

                when (need_rollback) {
                    rollback_flag := !rollback_flag
                }
            } otherwise {
                input_need_rollback.valid   := False
                input_need_rollback.payload.assignDontCare()
            }

            stageDrive(io.read_data(portId)) <-< resp
                .throwWhen(!match_flag || !match_tag)
                .translateInto(cloneOf(io.read_data(portId))) { (to, from) => 
                    to.fragment.data    := Vec.tabulate(sizeLine) { i => B(resp_data.map(_.data(i))).toDataType(genT()) }
                    to.fragment.context := resp_tag.context.context
                    to.last := resp_tag.context.last
                }
        }
    }}
}