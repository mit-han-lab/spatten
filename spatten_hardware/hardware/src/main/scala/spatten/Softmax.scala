package spatten

import spinal.core._
import spinal.lib._

object Softmax {
    case class Request[TContext <: Data](genContext: HardType[TContext])(implicit config: SpAttenConfig) extends Bundle {
        val data    = Vec(config.genFix(), config.numSoftMaxUnit)
        val context = genContext()
    }
    case class Response[TContext <: Data](genContext: HardType[TContext])(implicit config: SpAttenConfig) extends Bundle {
        val results = Vec(config.genFix(), config.numSoftMaxUnit)
        val context = genContext()
    }
}

// A dummy module since we don't get access to the floating point IPs in verilator
// We use external area/power numbers to evaluate the cost of Softmax 
// TODO: maybe use a DPI-based FP units during simulation
class Softmax[TContext <: Data](genContext: HardType[TContext])(implicit config: SpAttenConfig) extends Component {

    import Softmax._
    import config._

    val io = new Bundle {
        val req  = slave (Stream(Fragment(Request(genContext))))
        val resp = master(Stream(Fragment(Response(genContext))))
    }

    io.resp.translateFrom(io.req) { (to, from) => 
        to.fragment.results := from.fragment.data
        to.fragment.context := from.fragment.context
        to.last := from.last
    }
/*
    class StageFindMaxOutput extends Bundle {
        val rowid  = UInt(log2Up(config.sizeN) bits)
        val value  = config.genFP
        val maxv   = config.genFP
    }
    class StageSubtractOutput extends Bundle {
        val rowid = UInt(log2Up(config.sizeN) bits)
        val value = config.genFP
    }
    class StageExponentLUTOutput extends Bundle {
        val rowid        = UInt(log2Up(config.sizeN) bits)
        val value_top    = config.genFP
        val value_bottom = config.genFP
    }
    class StageMultiplyOutput extends StageSubtractOutput {
    }
    class StageSumOutput extends StageSubtractOutput {
        val sum = config.genFP
    }

    class StreamStatistics(genAccumulator: => Accumulator) extends Component {
        case class StreamStatisticsInput() extends Bundle {
            val rowid  = UInt(log2Up(config.sizeN) bits)
            val value  = config.genFP
        }
        case class StreamStatisticsOutput() extends Bundle {
            val rowid = UInt(log2Up(config.sizeN) bits)
            val value = config.genFP
            val stats = config.genFP
        }
        val io = new Bundle {
            val input = slave Stream(StreamStatisticsInput())
            val output = master Stream(StreamStatisticsOutput())
        }

        val fifo_raw  = StreamFifo(StreamStatisticsInput(), config.sizeN)
        val fifo_stats = StreamFifo(config.genFP, 2)

        val accum_inst = genAccumulator

        fifo_stats.io.push.valid := False
        fifo_stats.io.push.payload.assignDontCare()
        fifo_stats.io.pop.ready := False

        val (input1, input2) = StreamFork2(io.input)
        fifo_raw.io.push << input1
        accum_inst.io.input.translateFrom(input2) { (to, from) => 
            to.fragment := from.value
            to.last     := from.rowid === config.sizeN - 1
        }
        fifo_stats.io.push << accum_inst.io.output

        io.output << fifo_raw.io.pop.haltWhen(!fifo_stats.io.pop.valid).translateInto(Stream(StreamStatisticsOutput())) { (to, from) =>
            to.rowid := from.rowid
            to.value := from.value
            to.stats := fifo_stats.io.pop.payload
        }

        when (io.output.fire) {
            when (io.output.rowid === config.sizeN - 1) {
                fifo_stats.io.pop.ready := True
            }
        }
    }

    val stageFindMax = new Area {
        val stats = new StreamStatistics((a, b) => Max(a.asUInt, b.asUInt).asBits, 0)

        stats.io.input.translateFrom(io.dotproduct_result) { (to, from) =>
            to.assignAllByName(from)
        }

        val output = stats.io.output.translateInto(Stream(new StageFindMaxOutput())) { (to, from) =>
            to.rowid := from.rowid
            to.value := from.value
            to.maxv  := from.stats
        }
    }

    val stageSubtract = new Area {
        val input = stageFindMax.output

        val output = input.translateInto(Stream(new StageSubtractOutput())) { (to, from) =>
            to.rowid := from.rowid
            to.value := (from.value.asUInt - from.maxv.asUInt).asBits
        }.stage()
    }

    val stageExponentLUT = new Area {
        val input = stageSubtract.output

        val lutTop = Mem(for (i <- 0 until (1 << config.widthValue / 2)) yield { 
            // TODO: Fill actual values here
            U(0, config.widthValue bits) 
        })
        val lutBottom = Mem(for (i <- 0 until (1 << config.widthValue / 2)) yield { 
            // TODO: Fill actual values here
            U(0, config.widthValue bits) 
        })

        val value_top    = lutTop.readSync(input.value(config.widthValue - 1 downto config.widthValue / 2).asUInt, input.fire)
        val value_bottom = lutBottom.readSync(input.value(config.widthValue / 2 - 1 downto 0).asUInt, input.fire)

        val output = input.stage().translateInto(Stream(new StageExponentLUTOutput())) { (to, from) =>
            to.rowid        := from.rowid
            to.value_top    := value_top.asBits
            to.value_bottom := value_bottom.asBits
        }
    }

    val stageMult = new Area {
        val input = stageExponentLUT.output

        val multiplier = new ALUArray("FixedMult", config.widthValue, 1, input.rowid.getBitsWidth)

        multiplier.io.input.translateFrom(input) { (to, from) => 
            to.src1(0) := from.value_top
            to.src2(0) := from.value_bottom
            to.passthrough := from.rowid.asBits
        }

        val output = multiplier.io.output.translateInto(Stream(new StageMultiplyOutput())) { (to, from) =>
            to.value := from.result(0)
            to.rowid.assignFromBits(from.passthrough)
        }
    }

    val stageSum = new Area {
        val input = stageMult.output

        val stats = new StreamStatistics(new Accumulator(ALUnit("FPAdd", config.widthValue)))

        stats.io.input.translateFrom(input) { (to, from) =>
            to.assignAllByName(from)
        }

        val output = stats.io.output.translateInto(Stream(new StageSumOutput())) { (to, from) =>
            to.rowid := from.rowid
            to.value := from.value
            to.sum   := from.stats
        }
    }

    val stageDiv = new Area {
        val input = stageSum.output

        val divider = new ALUArray("FixedDiv", config.widthValue, 1, input.rowid.getBitsWidth)

        divider.io.input.translateFrom(input) { (to, from) =>
            to.src1(0) := from.value
            to.src2(0) := from.sum
            to.passthrough := from.rowid.asBits
        }

        io.output << divider.io.output.translateInto(Stream(ExponentOutput(config))) { (to, from) => 
            to.value := from.result(0)
            to.rowid.assignFromBits(from.passthrough)
        }
    }
    */
}