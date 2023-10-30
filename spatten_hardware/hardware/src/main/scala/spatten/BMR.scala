package spatten

import spinal.core._
import spinal.lib._

// BMR: Broadcast-Multiply-Reduce

object BMR {
    case class Config[T <: Data, TContext <: Data](
        genT: HardType[T],
        genContext: HardType[TContext],
        numMultipliers: Int,
        minBroadcast: Int,
        maxBroadcast: Int,
        useRevbit: Boolean,
        aluName: String,
        reduceFunc: (T, T) => T
    ) {
        val sizeIn  = numMultipliers / minBroadcast
        val sizeOut = maxBroadcast
    }

    case class Request[T <: Data, TContext <: Data](config: Config[T, TContext]) extends Bundle {
        val src1          = Vec(config.genT(), config.sizeIn)
        val src2          = Vec(config.genT(), config.numMultipliers)
        val num_broadcast = UInt(log2Up(config.maxBroadcast + 1) bits)
        val context       = config.genContext()
    }
    case class Response[T <: Data, TContext <: Data](config: Config[T, TContext]) extends Bundle {
        val results       = Vec(config.genT(), config.sizeOut)
        val context       = config.genContext()
    }
}

class BMR[T <: Data, TContext <: Data](config: BMR.Config[T, TContext]) extends Component {
    import BMR._
    import config._

    val io = new Bundle {
        val req  = slave Stream(Request(config))
        val resp = master Stream(Response(config))
    }

    require(isPow2(numMultipliers))

    
    case class StageBroadcastOutput() extends Bundle {
        val src1          = Vec(genT(), numMultipliers)
        val src2          = Vec(genT(), numMultipliers)
        val num_broadcast = UInt(log2Up(maxBroadcast + 1) bits)
        val context       = genContext()
    }
    case class StageALUOutput() extends Bundle {
        val mult_result   = Vec(genT(), numMultipliers)
        val num_broadcast = UInt(log2Up(maxBroadcast + 1) bits)
        val context       = genContext()
    }

    private def revbit(x: Int, width: Int): Int = {
        var ret = 0
        for (i <- 0 until width) {
            ret |= ((x >> i) & 1) << (width - 1 - i)
        }
        ret
    }

    val stageBroadcast = new Area {
        val input = io.req

        // TODO: We need two broadcast mode
        val broadcast = new Broadcast(genT, cloneOf(io.req.payload), numMultipliers, minBroadcast, maxBroadcast)
        
        broadcast.io.input.translateFrom(input) { (to, from) => 
            to.context  := from
            if (useRevbit) {
                to.data := Vec.tabulate(sizeIn) { i => from.src1(revbit(i, log2Up(sizeIn))) }
            } else {
                to.data := from.src1
            }
            to.copy     := from.num_broadcast
        }
        
        val output = broadcast.io.output.translateInto(Stream(StageBroadcastOutput())) { (to, from) => 
            if (useRevbit) {
                to.src1      := Vec.tabulate(numMultipliers) { i => from.data(revbit(i, log2Up(numMultipliers))) }
            } else {
                to.src1      := from.data
            }
            to.src2          := from.context.src2
            to.num_broadcast := from.context.num_broadcast
            to.context       := from.context.context
        }
    }

    val stageALU = new Area {
        val input = stageBroadcast.output

        val mult_array = new ALUArray(StageBroadcastOutput(), aluName, widthOf(genT()), numMultipliers)

        mult_array.io.input.translateFrom(input) { (to, from) =>
            to.src1    := Vec(from.src1.map(_.asBits))
            to.src2    := Vec(from.src2.map(_.asBits))
            to.context := from
        }

        val output = mult_array.io.output.translateInto(Stream(StageALUOutput())) { (to, from) =>
            to.mult_result   := Vec(from.result.map({ bits => val ret = genT(); ret.assignFromBits(bits); ret }))
            to.num_broadcast := from.context.num_broadcast
            to.context       := from.context.context
        }
    }

    val stageReduce = new Area {
        val input = stageALU.output

        val reduce_inst = new ReductionTree(ReductionTree.Config(
            genT(), genContext(), 
            numMultipliers, maxBroadcast, minBroadcast, 
            reduceFunc))

        reduce_inst.io.req << input.translateInto(cloneOf(reduce_inst.io.req)) { (to, from) =>
            if (useRevbit) {
                to.data   := Vec.tabulate(numMultipliers) { i => from.mult_result(revbit(i, log2Up(numMultipliers))) }
            } else {
                to.data   := from.mult_result
            }
            to.num_output := from.num_broadcast
            to.context    := from.context
        }

        io.resp << reduce_inst.io.resp.translateInto(cloneOf(io.resp)) { (to, from) => 
            to.results := from.data
            to.context := from.context
        }
    }
}