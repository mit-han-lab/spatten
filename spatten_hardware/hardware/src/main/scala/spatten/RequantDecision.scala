package spatten

import spinal.core._
import spinal.lib._

object RequantDecision {
    case class Request[TContext <: Data](genContext: HardType[TContext])(implicit config: SpAttenConfig) extends Bundle {
        val data      = Vec(config.genFix(), config.numSoftMaxUnit)
        val threshold = config.genFix()
        val context   = genContext()
    }
    case class Response[TContext <: Data](genContext: HardType[TContext])(implicit config: SpAttenConfig) extends Bundle {
        val data         = Vec(config.genFix(), config.numSoftMaxUnit)
        val need_requant = Bool
        val context      = genContext()
    }
}

class RequantDecision[TContext <: Data](genContext: HardType[TContext])(implicit config: SpAttenConfig) extends Component {
    import RequantDecision._
    import config._

    val io = new Bundle {
        val req  = slave  Stream(Fragment(Request (genContext)))
        val resp = master Stream(Fragment(Response(genContext)))
    }

    val areaReduce = new Area {
        val input = io.req

        val reduce_inst = new ReductionTree(ReductionTree.Config(
            genFix(), cloneOf(input.payload), 
            numSoftMaxUnit, 1, 1, 
            (x: SInt, y: SInt) => x max y))

        input.translateInto(reduce_inst.io.req) { (to, from) => 
            to.data    := from.fragment.data
            to.context := from
        }

        val output = reduce_inst.io.resp
    }

    val areaMax = new Area {
        val input = areaReduce.output

        val score_max = Reg(genFix()) init(0)

        when (input.fire) {
            when (input.context.last) {
                score_max := 0
            } otherwise {
                score_max := score_max max input.data(0)
            }
        }

        val fifo_scores = StreamFifo(cloneOf(input.payload), numBufferLines)
        val fifo_max    = StreamFifo(genFix(), 1)

        input.ready := fifo_scores.io.push.ready && (!input.context.last || fifo_max.io.push.ready)

        fifo_scores.io.push.valid   := input.fire
        fifo_scores.io.push.payload := input.payload
        fifo_max.io.push.valid      := input.fire && input.context.last
        fifo_max.io.push.payload    := score_max max input.data(0)

        fifo_max.io.pop.ready := fifo_scores.io.pop.fire && fifo_scores.io.pop.context.last

        io.resp <-< fifo_scores.io.pop
            .continueWhen(fifo_max.io.pop.valid)
            .translateInto(cloneOf(io.resp)) { (to, from) => 
                to.fragment.need_requant := fifo_max.io.pop.payload < from.context.fragment.threshold
                to.fragment.data         := from.context.fragment.data
                to.fragment.context      := from.context.fragment.context
                to.last                  := from.context.last
            }
    }
}