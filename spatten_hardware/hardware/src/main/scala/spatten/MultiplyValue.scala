package spatten

import spinal.core._
import spinal.lib._

object MultiplyValue {
    case class Request[TContext <: Data](genContext: HardType[TContext])(implicit config: SpAttenConfig) extends Bundle {
        val scores  = Vec(config.genFix(), config.numSoftMaxUnit)
        val values  = Vec(config.genFix(), config.numMultipliers)
        val size_d  = UInt(log2Up(config.sizeD + 1) bits)
        val context = genContext()
    }
    case class Response[TContext <: Data](genContext: HardType[TContext])(implicit config: SpAttenConfig) extends Bundle {
        val results = Vec(config.genFix(), config.sizeD)
        val context = genContext()
    }
}

class MultiplyValue[TContext <: Data](genContext: HardType[TContext])(implicit config: SpAttenConfig) extends Component {
    import MultiplyValue._
    import config._

    val io = new Bundle {
        val req  = slave  Stream(Fragment(Request(genContext)))
        val resp = master Stream(Response(genContext))
    }

    val areaBMR = new Area {
        val input = io.req

        val bmr_inst = new BMR(BMR.Config(
            genFix(), Fragment(genContext()), 
            numMultipliers, minD, sizeD,
            true, "FixedMult", (x: SInt, y: SInt) => x + y))

        bmr_inst.io.req.translateFrom(input) { (to, from) => 
            to.src1             := from.fragment.scores
            to.src2             := from.fragment.values
            to.num_broadcast    := from.fragment.size_d
            to.context.fragment := from.fragment.context
            to.context.last     := from.last
        }

        val output = bmr_inst.io.resp
    }

    val areaAccum = new Area {
        val input = areaBMR.output

        val sum = Vec(Reg(genFix()) init(0), sizeD)

        when (input.fire) {
            for (i <- 0 until sizeD) {
                when (input.context.last) {
                    sum(i) := 0
                } otherwise {
                    sum(i) := sum(i) + input.results(i)
                }
            }
        }

        io.resp <-< input.throwWhen(!input.context.last).translateInto(cloneOf(io.resp)) { (to, from) => 
            to.results := Vec((sum, input.results).zipped.map(_ + _))
            to.context := from.context.fragment
        }
    }
}