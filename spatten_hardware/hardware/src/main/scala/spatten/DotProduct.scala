package spatten

import spinal.core._
import spinal.lib._
import spinal.lib.fsm._

import scala.collection.mutable.ArrayBuffer

object DotProduct {
    case class Request[TContext <: Data](genContext: HardType[TContext])(implicit config: SpAttenConfig) extends Bundle {
        val key     = Vec(config.genFix(), config.numMultipliers)
        val query   = Vec(config.genFix(), config.sizeD)
        val size_d  = UInt(log2Up(config.sizeD + 1) bits)
        val context = genContext()
    }
    case class Response[TContext <: Data](genContext: HardType[TContext])(implicit config: SpAttenConfig) extends Bundle {
        val results = Vec(config.genFix(), config.numSoftMaxUnit)
        val context = genContext()
    }
}

class DotProduct[TContext <: Data](genContext: HardType[TContext])(implicit config: SpAttenConfig) extends Component {
    import DotProduct._
    import config._

    val io = new Bundle {
        val req  = slave  Stream(Fragment(Request (genContext)))
        val resp = master Stream(Fragment(Response(genContext)))
    }

    case class StageBroadcastOutput() extends Bundle {
        val key     = Vec(genFix(), numMultipliers)
        val query   = Vec(genFix(), numMultipliers)
        val size_d  = UInt(log2Up(sizeD + 1) bits)
        val context = genContext()
    }
    case class StageALUOutput() extends Bundle {
        val mult_result = Vec(genFix, numMultipliers)
        val size_d      = UInt(log2Up(sizeD + 1) bits)
        val context     = genContext()
    }

    val bmr_inst = new BMR(BMR.Config(
        genFix(), Fragment(genContext()), 
        numMultipliers, numMultipliers / sizeD, numMultipliers / minD,
        false, "FixedMult", (x: SInt, y: SInt) => x + y))

    bmr_inst.io.req.translateFrom(io.req) { (to, from) => 
        to.src1             := from.fragment.query
        to.src2             := from.fragment.key
        to.num_broadcast    := numMultipliers / sizeD   // should be numMultipliers / size_d
        to.context.fragment := from.fragment.context
        to.context.last     := from.last
    }

    bmr_inst.io.resp.translateInto(io.resp) { (to, from) => 
        to.fragment.results := from.results
        to.fragment.context := from.context.fragment
        to.last             := from.context.last
    }

}