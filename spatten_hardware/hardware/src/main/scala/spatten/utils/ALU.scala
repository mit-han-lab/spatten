package spatten

import spinal.core._
import spinal.lib._

class ALUArray[TContext <: Data](genContext: HardType[TContext], val op: String, val widthFP: Int, val size: Int) extends Component {
    case class Request() extends Bundle {
        val src1, src2  = Vec(Bits(widthFP bits), size)
        val context     = genContext()
    }
    case class Response() extends Bundle {
        val result      = Vec(Bits(widthFP bits), size)
        val context     = genContext()
    }
    val io = new Bundle {
        val input  = slave  Stream(Request())
        val output = master Stream(Response())
    }

    val insts = Array.fill(size) { ALUnit(op, widthFP) }

    require(insts(0) != null, s"No such configuration for FPUnit: ${widthFP}-bit-${op}")

    val latency = insts(0).latency

    io.input.ready := io.output.ready

    io.output.valid   := Delay(io.input.valid, latency, io.output.ready, False)
    io.output.context := Delay(io.input.context, latency, io.output.ready)

    for (i <- 0 until size) {
        insts(i).io.enable := io.output.ready
        insts(i).io.src1   := io.input.src1(i)
        insts(i).io.src2   := io.input.src2(i)

        io.output.result(i) := insts(i).io.dst
    }

}

abstract class ALUnit(val width: Int, val latency: Int) extends BlackBox {
    val io = new Bundle {
        val clk    = in  Bool()
        val enable = in  Bool()
        val src1   = in  Bits(width bits)
        val src2   = in  Bits(width bits)
        val dst    = out Bits(width bits)
    }
    
    mapClockDomain(clock = io.clk)
}

object ALUnit {
    class FPMultiplierSingle extends ALUnit(32, 7)  { val dummy = 0 }
    class FPAdderSingle      extends ALUnit(32, 12) { val dummy = 0 }
    class FixedMultiplier12  extends ALUnit(12, 1)  { 
        clearBlackBox()
        io.dst := RegNextWhen((io.src1.asSInt * io.src2.asSInt).asBits.resizeLeft(width), io.enable)
    }
    class FixedMultiplier32  extends ALUnit(32, 1)  { 
        clearBlackBox()
        io.dst := RegNextWhen((io.src1.asSInt * io.src2.asSInt).asBits.resizeLeft(width), io.enable)
    }
    class FixedDivider16     extends ALUnit(16, 20) { val dummy = 0 }

    def apply(op: String, widthFP: Int): ALUnit = {
        (op, widthFP) match  {
            case ("FPAdd", 32)  =>    new FPAdderSingle
            case ("FPMult", 32) =>    new FPMultiplierSingle
            case ("FixedMult", 12) => new FixedMultiplier12
            case ("FixedDiv", 16) =>  new FixedDivider16
            case ("FixedMult", 32) => new FixedMultiplier32
            case _ => null
        }
    }
}