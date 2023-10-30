package spatten

import spinal.core._
import spinal.lib._

abstract class BroadcastImpl[T <: Data](genValue: HardType[T], size: Int) extends Component {
    val io = new Bundle {
        val input   = in  Vec(genValue(), size)
        val output  = out Vec(genValue(), size)
        val copy    = in  UInt(log2Up(size + 1) bits)
    }

    def getCycleCount(): Int
}

class BroadcastImplMux[T <: Data](genValue: HardType[T], size: Int) extends BroadcastImpl(genValue, size) {
    val output = Reg(Vec(genValue(), size))

    for (i <- 0 until size) {
        switch (io.copy) {
            for (j <- 0 until log2Up(size + 1)) { 
                is (1 << j) {
                    output(i) := io.input(i % (size / (1 << j)))
                }
            }
            default {
                // assert(False)
                output(i).assignDontCare()
            }
        }
    }

    io.output := output

    def getCycleCount(): Int = 1
}

object Broadcast {
    class BroadcastArray[T <: Data, TContext <: Data](genValue: HardType[T], genContext: HardType[TContext], size: Int) extends Bundle {
        val data    = Vec(genValue(), size)
        val context = genContext()
    }

    case class BroadcastInput[T <: Data, TContext <: Data](genValue: HardType[T], genContext: HardType[TContext], size: Int, maxCopy: Int)  
        extends BroadcastArray(genValue, genContext, size) {

        val copy = UInt(log2Up(maxCopy + 1) bits)
    }
    case class BroadcastOutput[T <: Data, TContext <: Data](genValue: HardType[T], genContext: HardType[TContext], size: Int)
        extends BroadcastArray(genValue, genContext, size)
}

class Broadcast[T <: Data, TContext <: Data](
    genValue: HardType[T], genContext: HardType[TContext], size: Int, minCopy: Int, maxCopy: Int) extends Component {
    
    import Broadcast._

    val io = new Bundle {
        val input  = slave  Stream(BroadcastInput(genValue, genContext, size / minCopy, maxCopy))
        val output = master Stream(BroadcastOutput(genValue, genContext, size))
    }

    require(isPow2(size))
    require(isPow2(minCopy))
    require(isPow2(maxCopy))
    require(minCopy >= 1 && minCopy <= size)
    require(maxCopy >= 1 && maxCopy <= size)
    require(minCopy <= maxCopy)

    assert(!io.input.valid || (io.input.copy & (io.input.copy - 1)) === 0)  // io.input.copy is power of 2

    val areaImpl = new ClockEnableArea(io.output.isFree) {
        val groupSize  = size / maxCopy
        val groupCount = maxCopy / minCopy

        val impl = new BroadcastImplMux(Vec(genValue(), groupSize), groupCount)

        impl.io.copy := io.input.copy >> log2Up(minCopy)    // io.input.copy / minCopy

        for (i <- 0 until groupCount) {
            for (j <- 0 until groupSize) {
                impl.io.input(i)(j) := io.input.data(i * groupSize + j)
            }
        }

        for (k <- 0 until minCopy) {
            for (i <- 0 until groupCount) {
                for (j <- 0 until groupSize) {
                    io.output.data(k * groupCount * groupSize + i * groupSize + j) := impl.io.output(i)(j)
                }
            }
        }
    }

    io.input.ready    := io.output.isFree
    io.output.valid   := Delay(io.input.valid,   areaImpl.impl.getCycleCount(), io.output.isFree, False)
    io.output.context := Delay(io.input.context, areaImpl.impl.getCycleCount(), io.output.isFree)
}