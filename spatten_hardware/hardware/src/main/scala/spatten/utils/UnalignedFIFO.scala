package spatten

import spinal.core._
import spinal.lib._

object UnalignedFIFO {
    case class UnalignedPop[T <: Data](genT: HardType[T], sizePop: Int) extends Bundle {
        val data           = Vec(genT(), sizePop)
        val num_valid_data = UInt(log2Up(sizePop + 1) bits)
    }
}

class FIFOUnalignedPop[T <: Data](genT: HardType[T], sizePush: Int, sizePop: Int) extends Component {
    import UnalignedFIFO._

    val io = new Bundle {
        val push    = slave  Stream(Vec(genT(), sizePush))
        val pop     = master Stream(UnalignedPop(genT, sizePop))  // valid is True when there's at least one element in the FIFO
        val num_pop = in UInt(log2Up(sizePop + 1) bits)           // pop num_pop elements from the FIFO when pop.ready is True
    }

    val capacity = (sizePush max sizePop) * 2

    val data = Vec(Reg(genT()), capacity)
    val size = Reg(UInt(log2Up(capacity + 1) bits)) init(0)
    val head = Reg(UInt(log2Up(capacity) bits)) init(0)
    val tail = Reg(UInt(log2Up(capacity) bits)) init(0)

    io.push.ready := (size + sizePush) <= capacity

    size := (size + io.push.fire.asUInt * sizePush - ((io.pop.fire.asUInt * io.num_pop) min size)).resized

    when (io.push.fire) {
        for (i <- 0 until capacity) {
            for (j <- 0 until sizePush) {
                when (i === (tail + j).resize(log2Up(capacity))) {
                    data(i) := io.push.payload(j)
                }
            }
        }
        tail := (tail + sizePush).resized
    }

    when (io.pop.fire) {
        head := (head + io.num_pop).resized
    }

    for (i <- 0 until sizePop) {
        io.pop.data(i) := data((head + i).resize(log2Up(capacity)))
    }

    io.pop.valid  := size >= 1
    io.pop.num_valid_data := (size min sizePop).resized

    // val blackbox = new BlackBox {
    //     val io = new Bundle {
    //         val push    = slave  Stream(Vec(genT(), sizePush))
    //         val pop     = master Stream(UnalignedPop(genT, sizePop))  // valid is True when there's at least one element in the FIFO
    //         val num_pop = in UInt(log2Up(sizePop + 1) bits)           // pop num_pop elements from the FIFO when pop.ready is True
    //     }
    // }

    // blackbox.io.push << io.push
    // blackbox.io.pop  >> io.pop
    // blackbox.io.num_pop := io.num_pop
}

