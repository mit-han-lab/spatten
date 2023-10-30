package spatten

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._

object AXIDataLoader {
    case class Request[TContext <: Data](axiConfig: Axi4Config, genContext: HardType[TContext]) extends Bundle {
        val addr         = UInt(axiConfig.addressWidth bits)
        val len          = UInt(axiConfig.addressWidth bits)
        val read_enable  = Bool
        
        val context      = genContext()
    }
    case class Response[TContext <: Data](axiConfig: Axi4Config, genContext: HardType[TContext]) extends Bundle {
        val addr        = UInt(axiConfig.addressWidth bits)
        val len         = UInt(axiConfig.addressWidth bits)
        val read_enable = Bool

        val data        = Bits(axiConfig.dataWidth bits)
        // val partid      = UInt(axiConfig.addressWidth bits)

        val context     = genContext()
    }
}

class AXIDataLoader[TContext <: Data](axiConfig: Axi4Config, genContext: HardType[TContext]) extends Component {
    import AXIDataLoader._

    val io = new Bundle {
        val req  = slave  Stream(Request (axiConfig, genContext))
        val resp = master Stream(Fragment(Response(axiConfig, genContext)))
        val bus  = master(Axi4ReadOnly(axiConfig))
    }

    val req_fifo = StreamFifo(Request(axiConfig, genContext), 64)

    val areaSendAddr = new Area {
        val input = io.req

        assert(!input.valid || !input.read_enable || input.len === axiConfig.dataWidth / 8,
            Seq(s"AXI DataLoader only supports transaction len == dataWidth, expected ${axiConfig.dataWidth / 8} got ", input.len))

        val input_forked = StreamFork2(input, true)

        req_fifo.io.push <-< input_forked._1

        io.bus.ar <-< input_forked._2
            .throwWhen(!input_forked._2.read_enable)
            .translateInto(cloneOf(io.bus.ar)) { (to, from) => 
                to.addr := from.addr
            }
    }

    val areaRecvData = new Area {
        val input    = io.bus.r
        val fifo_pop = req_fifo.io.pop

        val dummy_input = cloneOf(input)
        dummy_input.valid := True
        dummy_input.payload.assignDontCare()

        val input_or_dummy = StreamMux(fifo_pop.read_enable.asUInt, List(dummy_input, input))

        io.resp << StreamJoin.arg(input_or_dummy, fifo_pop).translateInto(cloneOf(io.resp)) { (to, from) => 
            to.fragment.addr        := fifo_pop.addr
            to.fragment.len         := fifo_pop.len
            to.fragment.read_enable := fifo_pop.read_enable
            to.fragment.data        := input.data
            // to.fragment.partid      := 0
            to.fragment.context     := fifo_pop.context
            to.last := True
        }
    }

    // val blackbox = new BlackBox {
    //     val io = new Bundle {
    //         val req  = slave  Stream(Request (axiConfig, genContext))
    //         val resp = master Stream(Fragment(Response(axiConfig, genContext)))
    //         val bus  = master(Axi4ReadOnly(axiConfig))
    //     }
    // }

    // blackbox.io.req  << io.req
    // blackbox.io.resp >> io.resp
    // blackbox.io.bus  >> io.bus
}