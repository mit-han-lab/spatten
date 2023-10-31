package spatten

import spinal.core._
import spinal.lib._

object TopK {
    case class Config[T <: Data, TContext <: Data](
        genT: HardType[T], genContext: HardType[TContext],
        parallelism: Int,
        capacity: Int,
        funcComp: (T, T) => Bool,
        funcEqual: (T, T) => Bool
    )
    case class Request[T <: Data, TContext <: Data](config: Config[T, TContext]) extends Bundle {
        val data    = Vec(config.genT(), config.parallelism)
        val len     = UInt(log2Up(config.parallelism + 1) bits)
        val target  = UInt(log2Up(config.capacity) bits)
        val context = config.genContext()
    }
    case class Response[T <: Data, TContext <: Data](config: Config[T, TContext]) extends Bundle {
        val data    = Vec(config.genT(), config.parallelism)
        val len     = UInt(log2Up(config.parallelism + 1) bits)
        val context = config.genContext()
    }
}

class TopKBase[T <: Data, TContext <: Data](val config: TopK.Config[T, TContext]) extends Component {
    import TopK._
    val io = new Bundle {
        val req  = slave Stream (Fragment(Request(config)))
        val resp = master Stream(Fragment(Response(config)))
    }
}

// A dummy module which uses precomputed latency to simulate the behaviors of TopK module
// TODO: switch to the real TopK module during simulation
class TopKDummy[T <: Data, TContext <: Data](config: TopK.Config[T, TContext], funcGetCycle: TContext => UInt) extends TopKBase(config) {
    import TopK._
    import config._

    val cycle = Counter(32 bits)
    // cycle.increment()

    case class ReqData() extends Bundle {
        val req      = Request(config)
        // val in_cycle = UInt(32 bits)
    }

    val data_fifo = StreamFifo(Fragment(ReqData()), capacity / parallelism)

    data_fifo.io.push << io.req.translateInto(cloneOf(data_fifo.io.push)) { (to, from) => 
        to.fragment.req      := from.fragment
        // to.fragment.in_cycle := cycle.value
        to.last := from.last
    }

    val sum_len = Reg(UInt(log2Up(capacity + 1) bits)) init(0)

    when (data_fifo.io.pop.fire) {
        when (data_fifo.io.pop.last) {
            sum_len := 0
        } otherwise {
            sum_len := sum_len + data_fifo.io.pop.req.len
        }
    }

    when (!data_fifo.io.pop.valid || data_fifo.io.pop.fire) {
        cycle.clear()
    } otherwise {
        cycle.increment()
    }

    io.resp << data_fifo.io.pop
        .haltWhen(data_fifo.io.pop.isFirst && cycle < funcGetCycle(data_fifo.io.pop.req.context))
        .throwWhen(sum_len >= data_fifo.io.pop.req.target + 1).translateInto(cloneOf(io.resp)) { (to, from) => 
            
            when (sum_len + from.fragment.req.len >= data_fifo.io.pop.req.target + 1) {
                to.fragment.len := (data_fifo.io.pop.req.target + 1 - sum_len).resized
                to.last := True
            } otherwise {
                to.fragment.len := from.fragment.req.len
                to.last := from.last
            }
            to.fragment.data    := from.fragment.req.data
            to.fragment.context := from.fragment.req.context
        }

}

// The real topk module
class TopK[T <: Data, TContext <: Data](config: TopK.Config[T, TContext]) extends TopKBase(config) {
    import TopK._
    import config._

    // val io = new Bundle {
    //     val req  = slave Stream (Fragment(Request(config)))
    //     val resp = master Stream(Fragment(Response(config)))
    // }

    case class ReqData() extends Bundle {
        val data = Vec(genT(), parallelism)
        val len  = UInt(log2Up(parallelism + 1) bits)
    }

    case class ReqDataWithCompResults() extends Bundle {
        val data             = Vec(genT(), parallelism)
        val less_than_k      = Vec(Bool,   parallelism)
        val equal_to_k       = Vec(Bool,   parallelism)
        val equal_prefix_sum = Vec(UInt(log2Up(parallelism + 1) bits), parallelism)
        val keep_equal_num   = UInt(log2Up(capacity + 1) bits)
    }

    val quickSelectConfig = QuickSelect.Config(
        genT, 
        parallelism, 
        capacity, 
        parallelism * 2, 
        funcComp)
    val quickselect_inst = new QuickSelect(quickSelectConfig)
    val data_fifo        = StreamFifo(Fragment(ReqData()), capacity / parallelism)
    val context_fifo     = StreamFifo(genContext(), 2)

    val areaQuickSelectReq = new Area {
        val input = io.req

        val input_forked = StreamFork(input, 3)

        quickselect_inst.io.req << input_forked(0).translateInto(cloneOf(quickselect_inst.io.req)) { (to, from) => 
            to.fragment.data   := from.fragment.data
            to.fragment.len    := from.fragment.len
            to.fragment.target := from.fragment.target
            to.last            := from.last
        }

        data_fifo.io.push << input_forked(1).translateInto(cloneOf(data_fifo.io.push)) { (to, from) => 
            to.fragment.data   := from.fragment.data
            to.fragment.len    := from.fragment.len
            to.last            := from.last
        }

        context_fifo.io.push << input_forked(2)
            .throwWhen(input_forked(2).isTail)
            .translateInto(cloneOf(context_fifo.io.push)) { (to, from) => 
                to := from.fragment.context
            }
    }

    val areaCompareK = new Area {
        val input = data_fifo.io.pop

        quickselect_inst.io.resp.ready := input.fire && input.last

        val output = input.continueWhen(quickselect_inst.io.resp.valid)
            .translateInto(Stream(Fragment(ReqDataWithCompResults()))) { (to, from) => 
                for (i <- 0 until parallelism) {
                    val in_range = i < from.fragment.len

                    to.fragment.data(i)        := from.fragment.data(i)
                    to.fragment.less_than_k(i) := funcComp(from.fragment.data(i), quickselect_inst.io.resp.result) && in_range
                    to.fragment.equal_to_k(i)  := funcEqual(from.fragment.data(i), quickselect_inst.io.resp.result) && in_range
                }
                to.fragment.keep_equal_num := quickselect_inst.io.resp.target - quickselect_inst.io.resp.num_less
                to.fragment.equal_prefix_sum.assignDontCare()
                to.last := from.last
            }.stage()
    }

    val areaPrefixSum = new Area {
        val input = areaCompareK.output

        val psumConfig = PrefixSum.Config(
            UInt(log2Up(parallelism + 1) bits), 
            Fragment(ReqDataWithCompResults()), 
            parallelism, 
            (x: UInt, y: UInt) => x + y
        )
        val prefixsum_inst = new PrefixSum(psumConfig)

        prefixsum_inst.io.req << input.translateInto(cloneOf(prefixsum_inst.io.req)) { (to, from) => 
            to.data    := Vec(from.fragment.equal_to_k.map(_.asUInt.resize(log2Up(parallelism + 1))))
            to.context := from
        }

        val output = prefixsum_inst.io.resp.translateInto(Stream(Fragment(ReqDataWithCompResults()))) { (to, from) => 
            to := from.context
            to.fragment.equal_prefix_sum.allowOverride := from.data
        }
    }

    val areaZeroEliminator = new Area {
        val input = areaPrefixSum.output

        val cnt_equal_to_k = Reg(UInt(capacity + 1 bits)) init(0)

        when (input.fire) {
            when (input.last) {
                cnt_equal_to_k := 0
            } otherwise {
                cnt_equal_to_k := cnt_equal_to_k + input.equal_prefix_sum(parallelism - 1)
            }
        }

        val zeConfig = ZeroEliminator.Config(genT, NoData, parallelism, parallelism)
        val ze_inst = new ZeroEliminatorFrag(zeConfig)

        ze_inst.io.req <-< input.translateInto(cloneOf(ze_inst.io.req)) { (to, from) => 
            for (i <- 0 until parallelism) { 
                to.fragment.data(i)      := from.fragment.data(i)
                to.fragment.keep_data(i) := 
                    from.fragment.less_than_k(i) || 
                    (from.fragment.equal_to_k(i) && from.fragment.equal_prefix_sum(i) + cnt_equal_to_k <= from.fragment.keep_equal_num)
            }
            to.last := from.last
        }

        context_fifo.io.pop.ready := ze_inst.io.resp.fire && ze_inst.io.resp.last

        io.resp << ze_inst.io.resp.continueWhen(context_fifo.io.pop.valid).translateInto(cloneOf(io.resp)) { (to, from) => 
            to.fragment.data    := from.fragment.data
            to.fragment.len     := from.fragment.size
            to.fragment.context := context_fifo.io.pop.payload
            to.last             := from.last
        }
    }

}