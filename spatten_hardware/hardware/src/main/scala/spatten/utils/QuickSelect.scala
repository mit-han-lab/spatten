package spatten

import spinal.core._
import spinal.lib._
import spinal.lib.fsm._

import scala.collection.mutable.ArrayBuffer

object QuickSelect {
    // funcComp: The less comparator '<'
    case class Config[T <: Data](
        genT: HardType[T],
        sizeIn: Int,
        capacity: Int,
        parallelism: Int,
        funcComp: (T, T) => Bool
    )
    case class Request[T <: Data](val config: Config[T]) extends Bundle {
        import config._

        val data = Vec(genT(), sizeIn)
        val len  = UInt(log2Up(sizeIn + 1) bits)

        val target = UInt(log2Up(capacity) bits)
    }
    case class Response[T <: Data](val config: Config[T]) extends Bundle {
        import config._

        val result = genT()

        val target            = UInt(log2Up(capacity) bits)
        val num_less          = UInt(log2Up(capacity + 1) bits)
        val num_less_or_equal = UInt(log2Up(capacity + 1) bits)
    }
}

class QuickSelect[T <: Data](val config: QuickSelect.Config[T]) extends Component {
    import QuickSelect._
    import config._

    val io = new Bundle {
        val req  = slave  Stream(Fragment(Request(config)))
        val resp = master Stream(Response(config))
    }

    require(parallelism % sizeIn == 0)
    require(capacity % parallelism == 0)

    assert(!io.req.valid || io.req.last || io.req.len === sizeIn)

    type Stage = StageComponent

    val stageWidthAdapter = if (sizeIn == parallelism) new Stage("StageWidthAdapterDummy") {
        val input = stageIn(io.req)
        val output = stageOut(input)
    } else new Stage("StageWidthAdapter") {  
        // sizeIn < parallelism
        val input  = stageIn(io.req)

        val data          = Vec(Reg(Vec(genT(), sizeIn)), parallelism / sizeIn - 1)
        val buffered_data = Reg(UInt(log2Up(parallelism / sizeIn) bits)) init(0)
        val buffered_len  = Reg(UInt(log2Up(parallelism + 1) bits)) init(0)

        when (input.fire) {
            when (buffered_data === parallelism / sizeIn - 1) {
                buffered_data := 0
                buffered_len  := 0
            } otherwise {
                buffered_data := buffered_data + 1
                buffered_len  := buffered_len  + input.len
            }
            for (i <- 0 until parallelism / sizeIn - 1) {
                when (buffered_data === i) {
                    data(i) := input.data
                }
            }
        }
        
        val output = stageOut(input.throwWhen(buffered_data =/= parallelism / sizeIn - 1 && !input.last)
            .translateInto(Stream(Fragment(Request(config)))) { (to, from) => 
            to.last            := from.last
            to.fragment.len    := buffered_len + from.fragment.len
            to.fragment.target := from.fragment.target
            to.fragment.data   := Vec(data.foldLeft(new ArrayBuffer[T])(_ ++ _) ++ from.fragment.data)
        })
    }

    class RandomPivot extends Component {
        case class Request() extends Bundle {
            val data = Vec(genT(), parallelism)
            val len  = UInt(log2Up(parallelism + 1) bits)
        }
        val io = new Bundle {
            val input  = slave  Flow(Fragment(Request()))
            val pivot  = out   (genT())
        }

        val numRows = capacity / parallelism

        val random = HashRandomMatrix(0)(Counter(log2Up(numRows + 1) bits).asBits, log2Up(numRows + 1) bits)

        val pivot = Reg(genT())
        val count = Reg(UInt(log2Up(numRows + 1) bits)) init(0)

        val mask = Vec(Bool, log2Up(numRows + 1))       // mask = nextPow2(count) - 1
        mask(mask.size - 1) := count(mask.size - 1)
        for (i <- mask.size - 2 downto 0) {
            mask(i) := mask(i + 1) || count(i)
        }

        when (io.input.fire) {
            when ((random & mask.asBits.asUInt) === 0) {
                pivot := io.input.data(0)
            }

            when (io.input.last) {
                count := 0
            } otherwise {
                count := count + 1
            }
        }

        io.pivot := pivot
    }

    val buffer          = Array.fill(2) { StreamFifo(Vec(genT(), parallelism), capacity / parallelism) }
    val target          = Reg(UInt(log2Up(capacity) bits))
    val target_original = Reg(UInt(log2Up(capacity) bits))
    val current_bufid   = Reg(UInt(1 bit)) init(0)
    val buffer_lastlen  = Reg(Vec(UInt(log2Up(parallelism + 1) bits), 2))    // the length of last row in the buffer
    val count_equal_to_pivot = Reg(UInt(log2Up(capacity + 1) bits))         // number of elements that equal to pivot in the last round
    val finish_write    = Reg(Vec(Bool, 2))  // if write to buffer(i) is finished

    val buffer_remain_data = Reg(UInt(log2Up(capacity + 1) bits))   // number of elements in the current buffer
    val pivot              = Reg(genT())                            // current pivot

    val zeConfig = ZeroEliminator.Config(genT, NoData, parallelism, parallelism)
    val ze_inst    = Seq.fill(2) { new ZeroEliminatorFrag(zeConfig) }
    val pivot_inst = Seq.fill(2) { new RandomPivot }

    val response = Stream(Response(config))
    io.resp <-< response

    val areaFSM = new StateMachine {
        val input = stageWidthAdapter.output

        input.ready := False

        buffer.foreach  { _.io.push.valid := False }
        buffer.foreach  { _.io.push.payload.assignDontCare() }
        buffer.foreach  { _.io.pop.ready  := False }
        buffer.foreach  { _.io.flush      := False }

        ze_inst.foreach { _.io.req.valid  := False }
        ze_inst.foreach { _.io.req.payload.assignDontCare()  }
        ze_inst.foreach { _.io.resp.ready := False }

        pivot_inst.foreach { _.io.input.valid := False }
        pivot_inst.foreach { _.io.input.payload.assignDontCare() }
        
        response.valid := False
        response.payload.assignDontCare()

        val stateFill: State = new State with EntryPoint {
            // Fill data to buffer(0)
            whenIsActive {
                current_bufid := 0

                buffer(0).io.push.translateFrom(input) { (to, from) => 
                    to := from.fragment.data
                }
                buffer(1).io.flush := True

                when (input.fire) {
                    target          := input.target
                    target_original := input.target
                    when (input.last) {
                        buffer_lastlen(0) := input.len
                        buffer_lastlen(1) := 0
                        goto(stateRun)
                    }
                }

                pivot_inst(0).io.input.translateFrom(input.asFlow) { (to, from) => 
                    to.fragment.data := from.fragment.data
                    to.fragment.len  := from.fragment.len
                    to.last := from.last
                }
            }
        }

        def sizeOfBuffer: Vec[UInt] = {
            Vec((buffer, buffer_lastlen).zipped.map { (buf, lastlen) => 
                (buf.io.occupancy * parallelism + lastlen).resize(log2Up(capacity + 1))
            })
        }

        def prepareRun(): Unit = {
            assert(target < sizeOfBuffer(current_bufid))

            finish_write         := Vec.fill(2) { False }
            buffer_remain_data   := sizeOfBuffer(current_bufid)
            pivot                := Vec(pivot_inst.map(_.io.pivot))(current_bufid)
            count_equal_to_pivot := 0
        }

        val stateBeforeRun: State = new State {
            whenIsActive {
                prepareRun()
                goto(stateRun)
            }
        }

        val stateRun: State = new State {
            whenIsActive {
                // compare each element with pivot
                val pivot = Vec(pivot_inst.map(_.io.pivot)) (current_bufid)

                case class DataWithCompResults() extends Bundle {
                    val data     = Vec(genT(), parallelism)
                    val less     = Vec(Bool, parallelism)
                    val greater  = Vec(Bool, parallelism)
                    val in_range = Vec(Bool, parallelism)
                }

                val fifo_output = StreamMux(current_bufid, buffer.map(_.io.pop))

                // compare each element with pivot
                val comp_results = 
                    fifo_output.translateInto(Stream(Fragment(DataWithCompResults()))) { (to, from) =>
                        to.fragment.data     := from
                        to.fragment.less     := Vec(from.map(funcComp(_, pivot)))
                        to.fragment.greater  := Vec(from.map(funcComp(pivot, _)))
                        to.fragment.in_range := Vec.tabulate(parallelism) { _ < buffer_remain_data }
                        to.last := buffer_remain_data <= parallelism
                    }.stage()

                when (fifo_output.fire) {
                    buffer_remain_data := buffer_remain_data - parallelism
                }

                // count how many elements are equal to pivot
                when (comp_results.fire) {
                    count_equal_to_pivot := count_equal_to_pivot + 
                        CountOne(~(comp_results.less.asBits | comp_results.greater.asBits))
                }

                val comp_results_forked = StreamFork2(comp_results)

                val less_results    = comp_results_forked._1.translateInto(cloneOf(ze_inst(0).io.req)) { (to, from) => 
                    to.fragment.data      := from.fragment.data
                    to.fragment.keep_data := Vec((from.fragment.less zip from.fragment.in_range).map(x => x._1 && x._2))
                    to.last               := from.last
                }
                val greater_results = comp_results_forked._2.translateInto(cloneOf(ze_inst(1).io.req)) { (to, from) => 
                    to.fragment.data      := from.fragment.data
                    to.fragment.keep_data := Vec((from.fragment.greater zip from.fragment.in_range).map(x => x._1 && x._2))
                    to.last               := from.last
                }

                ze_inst(0).io.req << less_results
                ze_inst(1).io.req << greater_results

                // store the output results of zero eliminators to FIFOs
                for (i <- 0 until 2) {
                    ze_inst(i).io.resp.throwWhen(ze_inst(i).io.resp.size === 0).translateInto(buffer(i).io.push) { (to, from) => 
                        to := from.fragment.data
                    }
                    when (ze_inst(i).io.resp.isLast) {
                        buffer_lastlen(i) := ze_inst(i).io.resp.size
                        finish_write(i) := True
                    }
                }

                when ((finish_write(0) || ze_inst(0).io.resp.isLast) && (finish_write(1) || ze_inst(1).io.resp.isLast)) {
                    goto(stateAfterRun)
                }
            }
        }

        val stateAfterRun: State = new State {
            whenIsActive {
                when (sizeOfBuffer(0) <= target) {
                    buffer(0).io.flush := True

                    when (sizeOfBuffer(0) + count_equal_to_pivot > target) {
                        response.valid    := True
                        response.result   := pivot
                        response.target   := target_original
                        response.num_less := target_original - (target - sizeOfBuffer(0))
                        response.num_less_or_equal := target_original + (sizeOfBuffer(0) + count_equal_to_pivot - target)
                        when (response.ready) {
                            goto(stateFill)
                        }
                    } otherwise { // target is in the right buffer
                        current_bufid      := 1
                        target := (target - sizeOfBuffer(0) - count_equal_to_pivot).resized
                        prepareRun()
                        goto(stateRun)
                    }
                } otherwise {     // target is in the left buffer
                    buffer(1).io.flush := True

                    current_bufid      := 0
                    prepareRun()
                    goto(stateRun)
                }
            }
        }
    }
}