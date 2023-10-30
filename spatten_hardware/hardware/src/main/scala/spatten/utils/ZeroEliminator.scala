package spatten

import spinal.core._
import spinal.lib._

object ZeroEliminator {
    sealed trait RegisterPolicy
    case object LowLatency   extends RegisterPolicy
    case object MaxFrequency extends RegisterPolicy

    case class Config[T <: Data, TContext <: Data](
        genT: HardType[T], 
        genContext: HardType[TContext], 
        sizeIn: Int,
        sizeOut: Int,
        policy: RegisterPolicy = LowLatency
    )
    case class Request[T <: Data, TContext <: Data](config: Config[T, TContext]) extends Bundle {
        import config._

        val data      = Vec(genT(), sizeIn)
        val keep_data = Vec(Bool, sizeIn)

        val context = genContext()
    }
    case class Response[T <: Data, TContext <: Data](config: Config[T, TContext]) extends Bundle {
        import config._

        val data = Vec(genT(), sizeOut)
        val size = UInt(log2Up(sizeOut + 1) bits)

        val context = genContext()
    }
}

object ShiftVec {
    def shiftRight[T <: Data](vec: Vec[T], n: UInt, maxN: Int): Vec[T] = {
        if (vec.isEmpty) {
            vec
        } else {
            val vecBits = vec.map(_.asBits)
            val width = vecBits.head.getBitsWidth
            vecBits.foreach(x => require(x.getBitsWidth == width))

            val transposed = Vec.tabulate(width) { i => B(vecBits.map(_(i))) }
            val shifted = Vec(transposed.map(_ << n))
            val result = Vec.tabulate(vec.size + maxN) { i => 
                val x = cloneOf(vec.head)
                x.assignFromBits(B(shifted.map(_(i))))
                x
            }

            result
        }
    }
}

class ZeroEliminatorFrag[T <: Data, TContext <: Data](val config: ZeroEliminator.Config[T, TContext]) extends Component {
    import ZeroEliminator._
    val io = new Bundle {
        val req  = slave  Stream(Fragment(Request (config)))
        val resp = master Stream(Fragment(Response(config)))
    }

    type Stage = StageComponent

    val stageZE = new Stage("StageZE") {
        val input = stageIn(io.req)

        val inst = new ZeroEliminator(Config(config.genT, Fragment(config.genContext()), config.sizeIn, config.sizeIn max config.sizeOut, config.policy))

        inst.io.req.translateFrom(io.req) { (to, from) => 
            to.data             := from.data
            to.keep_data        := from.keep_data
            to.context.fragment := from.context
            to.context.last     := from.last
        }

        val output = stageOut(inst.io.resp.translateInto(cloneOf(io.resp)) { (to, from) => 
            to.data    := from.data
            to.size    := from.size
            to.context := from.context.fragment
            to.last    := from.context.last
        })
    }

    val stageRealign = new Stage("StageRealign") {
        val input = stageIn(stageZE.output)
        val output = stageOutDef(cloneOf(io.resp))

        val valid = Reg(Bool()) init(False)
        val last = Reg(Bool()) init(False)
        val size = Reg(UInt(log2Up(config.sizeOut + input.data.size + 1) bits)) init(0)
        val data = Vec(Reg(config.genT()), config.sizeOut + input.data.size)

        when (input.isFirst) {
            input.ready := !valid || (output.last && output.fire)
        } otherwise {
            input.ready := size + input.size <= config.sizeOut + input.data.size
        }

        output.valid := valid && (last || size >= config.sizeOut)
        output.data := Vec(data.slice(0, config.sizeOut))
        output.last := size <= config.sizeOut
        output.size := Mux(size > config.sizeOut, U(config.sizeOut), size)

        when (input.fire) {
            valid := True
        } elsewhen (output.fire && output.last) {
            valid := False
        }

        when (input.fire) {
            last := input.last
        }

        val size_pop = output.fire ? output.size | U(0)
        val size_push = input.fire ? input.size | U(0)
        size := size + size_push - size_pop

        val input_shifted = ShiftVec.shiftRight(input.data, size - size_pop, config.sizeOut)
        for (i <- 0 until config.sizeOut + input.data.size) {
            when (i >= size - size_pop) {
                data(i) := input_shifted(i)
            } elsewhen (output.fire) {
                data(i) := data(i + config.sizeOut)
            }
        }
    }

    io.resp << stageRealign.output.stage()
}

class ZeroEliminator[T <: Data, TContext <: Data](val config: ZeroEliminator.Config[T, TContext]) extends Component {
    import ZeroEliminator._
    import config._

    val io = new Bundle {
        val req  = slave  Stream(Request (config))
        val resp = master Stream(Response(config))
    }

    require(isPow2(sizeIn))

    case class LayerData() extends Bundle {
        val data      = Vec(genT(), sizeIn)
        val psum      = Vec(UInt(log2Up(sizeIn + 1) bits), sizeIn)
        // val invalid   = Vec(Bool, sizeIn)
    }

    val psumConfig = PrefixSum.Config(
        UInt(log2Up(sizeIn + 1) bits), 
        Request(config), 
        sizeIn, 
        (x: UInt, y: UInt) => x + y
    )
    val prefixsum_inst = new PrefixSum(psumConfig)

    prefixsum_inst.io.req << io.req.translateInto(cloneOf(prefixsum_inst.io.req)) { (to, from) => 
        to.data    := Vec(from.keep_data.map((x: Bool) => (!x).asUInt))
        to.context := from
    }

    val input = cloneOf(prefixsum_inst.io.resp)
    input << prefixsum_inst.io.resp

    val numLayers = log2Up(sizeIn)
    val layers = Vec(LayerData(), numLayers + 1)

    var numRegisterLayers = 0

    // data: the partially shifted data
    // psum: the partially shifted prefix sum, bit (k-1)~(PSUM_BIT_WIDTH-1) is useful
    layers(0).data    := input.context.data
    layers(0).psum    := Vec((input.context.keep_data, input.data).zipped.map(Mux(_, _, U(0))))

    for (k <- 0 until numLayers) {
        for (i <- 0 until sizeIn) {
            // for next layer
            val data    = genT()
            val psum    = UInt(log2Up(sizeIn + 1) bits)

            // ** RULE: psum[i][k] MUST be 0 if [i] is a invalid element **

            // data[i + (1<<k)]
            val data_shift = cloneOf(data)
            // psum[i + (1<<k)]
            val psum_shift = cloneOf(psum)
            if (i + (1 << k) < sizeIn) {
                psum_shift := layers(k).psum(i + (1 << k))
                data_shift := layers(k).data(i + (1 << k))
            } else {
                psum_shift := 0
                data_shift.assignDontCare()
            }

            // The element [i + 2^k] has higher priority,
            // because when [i] is a invalid element, psum[i] = 0 (satisfying psum[i][k] == 0) and data[i] is an arbitrary value
            //  if we use a OR to combine data[i] and data[i + 2^k], data[i] and data[i + 2^k] may be ORed together, which causes an incorrect output
            when (psum_shift(k)) {
                data := data_shift
                psum := psum_shift
            } elsewhen (~layers(k).psum(i)(k)) {
                // psum[i][k] == 0 <=> either [i] is a invalid element or [i] is valid but it doesn't shift in this stage
                data := layers(k).data(i)
                psum := layers(k).psum(i)
            } otherwise {
                // otherwise [i] is shifted and it now becomes a invalid element, we need to set psum[k+1] to 0 (see RULE (*) above)
                data.assignDontCare()
                psum := 0
            }

            if (policy == MaxFrequency || k == numLayers - 1) {
                layers(k + 1).data(i) := RegNextWhen(data, input.ready)
                layers(k + 1).psum(i) := RegNextWhen(psum, input.ready)
                numRegisterLayers += 1
            } else {
                layers(k + 1).data(i) := data
                layers(k + 1).psum(i) := psum
            }
        }
    }

    val delay_valid   = Delay(input.valid,              numRegisterLayers, input.ready, False)
    val delay_context = Delay(input.context.context,    numRegisterLayers, input.ready)
    val delay_size    = Delay(sizeIn - input.data.last, numRegisterLayers, input.ready)

    input.ready := !delay_valid || io.resp.ready

    io.resp.valid   := delay_valid
    io.resp.data    := Vec(layers(numLayers).data.slice(0, sizeOut))
    io.resp.size    := Mux(delay_size <= sizeOut, delay_size.resized, U(sizeOut))
    io.resp.context := delay_context
}