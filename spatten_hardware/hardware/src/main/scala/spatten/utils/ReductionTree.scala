package spatten

import spinal.core._
import spinal.lib._

object ReductionTree {
    case class Config[T <: Data, TContext <: Data](
        genT: HardType[T],
        genContext: HardType[TContext],
        sizeIn: Int,
        sizeOut: Int,
        sizeMinOut: Int,
        reduceFunc: (T, T) => T
    )
    case class Request[T <: Data, TContext <: Data](config: Config[T, TContext]) extends Bundle {
        val data       = Vec(config.genT(), config.sizeIn)
        val num_output = UInt(log2Up(config.sizeOut + 1) bits)
        val context    = config.genContext()
    }
    case class Response[T <: Data, TContext <: Data](config: Config[T, TContext]) extends Bundle {
        val data    = Vec(config.genT(), config.sizeOut)
        val context = config.genContext()
    }
}

class ReductionTree[T <: Data, TContext <: Data](config: ReductionTree.Config[T, TContext]) extends Component {
    import ReductionTree._
    import config._

    val io = new Bundle {
        val req  = slave  Stream(Request (config))
        val resp = master Stream(Response(config))
    }

    require(isPow2(sizeIn))
    require(isPow2(sizeOut))
    require(isPow2(sizeMinOut))
    require(sizeIn >= sizeOut)
    require(sizeOut >= sizeMinOut)

    val numLayers = log2Up(sizeIn / sizeMinOut)
    
    val layers = Vec.tabulate(numLayers + 1) { layerId => new Bundle { 
        val data       = Vec (genT(), (sizeIn >> layerId) max sizeOut)
        val num_output = UInt(log2Up(config.sizeOut + 1) bits)
    }}

    layers(0).data       := io.req.data
    layers(0).num_output := io.req.num_output

    for (layerId <- 0 until numLayers) {
        val prev_layer = layers(layerId)

        for (i <- 0 until layers(layerId + 1).data.size) {
            val next = if (i < (sizeIn >> (layerId + 1))) { 
                val reduced_result = reduceFunc(prev_layer.data(i * 2), prev_layer.data(i * 2 + 1))

                if (sizeOut <= (sizeIn >> (layerId + 1))) {
                    reduced_result
                } else {
                    Mux(prev_layer.num_output <= (sizeIn >> (layerId + 1)), 
                        reduced_result,
                        prev_layer.data(i))
                }
            } else {
                layers(layerId).data(i)
            }

            layers(layerId + 1).data(i)    := RegNextWhen(next, io.req.ready)
        }

        layers(layerId + 1).num_output := RegNextWhen(prev_layer.num_output, io.req.ready)
    }

    val delay_valid   = Delay(io.req.valid,   numLayers, io.req.ready, False)
    val delay_context = Delay(io.req.context, numLayers, io.req.ready)

    io.req.ready := !delay_valid || io.resp.ready

    io.resp.valid   := delay_valid
    io.resp.data    := layers(numLayers).data
    io.resp.context := delay_context
}
