package spatten

import spinal.core._
import spinal.lib._

object PrefixSum {
    sealed trait Structure 
    case object Parallel extends Structure
    
    sealed trait RegisterPolicy
    case object LowLatency   extends RegisterPolicy
    case object MaxFrequency extends RegisterPolicy

    case class Config[T <: Data, TContext <: Data](
        genT:       HardType[T], 
        genContext: HardType[TContext], 
        size:       Int, 
        funcAdd:    (T, T) => T, 
        structure:  Structure = Parallel,
        policy:     RegisterPolicy = LowLatency
    )

    case class Request[T <: Data, TContext <: Data](config: Config[T, TContext]) extends Bundle {
        val data    = Vec(config.genT(), config.size)
        val context = config.genContext()
    }
    case class Response[T <: Data, TContext <: Data](config: Config[T, TContext]) extends Bundle {
        val data    = Vec(config.genT(), config.size)
        val context = config.genContext()
    }
}

class PrefixSum[T <: Data, TContext <: Data](val config: PrefixSum.Config[T, TContext]) extends Component {
    import PrefixSum._
    import config._
    
    val io = new Bundle {
        val req  = slave  Stream(Request (config))
        val resp = master Stream(Response(config))
    }

    require(size > 0)

    val numLayers = log2Up(size)
    val layers        = Vec(Vec(genT(), size), numLayers + 1)
    val layers_output = Vec(Vec(genT(), size), numLayers)

    var numRegisterLayers = 0

    layers(0) := io.req.data

    structure match {
        case Parallel => 
            for (layerId <- 0 until numLayers) {
                for (i <- 0 until size) {
                    if (i < (1 << layerId)) {
                        layers_output(layerId)(i) := layers(layerId)(i)
                    } else {
                        layers_output(layerId)(i) := funcAdd(layers(layerId)(i - (1 << layerId)), layers(layerId)(i))
                    }
                }
            }
            for (layerId <- 0 until numLayers) {
                if (policy == MaxFrequency || layerId == numLayers - 1) {
                    layers(layerId + 1) := RegNextWhen(layers_output(layerId), io.req.ready)
                    numRegisterLayers += 1
                } else {
                    layers(layerId + 1) := layers_output(layerId)
                }
            }
        case _ => throw new IllegalArgumentException("Unsupported structure for PrefixSum")
    }

    // println(s"policy = ${policy}, numRegisterLayers = ${numRegisterLayers}")

    val delay_valid   = Delay(io.req.valid, numRegisterLayers, io.req.ready, False)
    val delay_context = Delay(io.req.context, numRegisterLayers, io.req.ready)

    io.req.ready := !delay_valid || io.resp.ready

    io.resp.valid   := delay_valid
    io.resp.data    := layers(numLayers)
    io.resp.context := delay_context
}

