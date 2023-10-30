package spatten

import spinal.core._
import spinal.lib._
import scala.util.Random

object HashRandomMatrix {
    def apply(seed: Int)(input: Bits, bitCountOfOutput: BitCount): UInt = {
        val widthInput  = input.getBitsWidth
        val widthOutput = bitCountOfOutput.value

        val randomMatrix = Array.ofDim[Boolean](widthOutput, widthInput)
        val rng = new Random(seed)

        for (i <- 0 until widthOutput; j <- 0 until widthInput) randomMatrix(i)(j) = rng.nextBoolean()

        val inst = new Component {
            setDefinitionName("HashRandomMatrixSeed" + seed)
            setPartialName("HashRandomMatrixSeed" + seed)

            val io = new Bundle {
                val input  = in Bits(widthInput bits)
                val output = out UInt(widthOutput bits)
            }

            for (i <- 0 until widthOutput) {
                var result = False
                for (j <- 0 until widthInput) {
                    if (randomMatrix(i)(j)) {
                        result \= result ^ io.input(j)
                    }
                }
                io.output(i) := result
            }
        }

        inst.io.input := input
        inst.io.output
    }
}