package spatten.sim

import scala.util.Random
import scala.math._

object TopKLatencyModel {
    val rng = new Random
    rng.setSeed(1)

    def topKLatency(size: Int, target: Int, parallelism: Int) = {

        def divide(input: Seq[Int]): (Seq[Int], Seq[Int], Int) = {
            val pivot = input(rng.nextInt(input.size))
            (input.filter(_ < pivot), input.filter(_ > pivot), input.count(_ == pivot))
        }

        var seq = Seq.fill(size) { rng.nextInt() }

        var cycleCount = 0
        var currentTarget = target
        var finish = false

        while (!finish && seq.size > 0) {
            cycleCount += (seq.size + parallelism - 1) / parallelism + 3
            val (l, r, cnt) = divide(seq)
            if (l.size <= currentTarget) {
                if (l.size + cnt > currentTarget) {
                    finish = true
                } else {
                    currentTarget -= l.size + cnt
                    seq = r
                }
            } else {
                seq = l
            }
        }

        cycleCount
    }

    def main(args: Array[String]): Unit = {
        val data = Seq(
            (1024, 512),
            (64,32   ),
            (64,32   ),
            (64,16   ),
            (128,64  ),
            (128,128 ),
            (128,64  ),
            (128,64  ),
            (256,32  ),
            (256,32  ),
            (256,16  ),
            (512,64  ),
            (512,64  ),
            (512,128 ),
            (512,64  ),
            (1024,128),
            (1024,64 ),
            (1024,32 ),
            (1024,32 ),
            (1024,32 ),
            (1024,16 ),
            (2048,64 ),
            (2048,128),
            (2048,64 ),
            (2048,32 ),
            (4096,32 ),
            (4096,32 ),
            (4096,16 ),
        )
        for ((size, target) <- data) {
            val latency = for (i <- 0 until 100) yield {
                topKLatency(size, target, 16)
            }
            val avgLatency = latency.sum.toDouble / latency.size
            val stdDev     = sqrt(latency.map(x => (x - avgLatency) * (x - avgLatency)).sum.toDouble / latency.size)
            println(s"${size},${target},${avgLatency},${stdDev}")
        }
    }
}