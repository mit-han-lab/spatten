package spatten

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinal.core.sim._

import sys.process._
import scala.concurrent._
import scala.collection.mutable.Queue

import scala.util.Random

import java.io.File

case class DRAMSimConfig(
    val addressWidth: Int,
    val dataWidth: Int) {}

case class DRAMSimTransaction(config: DRAMSimConfig) extends Bundle {
    val addr     = UInt(config.addressWidth bits)
    val data     = Bits(config.dataWidth bits)
    val is_write = Bool
}
case class DRAMSim(config: DRAMSimConfig) extends Bundle with IMasterSlave {
    val req  = Stream(DRAMSimTransaction(config))
    val resp = Stream(DRAMSimTransaction(config))

    override def asMaster(): Unit = {
        master(req)
        slave(resp)
    }

    def fromAxi4(bus: Axi4ReadOnly) = {
        val fifo_id = StreamFifo(cloneOf(bus.ar.id), 64)

        val ar_forked = StreamFork2(bus.ar)

        ar_forked._1.translateInto(fifo_id.io.push) { (to, from) => to := from.id }

        ar_forked._2.translateInto(req) { (to, from) => 
            to.addr := from.addr
            to.data.assignDontCare()
            to.is_write := False
        }
        
        StreamJoin.arg(fifo_id.io.pop, resp).translateInto(bus.r) { (to, from) => 
            to.data := resp.data
            to.last := True
            to.id   := fifo_id.io.pop.payload
        }
    }
}

class DRAMSimDPIDriver(val config: DRAMSimConfig) extends BlackBox {
    val io = new Bundle {
        val clk = in Bool()
        val rst = in Bool()
        val dram = slave(DRAMSim(config))
    }
    addGeneric("addressWidth", config.addressWidth)
    addGeneric("dataWidth", config.dataWidth)
    noIoPrefix()
    mapCurrentClockDomain(clock = io.clk, reset = io.rst)
}

class DRAMSimDriver(val id: String = "") {
    val transactionLength = 32

    private var inputStream: java.io.InputStream = null
    private var outputStream: java.io.OutputStream = null
    private var process: Process = null

    def init(): Unit = {
        val inputStreamSync = new SyncVar[java.io.InputStream]
        val outputStreamSync = new SyncVar[java.io.OutputStream]
        val io = new ProcessIO(stdin => outputStreamSync.put(stdin), stdout => inputStreamSync.put(stdout), stderr => ())
        if (id != "") {
            s"mkdir -p ./dram-${id}".!
            s"cp ./ramulator ./dram-${id}/".!
            s"cp ./HBM-config.cfg ./dram-${id}/".!
            process = Process(Seq("./ramulator", "HBM-config.cfg", "mode=interactive"), new File(s"./dram-${id}")).run(io)
        } else {
            process = Process(Seq("./ramulator", "HBM-config.cfg", "mode=interactive")).run(io)
        }
        inputStream = inputStreamSync.get
        outputStream = outputStreamSync.get
    }

    def exit(): Unit = {
        if (process == null) return
        outputStream.close()
        inputStream = null
        outputStream = null
        process = null
    }

    def drive(bus: DRAMSim, clkDomain: ClockDomain): Unit = {
        assert(process != null)
        assert(process.isAlive())

        val lineIter = scala.io.Source.fromInputStream(inputStream).getLines

        val data: Map[Long, BigInt] = Map()

        val dramThread = fork {
            bus.req.ready  #= false
            bus.resp.valid #= false
            bus.resp.addr  #= 0
            bus.resp.data  #= 0

            val addrFIFO = new Queue[Long]
            val dataFIFO = new Queue[Long]

            while (true) {
                clkDomain.waitSampling()

                val fifoDepth = 8    

                val req_valid  = bus.req.valid.toBoolean
                val resp_ready = bus.resp.ready.toBoolean
                val req_addr   = bus.req.addr.toLong
                val req_write  = bus.req.is_write.toBoolean

                if (resp_ready && !dataFIFO.isEmpty) {
                    dataFIFO.dequeue()
                }
                if (req_valid && addrFIFO.size < fifoDepth) {
                    addrFIFO.enqueue(req_addr)
                }

                assert(!req_valid || req_addr % transactionLength == 0, "Requested address must be aligned to transaction length")

                outputStream.write("M ".getBytes())
                if (dataFIFO.size < fifoDepth) 
                    outputStream.write("R ".getBytes())
                else
                    outputStream.write("- ".getBytes())
                if (!addrFIFO.isEmpty) {
                    outputStream.write("V 0x".getBytes())
                    outputStream.write(addrFIFO.front.toHexString.getBytes())
                    // if (req_write)
                    //     outputStream.write(" W\n".getBytes())
                    // else
                    outputStream.write(" R\n".getBytes())
                } else {
                    outputStream.write("\n".getBytes())
                }
                outputStream.flush()

                // println(s"WAITING... ${req_addr.toHexString} ${req_valid}")

                val line = lineIter.next().split(" +")

                // println("DONE...")


                assert(line(0) == "S")

                if (line(1) == "R") {
                    addrFIFO.dequeue()
                }
                if (line(2) == "V" && dataFIFO.size < fifoDepth) {
                    dataFIFO.enqueue(java.lang.Long.decode(line(3)))
                }

                // bus.req.ready     #= line(1) == "R"
                // bus.resp.valid    #= line(2) == "V"
                // if (line(2) == "V") {
                //     bus.resp.addr     #= 
                //     bus.resp.is_write #= line(4) == "W"
                //     bus.resp.data.randomize
                // }
                bus.req.ready  #= addrFIFO.size < fifoDepth
                bus.resp.valid #= !dataFIFO.isEmpty
                if (!dataFIFO.isEmpty) {
                    val rng = new Random
                    val addr = dataFIFO.front
                    rng.setSeed(addr)
                    bus.resp.addr  #= addr
                    bus.resp.data  #= BigInt(bus.resp.data.getBitsWidth, Random)
                }
            }
        }
    }
}