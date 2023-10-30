package spatten

import spinal.core._
import spinal.lib._

object MultiBankRAM {
    case class Config[T <: Data, TWriteContext <: Data, TReadContext <: Data](
        genT: HardType[T],
        genWriteContext: HardType[TWriteContext],
        genReadContext: HardType[TReadContext],
        capacity:      Int,
        maskWidth:     Int,
        numBanks:      Int,
        numWritePorts: Int,
        numReadPorts:  Int,
        initValue: Option[() => T] = None
    ) {
        def genAddr() = UInt(log2Up(capacity) bits)
        def genMask() = Bits(maskWidth bits)
    }

    case class WriteRequest[T <: Data, TWriteContext <: Data](config: Config[T, TWriteContext, _ <: Data]) extends Bundle {
        val addr    = config.genAddr()
        val data    = config.genT()
        val mask    = config.genMask()
        val context = config.genWriteContext()
    }
    case class WriteResponse[TWriteContext <: Data](config: Config[_ <: Data, TWriteContext, _ <: Data]) extends Bundle {
        val context = config.genWriteContext()
    }

    case class ReadRequest[TReadContext <: Data](config: Config[_ <: Data, _ <: Data, TReadContext]) extends Bundle {
        val addr    = config.genAddr()
        val context = config.genReadContext()
    }
    case class ReadResponse[T <: Data, TReadContext <: Data](config: Config[T, _ <: Data, TReadContext]) extends Bundle {
        val data    = config.genT()
        val context = config.genReadContext()
    }
}

class StreamRAM[T <: Data, TWriteContext <: Data, TReadContext <: Data](
    val config: MultiBankRAM.Config[T, TWriteContext, TReadContext]) extends Component { 
    
    import MultiBankRAM._
    import config._

    val io = new Bundle {
        val write_req  = slave  Stream(WriteRequest (config))
        val write_resp = master Stream(WriteResponse(config))
        val read_req   = slave  Stream(ReadRequest  (config))
        val read_resp  = master Stream(ReadResponse (config))
    }

    val ram = Mem(genT(), capacity)

    // for simulation
    val areaCounter = new Area {
        val depth       = config.capacity
        val width       = config.genT().getBitsWidth
        val count_read  = Counter(32 bits, io.write_req.fire)
        val count_write = Counter(32 bits, io.read_req.fire)
    }
    //////

    val areaInit = if (!initValue.isEmpty) new Area {
        val inited    = Reg(Bool) init(False)
        val init_addr = Reg(genAddr()) init(0)

        when (!inited) {
            init_addr  := init_addr + 1
            when (init_addr === capacity - 1) {
                inited := True
            }
        }
    } else new Area {
        val inited    = True
        val init_addr = genAddr()
        init_addr := 0
    }

    val areaWrite = new Area {
        val input = io.write_req.haltWhen(!areaInit.inited)

        val write_addr   = input.addr
        val write_data   = input.data
        val write_enable = input.fire

        if (!initValue.isEmpty) {
            when (!areaInit.inited) {
                write_addr   := areaInit.init_addr
                write_data   := initValue.get()
                write_enable := True
            }
        }

        ram.write(write_addr, write_data, write_enable, if (maskWidth > 0) input.mask else null)

        io.write_resp << input.stage().translateInto(cloneOf(io.write_resp)) { (to, from) => 
            to.context := from.context
        }
    }

    val areaRead = new Area {
        val input = io.read_req.haltWhen(!areaInit.inited)

        val read_results = ram.readSync(input.addr, input.fire)

        io.read_resp <-< input.stage().translateInto(cloneOf(io.read_resp)) { (to, from) => 
            to.data    := read_results
            to.context := from.context
        }
    }
}


class MultiBankRAM[T <: Data, TWriteContext <: Data, TReadContext <: Data](
    config: MultiBankRAM.Config[T, TWriteContext, TReadContext]) extends Component {

    import MultiBankRAM._
    import config._

    require(isPow2(capacity))
    require(isPow2(numBanks))
    require(numWritePorts >= 1)
    require(numReadPorts >= 1)
    require(capacity % numBanks == 0)
    require(maskWidth >= 0)
    if (maskWidth > 0) require(widthOf(genT()) % maskWidth == 0)

    val io = new Bundle {
        val write_req  = Vec(slave  Stream(WriteRequest (config)), numWritePorts)
        val write_resp = Vec(master Stream(WriteResponse(config)), numWritePorts)
        val read_req   = Vec(slave  Stream(ReadRequest  (config)), numReadPorts)
        val read_resp  = Vec(master Stream(ReadResponse (config)), numReadPorts)
    }

    if (numWritePorts > 1) {
        for (resp <- io.write_resp) {
            assert(resp.ready)
        }
    }
    if (numReadPorts > 1) {
        for (resp <- io.read_resp) {
            assert(resp.ready)
        }
    }

    case class WriteContext() extends Bundle {
        val src_port = UInt(log2Up(numWritePorts) bits)
        val context  = genWriteContext()
    }
    case class ReadContext() extends Bundle {
        val src_port = UInt(log2Up(numReadPorts) bits)
        val context  = genReadContext()
    }

    val ram_insts = Array.fill(numBanks) { new StreamRAM(Config(
        genT = genT, genWriteContext = WriteContext(), genReadContext = ReadContext(),
        capacity = capacity / numBanks, 
        maskWidth = maskWidth, 
        numBanks = 1, numWritePorts = 1, numReadPorts = 1,
        initValue = initValue)) }

    val write_reqs = Vec(io.write_req.map({ req => StreamDemux(req, (req.addr % numBanks).resized, numBanks)}))
    val read_reqs  = Vec(io.read_req.map ({ req => StreamDemux(req, (req.addr % numBanks).resized, numBanks)}))

    val areaWrite = Array.tabulate(numBanks) { bankId => if (numWritePorts > 1) new Area {
        val inputs = Vec(write_reqs.map(_(bankId)))

        val arbiter = StreamArbiterFactory.roundRobin.transactionLock.build(WriteRequest(config), numWritePorts)

        (arbiter.io.inputs, inputs).zipped.foreach { _ << _ }

        ram_insts(bankId).io.write_req <-< arbiter.io.output.translateInto(cloneOf(ram_insts(bankId).io.write_req)) { (to, from) => 
            to.addr             := (from.addr / numBanks).resized
            to.data             := from.data
            to.mask             := from.mask
            to.context.src_port := arbiter.io.chosen
            to.context.context  := from.context
        }

        val outputs = Vec(StreamDemux(ram_insts(bankId).io.write_resp, ram_insts(bankId).io.write_resp.context.src_port, numWritePorts)
            .map(_.translateInto(cloneOf(io.write_resp(0))) { (to, from) => 
                to.context := from.context.context
            }))
    } else new Area {
        val inputs = Vec(write_reqs.map(_(bankId)))

        ram_insts(bankId).io.write_req << inputs(0).translateInto(cloneOf(ram_insts(bankId).io.write_req)) { (to, from) => 
            to.addr            := (from.addr / numBanks).resized
            to.data            := from.data
            to.context.context := from.context
        }

        val outputs = Vec.fill(1) { ram_insts(bankId).io.write_resp.translateInto(cloneOf(io.write_resp(0))) { (to, from) => 
            to.context := from.context.context
        }}
    }} 

    val areaRead = Array.tabulate(numBanks) { bankId => if (numReadPorts > 1) new Area {
        val inputs = Vec(read_reqs.map(_(bankId)))

        val arbiter = StreamArbiterFactory.roundRobin.transactionLock.build(ReadRequest(config), numReadPorts)

        (arbiter.io.inputs, inputs).zipped.foreach { _ << _ }

        ram_insts(bankId).io.read_req <-< arbiter.io.output.translateInto(cloneOf(ram_insts(bankId).io.read_req)) { (to, from) => 
            to.addr             := (from.addr / numBanks).resized
            to.context.src_port := arbiter.io.chosen
            to.context.context  := from.context
        }

        val outputs = Vec(StreamDemux(ram_insts(bankId).io.read_resp, ram_insts(bankId).io.read_resp.context.src_port, numReadPorts)
            .map(_.translateInto(cloneOf(io.read_resp(0))) { (to, from) => 
                to.data    := from.data
                to.context := from.context.context
            }))
    } else new Area {
        val inputs = Vec(read_reqs.map(_(bankId)))

        ram_insts(bankId).io.read_req << inputs(0).translateInto(cloneOf(ram_insts(bankId).io.read_req)) { (to, from) => 
            to.addr            := (from.addr / numBanks).resized
            to.context.context := from.context
        }

        val outputs = Vec.fill(1) { ram_insts(bankId).io.read_resp.translateInto(cloneOf(io.read_resp(0))) { (to, from) => 
            to.data    := from.data
            to.context := from.context.context
        }}
    }} 

    for (i <- 0 until numWritePorts) {
        io.write_resp(i) <-< StreamArbiterFactory.lowerFirst.transactionLock.on(areaWrite.map(_.outputs(i)))
    }
    for (i <- 0 until numReadPorts) {
        io.read_resp(i) <-< StreamArbiterFactory.lowerFirst.transactionLock.on(areaRead.map(_.outputs(i)))
    }
}
