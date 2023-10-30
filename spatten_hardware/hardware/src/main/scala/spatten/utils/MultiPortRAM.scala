package spatten

import spinal.core._
import spinal.lib._
import scala.collection.mutable.ArrayBuffer

class MultiPortRAMFactory[T <: Data](genT: HardType[T], size: Int) {
    
    def genAddr() = UInt(log2Up(size) bits)

    case class ReadPort(val useEnable: Boolean = false, val readUnderWrite: ReadUnderWritePolicy = dontCare) extends Bundle with IMasterSlave {
        val address = genAddr()
        val data    = genT()
        val enable  = Bool genIf(useEnable)

        override def asMaster(): Unit = {
            out(address)
            in(data)
            if (useEnable) out(enable)
        }
    }

    case class WritePort(val useEnable: Boolean = false, val widthMask: Int = 0) extends Bundle {
        val address = genAddr()
        val data    = genT()
        val enable  = Bool genIf(useEnable)
        val mask    = Bits(widthMask bits) genIf(widthMask > 0)
    }

    val readPorts  = new ArrayBuffer[ReadPort]()
    val writePorts = new ArrayBuffer[WritePort]()

    val attributes = new ArrayBuffer[Attribute]()
    var initContent: Seq[T] = null

    def addRAMAttribute(attribute: Attribute): this.type = {
        attributes += attribute
        this
    }
    
    def init(initialContent: Seq[T]): this.type = {
        initContent = initialContent
        this
    }

    def write(address: UInt, data: T, enable: Bool = null, mask: Bits = null): Unit = {
        val port = WritePort(true, if (mask == null) 0 else mask.getWidth)

        port.address := address
        port.data    := data
        if (enable != null) {
            port.enable := enable
        } else {
            port.enable := ConditionalContext.isTrue
        }
        if (port.widthMask > 0) port.mask := mask

        writePorts += port
    }

    def readSync(address: UInt, enable: Bool = null, readUnderWrite: ReadUnderWritePolicy = dontCare): T = {
        val port = ReadPort(enable != null, readUnderWrite)

        port.address := address
        if (port.useEnable) port.enable := enable
        
        readPorts += port

        port.data
    }

    class MultiPortRAM extends Component {
        val io = new Bundle {
            val read_ports  = readPorts  map { x => slave(new ReadPort(x.useEnable, x.readUnderWrite)) } toArray
            val write_ports = writePorts map { x => in(new WritePort(x.useEnable, x.widthMask)) } toArray
        }

        // println("#ReadPorts=" + io.read_ports.size)

        require(io.write_ports.size <= 1, "Multiple write ports are currently not supported.")

        val mems = Array.fill(io.read_ports.size) { Mem(genT(), size) }
        
        for (port <- io.write_ports) {
            mems.foreach { _.write(
                port.address, 
                port.data, 
                if (port.useEnable)     port.enable else null, 
                if (port.widthMask > 0) port.mask   else null)
            }
        }

        for (portId <- 0 until io.read_ports.size) {
            val port = io.read_ports(portId)
            port.data := mems(portId).readSync(
                port.address,
                if (port.useEnable) port.enable else null,
                port.readUnderWrite)
        }

        if (initContent != null) {
            mems.foreach { _.init(initContent) }
        }
        for (attr <- attributes) {
            mems.foreach { _.addAttribute(attr) }
        }
        
    }

    def build(): MultiPortRAM = {
        val inst = new MultiPortRAM()

        for ((x, y) <- readPorts zip inst.io.read_ports) {
            y <> x
        }
        for ((x, y) <- writePorts zip inst.io.write_ports) {
            y := x
        }

        inst
    }

}

// class MultiReadRAMWrapper() extends Component {

//     val io = new Bundle {
//         val addr = in UInt(9 bits)
//         val data = in UInt(64 bits)
//         val odata1 = out UInt(64 bits)
//         val odata2 = out UInt(64 bits)
//         // val odata3 = out UInt(64 bits)
//     }
//     val addr_counter = Reg(UInt(9 bits)) init(0)
//     val ram = new MultiReadRAM(UInt(64 bits), 512, 2)
//     addr_counter := addr_counter + 1

//     ram.write(io.addr, io.data)
//     io.odata1 := ram.readSync(0, addr_counter)
//     io.odata2 := ram.readSync(1, io.addr, readUnderWrite=readFirst)
//     // io.odata3 := ram.readSync(1, io.addr)
//     ram.addAttribute(new AttributeString("style", "distributed"))
//     // ram.setTechnology(distributedLut)
//     // ram.generateAsBlackBox()
// }

