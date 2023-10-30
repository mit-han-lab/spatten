package spatten

import spinal.core._
import spinal.lib._
import spinal.lib.fsm._

object BufferManager {
    case class Config(capacity: Int, maxReq: Int) {
        def genAddr()     = UInt(log2Up(capacity) bits)
        def genSize()     = UInt(log2Up(capacity + 1) bits)
        def genHandle()   = UInt(log2Up(maxReq) bits)
    }
    case class AllocRequest[TContext <: Data](config: Config, genContext: HardType[TContext]) extends Bundle {
        val size       = config.genSize()
        val context    = genContext()
    }
    case class AllocResponse[TContext <: Data](config: Config, genContext: HardType[TContext]) extends Bundle {
        val start_addr = config.genAddr()
        val size       = config.genSize()
        val handle     = config.genHandle()
        val context    = genContext()
    }
    case class ReleaseRequest[TContext <: Data](config: Config, genContext: HardType[TContext]) extends Bundle {
        val handle     = config.genHandle()
        val context    = genContext()
    }
    case class ReleaseResponse[TContext <: Data](config: Config, genContext: HardType[TContext]) extends Bundle {
        val handle     = config.genHandle()
        val context    = genContext()
    }
}

class BufferManager[TAllocContext <: Data, TRefContext <: Data](
    config: BufferManager.Config, 
    genContextAlloc: HardType[TAllocContext], genContextRef: HardType[TRefContext]) extends Component {

    import BufferManager._
    import config._
    
    val io = new Bundle {
        val alloc_req    = slave  Stream(AllocRequest(config, genContextAlloc))
        val alloc_resp   = master Stream(AllocResponse(config, genContextAlloc))
        val release_req  = slave  Stream(ReleaseRequest(config, genContextRef))
        val release_resp = master Stream(ReleaseResponse(config, genContextRef))
    }

    case class Tag() extends Bundle {
        val released_diff = Bool            // released = tag.released_diff ^ tag_diff.released_diff
        val start_addr    = genAddr()
        val size          = genSize()
    }
    case class TagDiff() extends Bundle {
        val released_diff = Bool
    }

    require(isPow2(capacity))

    val tags      = Mem(Tag(), maxReq)
    val tags_diff = new MultiPortRAMFactory(TagDiff(), maxReq)

    val tags_head      = Counter(0 until maxReq)
    val tags_tail      = Counter(0 until maxReq)
    val tags_occupancy = Reg(UInt(log2Up(maxReq + 1) bits)) init(0)
    val buf_head       = Reg(genAddr()) init(0)
    val buf_tail       = Reg(genAddr()) init(0)
    val buf_occupancy  = Reg(genSize()) init(0)

    val tags_head_inc = Bool
    val tags_tail_inc = Bool
    val buf_head_inc  = genSize()
    val buf_tail_inc  = genSize()
    tags_head_inc := False
    tags_tail_inc := False
    buf_head_inc  := 0
    buf_tail_inc  := 0

    when (tags_head_inc) { tags_head.increment() }
    when (tags_tail_inc) { tags_tail.increment() }
    tags_occupancy := (tags_occupancy + tags_tail_inc.asUInt - tags_head_inc.asUInt).resized

    buf_head      := (buf_head + buf_head_inc).resized
    buf_tail      := (buf_tail + buf_tail_inc).resized
    buf_occupancy := (buf_occupancy + buf_tail_inc - buf_head_inc).resized

    val areaInit = new Area {
        val inited      = RegInit(False)
        val init_handle = Reg(genHandle()) init(0)

        when (init_handle === maxReq - 1) {
            inited := True
        } otherwise {
            init_handle := init_handle + 1
        }
    }

    val areaAlloc = new Area {
        val input = io.alloc_req

        val tag_diff_at_tail = tags_diff.readSync(tags_tail.valueNext)

        when (input.fire && input.size > 0) {
            val tag = Tag()
            tag.released_diff := tag_diff_at_tail.released_diff
            tag.start_addr    := buf_tail
            tag.size          := input.size
            tags.write(tags_tail, tag)

            tags_tail_inc := True
            buf_tail_inc  := input.size
        }


        // The width of buf_occupancy + input.size is max(widthOf(buf_occupancy),widthOf(input.size)) !!!
        // USE  buf_occupancy +^ input.size

        io.alloc_resp << 
            input.haltWhen(!areaInit.inited || tags_occupancy === maxReq || buf_occupancy +^ input.size > capacity)
            .translateInto(cloneOf(io.alloc_resp)) { (to, from) => 
                to.assignSomeByName(from)
                to.handle     := tags_tail
                to.start_addr := buf_tail
            }.stage()
    }

    val areaRelease = new Area {
        val input = io.release_req

        when (input.valid) {
            // assert(input.handle >= 0 && input.handle.resized < maxReq)
            when (tags_tail > tags_head) {
                assert(input.handle >= tags_head && input.handle < tags_tail)
            } elsewhen (tags_tail < tags_head) {
                assert(input.handle >= tags_head || input.handle < tags_tail)
            } otherwise {
                assert(tags_occupancy === maxReq)
            }
        }
        
        val last_handle   = RegNextWhen(input.handle, input.fire)
        val last_fire     = RegNext(input.fire, False)
        val last_tag_diff = TagDiff()

        last_tag_diff := tags_diff.readSync(input.handle, input.fire)

        val reversed_tag  = TagDiff()
        reversed_tag.released_diff := !last_tag_diff.released_diff

        val tags_diff_write_handle = genHandle()
        val tags_diff_write_data   = TagDiff()
        val tags_diff_write_enable = Bool()

        when (!areaInit.inited) {
            tags_diff_write_handle := areaInit.init_handle
            tags_diff_write_data   := TagDiff().getZero
            tags_diff_write_enable := True
        } otherwise {
            tags_diff_write_handle := last_handle
            tags_diff_write_data   := reversed_tag
            tags_diff_write_enable := last_fire
        }

        tags_diff.write(tags_diff_write_handle, tags_diff_write_data, tags_diff_write_enable)

        io.release_resp << input.haltWhen(!areaInit.inited).translateInto(cloneOf(io.release_resp)) { (to, from) => 
            to.assignAllByName(from)
        }
    }

    val areaRecycle = new Area {
        // TODO: better to use writeFirst
        val tag_diff_at_head     = TagDiff() 
        val tag_at_head          = Tag()
        tag_diff_at_head        := tags_diff.readSync(tags_head.valueNext)
        tag_at_head             := tags.readSync(tags_head.valueNext)
        val tag_read_data_valid  = RegNext(tags_head.valueNext =/= tags_tail || !tags_tail_inc) // avoid write conflict

        when (tags_occupancy > 0) {
            when (tag_read_data_valid && (tag_at_head.released_diff ^ tag_diff_at_head.released_diff)) {
                tags_head_inc := True
                buf_head_inc  := tag_at_head.size
                assert(buf_head === tag_at_head.start_addr)
            }
        }
    }

    val tags_diff_inst = tags_diff.build()
}