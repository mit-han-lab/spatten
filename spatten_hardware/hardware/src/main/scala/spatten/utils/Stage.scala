package spatten

import spinal.core._
import spinal.lib._
import scala.collection.mutable.ArrayBuffer

trait StageImplementation {
    def stageInDef[T <: Data](that: HardType[T]): T
    def stageIn[T <: Data](that: T): T
    def stageOutDef[T <: Data](that: HardType[T]): T
    def stageOut[T <: Data](that: T): T
    def stageDrive[T <: Data](that: T): T
}

trait Stageable extends Component {
    val stageTasks = ArrayBuffer[() => Unit]()
    def addStageTask(task: () => Unit) {
        stageTasks += task
    }
    addPrePopTask({ () =>
        for (task <- stageTasks) {
            task()
        }
    })
}

abstract class StageComponent(name: String = "") extends Component with StageImplementation {

    if (name != "") {
        this.definitionName = 
            if (parent.definitionName != null && parent.definitionName.nonEmpty) {
                s"${parent.name}_${name}"
            } else {
                name
            }
    }

    override def stageInDef[T <: Data](that: HardType[T]): T = {
        val data = that()
        data match {
            case p: IMasterSlave => 
                slave(data.asInstanceOf[T with IMasterSlave])
            case _ =>
                in(data)
        }
    }

    override def stageIn[T <: Data](that: T): T = {
        val data = stageInDef(that)
        parent.asInstanceOf[Stageable].addStageTask { () =>
            data <> that
        }
        data
    }

    override def stageOutDef[T <: Data](that: HardType[T]): T = {
        val data = that()
        data match {
            case p: IMasterSlave => 
                master(data.asInstanceOf[T with IMasterSlave])
            case _ =>
                out(data)
        }
    }

    override def stageOut[T <: Data](that: T): T = {
        val data = stageOutDef(that)
        data <> that
        data
    }

    override def stageDrive[T <: Data](that: T): T = {
        val data = stageOutDef(that)
        parent.asInstanceOf[Stageable].addStageTask { () =>
            data <> that
        }
        data
    }
}

abstract class StageArea(name: String = "") extends Area with StageImplementation {
    override def stageInDef[T <: Data](that: HardType[T]): T = that()
    override def stageIn[T <: Data](that: T): T = that
    override def stageOutDef[T <: Data](that: HardType[T]): T = that()
    override def stageOut[T <: Data](that: T): T = that
    override def stageDrive[T <: Data](that: T): T = that
}