name := "spatten"

version := "1.0"

scalaVersion := "2.13.12"
val spinalVersion = "1.9.3"

libraryDependencies ++= Seq(
    "com.github.spinalhdl" % "spinalhdl-core_2.13" % spinalVersion,
    "com.github.spinalhdl" % "spinalhdl-lib_2.13" % spinalVersion,
    compilerPlugin("com.github.spinalhdl" % "spinalhdl-idsl-plugin_2.13" % spinalVersion),
    "com.github.tototoshi" % "scala-csv_2.13" % "1.3.10",
    "org.scalatest" % "scalatest_2.13" % "3.2.12" % "test",
)

ThisBuild / assemblyMergeStrategy := {
    case PathList("META-INF", _*) => MergeStrategy.discard
    case _                        => MergeStrategy.first
}

fork := true