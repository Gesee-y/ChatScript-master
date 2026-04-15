import std/os
import Training

when isMainModule:
  let args = @[
    "--train", "KG" / "DATA" / "train_instance_4_train_pruned.txt",
    "--answers", "KG" / "DATA" / "train_instance_4_answers.csv",
    "--outdir", "KG" / "artifacts" / "instance_4"
  ] & commandLineParams()
  runTrainingPipeline(args)
