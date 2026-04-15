import std/os
import Training

when isMainModule:
  let args = @[
    "--train", "KG" / "DATA" / "train_instance_3_train_pruned.txt",
    "--answers", "KG" / "DATA" / "train_instance_3_answers.csv",
    "--outdir", "KG" / "artifacts" / "instance_3"
  ] & commandLineParams()
  runTrainingPipeline(args)
