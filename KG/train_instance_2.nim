import std/os
import Training

when isMainModule:
  let args = @[
    "--train", "KG" / "DATA" / "train_instance_2_train_pruned.txt",
    "--answers", "KG" / "DATA" / "train_instance_2_answers.csv",
    "--outdir", "KG" / "artifacts" / "instance_2"
  ] & commandLineParams()
  runTrainingPipeline(args)
